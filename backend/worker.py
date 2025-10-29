import logging
import time
from datetime import datetime
from sqlmodel import Session, select
import numpy as np
import sys
import redis

# Import core components
from rag_core import (
    redis_client,
    get_session,
    engine,
    QUEUE_NAME,
    IngestionJob,
    Chunk,
    fetch_and_clean_text,
    chunk_text,
    embed_texts,
    get_faiss_index,
    save_faiss_index
)

# --- Worker Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - WORKER - %(levelname)s - %(message)s')

# --- Main Worker Functions ---

def process_job(job_id: int):
    """
    Processes a single ingestion job.
    Fetches, chunks, embeds, and stores the content.
    """
    logging.info(f"Processing job {job_id}...")
    
    # We use a new session per job
    with Session(engine) as session:
        try:
            # 1. Get job and set status to 'processing'
            job = session.get(IngestionJob, job_id)
            if not job:
                logging.error(f"Job {job_id} not found in database.")
                return
            
            job.status = "processing"
            job.updated_at = datetime.utcnow()
            session.add(job)
            session.commit()
            session.refresh(job)

            # 2. Fetch and clean text
            logging.info(f"Fetching {job.url}...")
            text = fetch_and_clean_text(job.url)
            if not text:
                raise Exception("No text content found.")

            # 3. Chunk text
            logging.info("Chunking text...")
            chunks = chunk_text(text)
            if not chunks:
                raise Exception("Failed to chunk text.")

            # 4. Embed chunks
            # Note: This is CPU/GPU intensive
            embeddings = embed_texts(chunks)

            # 5. Load FAISS index
            # This logic needs to be robust. For a single worker,
            # loading, modifying, and saving is fine.
            # For multiple workers, you'd need a lock or a dedicated vector DB.
            index = get_faiss_index()
            start_vector_id = index.ntotal
            
            # 6. Add new data to DB and FAISS
            logging.info(f"Adding {len(chunks)} chunks to DB and FAISS...")
            new_chunks_to_db = []
            for i, chunk in enumerate(chunks):
                vector_id = start_vector_id + i
                new_chunk = Chunk(
                    job_id=job.id,
                    chunk_text=chunk,
                    vector_id=vector_id
                )
                new_chunks_to_db.append(new_chunk)

            session.add_all(new_chunks_to_db)
            
            # Add to FAISS index
            index.add(embeddings)
            
            # 7. Save index and commit DB changes
            save_faiss_index(index)
            
            job.status = "completed"
            job.updated_at = datetime.utcnow()
            session.add(job)
            session.commit()
            
            logging.info(f"Job {job_id} completed successfully.")

        except Exception as e:
            # Handle failure
            logging.error(f"Job {job_id} failed: {e}", exc_info=True)
            if 'job' in locals():
                job.status = "failed"
                job.error_message = str(e)
                job.updated_at = datetime.utcnow()
                session.add(job)
                session.commit()

def main_loop():
    """
    The main worker loop, continuously polling Redis for jobs.
    """
    if redis_client is None:
        logging.critical("Redis client is not available. Worker cannot start.")
        sys.exit(1)
        
    logging.info("Worker started. Waiting for jobs...")
    while True:
        try:
            # blpop is 'blocking list pop'. It waits until an item is available.
            # timeout=0 means wait forever
            packed_job = redis_client.blpop([QUEUE_NAME], timeout=0)
            
            # blpop returns a tuple (queue_name, item_bytes)
            job_id_bytes = packed_job[1]
            job_id = int(job_id_bytes.decode('utf-8'))
            
            logging.info(f"Received job {job_id}.")
            process_job(job_id)

        except redis.exceptions.ConnectionError as e:
            logging.error(f"Redis connection error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            # Don't let one failed job kill the worker
            time.sleep(1)

if __name__ == "__main__":
    main_loop()
