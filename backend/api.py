import uvicorn
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from sqlmodel import Session, select
from contextlib import asynccontextmanager
import numpy as np

# Import core components
from rag_core import (
    get_session,
    create_db_and_tables,
    redis_client,
    QUEUE_NAME,
    IngestionJob,
    Chunk,
    embedding_model,
    get_faiss_index,
    get_grounded_answer,
    llm,
    embedding_model
)

# --- FastAPI App Initialization ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    logging.info("FastAPI startup: Initializing database...")
    create_db_and_tables()
    
    # Check for models
    if redis_client is None:
        logging.error("Redis client not available. Ingestion will fail.")
    if llm is None:
        logging.error("Gemini LLM not available. Querying will fail.")
    if embedding_model is None:
        logging.error("Embedding model not available. All operations will fail.")
        
    logging.info("FastAPI startup complete.")
    yield
    # Code to run on shutdown
    logging.info("FastAPI shutdown.")

app = FastAPI(lifespan=lifespan)
logging.basicConfig(level=logging.INFO)

# --- API Models ---

class IngestRequest(BaseModel):
    url: str

class IngestResponse(BaseModel):
    message: str
    job_id: int

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

# --- API Endpoints ---

@app.post("/ingest-url", status_code=202, response_model=IngestResponse)
async def ingest_url(request: IngestRequest, session: Session = Depends(get_session)):
    """
    Accepts a URL for ingestion.
    Creates a job in the DB and pushes it to the Redis queue.
    """
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Queueing service (Redis) is not available.")

    # 1. Check if URL already exists and its status
    existing_job = session.exec(
        select(IngestionJob).where(IngestionJob.url == request.url)
    ).first()

    if existing_job:
        if existing_job.status in ["pending", "processing"]:
            raise HTTPException(status_code=400, detail="This URL is already processing.")
        elif existing_job.status == "completed":
            # Re-queue it? For this demo, we'll just say it's done.
            # For a real app, you might want to allow re-ingestion.
            return IngestResponse(
                message="URL has already been ingested successfully.", 
                job_id=existing_job.id
            )
        # If 'failed', we'll create a new job, which will overwrite
        # We can also just re-use the old job
        job = existing_job
        job.status = "pending" # Reset status
        job.error_message = None
    else:
        # 2. Create a new job
        job = IngestionJob(url=request.url, status="pending")
        session.add(job)

    session.commit()
    session.refresh(job)

    # 3. Push job ID to Redis queue
    try:
        redis_client.rpush(QUEUE_NAME, str(job.id))
        logging.info(f"Queued job {job.id} for URL: {request.url}")
    except Exception as e:
        logging.error(f"Failed to queue job {job.id}: {e}")
        job.status = "failed"
        job.error_message = "Failed to queue job"
        session.commit()
        raise HTTPException(status_code=500, detail="Failed to queue ingestion job.")

    return IngestResponse(message="URL accepted for ingestion", job_id=job.id)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, session: Session = Depends(get_session)):
    """
    Answers a query based on the ingested knowledge base.
    """
    if embedding_model is None or llm is None:
        raise HTTPException(status_code=503, detail="Core services (Embedding or LLM) are not available.")

    try:
        # 1. Load the FAISS index
        index = get_faiss_index()
        if index.ntotal == 0:
            raise HTTPException(status_code=404, detail="Knowledge base is empty. Please ingest content first.")

        # 2. Embed the query
        logging.info(f"Embedding query: {request.query}")
        query_embedding = embedding_model.encode([request.query]).astype('float32')

        # 3. Search FAISS
        K = 5 # Number of chunks to retrieve
        distances, indices = index.search(query_embedding, K)
        
        vector_ids = [int(i) for i in indices[0]]
        if not vector_ids:
            raise HTTPException(status_code=404, detail="No relevant information found.")

        # 4. Retrieve chunks and their job (source URL) from
        # We use a join to get the chunk text and the source URL in one go
        results = session.exec(
            select(Chunk, IngestionJob)
            .join(IngestionJob)
            .where(Chunk.vector_id.in_(vector_ids))
        ).all()

        if not results:
             raise HTTPException(status_code=404, detail="Could not retrieve chunk metadata.")

        # 5. Format context and get sources
        context = ""
        sources = set()
        for chunk, job in results:
            context += f"Source: {job.url}\nContent: {chunk.chunk_text}\n\n"
            sources.add(job.url)
        
        logging.info(f"Context retrieved from {len(sources)} sources.")

        # 6. Call LLM for grounded answer
        answer = get_grounded_answer(request.query, context)

        return QueryResponse(answer=answer, sources=list(sources))

    except Exception as e:
        logging.error(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
