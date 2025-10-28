import os
import time
import redis
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Enum, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import uuid
import logging
import enum
import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import fcntl
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Load Environment ---
log.info("Worker starting up, loading environment...")
load_dotenv(".env")
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "/data/faiss_index")

if not DATABASE_URL or not REDIS_URL:
    log.error("DATABASE_URL and REDIS_URL must be set.")
    exit(1)

# --- Code Duplication Warning ---
# The following classes (Settings, DB Models, Vector Store) are duplicated
# from the `backend` app. In a larger project, this would be a shared
# 'common' Python package. For this minimal structure, duplication
# keeps the 'worker' and 'backend' services independently runnable.

# --- Config Mock ---
class Settings:
    DATABASE_URL: str = DATABASE_URL
    REDIS_URL: str = REDIS_URL
    EMBEDDING_MODEL_NAME: str = EMBEDDING_MODEL_NAME
    VECTOR_STORE_PATH: str = VECTOR_STORE_PATH
settings = Settings()

# --- Database Setup ---
try:
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except Exception as e:
    log.error(f"Worker failed to create DB engine: {e}")
    exit(1)

class IngestionStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Url(Base):
    __tablename__ = "urls"
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, index=True)
    status = Column(Enum(IngestionStatus), default=IngestionStatus.PENDING)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(Integer, primary_key=True, index=True)
    url_id = Column(Integer, index=True) 
    chunk_text = Column(Text)
    vector_id = Column(String, unique=True) 

# Create tables if they don't exist
try:
    Base.metadata.create_all(bind=engine)
    log.info("Worker verified database tables.")
except Exception as e:
    log.warning(f"Worker had an issue creating/verifying tables (may be fine): {e}")

# --- CRUD Functions ---
from sqlalchemy.orm import Session
def get_db_session():
    return SessionLocal()

def get_url_by_url(session: Session, url: str):
    return session.query(Url).filter(Url.url == url).first()

def update_url_status(session: Session, url_str: str, status: IngestionStatus):
    try:
        db_url = get_url_by_url(session, url_str)
        if db_url:
            db_url.status = status
            session.commit()
            session.refresh(db_url)
            log.info(f"Updated URL {url_str} to status {status}")
        return db_url
    except Exception as e:
        log.error(f"Error updating URL status for {url_str}: {e}")
        session.rollback()
        return None

def add_document_chunk(session: Session, url_id: int, chunk_text: str, vector_id: str):
    try:
        db_chunk = DocumentChunk(
            url_id=url_id,
            chunk_text=chunk_text,
            vector_id=vector_id
        )
        session.add(db_chunk)
        session.commit()
        session.refresh(db_chunk)
        return db_chunk
    except Exception as e:
        log.error(f"Error adding document chunk: {e}")
        session.rollback()
        return None

# --- Vector Store Functions ---
_model = None

class FileLock:
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.handle = None
    def __enter__(self):
        self.handle = open(self.lock_file, 'w')
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        return self.handle
    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()

def get_embedding_model():
    global _model
    if _model is None:
        log.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        log.info("Embedding model loaded.")
    return _model

def get_index_paths():
    base_path = settings.VECTOR_STORE_PATH
    index_path = f"{base_path}.index"
    meta_path = f"{base_path}.meta"
    lock_path = f"{base_path}.lock"
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    return index_path, meta_path, lock_path

def load_index():
    index_path, meta_path, _ = get_index_paths()
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            index_to_vector_id = pickle.load(f)
    else:
        model = get_embedding_model()
        dimension = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)
        index_to_vector_id = {}
    return index, index_to_vector_id

def save_index(index, index_to_vector_id):
    index_path, meta_path, _ = get_index_paths()
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(index_to_vector_id, f)

def add_to_index(vectors: np.ndarray, vector_ids: list[str]):
    _, _, lock_path = get_index_paths()
    with FileLock(lock_path):
        log.info("Worker acquired lock to add to FAISS index...")
        index, index_to_vector_id = load_index()
        start_index = index.ntotal
        index.add(vectors)
        for i, vec_id in enumerate(vector_ids):
            index_to_vector_id[start_index + i] = vec_id
        save_index(index, index_to_vector_id)
        log.info(f"Worker added {len(vectors)} vectors. Total size: {index.ntotal}")

def embed_text(text: str | list[str]):
    model = get_embedding_model()
    return model.encode(text, convert_to_tensor=False)

# --- End of Duplicated Code ---


# --- Worker Core Logic ---

def fetch_and_clean_text(url: str) -> str | None:
    """Fetches a URL and returns its clean, textual content."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script, style, nav, footer
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        
        text = soup.get_text()
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        log.info(f"Fetched and cleaned {url}. Text length: {len(text)}")
        return text
    except Exception as e:
        log.error(f"Failed to fetch or clean URL {url}: {e}")
        return None

def process_url_job(url: str):
    """The main processing pipeline for a single URL."""
    db = get_db_session()
    try:
        log.info(f"Processing URL: {url}")
        db_url = update_url_status(db, url, IngestionStatus.PROCESSING)
        if not db_url:
            log.error(f"URL not found in DB, skipping: {url}")
            return

        # 1. Fetch and Clean
        text = fetch_and_clean_text(url)
        if not text:
            update_url_status(db, url, IngestionStatus.FAILED)
            return

        # 2. Chunk
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        if not chunks:
            log.warning(f"No text chunks extracted from {url}")
            update_url_status(db, url, IngestionStatus.COMPLETED) # No error, just no content
            return
        log.info(f"Split {url} into {len(chunks)} chunks.")

        # 3. Embed
        embeddings = embed_text(chunks)
        
        # 4. Store in DB and Vector Store
        vector_ids = [uuid.uuid4().hex for _ in chunks]
        
        for i, chunk_text in enumerate(chunks):
            add_document_chunk(
                session=db,
                url_id=db_url.id,
                chunk_text=chunk_text,
                vector_id=vector_ids[i]
            )
        
        add_to_index(embeddings, vector_ids)

        # 5. Mark as Completed
        update_url_status(db, url, IngestionStatus.COMPLETED)
        log.info(f"Successfully processed URL: {url}")

    except Exception as e:
        log.error(f"Unhandled exception processing {url}: {e}", exc_info=True)
        update_url_status(db, url, IngestionStatus.FAILED)
    finally:
        db.close()

def main_loop():
    """The main worker loop, listens to Redis for jobs."""
    log.info("Worker connecting to Redis...")
    redis_client = None
    while True:
        if not redis_client:
            try:
                redis_client = redis.from_url(REDIS_URL, decode_responses=True)
                redis_client.ping()
                log.info("Worker connected to Redis.")
            except Exception as e:
                log.error(f"Worker failed to connect to Redis: {e}. Retrying in 5s...")
                time.sleep(5)
                continue
        
        try:
            # Blocking pop from the right (FIFO with lpush)
            log.info("Worker listening for jobs on 'url_ingestion_queue'...")
            _, url = redis_client.brpop("url_ingestion_queue")
            if url:
                process_url_job(url)
        except redis.exceptions.ConnectionError:
            log.error("Redis connection lost. Reconnecting...")
            redis_client = None
            time.sleep(5)
        except Exception as e:
            log.error(f"An error occurred in the main worker loop: {e}", exc_info=True)
            time.sleep(5) # Avoid rapid-fire errors

if __name__ == "__main__":
    main_loop()
