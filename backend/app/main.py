import google.generativeai as genai
import redis
import logging
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session

from . import crud, models, database, vector_store
from .config import settings

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- App Initialization ---
log.info("Starting FastAPI application...")

# Initialize database tables
database.init_db()

app = FastAPI(title="Web RAG Engine")

# --- Connect to Redis ---
try:
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    redis_client.ping()
    log.info("Connected to Redis successfully!")
except Exception as e:
    log.error(f"Failed to connect to Redis: {e}")
    redis_client = None

# --- Configure Gemini ---
try:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    generation_model = genai.GenerativeModel('gemini-2.5-flash')
    log.info("Gemini model configured.")
except Exception as e:
    log.error(f"Failed to configure Gemini: {e}")
    generation_model = None

# --- API Endpoints ---

@app.post("/ingest-url", response_model=models.IngestResponse, status_code=status.HTTP_202_ACCEPTED)
def ingest_url(request: models.IngestRequest, db: Session = Depends(database.get_db)):
    """
    Accepts a URL for ingestion.
    Adds it to the Redis queue for background processing.
    """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection not available")

    url_str = str(request.url)
    db_url = crud.get_url_by_url(db, url_str)

    # Check if already processed or in progress
    if db_url:
        if db_url.status == database.IngestionStatus.COMPLETED:
            log.info(f"URL already processed: {url_str}")
            return models.IngestResponse(
                message="URL already processed", url=url_str, status=db_url.status
            )
        if db_url.status in [database.IngestionStatus.PENDING, database.IngestionStatus.PROCESSING]:
            log.info(f"URL ingestion in progress: {url_str}")
            return models.IngestResponse(
                message="URL ingestion in progress", url=url_str, status=db_url.status
            )
        # If FAILED, we can allow re-queueing
        
    # Create new entry if it doesn't exist (or was FAILED)
    if not db_url:
        db_url = crud.create_url_entry(db, url_str)
    
    # Push to Redis queue
    try:
        redis_client.lpush("url_ingestion_queue", url_str)
        log.info(f"Pushed URL to queue: {url_str}")
    except Exception as e:
        log.error(f"Failed to push to Redis queue: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to push to Redis queue: {e}")

    return models.IngestResponse(
        message="URL accepted for ingestion", 
        url=url_str, 
        status=database.IngestionStatus.PENDING
    )

@app.post("/query", response_model=models.QueryResponse)
def query_index(request: models.QueryRequest, db: Session = Depends(database.get_db)):
    """
    Queries the ingested knowledge base.
    1. Embeds the query.
    2. Searches the FAISS index.
    3. Retrieves text chunks from Postgres.
    4. Generates a grounded answer with Gemini.
    """
    if not generation_model:
        raise HTTPException(status_code=503, detail="Gemini model not available")

    # 1. Embed query
    log.info(f"Received query: {request.query}")
    query_embedding = vector_store.embed_text(request.query)
    
    # 2. Search vector store
    vector_ids, distances = vector_store.search_index(query_embedding, k=5)
    
    if not vector_ids:
        log.warning("No relevant vector IDs found.")
        raise HTTPException(status_code=404, detail="No relevant information found.")

    # 3. Get chunks from Postgres
    chunks = crud.get_chunks_by_vector_ids(db, vector_ids)
    
    # Get unique source URLs
    url_ids = list(set([chunk.url_id for chunk in chunks]))
    urls = [crud.get_url_by_id(db, url_id).url for url_id in url_ids if crud.get_url_by_id(db, url_id)]

    # 4. Build context and prompt for Gemini
    context = "\n\n---\n\n".join([chunk.chunk_text for chunk in chunks])
    
    prompt = f"""
    You are a helpful AI assistant. Answer the user's query based *only* on the provided context.
    If the answer is not in the context, say "I could not find an answer in the provided documents."
    Do not use any prior knowledge.

    CONTEXT:
    {context}

    QUERY:
    {request.query}

    ANSWER:
    """
    
    try:
        response = generation_model.generate_content(prompt)
        answer = response.text
        log.info(f"Generated answer for query: {request.query}")
    except Exception as e:
        log.error(f"Error generating answer with Gemini: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer with Gemini: {e}")

    return models.QueryResponse(answer=answer, sources=urls)

@app.get("/health", response_model=models.HealthResponse)
def health_check(db: Session = Depends(database.get_db)):
    """Checks the health of the API and its dependencies (Redis, DB)."""
    redis_status = "error"
    db_status = "error"

    # Check Redis
    if redis_client:
        try:
            redis_client.ping()
            redis_status = "ok"
        except Exception as e:
            log.warning(f"Redis health check failed: {e}")
            
    # Check Database
    try:
        db.execute("SELECT 1")
        db_status = "ok"
    except Exception as e:
        log.warning(f"Database health check failed: {e}")

    return models.HealthResponse(api="ok", redis=redis_status, database=db_status)

