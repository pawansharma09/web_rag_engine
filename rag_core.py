import os
import faiss
import requests
import google.generativeai as genai
import redis
import logging
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional
from sqlmodel import Field, Session, SQLModel, create_engine, select
from datetime import datetime

# --- Configuration & Initialization ---

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# API Keys and URLs
UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not UPSTASH_REDIS_URL or not GEMINI_API_KEY:
    logging.error("Missing required environment variables (UPSTASH_REDIS_URL or GEMINI_API_KEY)")
    # In a real app, you'd exit, but here we'll let it fail on use
    
# --- Database (SQLite) Setup ---
DATABASE_FILE = "metadata.db"
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"

class IngestionJob(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str = Field(unique=True, index=True)
    status: str = Field(default="pending", index=True) # pending, processing, completed, failed
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class Chunk(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="ingestionjob.id")
    chunk_text: str
    vector_id: int = Field(index=True, unique=True) # This is the row number in FAISS

# Engine connects SQLModel to the DB
engine = create_engine(DATABASE_URL)

def create_db_and_tables():
    """Initializes the SQLite database and tables."""
    SQLModel.metadata.create_all(engine)

def get_session():
    """Returns a new database session."""
    return Session(engine)

# --- Vector DB (FAISS) Setup ---
FAISS_INDEX_FILE = "vector_store.index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Load embedding model
logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    logging.error(f"Failed to load SentenceTransformer model: {e}")
    embedding_model = None # Handle this in functions

MODEL_DIMENSION = embedding_model.get_sentence_embedding_dimension() if embedding_model else 384
logging.info(f"Embedding model loaded. Dimension: {MODEL_DIMENSION}")

def get_faiss_index():
    """Loads FAISS index from disk or creates a new one."""
    if os.path.exists(FAISS_INDEX_FILE):
        logging.info("Loading existing FAISS index.")
        try:
            return faiss.read_index(FAISS_INDEX_FILE)
        except Exception as e:
            logging.error(f"Failed to read FAISS index: {e}. Creating new one.")
            return faiss.IndexFlatL2(MODEL_DIMENSION)
    else:
        logging.info("Creating new FAISS index.")
        return faiss.IndexFlatL2(MODEL_DIMENSION)

def save_faiss_index(index):
    """Saves the FAISS index to disk."""
    logging.info(f"Saving FAISS index to {FAISS_INDEX_FILE}...")
    try:
        faiss.write_index(index, FAISS_INDEX_FILE)
        logging.info("FAISS index saved.")
    except Exception as e:
        logging.error(f"Failed to save FAISS index: {e}")

# --- Queue (Redis) Setup ---
try:
    redis_client = redis.from_url(UPSTASH_REDIS_URL)
    redis_client.ping()
    logging.info("Connected to Redis successfully.")
except Exception as e:
    logging.error(f"Failed to connect to Redis: {e}")
    redis_client = None

QUEUE_NAME = "ingestion_queue"

# --- LLM (Gemini) Setup ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    llm = genai.GenerativeModel('gemini-2.5-flash')
    logging.info("Gemini model configured.")
except Exception as e:
    logging.error(f"Failed to configure Gemini: {e}")
    llm = None

def get_grounded_answer(query: str, context: str) -> str:
    """
    Generates an answer using the LLM, attempting to ground it in the context first.
    If the answer is not in the context, it uses its general knowledge and states so.
    """
    if not llm:
        logging.error("LLM client is not initialized.")
        return "Error: LLM not configured."
        
    # This new prompt instructs the LLM on the desired fallback behavior
    prompt = f"""
    You are a helpful assistant with a specific task.

    1.  First, carefully review the "CONTEXT" below to find an answer to the "QUERY".
    2.  If the answer is found in the "CONTEXT", provide that answer directly.
    3.  If the answer is *not* found in the "CONTEXT", you MUST begin your response with the exact phrase:
        "Based on my understanding, as this information was not found in the provided knowledge base, "
        ...and then, after that exact phrase, provide the answer using your general knowledge.
    
    Do not use your general knowledge unless the answer is missing from the context.

    CONTEXT:
    ---
    {context}
    ---

    QUERY:
    {query}

    ANSWER:
    """
    try:
        # The 'llm' object is assumed to be the user's configured Gemini client
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        return f"Error generating answer: {e}"


# --- Processing Logic ---

def fetch_and_clean_text(url: str) -> str:
    """Fetches a URL and returns clean, visible text."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise error for bad responses
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script/style tags
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
            
        # Get text, strip whitespace, and join lines
        text = ' '.join(t.strip() for t in soup.stripped_strings)
        return text
        
    except requests.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        raise

def chunk_text(text: str) -> list[str]:
    """Splits text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generates embeddings for a list of texts."""
    if not embedding_model:
        raise Exception("Embedding model is not loaded.")
    
    logging.info(f"Embedding {len(texts)} chunks...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    logging.info("Embedding complete.")
    return embeddings.astype('float32') # FAISS requires float32

# --- Main block for initialization ---
if __name__ == "__main__":
    logging.info("Initializing database...")
    create_db_and_tables()
    logging.info("Database initialized.")
    # This just checks that the FAISS index can be created/loaded
    _ = get_faiss_index()
