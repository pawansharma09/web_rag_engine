import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import settings
import os
import pickle
import fcntl # For file locking on Unix (Render runs on Linux)
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Globals ---
_model = None

# --- File Lock ---
class FileLock:
    """A simple file lock for safe concurrent writes to the FAISS index."""
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.handle = None

    def __enter__(self):
        self.handle = open(self.lock_file, 'w')
        # Acquire an exclusive lock
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        return self.handle

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle:
            # Release the lock
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()

# --- Helper Functions ---

def get_embedding_model():
    """Lazily loads and returns the SentenceTransformer model."""
    global _model
    if _model is None:
        log.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        log.info("Embedding model loaded.")
    return _model

def get_index_paths():
    """Returns the paths for the index, metadata, and lock files."""
    base_path = settings.VECTOR_STORE_PATH
    index_path = f"{base_path}.index"
    meta_path = f"{base_path}.meta"
    lock_path = f"{base_path}.lock"
    # Ensure the directory exists (for Render's persistent disk)
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    return index_path, meta_path, lock_path

def load_index():
    """
    Loads the FAISS index and its ID-mapping metadata from disk.
    Creates new ones if they don't exist.
    """
    index_path, meta_path, _ = get_index_paths()
    if os.path.exists(index_path):
        log.info(f"Loading existing FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            # Metadata maps {faiss_internal_int_id: our_string_vector_id}
            index_to_vector_id = pickle.load(f)
    else:
        log.info("No FAISS index found, creating new one.")
        model = get_embedding_model()
        dimension = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)  # Simple L2 distance index
        index_to_vector_id = {}
    return index, index_to_vector_id

def save_index(index, index_to_vector_id):
    """Saves the FAISS index and metadata to disk."""
    index_path, meta_path, _ = get_index_paths()
    log.info(f"Saving FAISS index to {index_path} (Total vectors: {index.ntotal})")
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(index_to_vector_id, f)

# --- Public Interface ---

def embed_text(text: str | list[str]) -> np.ndarray:
    """Embeds a single string or a list of strings."""
    model = get_embedding_model()
    return model.encode(text, convert_to_tensor=False)

def add_to_index(vectors: np.ndarray, vector_ids: list[str]):
    """
    Atomically adds new vectors and their string IDs to the FAISS index.
    This operation is write-locked.
    """
    _, _, lock_path = get_index_paths()
    with FileLock(lock_path):
        log.info("Acquired lock to add to FAISS index...")
        index, index_to_vector_id = load_index()
        
        start_index = index.ntotal
        index.add(vectors)
        
        # Add new ID mappings
        for i, vec_id in enumerate(vector_ids):
            index_to_vector_id[start_index + i] = vec_id
        
        save_index(index, index_to_vector_id)
        log.info(f"Added {len(vectors)} vectors. Total size: {index.ntotal}")

def search_index(query_vector: np.ndarray, k=5) -> tuple[list[str], list[float]]:
    """
    Searches the FAISS index for the k-nearest neighbors.
    This operation is read-only and does not lock.
    """
    index_path, _, _ = get_index_paths()
    if not os.path.exists(index_path):
        log.warning("Search called but no index file found.")
        return [], []
    
    # We accept eventually consistent reads. No lock needed.
    index, index_to_vector_id = load_index()

    if index.ntotal == 0:
        log.warning("Search called but index is empty.")
        return [], []
    
    log.info(f"Searching index with {index.ntotal} vectors for k={k}")
    distances, faiss_indices = index.search(query_vector.reshape(1, -1), k)
    
    # Map internal FAISS int IDs back to our string vector IDs
    found_vector_ids = [
        index_to_vector_id[i] for i in faiss_indices[0] if i in index_to_vector_id
    ]
    log.info(f"Found {len(found_vector_ids)} matching vector IDs.")
    return found_vector_ids, distances[0].tolist()
