import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Loads environment variables for the application."""
    DATABASE_URL: str
    REDIS_URL: str
    GEMINI_API_KEY: str
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # Use /data/ for Render's persistent disk mount path
    VECTOR_STORE_PATH: str = "/data/faiss_index" 

    class Config:
        env_file = ".env"
        # Allow loading from .env file for local development
        if os.path.exists(".env"):
            env_file = ".env"
        else:
            # In production (Render), env vars are set directly
            pass

settings = Settings()
