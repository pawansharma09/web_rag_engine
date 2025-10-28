from sqlalchemy import create_engine, Column, String, Integer, DateTime, Enum, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import enum
import datetime
from .config import settings

# Create the SQLAlchemy engine
try:
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    print("Database engine created successfully.")
except Exception as e:
    print(f"Error creating database engine: {e}")
    # Handle error appropriately

class IngestionStatus(str, enum.Enum):
    """Enum for the status of a URL ingestion job."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Url(Base):
    """Database model for tracking ingested URLs."""
    __tablename__ = "urls"
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, index=True)
    status = Column(Enum(IngestionStatus), default=IngestionStatus.PENDING)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class DocumentChunk(Base):
    """Database model for storing individual text chunks."""
    __tablename__ = "document_chunks"
    id = Column(Integer, primary_key=True, index=True)
    url_id = Column(Integer, index=True) # Foreign key to Url.id
    chunk_text = Column(Text)
    vector_id = Column(String, unique=True) # ID used in FAISS

def init_db():
    """Initializes the database by creating all tables."""
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created.")
    except Exception as e:
        print(f"Error creating database tables: {e}")

def get_db():
    """Dependency injector to get a DB session for an API request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
