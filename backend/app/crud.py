from sqlalchemy.orm import Session
from . import database as db_models
import logging

log = logging.getLogger(__name__)

def create_url_entry(session: Session, url: str) -> db_models.Url:
    """Creates a new URL entry in the database with PENDING status."""
    db_url = db_models.Url(url=url, status=db_models.IngestionStatus.PENDING)
    session.add(db_url)
    session.commit()
    session.refresh(db_url)
    log.info(f"Created new URL entry for: {url}")
    return db_url

def get_url_by_url(session: Session, url: str) -> db_models.Url | None:
    """Retrieves a URL entry by its URL string."""
    return session.query(db_models.Url).filter(db_models.Url.url == url).first()

def get_url_by_id(session: Session, url_id: int) -> db_models.Url | None:
    """Retrievies a URL entry by its primary key ID."""
    return session.query(db_models.Url).filter(db_models.Url.id == url_id).first()

def update_url_status(session: Session, url_str: str, status: db_models.IngestionStatus) -> db_models.Url | None:
    """Updates the status of a URL entry."""
    db_url = get_url_by_url(session, url_str)
    if db_url:
        db_url.status = status
        session.commit()
        session.refresh(db_url)
        log.info(f"Updated status for {url_str} to {status}")
    return db_url

def add_document_chunk(session: Session, url_id: int, chunk_text: str, vector_id: str) -> db_models.DocumentChunk:
    """Adds a new document chunk to the database."""
    db_chunk = db_models.DocumentChunk(
        url_id=url_id,
        chunk_text=chunk_text,
        vector_id=vector_id
    )
    session.add(db_chunk)
    session.commit()
    session.refresh(db_chunk)
    return db_chunk

def get_chunks_by_vector_ids(session: Session, vector_ids: list[str]) -> list[db_models.DocumentChunk]:
    """Retrieves a list of document chunks based on their vector IDs."""
    return session.query(db_models.DocumentChunk).filter(
        db_models.DocumentChunk.vector_id.in_(vector_ids)
    ).all()
