from pydantic import BaseModel, HttpUrl

# Pydantic models define the shape of API requests/responses

class IngestRequest(BaseModel):
    """Request model for /ingest-url endpoint."""
    url: HttpUrl

class IngestResponse(BaseModel):
    """Response model for /ingest-url endpoint."""
    message: str
    url: str
    status: str

class QueryRequest(BaseModel):
    """Request model for /query endpoint."""
    query: str

class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    answer: str
    sources: list[str]

class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    api: str
    redis: str
    database: str
