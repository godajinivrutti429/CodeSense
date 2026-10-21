"""Pydantic models shared across the API."""
from typing import List, Optional
from pydantic import BaseModel


class IngestRequest(BaseModel):
    repo_path: Optional[str] = None   # local path on server/container
    repo_url: Optional[str] = None    # git URL to clone
    collection_name: str              # namespace for this repo's vectors


class IngestResponse(BaseModel):
    collection_name: str
    files_processed: int
    chunks_indexed: int
    embedding_model: str


class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    collection_name: str
    question: str
    history: List[ChatMessage] = []
    top_k: int = 5


class SourceChunk(BaseModel):
    file_path: str
    function_name: Optional[str] = None
    line_start: int
    line_end: int
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class UserCreate(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
