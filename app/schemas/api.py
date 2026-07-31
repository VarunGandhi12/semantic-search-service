"""Pydantic request/response contracts — the only shapes the API exposes."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.search.hybrid import SearchMode


class DocumentIn(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = ""


class IndexRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1)


class IndexResponse(BaseModel):
    indexed: int
    total: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    mode: SearchMode = SearchMode.HYBRID


class Hit(BaseModel):
    id: str
    text: str
    source: str
    score: float


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    hits: list[Hit]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=4, ge=1, le=20)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Hit]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    documents_indexed: int
