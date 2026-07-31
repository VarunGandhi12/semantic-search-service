"""HTTP surface: index, search, ask, health. Thin — all logic lives in services."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_rag_service, get_search_service
from app.core.config import get_settings
from app.domain.models import Document, ScoredDocument
from app.rag.service import RagService
from app.schemas.api import (
    AskRequest,
    AskResponse,
    HealthResponse,
    Hit,
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
)
from app.service import SearchService

router = APIRouter()


def _to_hit(scored: ScoredDocument) -> Hit:
    doc = scored.document
    return Hit(id=doc.id, text=doc.text, source=doc.source, score=round(scored.score, 6))


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(
    service: Annotated[SearchService, Depends(get_search_service)],
) -> HealthResponse:
    return HealthResponse(
        version=get_settings().version,
        documents_indexed=service.size,
    )


@router.post("/v1/documents", response_model=IndexResponse, tags=["index"])
def index_documents(
    request: IndexRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> IndexResponse:
    documents = [Document(id=d.id, text=d.text, source=d.source) for d in request.documents]
    indexed = service.index(documents)
    return IndexResponse(indexed=indexed, total=service.size)


@router.post("/v1/search", response_model=SearchResponse, tags=["search"])
def search(
    request: SearchRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> SearchResponse:
    hits = service.search(request.query, k=request.k, mode=request.mode)
    return SearchResponse(
        query=request.query,
        mode=request.mode,
        hits=[_to_hit(h) for h in hits],
    )


@router.post("/v1/ask", response_model=AskResponse, tags=["rag"])
def ask(
    request: AskRequest,
    rag: Annotated[RagService, Depends(get_rag_service)],
) -> AskResponse:
    answer = rag.ask(request.question, k=request.k)
    return AskResponse(
        question=request.question,
        answer=answer.text,
        sources=[_to_hit(s) for s in answer.sources],
    )
