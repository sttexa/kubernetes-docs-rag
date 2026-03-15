from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models import AskRequest, AskResponse
from app.services.retrieval import RagService

logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(title=settings.app_name)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
rag_service = RagService(settings)


def _validate_settings() -> None:
    if settings.embedding_provider not in {"demo", "openai"}:
        raise RuntimeError("EMBEDDING_PROVIDER must be either 'demo' or 'openai'.")
    if settings.chat_provider not in {"demo", "openai"}:
        raise RuntimeError("CHAT_PROVIDER must be either 'demo' or 'openai'.")
    if settings.embedding_provider == "openai" and not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai.")
    if settings.chat_provider == "openai" and not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required when CHAT_PROVIDER=openai.")


@app.on_event("startup")
def startup() -> None:
    # Fail fast on config issues instead of surfacing them on the first request.
    _validate_settings()


@app.get("/health")
def health() -> dict[str, str]:
    qdrant_status = "unavailable"
    if rag_service.qdrant_available:
        try:
            assert rag_service.qdrant is not None
            rag_service.qdrant.get_collections()
            qdrant_status = "ok"
        except Exception:
            logger.warning("Qdrant health check failed", exc_info=True)
            qdrant_status = "degraded"
    return {"status": "ok", "qdrant": qdrant_status}


@app.get("/")
def root() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    try:
        route, answer, sources = rag_service.answer(payload.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to answer question")
        raise HTTPException(status_code=500, detail="Failed to answer the question.") from exc
    return AskResponse(question=payload.question, route=route, answer=answer, sources=sources)
