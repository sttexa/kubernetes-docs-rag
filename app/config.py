from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_env() -> None:
    cwd = Path.cwd()
    candidates = [
        cwd / ".env",
        cwd.parent / ".env",
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate, override=False)


_load_env()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    app_name: str = "Kubernetes Docs RAG"
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="kubernetes-docs", alias="QDRANT_COLLECTION")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    openai_chat_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_CHAT_MODEL")
    top_k: int = Field(default=6, alias="TOP_K")
    embedding_provider: str = Field(default="openai", alias="EMBEDDING_PROVIDER")
    chat_provider: str = Field(default="openai", alias="CHAT_PROVIDER")
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data")

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def chunks_path(self) -> Path:
        return self.processed_data_dir / "chunks.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
