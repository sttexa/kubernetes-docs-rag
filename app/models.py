from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


DocType = Literal["concepts", "tasks", "reference"]


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_type: DocType
    page_title: str
    heading_path: list[str]
    text: str
    url: HttpUrl


class AskRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    title: str
    url: HttpUrl
    doc_type: DocType
    excerpt: str
    score: float | None = None
    heading_path: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    question: str
    route: DocType
    answer: str
    sources: list[SourceItem]
