from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from app.config import get_settings
from app.models import ChunkRecord
from app.services.demo_mode import DEMO_VECTOR_SIZE, demo_embed
from app.services.retrieval import record_to_payload


def load_chunks(path: Path) -> list[ChunkRecord]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(ChunkRecord.model_validate(json.loads(line)))
    return records


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    existing = {collection.name for collection in client.get_collections().collections}
    if collection_name in existing:
        client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
    )


def main() -> None:
    settings = get_settings()
    if settings.embedding_provider == "openai" and not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    if not settings.chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {settings.chunks_path}")

    records = load_chunks(settings.chunks_path)
    qdrant_client = QdrantClient(url=settings.qdrant_url, check_compatibility=False)

    if settings.embedding_provider == "demo":
        vectors = [demo_embed(record.text) for record in records]
        vector_size = DEMO_VECTOR_SIZE
    else:
        openai_client = OpenAI(api_key=settings.openai_api_key)
        embeddings = openai_client.embeddings.create(
            model=settings.openai_embedding_model,
            input=[record.text for record in records],
        ).data
        vectors = [embedding.embedding for embedding in embeddings]
        vector_size = len(vectors[0])
    ensure_collection(qdrant_client, settings.qdrant_collection, vector_size)

    qdrant_client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            rest.PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload=record_to_payload(record),
            )
            for record, vector in zip(records, vectors, strict=True)
        ],
    )
    print(f"Indexed {len(records)} chunks into {settings.qdrant_collection}")


if __name__ == "__main__":
    main()
