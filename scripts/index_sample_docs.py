from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from scripts.index_docs import ensure_collection, load_chunks
from app.services.demo_mode import DEMO_VECTOR_SIZE, demo_embed
from app.services.retrieval import record_to_payload
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from uuid import uuid4


def main() -> None:
    settings = get_settings()
    sample_path = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_chunks.jsonl"
    if settings.embedding_provider == "openai" and not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample chunks file not found: {sample_path}")

    records = load_chunks(sample_path)
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
    print(f"Indexed {len(records)} sample chunks into {settings.qdrant_collection}")


if __name__ == "__main__":
    main()
