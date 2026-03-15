from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency in demo/tests
    OpenAI = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except ImportError:  # pragma: no cover - optional dependency in demo/tests
    QdrantClient = None
    rest = None

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


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int, recreate: bool = False) -> None:
    if rest is None:
        raise RuntimeError("qdrant-client package is not installed.")
    existing = {collection.name for collection in client.get_collections().collections}
    if collection_name in existing and recreate:
        client.delete_collection(collection_name=collection_name)
    if collection_name in existing and not recreate:
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
    )


def upsert_points(
    client: QdrantClient,
    collection_name: str,
    records: list[ChunkRecord],
    vectors: list[list[float]],
    batch_size: int,
) -> None:
    if rest is None:
        raise RuntimeError("qdrant-client package is not installed.")
    for offset in range(0, len(records), batch_size):
        batch_records = records[offset : offset + batch_size]
        batch_vectors = vectors[offset : offset + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=[
                rest.PointStruct(
                    # Qdrant requires numeric or UUID ids; uuid5 keeps imports deterministic.
                    id=str(uuid5(NAMESPACE_URL, record.chunk_id)),
                    vector=vector,
                    payload=record_to_payload(record),
                )
                for record, vector in zip(batch_records, batch_vectors, strict=True)
            ],
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index chunked Kubernetes docs into Qdrant.")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the collection before indexing.")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of vectors to upload in each batch.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    if settings.embedding_provider == "openai" and not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    if QdrantClient is None:
        raise RuntimeError("qdrant-client package is not installed.")
    if not settings.chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {settings.chunks_path}")

    records = load_chunks(settings.chunks_path)
    if not records:
        raise RuntimeError(f"No chunk records found in {settings.chunks_path}")
    qdrant_client = QdrantClient(url=settings.qdrant_url, check_compatibility=False)

    if settings.embedding_provider == "demo":
        vectors = [demo_embed(record.text) for record in records]
        vector_size = DEMO_VECTOR_SIZE
    else:
        if OpenAI is None:
            raise RuntimeError("OpenAI package is not installed.")
        openai_client = OpenAI(api_key=settings.openai_api_key)
        embeddings = openai_client.embeddings.create(
            model=settings.openai_embedding_model,
            input=[record.text for record in records],
        ).data
        vectors = [embedding.embedding for embedding in embeddings]
        vector_size = len(vectors[0])
    ensure_collection(qdrant_client, settings.qdrant_collection, vector_size, recreate=args.recreate)
    upsert_points(
        qdrant_client,
        settings.qdrant_collection,
        records,
        vectors,
        batch_size=max(1, args.batch_size),
    )
    print(f"Indexed {len(records)} chunks into {settings.qdrant_collection}")


if __name__ == "__main__":
    main()
