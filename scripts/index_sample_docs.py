from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from scripts.index_docs import OpenAI, QdrantClient, ensure_collection, load_chunks, upsert_points
from app.services.demo_mode import DEMO_VECTOR_SIZE, demo_embed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index the committed sample dataset into Qdrant.")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the collection before indexing.")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of vectors to upload in each batch.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    sample_path = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_chunks.jsonl"
    if settings.embedding_provider == "openai" and not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    if QdrantClient is None:
        raise RuntimeError("qdrant-client package is not installed.")
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample chunks file not found: {sample_path}")

    records = load_chunks(sample_path)
    if not records:
        raise RuntimeError(f"No sample chunks found in {sample_path}")
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
    print(f"Indexed {len(records)} sample chunks into {settings.qdrant_collection}")


if __name__ == "__main__":
    main()
