from __future__ import annotations

import json
import os
from pathlib import Path

from app.config import get_settings
from app.services.retrieval import RagService


def load_cases(path: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    eval_path = root / "data" / "sample" / "eval_queries.jsonl"
    sample_chunks_path = root / "data" / "sample" / "sample_chunks.jsonl"
    processed_path = root / "data" / "processed" / "chunks.jsonl"

    os.environ.setdefault("EMBEDDING_PROVIDER", "demo")
    os.environ.setdefault("CHAT_PROVIDER", "demo")
    os.environ.setdefault("TOP_K", "3")
    get_settings.cache_clear()
    settings = get_settings()
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.write_text(sample_chunks_path.read_text(encoding="utf-8"), encoding="utf-8")
    settings = settings.model_copy(update={"data_dir": root / "data", "top_k": 3})

    service = RagService(settings)
    total = 0
    route_hits = 0
    source_hits = 0

    for case in load_cases(eval_path):
        total += 1
        route, sources = service.search(str(case["question"]), limit=3)
        expected_route = case["expected_route"]
        expected_url = case["expected_url"]
        route_hits += int(route == expected_route)
        source_hits += int(any(str(source.url) == expected_url for source in sources))
        print(
            json.dumps(
                {
                    "question": case["question"],
                    "predicted_route": route,
                    "expected_route": expected_route,
                    "top_source": str(sources[0].url) if sources else None,
                    "expected_url": expected_url,
                },
                ensure_ascii=False,
            )
        )

    print(
        json.dumps(
            {
                "cases": total,
                "route_accuracy": round(route_hits / total, 3) if total else 0.0,
                "top_k_source_hit_rate": round(source_hits / total, 3) if total else 0.0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
