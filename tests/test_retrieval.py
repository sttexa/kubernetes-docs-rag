from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.services.retrieval import RagService, route_question


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tempdir.name) / "chunks.jsonl"
        sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_chunks.jsonl"
        self.chunks_path.write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")
        self.settings = SimpleNamespace(
            openai_api_key="",
            qdrant_url="http://localhost:6333",
            qdrant_collection="kubernetes-docs-test",
            openai_embedding_model="text-embedding-3-small",
            openai_chat_model="gpt-4.1-mini",
            top_k=3,
            embedding_provider="demo",
            chat_provider="demo",
            chunks_path=self.chunks_path,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_route_question_prefers_reference_for_spec_queries(self) -> None:
        route = route_question("What fields are in Deployment spec?")
        self.assertEqual(route.primary, "reference")

    def test_search_works_without_qdrant_using_local_fallback(self) -> None:
        service = RagService(self.settings)
        route, sources = service.search("How do I install kubectl on macOS?")

        self.assertEqual(route, "tasks")
        self.assertTrue(sources)
        self.assertEqual(str(sources[0].url), "https://kubernetes.io/docs/tasks/tools/install-kubectl-macos/")


if __name__ == "__main__":
    unittest.main()
