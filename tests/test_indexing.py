from __future__ import annotations

import unittest
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

from scripts import index_docs


class FakeCollections:
    def __init__(self, names: list[str]) -> None:
        self.collections = [SimpleNamespace(name=name) for name in names]


class FakeClient:
    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.deleted: list[str] = []
        self.created: list[str] = []
        self.upserts: list[list[object]] = []

    def get_collections(self) -> FakeCollections:
        return FakeCollections(self.names)

    def delete_collection(self, collection_name: str) -> None:
        self.deleted.append(collection_name)

    def create_collection(self, collection_name: str, vectors_config: object) -> None:
        self.created.append(collection_name)

    def upsert(self, collection_name: str, points: list[object]) -> None:
        self.upserts.append(points)


class IndexingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_rest = index_docs.rest
        index_docs.rest = SimpleNamespace(
            VectorParams=lambda size, distance: {"size": size, "distance": distance},
            Distance=SimpleNamespace(COSINE="cosine"),
            PointStruct=lambda id, vector, payload: {"id": id, "vector": vector, "payload": payload},
        )

    def tearDown(self) -> None:
        index_docs.rest = self.original_rest

    def test_ensure_collection_is_non_destructive_by_default(self) -> None:
        client = FakeClient(["kubernetes-docs"])

        index_docs.ensure_collection(client, "kubernetes-docs", vector_size=256, recreate=False)

        self.assertEqual(client.deleted, [])
        self.assertEqual(client.created, [])

    def test_upsert_points_uses_stable_chunk_ids(self) -> None:
        records = [
            SimpleNamespace(
                chunk_id="chunk-a",
                doc_type="concepts",
                page_title="Service",
                heading_path=["Service"],
                text="Example chunk text",
                url="https://kubernetes.io/docs/concepts/services-networking/service/",
            )
        ]
        client = FakeClient([])

        index_docs.upsert_points(client, "kubernetes-docs", records, [[0.1, 0.2]], batch_size=1)

        self.assertEqual(client.upserts[0][0]["id"], str(uuid5(NAMESPACE_URL, "chunk-a")))


if __name__ == "__main__":
    unittest.main()
