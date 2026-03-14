from __future__ import annotations

import re
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from app.config import Settings
from app.models import ChunkRecord, DocType, SourceItem
from app.services.demo_mode import demo_answer, demo_embed

DOC_TYPES: tuple[DocType, ...] = ("concepts", "tasks", "reference")
QUERY_EXPANSIONS = {
    "expose": ["service", "nodeport", "loadbalancer", "ingress", "external"],
    "outside": ["external", "ingress", "loadbalancer", "nodeport"],
    "cluster": ["service", "ingress"],
    "config": ["configmap", "configuration"],
    "sensitive": ["secret"],
    "secret": ["secret", "configmap"],
    "deploy": ["deployment"],
    "deployment": ["deployment", "replicaset"],
    "ingress": ["ingress", "service"],
    "service": ["service", "nodeport", "loadbalancer"],
}
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "kubernetes",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
}
GENERIC_QUERY_TOKENS = {"fields", "field", "what", "how", "install", "configure", "create", "use"}


@dataclass(frozen=True)
class RoutePlan:
    primary: DocType
    weights: dict[DocType, float]


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9-]+", text.lower())
        if token not in STOP_WORDS and len(token) > 1
    }


def route_question(question: str) -> RoutePlan:
    lowered = question.lower()
    weights: dict[DocType, float] = {doc_type: 0.0 for doc_type in DOC_TYPES}

    concept_keywords = ["what", "why", "difference", "overview", "concept", "explain", "architecture"]
    task_keywords = [
        "how",
        "install",
        "create",
        "configure",
        "deploy",
        "setup",
        "run",
        "steps",
        "enable",
        "disable",
        "upgrade",
        "use",
    ]
    reference_keywords = ["api", "field", "fields", "flag", "flags", "spec", "yaml", "kubectl", "reference"]

    for keyword in concept_keywords:
        if keyword in lowered:
            weights["concepts"] += 1.6
    for keyword in task_keywords:
        if keyword in lowered:
            weights["tasks"] += 1.8
    for keyword in reference_keywords:
        if keyword in lowered:
            weights["reference"] += 1.5

    if "kubectl" in lowered and any(word in lowered for word in ("install", "setup", "configure", "use")):
        weights["tasks"] += 1.8
    if any(word in lowered for word in ("what is", "what are", "difference between", "overview of")):
        weights["concepts"] += 2.0
    if any(word in lowered for word in ("macos", "linux", "windows", "cluster", "node")) and "install" in lowered:
        weights["tasks"] += 1.2
    if any(word in lowered for word in ("yaml", "manifest")) and "example" not in lowered:
        weights["reference"] += 0.8

    if all(value == 0.0 for value in weights.values()):
        weights["concepts"] = 1.0
        weights["tasks"] = 0.7
        weights["reference"] = 0.4

    primary = max(weights, key=weights.get)
    return RoutePlan(primary=primary, weights=weights)


def _payload_text(payload: dict[str, object]) -> str:
    heading_path = payload.get("heading_path", [])
    heading_text = " ".join(heading_path) if isinstance(heading_path, list) else ""
    return " ".join(
        str(part)
        for part in (
            payload.get("page_title", ""),
            heading_text,
            payload.get("text", ""),
            urlparse(str(payload.get("url", ""))).path.replace("/", " "),
        )
        if part
    )


def _expand_query_tokens(question_tokens: set[str]) -> set[str]:
    expanded = set(question_tokens)
    for token in list(question_tokens):
        expanded.update(QUERY_EXPANSIONS.get(token, []))
    return expanded


def _comparison_terms(question: str) -> list[str]:
    lowered = question.lower().strip()
    patterns = [
        r"difference between ([a-z0-9- ]+) and ([a-z0-9- ]+)",
        r"compare ([a-z0-9- ]+) and ([a-z0-9- ]+)",
        r"([a-z0-9- ]+)\s+vs\.?\s+([a-z0-9- ]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return [match.group(1).strip(), match.group(2).strip()]
    return []


def _target_tokens(question: str, question_tokens: set[str]) -> set[str]:
    lowered = question.lower()
    match = re.search(r"what\s+(?:is|are)(?:\s+a|\s+an)?\s+(.+)", lowered)
    if match:
        phrase = re.split(r"[?.!,]", match.group(1))[0]
        return {
            token
            for token in _tokenize(phrase)
            if token not in {"kubernetes", "spec", "fields", "field"}
        }
    match = re.search(r"fields?\s+are\s+in\s+(.+?)(?:\s+spec)?[?.!]?$", lowered)
    if match:
        return {token for token in _tokenize(match.group(1)) if token != "kubernetes"}
    return {token for token in question_tokens if token not in GENERIC_QUERY_TOKENS}


@lru_cache(maxsize=4)
def _load_chunk_payloads(chunks_path: str) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    path = Path(chunks_path)
    if not path.exists():
        return payloads
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payloads.append(json.loads(line))
    return payloads


def _rerank_score(
    question: str,
    question_tokens: set[str],
    route_plan: RoutePlan,
    payload: dict[str, object],
    vector_score: float,
) -> float:
    doc_type = str(payload.get("doc_type", "concepts"))
    payload_tokens = _tokenize(_payload_text(payload))
    overlap = len(question_tokens & payload_tokens)
    title_tokens = _tokenize(str(payload.get("page_title", "")))
    heading_tokens = _tokenize(" ".join(payload.get("heading_path", []))) if isinstance(payload.get("heading_path"), list) else set()
    url_tokens = _tokenize(urlparse(str(payload.get("url", ""))).path.replace("/", " "))
    focus_tokens = _target_tokens(question, question_tokens)

    title_overlap = len(question_tokens & title_tokens)
    heading_overlap = len(question_tokens & heading_tokens)
    url_overlap = len(question_tokens & url_tokens)
    focus_title_overlap = len(focus_tokens & title_tokens)
    focus_url_overlap = len(focus_tokens & url_tokens)
    route_weight = route_plan.weights.get(doc_type, 0.0)
    exact_install_bonus = 1.5 if "install" in question_tokens and "install" in payload_tokens else 0.0
    focus_bonus = 0.9 * focus_title_overlap + 0.7 * focus_url_overlap
    exact_subject_bonus = 0.0
    if focus_tokens:
        if focus_tokens <= title_tokens and len(title_tokens - focus_tokens) <= 1:
            exact_subject_bonus += 2.2
        if focus_tokens <= url_tokens and len(url_tokens - focus_tokens) <= 3:
            exact_subject_bonus += 1.6
        if len(focus_tokens) == 1 and title_tokens == focus_tokens:
            exact_subject_bonus += 4.0
        if len(focus_tokens) == 1 and focus_tokens <= url_tokens and len(url_tokens) <= 2:
            exact_subject_bonus += 2.0

    return (
        vector_score
        + (0.35 * overlap)
        + (0.45 * title_overlap)
        + (0.25 * heading_overlap)
        + (0.35 * url_overlap)
        + (0.22 * route_weight)
        + exact_install_bonus
        + focus_bonus
        + exact_subject_bonus
    )


class RagService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.qdrant = QdrantClient(url=settings.qdrant_url, check_compatibility=False)

    def _embed(self, text: str) -> list[float]:
        if self.settings.embedding_provider == "demo":
            return demo_embed(text)
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai.")
        response = self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _local_candidates(self, question: str, question_tokens: set[str], route_plan: RoutePlan, per_type_limit: int) -> list[tuple[float, dict[str, object]]]:
        candidates: list[tuple[float, dict[str, object]]] = []
        for payload in _load_chunk_payloads(str(self.settings.chunks_path)):
            payload_type = payload.get("doc_type")
            if payload_type not in DOC_TYPES:
                continue
            score = _rerank_score(question, question_tokens, route_plan, payload, 0.0)
            if score <= route_plan.weights.get(payload_type, 0.0):
                continue
            candidates.append((score, payload))

        ranked: list[tuple[float, dict[str, object]]] = []
        for doc_type in DOC_TYPES:
            per_type = [item for item in candidates if item[1]["doc_type"] == doc_type]
            per_type.sort(key=lambda item: item[0], reverse=True)
            ranked.extend(per_type[:per_type_limit])
        return ranked

    def _comparison_candidates(self, question: str, route_plan: RoutePlan, per_type_limit: int) -> list[tuple[float, dict[str, object]]]:
        terms = _comparison_terms(question)
        if len(terms) != 2:
            return []

        candidates: list[tuple[float, dict[str, object]]] = []
        for payload in _load_chunk_payloads(str(self.settings.chunks_path)):
            payload_type = payload.get("doc_type")
            if payload_type not in DOC_TYPES:
                continue
            payload_text = _payload_text(payload).lower()
            title_tokens = _tokenize(str(payload.get("page_title", "")))
            term_scores = []
            for term in terms:
                tokens = _tokenize(term)
                overlap = len(tokens & _tokenize(payload_text))
                if tokens and tokens <= title_tokens:
                    overlap += 3
                term_scores.append(overlap)
            if any(score > 0 for score in term_scores):
                score = sum(term_scores) + route_plan.weights.get(payload_type, 0.0)
                candidates.append((score, payload))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[:per_type_limit]

    def search(self, question: str, limit: int | None = None) -> tuple[DocType, list[SourceItem]]:
        route_plan = route_question(question)
        final_limit = limit or self.settings.top_k
        candidate_limit = max(final_limit * 5, 20) if self.settings.embedding_provider == "demo" else max(final_limit * 2, 8)
        vector = self._embed(question)
        question_tokens = _expand_query_tokens(_tokenize(question))
        gathered_hits = []

        for doc_type in DOC_TYPES:
            response = self.qdrant.query_points(
                collection_name=self.settings.qdrant_collection,
                query=vector,
                query_filter=rest.Filter(
                    must=[rest.FieldCondition(key="doc_type", match=rest.MatchValue(value=doc_type))]
                ),
                limit=candidate_limit,
                with_payload=True,
            )
            for hit in response.points:
                payload = hit.payload or {}
                gathered_hits.append(
                    (
                        _rerank_score(question, question_tokens, route_plan, payload, float(hit.score or 0.0)),
                        payload,
                    )
                )

        gathered_hits.extend(self._local_candidates(question, question_tokens, route_plan, per_type_limit=candidate_limit))
        gathered_hits.extend(self._comparison_candidates(question, route_plan, per_type_limit=candidate_limit))

        gathered_hits.sort(key=lambda item: item[0], reverse=True)
        seen_urls: set[tuple[str, str]] = set()
        sources = [
            SourceItem(
                title=payload["page_title"],
                url=payload["url"],
                doc_type=payload["doc_type"],
                excerpt=payload["text"][:320],
                score=score,
                heading_path=payload.get("heading_path", []),
            )
            for score, payload in gathered_hits
            if not ((payload["url"], payload["text"][:120]) in seen_urls or seen_urls.add((payload["url"], payload["text"][:120])))
        ]
        return route_plan.primary, sources[:final_limit]

    def answer(self, question: str) -> tuple[DocType, str, list[SourceItem]]:
        route, sources = self.search(question)
        if self.settings.chat_provider == "demo":
            return route, demo_answer(question, route, sources), sources
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY is required when CHAT_PROVIDER=openai.")
        context = "\n\n".join(
            f"[{index + 1}] {source.title}\nURL: {source.url}\nType: {source.doc_type}\nExcerpt: {source.excerpt}"
            for index, source in enumerate(sources)
        )
        prompt = (
            "You are a Kubernetes documentation assistant. "
            "Answer only from the provided official Kubernetes documentation excerpts. "
            "If the sources do not contain the answer, say that clearly. "
            "Cite source numbers inline, like [1]."
        )
        response = self.client.responses.create(
            model=self.settings.openai_chat_model,
            instructions=prompt,
            input=f"Question: {question}\n\nRoute: {route}\n\nSources:\n{context}",
        )
        return route, response.output_text, sources


def record_to_payload(record: ChunkRecord) -> dict[str, object]:
    return {
        "chunk_id": record.chunk_id,
        "doc_type": record.doc_type,
        "page_title": record.page_title,
        "heading_path": record.heading_path,
        "text": record.text,
        "url": str(record.url),
    }
