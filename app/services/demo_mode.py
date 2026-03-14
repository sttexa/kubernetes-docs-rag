from __future__ import annotations

import re
from hashlib import sha256

from app.models import SourceItem

DEMO_VECTOR_SIZE = 256


def demo_embed(text: str) -> list[float]:
    values = [0.0] * DEMO_VECTOR_SIZE
    for token in text.lower().split():
        digest = sha256(token.encode("utf-8")).digest()
        for index, byte in enumerate(digest):
            values[index % DEMO_VECTOR_SIZE] += (byte / 255.0) - 0.5
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [value / norm for value in values]


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


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9-]+", text.lower()))


def _dedupe_sources(sources: list[SourceItem], limit: int = 3) -> list[SourceItem]:
    unique: list[SourceItem] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        key = (str(source.url), source.title)
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
        if len(unique) >= limit:
            break
    return unique


def _first_sentence(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)
    return parts[0] if parts else normalized


def _source_summary(source: SourceItem) -> str:
    heading = " > ".join(source.heading_path[:2]) if source.heading_path else source.title
    sentence = _first_sentence(source.excerpt)
    return f"{heading}: {sentence}"


def demo_answer(question: str, route: str, sources: list[SourceItem]) -> str:
    if not sources:
        return "I could not find a matching answer in the indexed Kubernetes documentation."

    picked = _dedupe_sources(sources, limit=3)
    comparison = _comparison_terms(question)

    if comparison and len(picked) >= 2:
        comparison_tokens = [_tokenize(term) for term in comparison]
        filtered = []
        for source in picked:
            source_tokens = _tokenize(source.title + " " + " ".join(source.heading_path))
            if any(tokens & source_tokens for tokens in comparison_tokens):
                filtered.append(source)
        if len(filtered) >= 2:
            picked = filtered

        lines = [
            f"Working summary for: {question}",
            "",
            f"This looks like a comparison between `{comparison[0]}` and `{comparison[1]}`.",
            "Based on the retrieved Kubernetes docs, these are the most relevant reference points:",
        ]
        for index, source in enumerate(picked, start=1):
            lines.append(f"{index}. {_source_summary(source)}")
        lines.append("")
        lines.append("Suggested reading order:")
        for source in picked:
            lines.append(f"- {source.title}: {source.url}")
        lines.append("")
        lines.append("This answer is a local summary because CHAT_PROVIDER=demo.")
        return "\n".join(lines)

    lines = [
        f"Working summary for: {question}",
        "",
        f"The best matching Kubernetes docs are in the `{route}` category.",
    ]

    if picked:
        lines.append("Key points from the retrieved pages:")
        for index, source in enumerate(picked, start=1):
            lines.append(f"{index}. {_source_summary(source)}")

        lines.append("")
        lines.append("Sources:")
        for source in picked:
            lines.append(f"- {source.title}: {source.url}")

    lines.append("")
    lines.append("This answer is a local summary because CHAT_PROVIDER=demo.")
    return "\n".join(lines)
