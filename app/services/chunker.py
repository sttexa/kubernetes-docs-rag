from __future__ import annotations

import re
from hashlib import sha1
from typing import Iterable
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from app.models import ChunkRecord, DocType


def infer_doc_type(url: str) -> DocType:
    lowered = url.lower()
    if "/concepts/" in lowered:
        return "concepts"
    if "/reference/" in lowered:
        return "reference"
    return "tasks"


def extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.select_one("article h1, main article h1, .td-content h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return "Untitled"


def _main_content(soup: BeautifulSoup) -> Tag | BeautifulSoup:
    for selector in ("main article", "article", ".td-content"):
        node = soup.select_one(selector)
        if node:
            return node
    return soup


NOISE_SELECTORS = (
    "#pre-footer",
    "#feedback",
    "#auto-generated-edit-notice",
    ".feedback--prompt",
    ".feedback--response",
    ".feedback--link",
    ".td-page-meta",
    ".pageinfo",
    ".text-muted.mt-5",
    ".alert.alert-info",
    "nav",
    "aside",
    "footer",
    "script",
    "style",
)

NOISE_PATTERNS = (
    "was this page helpful",
    "thanks for the feedback",
    "report a problem",
    "suggest an improvement",
    "edit this page",
    "last modified",
    "copyright",
    "documentation distributed under",
)


def _clean_container(container: Tag | BeautifulSoup) -> Tag | BeautifulSoup:
    for selector in NOISE_SELECTORS:
        for node in container.select(selector):
            node.decompose()
    return container


def _text(node: Tag) -> str:
    text = node.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _iter_sections(container: Tag | BeautifulSoup) -> Iterable[tuple[list[str], list[str]]]:
    heading_path: list[str] = []
    body: list[str] = []

    for element in container.find_all(["h1", "h2", "h3", "p", "li", "pre", "code"], recursive=True):
        if not isinstance(element, Tag):
            continue
        if element.name in {"h1", "h2", "h3"}:
            if body and heading_path:
                yield heading_path.copy(), body.copy()
                body.clear()
            level = int(element.name[1])
            heading_text = _text(element)
            heading_path = heading_path[: max(level - 2, 0)]
            heading_path.append(heading_text)
            continue

        snippet = _text(element)
        lowered = snippet.lower()
        if snippet and not any(pattern in lowered for pattern in NOISE_PATTERNS):
            body.append(snippet)

    if body:
        yield heading_path.copy(), body.copy()


def chunk_html(html: str, url: str) -> list[ChunkRecord]:
    soup = BeautifulSoup(html, "lxml")
    container = _clean_container(_main_content(soup))
    page_title = extract_title(soup)
    doc_type = infer_doc_type(url)
    parsed_url = urlparse(url)
    clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    chunks: list[ChunkRecord] = []

    for index, (heading_path, body) in enumerate(_iter_sections(container)):
        deduped_body = list(dict.fromkeys(body))
        text = "\n".join(deduped_body).strip()
        if len(text) < 120:
            continue
        digest = sha1(f"{clean_url}|{' > '.join(heading_path)}|{text[:160]}".encode("utf-8")).hexdigest()
        chunks.append(
            ChunkRecord(
                chunk_id=f"{doc_type}-{index}-{digest[:16]}",
                doc_type=doc_type,
                page_title=page_title,
                heading_path=heading_path or [page_title],
                text=text,
                url=clean_url,
            )
        )
    return chunks
