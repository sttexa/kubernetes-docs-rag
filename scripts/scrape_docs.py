from __future__ import annotations

import json
import re
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from app.config import get_settings
from app.services.chunker import chunk_html

SECTION_ROOTS = [
    "https://kubernetes.io/docs/concepts/",
    "https://kubernetes.io/docs/tasks/",
    "https://kubernetes.io/docs/reference/",
]
PRIORITY_URLS = [
    "https://kubernetes.io/docs/concepts/services-networking/service/",
    "https://kubernetes.io/docs/concepts/configuration/configmap/",
    "https://kubernetes.io/docs/concepts/configuration/secret/",
    "https://kubernetes.io/docs/concepts/services-networking/ingress/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
    "https://kubernetes.io/docs/tasks/tools/install-kubectl-macos/",
    "https://kubernetes.io/docs/tasks/run-application/run-stateless-application-deployment/",
    "https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster-services/",
    "https://kubernetes.io/docs/reference/kubernetes-api/workload-resources/deployment-v1/",
    "https://kubernetes.io/docs/reference/kubectl/generated/kubectl_apply/",
]
ALLOWED_PREFIXES = ("/docs/concepts/", "/docs/tasks/", "/docs/reference/")
SKIP_PATTERNS = (
    "/docs/reference/generated/",
    "/docs/reference/node/",
    "/docs/reference/setup-tools/",
    "/docs/reference/glossary/",
    "/docs/reference/command-line-tools-reference/",
    "/docs/tasks/debug/",
)
MAX_PAGES = 140
PAGES_PER_SECTION = 45


def _safe_name(url: str) -> str:
    parsed = urlparse(url)
    slug = parsed.path.strip("/").replace("/", "__") or "index"
    return re.sub(r"[^a-zA-Z0-9._-]", "-", slug)


def _normalize_url(candidate: str) -> str:
    parsed = urlparse(candidate)
    cleaned = parsed._replace(query="", fragment="")
    path = cleaned.path
    if not path.endswith("/"):
        path = f"{path}/"
    cleaned = cleaned._replace(path=path)
    return urlunparse(cleaned)


def _is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc != "kubernetes.io":
        return False
    if parsed.path.endswith((".pdf", ".svg", ".png", ".jpg", ".xml")):
        return False
    if any(pattern in parsed.path for pattern in SKIP_PATTERNS):
        return False
    return parsed.path.startswith(ALLOWED_PREFIXES)


def discover_urls(client: httpx.Client) -> list[str]:
    discovered = [_normalize_url(url) for url in PRIORITY_URLS]
    seen: set[str] = set(discovered)

    for url in SECTION_ROOTS:
        normalized = _normalize_url(url)
        section_prefix = urlparse(normalized).path
        section_urls: list[str] = []
        if normalized not in seen:
            seen.add(normalized)
            section_urls.append(normalized)
        response = client.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        for anchor in soup.select("a[href]"):
            href = anchor.get("href")
            if not href:
                continue
            next_url = _normalize_url(urljoin(normalized, href))
            if _is_allowed(next_url) and urlparse(next_url).path.startswith(section_prefix) and next_url not in seen:
                seen.add(next_url)
                section_urls.append(next_url)
                if len(section_urls) >= PAGES_PER_SECTION:
                    break

        discovered.extend(section_urls)
        if len(discovered) >= MAX_PAGES:
            return discovered[:MAX_PAGES]

    return discovered


def main() -> None:
    settings = get_settings()
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = settings.chunks_path

    all_chunks = []
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        urls = discover_urls(client)
        print(f"Selected {len(urls)} official docs pages for indexing", flush=True)
        for url in urls:
            response = client.get(url)
            response.raise_for_status()
            html = response.text
            raw_path = settings.raw_data_dir / f"{_safe_name(url)}.html"
            raw_path.write_text(html, encoding="utf-8")
            all_chunks.extend(chunk_html(html, url))
            print(f"Fetched {url} -> {raw_path.name}", flush=True)

    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_chunks)} chunks to {chunks_path}", flush=True)


if __name__ == "__main__":
    main()
