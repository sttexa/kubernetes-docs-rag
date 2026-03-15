from __future__ import annotations

import argparse
import json
import re
import time
from collections import deque
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
MAX_DEPTH = 2
REQUEST_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5


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


def _fetch_with_retries(client: httpx.Client, url: str, retries: int = REQUEST_RETRIES) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    assert last_error is not None
    raise last_error


def _extract_links(html: str, base_url: str, section_prefix: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    for anchor in soup.select("a[href]"):
        href = anchor.get("href")
        if not href:
            continue
        next_url = _normalize_url(urljoin(base_url, href))
        if _is_allowed(next_url) and urlparse(next_url).path.startswith(section_prefix):
            links.append(next_url)
    return links


def discover_urls(
    client: httpx.Client,
    max_pages: int = MAX_PAGES,
    max_depth: int = MAX_DEPTH,
    per_section_limit: int = PAGES_PER_SECTION,
) -> tuple[list[str], dict[str, object]]:
    discovered: list[str] = []
    seen: set[str] = set()
    failures: list[str] = []
    section_counts: dict[str, int] = {}

    for url in [_normalize_url(item) for item in PRIORITY_URLS]:
        if url not in seen:
            seen.add(url)
            discovered.append(url)

    for root in SECTION_ROOTS:
        normalized_root = _normalize_url(root)
        section_prefix = urlparse(normalized_root).path
        # Keep discovery bounded so the demo stays fast and predictable.
        queue: deque[tuple[str, int]] = deque([(normalized_root, 0)])
        local_seen: set[str] = set()
        section_counts[section_prefix] = 0

        while queue and len(discovered) < max_pages and section_counts[section_prefix] < per_section_limit:
            current_url, depth = queue.popleft()
            if current_url in local_seen:
                continue
            local_seen.add(current_url)

            if current_url not in seen:
                seen.add(current_url)
                discovered.append(current_url)
                section_counts[section_prefix] += 1

            try:
                html = _fetch_with_retries(client, current_url).text
            except httpx.HTTPError:
                failures.append(current_url)
                continue

            if depth >= max_depth:
                continue

            for next_url in _extract_links(html, current_url, section_prefix):
                if next_url not in local_seen:
                    queue.append((next_url, depth + 1))

    report = {
        "max_pages": max_pages,
        "max_depth": max_depth,
        "discovered_pages": len(discovered),
        "section_counts": section_counts,
        "failures": failures,
    }
    return discovered[:max_pages], report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape selected Kubernetes docs pages and chunk them for indexing.")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--per-section-limit", type=int, default=PAGES_PER_SECTION)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = settings.chunks_path
    crawl_report_path = settings.processed_data_dir / "crawl_report.json"

    all_chunks = []
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        urls, report = discover_urls(
            client,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            per_section_limit=args.per_section_limit,
        )
        print(f"Selected {len(urls)} official docs pages for indexing", flush=True)
        for url in urls:
            html = _fetch_with_retries(client, url).text
            raw_path = settings.raw_data_dir / f"{_safe_name(url)}.html"
            raw_path.write_text(html, encoding="utf-8")
            all_chunks.extend(chunk_html(html, url))
            print(f"Fetched {url} -> {raw_path.name}", flush=True)

    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n")

    crawl_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(all_chunks)} chunks to {chunks_path}", flush=True)
    print(f"Wrote crawl report to {crawl_report_path}", flush=True)


if __name__ == "__main__":
    main()
