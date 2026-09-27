"""
Multi-backend web search + page scraping.

This is the same approach used by the BRAINF4RT/forecastingbot Metaculus bot
(research/scraper.py): search several DDGS backends independently so a
single backend going down doesn't kill search, then try to pull full page
text for each hit with trafilatura (falling back to BeautifulSoup, then to
the search-engine snippet).

No API key required. Exposed as a single function, `web_search()`, that
returns a plain string ready to be dropped into a tool result message for
a local LLM.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS

logger = logging.getLogger("lmstudio_bionic.websearch")

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0 Safari/537.36"
)

# Backends tried in order for each query. A failure on one backend never
# blocks the others.
SEARCH_BACKENDS = (
    "brave",
    "google",
    "bing",
    "duckduckgo",
    "yahoo",
)

SEARCH_DELAY_SECONDS = 0.25
SCRAPE_TIMEOUT = 12
MAX_CHARS_PER_SOURCE = 10000  # keep local-model context usage sane


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: str = ""
    method: str = ""
    backend: str = ""


def _normalise_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if "?" in url:
        base, query = url.split("?", 1)
        keep = [
            item
            for item in query.split("&")
            if item.split("=", 1)[0].lower() not in
            {"fbclid", "gclid", "ref", "ref_src"}
            and not item.split("=", 1)[0].lower().startswith("utm_")
        ]
        url = base + (("?" + "&".join(keep)) if keep else "")
    return url.rstrip("/")


def _valid_result(result: dict) -> bool:
    url = result.get("href") or result.get("url") or ""
    title = result.get("title") or ""
    body = result.get("body") or ""
    return bool(isinstance(url, str) and url.strip() and (title.strip() or body.strip()))


def ddgs_search(query: str, max_results: int = 3) -> list[dict]:
    """Search multiple DDGS backends independently, stopping once enough hits exist."""

    query = query.strip()
    if not query:
        return []

    all_results: list[dict] = []
    seen_urls: set[str] = set()

    for backend in SEARCH_BACKENDS:
        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        query,
                        region="us-en",
                        safesearch="moderate",
                        max_results=max_results,
                        backend=backend,
                    )
                )

            for result in results:
                if not _valid_result(result):
                    continue

                url = _normalise_url(result.get("href") or result.get("url") or "")
                if not url or url in seen_urls:
                    continue

                seen_urls.add(url)
                result = dict(result)
                result["_backend"] = backend
                result["_normalised_url"] = url
                all_results.append(result)

            if len(all_results) >= max_results:
                break

        except Exception as exc:
            logger.warning("backend=%s failed for %r: %s", backend, query, exc)

        time.sleep(SEARCH_DELAY_SECONDS)

    return all_results[:max_results]


def scrape_with_trafilatura(url: str) -> str | None:
    try:
        response = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=SCRAPE_TIMEOUT, allow_redirects=True
        )
        response.raise_for_status()
    except Exception as exc:
        logger.debug("trafilatura fetch failed for %s: %s", url, exc)
        return None

    try:
        text = trafilatura.extract(response.text, include_comments=False, include_tables=False)
        if not text:
            return None
        text = text.strip()
        return text if len(text) >= 100 else None
    except Exception as exc:
        logger.debug("trafilatura extract failed for %s: %s", url, exc)
        return None


def scrape_with_bs4(url: str) -> str | None:
    try:
        response = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=SCRAPE_TIMEOUT, allow_redirects=True
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
            tag.decompose()

        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(" ", strip=True)) >= 40
        ]
        text = "\n".join(paragraphs).strip()
        return text if len(text) >= 100 else None
    except Exception as exc:
        logger.debug("bs4 failed for %s: %s", url, exc)
        return None


def scrape_url(url: str) -> tuple[str, str]:
    """Try trafilatura first, then BeautifulSoup."""
    text = scrape_with_trafilatura(url)
    if text:
        return text, "trafilatura"
    text = scrape_with_bs4(url)
    if text:
        return text, "bs4"
    return "", "failed"


def gather_sources(query: str, results_per_query: int = 4, scrape: bool = True) -> list[SearchResult]:
    """Search a single query and return usable sources (optionally scraped)."""

    sources: list[SearchResult] = []
    hits = ddgs_search(query, max_results=results_per_query)

    for hit in hits:
        url = hit.get("_normalised_url") or ""
        title = hit.get("title") or url
        snippet = (hit.get("body") or "").strip()
        backend = hit.get("_backend") or "unknown"

        content, method = ("", "snippet_only")
        if scrape:
            content, method = scrape_url(url)

        if not content and len(snippet) >= 40:
            content = snippet
            method = "search_snippet"

        if not content:
            continue

        sources.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                content=content[:MAX_CHARS_PER_SOURCE],
                method=method,
                backend=backend,
            )
        )

    return sources


def web_search(query: str, max_results: int = 4, scrape: bool = True) -> str:
    """
    Run a web search and return a markdown string of results, ready to be
    used as a tool-call result for an LLM.
    """

    sources = gather_sources(query, results_per_query=max_results, scrape=scrape)

    if not sources:
        return f'No usable web results found for query: "{query}"'

    blocks = [f'Search results for: "{query}"\n']
    for i, source in enumerate(sources, start=1):
        blocks.append(
            f"[{i}] {source.title}\n"
            f"URL: {source.url}\n"
            f"(backend: {source.backend}, extraction: {source.method})\n"
            f"{source.content}\n"
        )

    return "\n---\n".join(blocks)


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="websearch.py",
        description="Search the web and print page contents as markdown.",
    )
    parser.add_argument("query", nargs="+", help="Search query (quote it, or pass multiple words).")
    parser.add_argument(
        "-n", "--max-results", type=int, default=4,
        help="Number of results to fetch (default: 4, max: 8).",
    )
    parser.add_argument(
        "--no-scrape", action="store_true",
        help="Skip fetching full page text; return search snippets only (faster).",
    )
    args = parser.parse_args()

    query = " ".join(args.query).strip()
    max_results = max(1, min(args.max_results, 8))

    if not query:
        print("Error: empty query.", file=__import__("sys").stderr)
        return 1

    try:
        result = web_search(query, max_results=max_results, scrape=not args.no_scrape)
    except Exception as exc:
        print(f"Error: web search failed ({exc}).", file=__import__("sys").stderr)
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
