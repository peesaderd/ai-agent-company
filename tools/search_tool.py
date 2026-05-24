"""Search and fetch tools using DuckDuckGo HTML + httpx (lightweight, no CrawlAI)."""

from typing import Optional
import re
import urllib.parse
import httpx
from bs4 import BeautifulSoup


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using DuckDuckGo HTML search."""
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.post(url, data=params, headers=headers)
            resp.raise_for_status()
    except Exception as e:
        return [{"title": "Error", "url": "", "snippet": f"Search failed: {e}"}]

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    for result in soup.select(".result")[:max_results]:
        title_el = result.select_one(".result__title a")
        snippet_el = result.select_one(".result__snippet")

        if title_el:
            href = title_el.get("href", "")
            # DuckDuckGo redirect URLs
            if "uddg=" in href:
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                href = parsed.get("uddg", [""])[0]
            title = title_el.get_text(strip=True)
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            results.append({"title": title, "url": href, "snippet": snippet})

    return results


def fetch_page(url: str, max_length: int = 5000) -> Optional[str]:
    """Fetch a webpage and extract clean text content."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
    except Exception as e:
        return f"Error fetching {url}: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    lines = [line for line in text.splitlines() if len(line.strip()) > 30]
    content = "\n".join(lines)

    if len(content) > max_length:
        content = content[:max_length] + "\n\n[เนื้อหาถูกตัด...]"

    return content


def search_and_fetch(query: str, max_results: int = 3, max_length: int = 5000) -> list[dict]:
    """Search then fetch full content from each result."""
    results = search_web(query, max_results=max_results)
    for r in results:
        content = fetch_page(r["url"], max_length=max_length)
        r["content"] = content
    return results
