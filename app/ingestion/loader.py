"""
Website loader module.

Uses LangChain's SitemapLoader to load company website content.
"""

import logging
import os
import re
import urllib.request
from urllib.parse import quote, urljoin, urlparse

from bs4 import BeautifulSoup
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_core.documents import Document

from app.config import MAX_PAGES, SITEMAP_URL

logger = logging.getLogger(__name__)


def _clean_html_content(content: str) -> str:
    """
    Clean HTML content extracted by the sitemap loader.
    
    Removes navigation, scripts, styles, cookie banners, and other
    irrelevant page chrome while preserving meaningful content.
    """
    if not content:
        return ""

    # Remove excessive whitespace and normalize
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r" {2,}", " ", content)
    content = re.sub(r"\t+", " ", content)

    # Remove common irrelevant patterns
    patterns_to_remove = [
        r"Cookie\s*(Policy|Notice|Consent|Banner).*?\n",
        r"Accept\s*(All\s*)?Cookies.*?\n",
        r"Privacy\s*Policy\s*\|.*?\n",
        r"©\s*\d{4}.*?All\s*[Rr]ights\s*[Rr]eserved.*?\n",
        r"Skip\s*to\s*(main\s*)?content.*?\n",
    ]
    for pattern in patterns_to_remove:
        content = re.sub(pattern, "", content, flags=re.IGNORECASE)

    return content.strip()


def _default_html_parser(content: str) -> str:
    """Parse HTML content using BeautifulSoup, extracting text."""
    soup = BeautifulSoup(content, "lxml")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        element.decompose()

    # Get text
    text = soup.get_text(separator="\n", strip=True)
    return _clean_html_content(text)


def _build_sitemap_url(base_url: str) -> str:
    """Build the sitemap URL from a base URL."""
    if SITEMAP_URL:
        return SITEMAP_URL

    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    return f"{base}/sitemap.xml"


def _url_filter(base_url: str):
    """Create a URL filter that only allows same-domain URLs."""
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc

    def _filter(url: str) -> bool:
        parsed = urlparse(url)
        # Only allow same domain
        if parsed.netloc != base_domain:
            return False
        # Skip auth/login pages
        skip_patterns = ["/login", "/auth", "/signin", "/signup", "/admin", "/wp-admin"]
        for pattern in skip_patterns:
            if pattern in parsed.path.lower():
                return False
        return True

    return _filter


def _crawl_website(base_url: str, max_pages: int) -> list[Document]:
    """Crawl internal pages starting from base_url when sitemap is missing."""
    headers = {
        "User-Agent": os.getenv(
            "USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    visited: set[str] = set()
    queue = [base_url]
    filter_fn = _url_filter(base_url)
    docs: list[Document] = []

    while queue and len(docs) < max_pages:
        current_url = queue.pop(0)
        norm_url = current_url.split("#")[0].rstrip("/")
        if norm_url in visited:
            continue
        visited.add(norm_url)

        try:
            encoded_url = quote(current_url, safe=":/#?&=")
            req = urllib.request.Request(encoded_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    continue
                html = resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            logger.debug("Failed to fetch %s: %s", current_url, e)
            continue

        soup = BeautifulSoup(html, "lxml")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        # Find linked internal pages
        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if not isinstance(href, str) or not href.strip():
                continue
            next_url = urljoin(current_url, href.strip()).split("#")[0].rstrip("/")
            if filter_fn(next_url) and next_url not in visited and next_url not in queue:
                queue.append(next_url)

        text = _default_html_parser(html)
        if text and len(text.strip()) >= 50:
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": current_url,
                        "title": title or current_url,
                        "url": current_url,
                    },
                )
            )
            logger.info("Loaded page (%d/%d): %s", len(docs), max_pages, current_url)

    return docs


def load_website(base_url: str, max_pages: int | None = None) -> list[Document]:
    """
    Load website content using LangChain's SitemapLoader, with crawler fallback.

    Args:
        base_url: The company website base URL.
        max_pages: Maximum number of pages to load. Defaults to MAX_PAGES config.

    Returns:
        A list of LangChain Document objects with metadata.
    """
    pages_limit = max_pages or MAX_PAGES
    sitemap_url = _build_sitemap_url(base_url)

    logger.info("Starting website ingestion...")
    logger.info("Base URL: %s", base_url)
    logger.info("Sitemap URL: %s", sitemap_url)
    logger.info("Max pages: %d", pages_limit)

    try:
        loader = SitemapLoader(
            web_path=sitemap_url,
            filter_urls=[base_url],
            parsing_function=_default_html_parser,
            is_local=False,
            continue_on_failure=True,
        )

        logger.info("Loading sitemap and pages...")
        documents = loader.load()
        logger.info("Pages discovered and loaded: %d", len(documents))

    except Exception as e:
        logger.error("Failed to load sitemap from %s: %s", sitemap_url, e)
        raise

    # Fall back to direct website crawling if sitemap returned no pages
    if not documents:
        logger.info(
            "Sitemap yielded 0 documents. Falling back to direct website crawling for %s...",
            base_url,
        )
        documents = _crawl_website(base_url, pages_limit)

    # Apply page limit
    if len(documents) > pages_limit:
        logger.info("Limiting to %d pages (from %d discovered).", pages_limit, len(documents))
        documents = documents[:pages_limit]

    # Filter out empty documents and ensure metadata
    valid_documents: list[Document] = []
    failed_count = 0

    for doc in documents:
        if not doc.page_content or len(doc.page_content.strip()) < 50:
            failed_count += 1
            continue

        # Ensure metadata has required fields
        if not doc.metadata:
            doc.metadata = {}
        if "source" not in doc.metadata:
            doc.metadata["source"] = base_url

        valid_documents.append(doc)

    logger.info("Successfully loaded: %d", len(valid_documents))
    logger.info("Failed/empty: %d", failed_count)

    return valid_documents
