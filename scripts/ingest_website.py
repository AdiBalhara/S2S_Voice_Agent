"""
Website ingestion script.

Usage:
    uv run python scripts/ingest_website.py https://example.com
    uv run python scripts/ingest_website.py https://example.com --max-pages 50
    uv run python scripts/ingest_website.py https://example.com --sitemap-url https://example.com/sitemap.xml
    uv run python scripts/ingest_website.py https://example.com --chunk-size 800 --chunk-overlap 100
"""

import argparse
import logging
import sys

# Ensure project root is in path
sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.database.connection import configure_windows_event_loop

# On Windows, psycopg async requires SelectorEventLoop instead of ProactorEventLoop
configure_windows_event_loop()

from app.config import validate_config, log_config_summary
from app.ingestion.pipeline import build_company_knowledge_base


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a company website into the vector knowledge base.",
    )
    parser.add_argument(
        "base_url",
        help="The company website base URL (e.g., https://trijotech.com)",
    )
    parser.add_argument(
        "--sitemap-url",
        default=None,
        help="Custom sitemap URL (default: {base_url}/sitemap.xml)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to ingest",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for text splitting (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap for text splitting (default: 150)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set custom sitemap URL if provided
    if args.sitemap_url:
        import app.config
        app.config.SITEMAP_URL = args.sitemap_url

    # Validate configuration
    errors = validate_config()
    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        sys.exit(1)

    log_config_summary()

    # Run ingestion
    try:
        summary = build_company_knowledge_base(
            base_url=args.base_url,
            max_pages=args.max_pages,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        print("\n" + "=" * 60)
        print("INGESTION SUMMARY")
        print("=" * 60)
        print(f"  Status: {summary['status']}")
        print(f"  Pages loaded: {summary.get('pages_loaded', 0)}")
        print(f"  Chunks created: {summary.get('chunks_created', 0)}")
        print(f"  Documents stored: {summary.get('documents_stored', 0)}")
        print("=" * 60)

        if summary["status"] == "success":
            print("\nIngestion complete. You can now test with:")
            print('  uv run python scripts/test_rag.py "What services does the company provide?"')
        else:
            print(f"\nIngestion finished with status: {summary['status']}")
            sys.exit(1)

    except Exception as e:
        logging.error("Ingestion failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
