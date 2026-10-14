"""
Markdown ingestion script.

Usage:
    uv run python scripts/ingest_markdown.py
    uv run python scripts/ingest_markdown.py data/dmrc_mock_data.md
    uv run python scripts/ingest_markdown.py data/dmrc_mock_data.md --chunk-size 800 --chunk-overlap 100
"""

import argparse
import logging
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.database.connection import configure_windows_event_loop

# On Windows, psycopg async requires SelectorEventLoop instead of ProactorEventLoop
configure_windows_event_loop()

from app.config import log_config_summary, validate_config
from app.ingestion.pipeline import build_knowledge_base_from_markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a Markdown file into the vector knowledge base.",
    )
    parser.add_argument(
        "markdown_path",
        nargs="?",
        type=Path,
        default=BACKEND_ROOT / "data" / "dmrc_mock_data.md",
        help="Markdown file to ingest (default: data/dmrc_mock_data.md)",
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

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    errors = validate_config()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        sys.exit(1)

    log_config_summary()
    try:
        summary = build_knowledge_base_from_markdown(
            markdown_path=args.markdown_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except Exception as error:
        logging.error("Markdown ingestion failed: %s", error, exc_info=True)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"  Status: {summary['status']}")
    print(f"  Documents loaded: {summary.get('documents_loaded', 0)}")
    print(f"  Chunks created: {summary.get('chunks_created', 0)}")
    print(f"  Documents stored: {summary.get('documents_stored', 0)}")
    print("=" * 60)

    if summary["status"] == "success":
        print("\nIngestion complete. You can now test with:")
        print('  uv run python scripts/test_rag.py "Which metro line serves the airport?"')
    else:
        print(f"\nIngestion finished with status: {summary['status']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
