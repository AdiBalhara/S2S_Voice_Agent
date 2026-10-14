"""
RAG testing script.

Usage:
    uv run python scripts/test_rag.py "What services does the company provide?"
    uv run python scripts/test_rag.py "What SAP services do you offer?" --top-k 5
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
from app.rag.search import search_company_knowledge


def main():
    parser = argparse.ArgumentParser(
        description="Test RAG retrieval against the company knowledge base.",
    )
    parser.add_argument(
        "query",
        help="The search query to test",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results to retrieve (default: RAG_TOP_K from config)",
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

    # Validate configuration
    errors = validate_config()
    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        sys.exit(1)

    log_config_summary()

    # Perform search
    print(f"\nSearching for: \"{args.query}\"")
    print("=" * 60)

    try:
        results = search_company_knowledge(args.query, top_k=args.top_k)

        if not results:
            print("\nNo results found.")
            print("This could mean:")
            print("  - The knowledge base is empty (run ingestion first)")
            print("  - The query doesn't match any stored content")
            print("  - There's a vector dimension mismatch")
            sys.exit(0)

        print(f"\nFound {len(results)} results:\n")

        for r in results:
            print(f"--- Result #{r['rank']} ---")
            print(f"  Source: {r['source']}")
            print(f"  Title:  {r['title']}")
            print(f"  Score:  {r['score']}")
            print(f"  Content:")
            # Print content with indentation, truncated
            content = r["content"][:500]
            for line in content.split("\n"):
                if line.strip():
                    print(f"    {line.strip()}")
            print()

        print("=" * 60)
        print(f"Retrieval is functioning. {len(results)} documents returned.")

    except Exception as e:
        logging.error("RAG test failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
