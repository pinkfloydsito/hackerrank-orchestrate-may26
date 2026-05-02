"""Main entry point for the support triage agent."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.processor import process_tickets


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Support Triage Agent - Process support tickets using AI"
    )
    parser.add_argument(
        "--input", "-i",
        default="../support_tickets/support_tickets.csv",
        help="Input CSV file with tickets (default: ../support_tickets/support_tickets.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="../support_tickets/output.csv",
        help="Output CSV file for results (default: ../support_tickets/output.csv)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of tickets to process (for testing)"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run ingestion pipeline first"
    )
    
    args = parser.parse_args()
    
    # Run ingestion if requested
    if args.ingest:
        print("Running ingestion pipeline...")
        from pipeline.ingest import main as ingest_main
        ingest_main()
        print()
    
    # Process tickets
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    if args.limit:
        print(f"Limit: {args.limit} tickets")
    print()
    
    asyncio.run(process_tickets(args.input, args.output, args.limit))


if __name__ == "__main__":
    main()
