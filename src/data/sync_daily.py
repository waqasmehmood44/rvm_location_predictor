"""Daily automation script to sync Coripet APIs into local SQLite database.

Usage:
    python -m src.data.sync_daily
    python -m src.data.sync_daily --days 10
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.db_manager import run_full_daily_sync, DB_DEFAULT_PATH


def main():
    parser = argparse.ArgumentParser(description="Sync Coripet live APIs into local SQLite database.")
    parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Number of previous rolling days of daily bottles to fetch (default: 10)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DB_DEFAULT_PATH),
        help=f"Path to SQLite database file (default: {DB_DEFAULT_PATH})"
    )
    args = parser.parse_args()

    print(f"[DAILY SYNC] Executing synchronization with days_back={args.days} into {args.db_path}...")
    result = run_full_daily_sync(db_path=Path(args.db_path), days_back=args.days)
    print("[DAILY SYNC] Completed successfully.")
    print(result)


if __name__ == "__main__":
    main()
