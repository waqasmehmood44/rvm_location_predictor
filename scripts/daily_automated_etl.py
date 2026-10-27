"""
Coripet RVM Predictor: Daily Automated ETL Ingestion & Master Preparation Pipeline

Automates the 2-step daily workflow:
  Step 1: Calls Coripet live API to fetch daily bottle collection records for the last 2 days
          and upserts them into PostgreSQL table `raw_daily_conferimenti`.
  Step 2: Enriches those records with physical point history, Nielsen census attributes,
          municipality demographics/economics, and 4-gate maturity rules, then inserts them
          into PostgreSQL table `master_daily_rvm`.

Usage:
  python scripts/daily_automated_etl.py
  python scripts/daily_automated_etl.py --days 2
"""

import os
import sys
import argparse
import datetime
import logging
from pathlib import Path

# Set up project root
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "daily_automated_etl.log"

# Configure unified logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DailyAutomatedETL")

from scripts.fetch_and_update_conferimenti import (
    fetch_api_conferimenti_range,
    clean_conferimenti,
    update_postgres_conferimenti,
    update_sqlite_conferimenti,
    verify_database_state as verify_conferimenti
)
from scripts.update_master_daily_rvm import run_pipeline as update_master_rvm_pipeline


def execute_daily_sync(days_back: int = 2, custom_start: str = None, custom_end: str = None):
    now = datetime.datetime.now()
    if custom_end:
        end_date = custom_end
    else:
        # Default: yesterday
        end_date = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    if custom_start:
        start_date = custom_start
    else:
        # Default: days_back days before end_date
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_dt - datetime.timedelta(days=days_back - 1)).strftime("%Y-%m-%d")

    logger.info("=" * 80)
    logger.info("CORIPET DAILY AUTOMATED ETL EXECUTION STARTED")
    logger.info(f"Execution Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Target Date Window: {start_date} to {end_date} (Rolling {days_back} Days)")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: Fetch from API -> raw_daily_conferimenti
    # -------------------------------------------------------------------------
    logger.info("\n>>> STEP 1: Fetching live daily bottles from Coripet API...")
    try:
        df_raw = fetch_api_conferimenti_range(start_date, end_date)
        if df_raw.empty:
            logger.warning("[STEP 1] No records returned from Coripet API. Aborting.")
            return False

        logger.info(f"[STEP 1] Cleaning {len(df_raw):,} raw records...")
        df_clean = clean_conferimenti(df_raw)

        logger.info(f"[STEP 1] Updating PostgreSQL raw_daily_conferimenti for {start_date} to {end_date}...")
        update_postgres_conferimenti(df_clean, start_date, end_date)

        logger.info("[STEP 1] Updating local SQLite backup...")
        update_sqlite_conferimenti(df_clean, start_date, end_date)

        verify_conferimenti(start_date, end_date)
        logger.info("[STEP 1] Completed successfully.")
    except Exception as ex:
        logger.error(f"[STEP 1 FAILED] Error during API fetch / raw insertion: {ex}", exc_info=True)
        return False

    # -------------------------------------------------------------------------
    # STEP 2: Enrich & prepare -> master_daily_rvm
    # -------------------------------------------------------------------------
    logger.info("\n>>> STEP 2: Preparing and adding enriched retail records into master_daily_rvm...")
    try:
        update_master_rvm_pipeline(start_date=start_date, end_date=end_date)
        logger.info("[STEP 2] Completed successfully.")
    except Exception as ex:
        logger.error(f"[STEP 2 FAILED] Error during master_daily_rvm integration: {ex}", exc_info=True)
        return False

    elapsed = (datetime.datetime.now() - now).total_seconds()
    logger.info("=" * 80)
    logger.info(f"CORIPET DAILY AUTOMATED ETL FINISHED SUCCESSFULLY in {elapsed:.1f}s")
    logger.info("=" * 80)
    return True


def main():
    parser = argparse.ArgumentParser(description="Automated Coripet ETL runner.")
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="Number of rolling days to fetch up to yesterday (default: 2)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional custom start date YYYY-MM-DD"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional custom end date YYYY-MM-DD"
    )
    args = parser.parse_args()

    success = execute_daily_sync(
        days_back=args.days,
        custom_start=args.start_date,
        custom_end=args.end_date
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
