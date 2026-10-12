"""
Syncs benchmark_tests.json and benchmark_tests.csv directly from the primary
test holdout dataset (data/processed/test_holdout_stores_summary.csv).
Ensures zero hardcoded/mock entries and guarantees 100% empirical ground truth.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))
from scripts.rebuild_dashboard_from_holdout import rebuild_from_holdout

def update_benchmarks():
    rebuild_from_holdout()

if __name__ == '__main__':
    update_benchmarks()
