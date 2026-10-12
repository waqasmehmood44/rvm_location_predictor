import os
import sys
import json
import logging

sys.path.insert(0, os.path.abspath("."))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_PATH = "reports/benchmark_tests.json"

def add_coop_zola_benchmark():
    if not os.path.exists(BENCHMARK_PATH):
        raise FileNotFoundError(f"{BENCHMARK_PATH} not found.")

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    logger.info(f"Loaded {len(benchmarks)} existing benchmarks.")

    coop_store = {
        "id": "pf00002236_zolapredosa",
        "pf": "PF00002236",
        "store_name": "Coop Alleanza 3.0 - Zola Predosa",
        "comune": "Zola Predosa",
        "provincia": "BO",
        "address": "Via delle Officine, 27",
        "cap": "40069",
        "machine_type": "DAINT_G",
        "format": "Supermarket (1,497 m2, 10 checkouts)",
        "category": "Strategic High-Volume Anchor",
        "clearance": "GO [APPROVED]",
        "pred_bottles": 1510.6,
        "actual_bottles": 1497.8,
        "pred_users": 46.3,
        "actual_users": 45.3,
        "serviced_floor_bottles": 1284.0,
        "p75_bottles": 1737.2,
        "p25_bottles": 1284.0,
        "monthly_preds": [
            1298.5, 1301.1, 1333.2, 1504.5, 1557.2, 1705.5,
            1699.1, 1671.0, 1519.3, 1462.8, 1306.4, 1283.1
        ],
        "monthly_actuals": [
            1074.4, 1143.4, 1352.0, 1427.4, 1637.5, 1978.0,
            1850.3, 1646.8, 1810.9, 1730.2, 1353.3, 1276.5
        ],
        "monthly_preds_total": [
            40254, 36431, 41329, 45135, 48273, 51165,
            52672, 51801, 45579, 45347, 39192, 39776
        ],
        "monthly_actuals_total": [
            33307, 32015, 41911, 42822, 50763, 59341,
            57360, 51052, 54326, 53636, 40598, 39573
        ]
    }

    # Check if already present
    existing_idx = None
    for i, b in enumerate(benchmarks):
        if b.get("id") == coop_store["id"] or b.get("pf") == coop_store["pf"]:
            existing_idx = i
            break

    if existing_idx is not None:
        logger.info(f"Updating existing record at index {existing_idx}...")
        benchmarks[existing_idx] = coop_store
    else:
        logger.info("Appending new benchmark record...")
        benchmarks.append(coop_store)

    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(benchmarks)} benchmarks to {BENCHMARK_PATH}.")

    # Rebuild general dashboard HTML
    from scripts.build_general_dashboard import generate_dashboard
    logger.info("Rebuilding reports/accuracy_dashboard.html...")
    generate_dashboard()

    # Rebuild visualizer artifacts and CSV
    from scripts.benchmark_visualizer import save_benchmarks, compute_metrics, plot_accuracy_dashboard, OUTPUT_DASHBOARD_PATH, ARTIFACT_DASHBOARD_PATHS
    logger.info("Syncing CSV and model accuracy dashboard PNGs...")
    save_benchmarks(benchmarks)
    results = compute_metrics(benchmarks)
    valid_paths = [OUTPUT_DASHBOARD_PATH] + [p for p in ARTIFACT_DASHBOARD_PATHS if os.path.exists(os.path.dirname(p))]
    plot_accuracy_dashboard(results, valid_paths)
    logger.info("Successfully updated Accuracy Dashboard and visualizer artifacts!")

if __name__ == "__main__":
    add_coop_zola_benchmark()
