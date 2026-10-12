import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Ensure UTF-8 output encoding
sys.stdout.reconfigure(encoding='utf-8')

BENCHMARK_REGISTRY_PATH = os.path.join("reports", "benchmark_tests.json")
OUTPUT_DASHBOARD_PATH = os.path.join("reports", "model_accuracy_dashboard.png")
ARTIFACT_DASHBOARD_PATHS = [
    r"C:\Users\waqas\.gemini\antigravity-ide\brain\dfcfda3e-08d6-4e08-8441-294e599fdf4a\model_accuracy_dashboard.png",
    r"C:\Users\waqas\.gemini\antigravity-ide\brain\2b763cb3-efcc-44bc-8c7a-65fece39af4d\model_accuracy_dashboard.png"
]

# Default seed benchmarks tested across the platform
DEFAULT_BENCHMARKS = [
    {
        "store_name": "Conad Sarre",
        "comune": "Sarre",
        "provincia": "AO",
        "pred_bottles": 536.3,
        "actual_bottles": 540.0,
        "pred_users": 12.4,
        "actual_users": 12.5,
        "category": "High-Volume Anchor"
    },
    {
        "store_name": "Conad Fano",
        "comune": "Fano",
        "provincia": "PU",
        "pred_bottles": 212.4,
        "actual_bottles": 215.0,
        "pred_users": 7.4,
        "actual_users": 7.5,
        "category": "Standard Supermarket"
    },
    {
        "store_name": "Decò Pontelatone",
        "comune": "Pontelatone",
        "provincia": "CE",
        "pred_bottles": 263.1,
        "actual_bottles": 304.4,
        "pred_users": 6.8,
        "actual_users": 7.0,
        "category": "Rural Monopoly Anchor"
    },
    {
        "store_name": "Merkante Caserta",
        "comune": "Caserta",
        "provincia": "CE",
        "pred_bottles": 173.3,
        "actual_bottles": 189.6,
        "pred_users": 5.2,
        "actual_users": 5.8,
        "category": "Diluted Cluster Store"
    },
    {
        "store_name": "Conad Trieste",
        "comune": "Trieste",
        "provincia": "TS",
        "pred_bottles": 339.4,
        "actual_bottles": 298.0,
        "pred_users": 14.2,
        "actual_users": 14.0,
        "category": "Urban Cluster Store"
    },
    {
        "store_name": "Penny Ascoli Piceno",
        "comune": "Ascoli Piceno",
        "provincia": "AP",
        "pred_bottles": 110.3,
        "actual_bottles": 143.7,
        "pred_users": 4.9,
        "actual_users": 5.1,
        "category": "Suburban Discount"
    },
    {
        "store_name": "Famila Santo Stefano",
        "comune": "Monte Argentario",
        "provincia": "GR",
        "pred_bottles": 120.6,
        "actual_bottles": 103.1,
        "pred_users": 4.8,
        "actual_users": 4.6,
        "category": "Coastal Tourist Supermarket"
    },
    {
        "store_name": "Iper Futura Battipaglia",
        "comune": "Battipaglia",
        "provincia": "SA",
        "pred_bottles": 267.1,
        "actual_bottles": 350.1,
        "pred_users": 8.3,
        "actual_users": 12.5,
        "category": "Southern Independent Anchor"
    },
    {
        "store_name": "Conad Marino",
        "comune": "Marino",
        "provincia": "RM",
        "pred_bottles": 869.3,
        "actual_bottles": 726.0,
        "pred_users": 27.0,
        "actual_users": 24.5,
        "category": "Flagship High-Volume Anchor"
    },
    {
        "store_name": "Interspar Castelfranco",
        "comune": "Castelfranco Veneto",
        "provincia": "TV",
        "pred_bottles": 997.3,
        "actual_bottles": 549.8,
        "pred_users": 29.7,
        "actual_users": 17.2,
        "category": "Mega Hypermarket (1-Machine Cap)"
    }
]

def load_benchmarks():
    if os.path.exists(BENCHMARK_REGISTRY_PATH):
        try:
            with open(BENCHMARK_REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_BENCHMARKS

def save_benchmarks(benchmarks):
    import csv
    os.makedirs(os.path.dirname(BENCHMARK_REGISTRY_PATH), exist_ok=True)
    with open(BENCHMARK_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2, ensure_ascii=False)
    
    # Also export CSV
    csv_path = os.path.splitext(BENCHMARK_REGISTRY_PATH)[0] + ".csv"
    results = compute_metrics(benchmarks)
    fieldnames = [
        'store_name', 'comune', 'provincia', 'category',
        'pred_bottles', 'actual_bottles', 'error_bottles', 'pct_error_bottles', 'accuracy_bottles',
        'pred_users', 'actual_users', 'error_users', 'pct_error_users', 'accuracy_users'
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

def compute_metrics(benchmarks):
    results = []
    for b in benchmarks:
        pred_b = b["pred_bottles"]
        act_b = b["actual_bottles"]
        pred_u = b.get("pred_users", 0.0)
        act_u = b.get("actual_users", 0.0)

        err_b = pred_b - act_b
        pct_err_b = (err_b / act_b) * 100.0 if act_b > 0 else 0.0
        acc_b = max(0.0, 100.0 - abs(pct_err_b))

        err_u = pred_u - act_u if act_u > 0 else 0.0
        pct_err_u = (err_u / act_u) * 100.0 if act_u > 0 else 0.0
        acc_u = max(0.0, 100.0 - abs(pct_err_u)) if act_u > 0 else 0.0

        results.append({
            **b,
            "error_bottles": round(err_b, 1),
            "pct_error_bottles": round(pct_err_b, 1),
            "accuracy_bottles": round(acc_b, 1),
            "error_users": round(err_u, 1),
            "pct_error_users": round(pct_err_u, 1),
            "accuracy_users": round(acc_u, 1)
        })
    return results

def plot_accuracy_dashboard(results, output_paths):
    # Overall summary metrics
    preds = np.array([r["pred_bottles"] for r in results])
    acts = np.array([r["actual_bottles"] for r in results])
    accuracies = np.array([r["accuracy_bottles"] for r in results])
    
    mae = np.mean(np.abs(preds - acts))
    wape = (np.sum(np.abs(preds - acts)) / np.sum(acts)) * 100.0
    mean_acc = np.mean(accuracies)
    median_acc = np.median(accuracies)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('#F8FAF9')

    # Color palette
    C_PRIMARY = '#104B35'     # Deep Forest Green
    C_SECONDARY = '#1E824C'   # Emerald Green
    C_AMBER = '#D97706'       # Amber Warning
    C_RED = '#DC2626'         # Crimson Red
    C_BLUE = '#2563EB'        # Slate Blue

    # Title header
    fig.suptitle(
        f"Coripet RVM Predictor — Model Accuracy & Benchmark Validation\n"
        f"Network Accuracy: {mean_acc:.1f}% (Mean) | {median_acc:.1f}% (Median) | WAPE: {wape:.1f}% | MAE: {mae:.1f} b/day",
        fontsize=15, fontweight='bold', color=C_PRIMARY, y=0.98
    )

    # -------------------------------------------------------------
    # GRAPH 1: Store-by-Store Daily Volume Comparison (Predicted vs Actual)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_facecolor('#FFFFFF')
    names = [r["store_name"].replace("Conad ", "").replace("Market ", "") for r in results]
    x = np.arange(len(names))
    bar_w = 0.38

    b1 = ax1.bar(x - bar_w/2, preds, width=bar_w, label='Model Predicted (b/day)', color=C_SECONDARY, alpha=0.9, edgecolor='black', linewidth=0.8)
    b2 = ax1.bar(x + bar_w/2, acts, width=bar_w, label='Power BI Actual (b/day)', color=C_BLUE, alpha=0.85, edgecolor='black', linewidth=0.8)

    # Annotate bar values
    for bar in b1:
        h = bar.get_height()
        if h > 0:
            ax1.annotate(f"{h:.0f}", (bar.get_x() + bar.get_width()/2, h), textcoords="offset points", xytext=(0, 3), ha='center', fontsize=7.5, fontweight='bold', color=C_SECONDARY)
    for bar in b2:
        h = bar.get_height()
        if h > 0:
            ax1.annotate(f"{h:.0f}", (bar.get_x() + bar.get_width()/2, h), textcoords="offset points", xytext=(0, 3), ha='center', fontsize=7.5, fontweight='bold', color=C_BLUE)

    ax1.set_title("1. Daily Collection Volume Comparison (Model Predicted vs. Power BI Ground Truth)", fontsize=12, fontweight='bold', color=C_PRIMARY, pad=10)
    ax1.set_ylabel("Daily Bottles (bottles/day)", fontsize=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha='right', fontsize=9, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # -------------------------------------------------------------
    # GRAPH 2: Store-by-Store Accuracy Percentage Ranking
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_facecolor('#FFFFFF')
    
    # Sort by accuracy
    sorted_res = sorted(results, key=lambda r: r["accuracy_bottles"])
    s_names = [r["store_name"].replace("Conad ", "").replace("Market ", "") for r in sorted_res]
    s_acc = [r["accuracy_bottles"] for r in sorted_res]
    s_err = [r["pct_error_bottles"] for r in sorted_res]
    y_pos = np.arange(len(s_names))

    bar_colors = [C_SECONDARY if a >= 75 else (C_AMBER if a >= 60 else C_RED) for a in s_acc]
    bars = ax2.barh(y_pos, s_acc, color=bar_colors, edgecolor='black', linewidth=0.8, height=0.65)

    # Reference lines
    ax2.axvline(75, color=C_SECONDARY, linestyle=':', linewidth=1.5, label='75% Target Benchmark')
    ax2.axvline(mean_acc, color=C_PRIMARY, linestyle='--', linewidth=2, label=f'Mean Network Accuracy ({mean_acc:.1f}%)')

    # Data labels on horizontal bars
    for idx, (b_bar, val, err) in enumerate(zip(bars, s_acc, s_err)):
        sign = "+" if err > 0 else ""
        ax2.text(
            val + 1.2,
            b_bar.get_y() + b_bar.get_height()/2,
            f"{val:.1f}% (Err: {sign}{err:.1f}%)",
            va='center',
            fontsize=8.5,
            fontweight='bold',
            color='#1F2937'
        )

    ax2.set_title("2. Store-by-Store Model Accuracy Ranking (%)", fontsize=12, fontweight='bold', color=C_PRIMARY, pad=10)
    ax2.set_xlabel("Accuracy Percentage (100% - Absolute Error %)", fontsize=10, fontweight='bold')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(s_names, fontsize=9, fontweight='bold')
    ax2.set_xlim(0, 115)
    ax2.legend(loc='lower right', fontsize=9, frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    for out in output_paths:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Accuracy dashboard saved to: {out}")
    plt.close(fig)

def print_summary_table(results):
    print("=" * 95)
    print(f"{'STORE NAME':<25} | {'PRED (b/d)':<10} | {'ACTUAL (b/d)':<12} | {'ERROR %':<10} | {'ACCURACY %':<10} | {'USERS ACC %':<10}")
    print("=" * 95)
    for r in sorted(results, key=lambda x: x["accuracy_bottles"], reverse=True):
        sign = "+" if r["pct_error_bottles"] > 0 else ""
        u_acc = f"{r['accuracy_users']:.1f}%" if r['accuracy_users'] > 0 else "N/A"
        print(f"{r['store_name']:<25} | {r['pred_bottles']:<10.1f} | {r['actual_bottles']:<12.1f} | {sign + str(r['pct_error_bottles']) + '%':<10} | {str(r['accuracy_bottles']) + '%':<10} | {u_acc:<10}")
    print("=" * 95)
    mean_acc = np.mean([r["accuracy_bottles"] for r in results])
    wape = (np.sum([abs(r["error_bottles"]) for r in results]) / np.sum([r["actual_bottles"] for r in results])) * 100.0
    print(f"OVERALL NETWORK ACCURACY: {mean_acc:.1f}% | OVERALL NETWORK WAPE: {wape:.1f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coripet RVM Model Accuracy Tracker & Visualizer")
    parser.add_argument("--add", type=str, help="Store name to add")
    parser.add_argument("--comune", type=str, default="N/A", help="Comune")
    parser.add_argument("--provincia", type=str, default="N/A", help="Provincia")
    parser.add_argument("--pred_b", type=float, help="Predicted daily bottles")
    parser.add_argument("--actual_b", type=float, help="Power BI actual daily bottles")
    parser.add_argument("--pred_u", type=float, default=0.0, help="Predicted daily users")
    parser.add_argument("--actual_u", type=float, default=0.0, help="Power BI actual daily users")
    parser.add_argument("--category", type=str, default="Benchmark Store", help="Category")
    args = parser.parse_args()

    benchmarks = load_benchmarks()

    if args.add and args.pred_b is not None and args.actual_b is not None:
        # Check if store already exists, update if so
        exists = False
        for b in benchmarks:
            if b["store_name"].strip().lower() == args.add.strip().lower():
                b["pred_bottles"] = args.pred_b
                b["actual_bottles"] = args.actual_b
                b["pred_users"] = args.pred_u
                b["actual_users"] = args.actual_u
                if args.category and args.category != "Benchmark Store":
                    b["category"] = args.category
                exists = True
                break
        if not exists:
            benchmarks.append({
                "store_name": args.add,
                "comune": args.comune,
                "provincia": args.provincia,
                "pred_bottles": args.pred_b,
                "actual_bottles": args.actual_b,
                "pred_users": args.pred_u,
                "actual_users": args.actual_u,
                "category": args.category
            })
        print(f"Saved benchmark test for: {args.add}")

    # Always ensure JSON and CSV are synced
    save_benchmarks(benchmarks)

    # Compute metrics and plot
    results = compute_metrics(benchmarks)
    print_summary_table(results)
    valid_paths = [OUTPUT_DASHBOARD_PATH] + [p for p in ARTIFACT_DASHBOARD_PATHS if os.path.exists(os.path.dirname(p))]
    plot_accuracy_dashboard(results, valid_paths)
