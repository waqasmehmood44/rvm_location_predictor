"""
Adds Coop Lombardia di Cantù - CC Mirabello (PF00001305) to the benchmark dashboard,
synchronizing benchmark_tests.json, benchmark_tests.csv, dashboard_monthly_records.csv,
dashboard_dow_records.csv, reports/accuracy_dashboard.html, and model_accuracy_dashboard.png.
"""

import os
import sys
import json
import shutil
import pandas as pd
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from scripts.build_general_dashboard import generate_dashboard
from scripts.benchmark_visualizer import compute_metrics, plot_accuracy_dashboard, OUTPUT_DASHBOARD_PATH, ARTIFACT_DASHBOARD_PATHS

def main():
    json_path = "reports/benchmark_tests.json"
    csv_path = "reports/benchmark_tests.csv"
    monthly_csv_path = "reports/dashboard_monthly_records.csv"
    dow_csv_path = "reports/dashboard_dow_records.csv"
    
    with open(json_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)
        
    coop_cantu_entry = {
        "id": "pf00001305_coop_lombardia_cantu",
        "pf": "PF00001305",
        "store_name": "Coop Lombardia Cantù - CC Mirabello",
        "comune": "Cantù",
        "provincia": "CO",
        "address": "Via Lombardia, 68",
        "cap": "22063",
        "machine_type": "DAINT_G",
        "format": "Regional Hypermarket (7,200 m2, 34 checkouts)",
        "category": "Regional Hypermarket Anchor",
        "clearance": "GO [PRIORITY APPROVED]",
        "pred_bottles": 720.9,
        "actual_bottles": 1030.0,
        "pred_users": 23.3,
        "actual_users": 37.0,
        "serviced_floor_bottles": 591.1,
        "p75_bottles": 901.1,
        "p25_bottles": 591.1,
        "monthly_preds": [
            627.1, 627.7, 642.6, 678.6, 728.7, 794.0,
            823.9, 790.6, 728.7, 672.2, 643.3, 632.1
        ],
        "monthly_actuals": [
            1008.1, 1068.5, 1083.9, 1078.2, 1148.7, 1045.1,
            1155.7, 1044.3, 1131.0, 1080.9, 953.4, 1006.2
        ],
        "monthly_preds_total": [
            19440, 17576, 19921, 20358, 22590, 23820,
            25541, 24509, 21861, 20838, 19299, 19595
        ],
        "monthly_actuals_total": [
            31252, 29918, 33602, 32346, 35610, 31354,
            35826, 32373, 33931, 33507, 28603, 31193
        ]
    }
    
    # Remove existing if any, then append
    benchmarks = [b for b in benchmarks if b.get("pf") != "PF00001305"]
    benchmarks.append(coop_cantu_entry)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2, ensure_ascii=False)
    print(f"Updated {json_path} (Total stores: {len(benchmarks)})")
    
    # 2. Update benchmark_tests.csv
    csv_rows = []
    for b in benchmarks:
        pb = b["pred_bottles"]
        ab = b["actual_bottles"]
        diff_b = round(pb - ab, 1)
        pct_b = round((pb - ab) / ab * 100.0, 1) if ab > 0 else 0.0
        acc_b = round(max(0.0, 100.0 - abs(pct_b)), 1)
        
        pu = b["pred_users"]
        au = b["actual_users"]
        diff_u = round(pu - au, 1)
        pct_u = round((pu - au) / au * 100.0, 1) if au > 0 else 0.0
        acc_u = round(max(0.0, 100.0 - abs(pct_u)), 1)
        
        csv_rows.append({
            "store_name": b["store_name"],
            "comune": b["comune"],
            "provincia": b["provincia"],
            "category": b["category"],
            "pred_bottles": pb,
            "actual_bottles": ab,
            "error_bottles": diff_b,
            "pct_error_bottles": pct_b,
            "accuracy_bottles": acc_b,
            "pred_users": pu,
            "actual_users": au,
            "error_users": diff_u,
            "pct_error_users": pct_u,
            "accuracy_users": acc_u
        })
    df_csv = pd.DataFrame(csv_rows)
    df_csv.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} with {len(df_csv)} benchmark rows.")
    
    # 3. Update dashboard_monthly_records.csv
    if os.path.exists(monthly_csv_path):
        df_monthly = pd.read_csv(monthly_csv_path)
        df_monthly = df_monthly[df_monthly["punto_fisico"] != "PF00001305"]
    else:
        df_monthly = pd.DataFrame()
        
    month_names = [
        "01_Gennaio", "02_Febbraio", "03_Marzo", "04_Aprile", "05_Maggio", "06_Giugno",
        "07_Luglio", "08_Agosto", "09_Settembre", "10_Ottobre", "11_Novembre", "12_Dicembre"
    ]
    days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    contexts = [
        "Winter baseline / off-peak",
        "Short winter cycle",
        "Early spring resumption",
        "Spring baseline",
        "Late spring ramp-up",
        "Early summer hydration ramp",
        "Summer peak hydration",
        "Mid-summer holiday rhythm",
        "School / work return",
        "Autumn steady-state",
        "Late autumn transition",
        "Holiday shopping period"
    ]
    
    new_monthly_rows = []
    for m_idx in range(12):
        pb_day = coop_cantu_entry["monthly_preds"][m_idx]
        ab_day = coop_cantu_entry["monthly_actuals"][m_idx]
        days = days_in_months[m_idx]
        pb_tot = coop_cantu_entry["monthly_preds_total"][m_idx]
        ab_tot = coop_cantu_entry["monthly_actuals_total"][m_idx]
        
        diff_day = round(pb_day - ab_day, 1)
        pct_err = round((pb_day - ab_day) / ab_day * 100.0, 1)
        acc_pct = round(max(0.0, 100.0 - abs(pct_err)), 1)
        
        # Presenze actuals from holdout
        act_users_monthly = [34.5, 37.7, 38.0, 36.8, 37.8, 36.1, 37.3, 37.0, 41.0, 39.1, 34.6, 34.2]
        au_day = act_users_monthly[m_idx]
        pu_day = 23.3
        
        new_monthly_rows.append({
            "store_name": coop_cantu_entry["store_name"],
            "punto_fisico": coop_cantu_entry["pf"],
            "comune": coop_cantu_entry["comune"],
            "provincia": coop_cantu_entry["provincia"],
            "month_num": m_idx + 1,
            "month_name": month_names[m_idx],
            "days_in_month": days,
            "pred_bottles_day": pb_day,
            "actual_bottles_day": ab_day,
            "diff_bottles_day": diff_day,
            "pct_error_bottles": pct_err,
            "accuracy_bottles_pct": acc_pct,
            "pred_monthly_total": pb_tot,
            "actual_monthly_total": ab_tot,
            "pred_presenze_day": pu_day,
            "actual_presenze_day": au_day,
            "pred_presenze_total": round(pu_day * days),
            "actual_presenze_total": round(au_day * days),
            "operational_context": contexts[m_idx]
        })
    df_new_monthly = pd.DataFrame(new_monthly_rows)
    df_combined_monthly = pd.concat([df_monthly, df_new_monthly], ignore_index=True)
    df_combined_monthly.to_csv(monthly_csv_path, index=False)
    print(f"Updated {monthly_csv_path} with {len(df_combined_monthly)} total monthly records.")
    
    # 4. Update dashboard_dow_records.csv
    if os.path.exists(dow_csv_path):
        df_dow = pd.read_csv(dow_csv_path)
        df_dow = df_dow[df_dow["punto_fisico"] != "PF00001305"]
    else:
        df_dow = pd.DataFrame()
        
    dow_rows = [
        {"day_of_week": "Monday", "mean": 711.2, "median": 681.6, "context": "Weekday baseline"},
        {"day_of_week": "Tuesday", "mean": 689.9, "median": 665.2, "context": "Weekday baseline"},
        {"day_of_week": "Wednesday", "mean": 691.4, "median": 669.8, "context": "Weekday baseline"},
        {"day_of_week": "Thursday", "mean": 697.0, "median": 675.4, "context": "Weekday baseline"},
        {"day_of_week": "Friday", "mean": 728.1, "median": 702.3, "context": "Pre-weekend shopping ramp"},
        {"day_of_week": "Saturday", "mean": 846.1, "median": 876.3, "context": "Peak Saturday family grocery trip"},
        {"day_of_week": "Sunday", "mean": 682.3, "median": 651.1, "context": "Sunday shopping"}
    ]
    new_dow_rows = []
    for d in dow_rows:
        new_dow_rows.append({
            "store_name": coop_cantu_entry["store_name"],
            "punto_fisico": coop_cantu_entry["pf"],
            "day_of_week": d["day_of_week"],
            "pred_bottles_mean": d["mean"],
            "pred_bottles_median": d["median"],
            "operational_context": d["context"]
        })
    df_new_dow = pd.DataFrame(new_dow_rows)
    df_combined_dow = pd.concat([df_dow, df_new_dow], ignore_index=True)
    df_combined_dow.to_csv(dow_csv_path, index=False)
    print(f"Updated {dow_csv_path} with {len(df_combined_dow)} total day-of-week records.")
    
    # 5. Generate accuracy_dashboard.html
    print("Rebuilding reports/accuracy_dashboard.html...")
    generate_dashboard()
    print("Successfully rebuilt reports/accuracy_dashboard.html")
    
    # 6. Rebuild model_accuracy_dashboard.png
    print("Rebuilding model_accuracy_dashboard.png...")
    results = compute_metrics(benchmarks)
    plot_accuracy_dashboard(results, [OUTPUT_DASHBOARD_PATH] + ARTIFACT_DASHBOARD_PATHS)
    print("Successfully rebuilt and saved model_accuracy_dashboard.png to reports and artifacts.")
    
    # 7. Print summary metrics
    preds = np.array([r["pred_bottles"] for r in results])
    acts = np.array([r["actual_bottles"] for r in results])
    accuracies = np.array([r["accuracy_bottles"] for r in results])
    mae = np.mean(np.abs(preds - acts))
    wape = (np.sum(np.abs(preds - acts)) / np.sum(acts)) * 100.0
    mean_acc = np.mean(accuracies)
    median_acc = np.median(accuracies)
    print("\n" + "=" * 60)
    print(f"DASHBOARD UPDATE COMPLETE — 7 BENCHMARK STORES ACTIVE")
    print(f"Network Mean Accuracy:   {mean_acc:.1f}%")
    print(f"Network Median Accuracy: {median_acc:.1f}%")
    print(f"Weighted APE (WAPE):     {wape:.1f}%")
    print(f"Mean Absolute Error:     {mae:.1f} bottles/day")
    print("=" * 60)

if __name__ == "__main__":
    main()
