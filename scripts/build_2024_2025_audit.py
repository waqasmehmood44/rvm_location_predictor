"""
Builds comprehensive 2024-2025 Power BI Telemetry Audit CSV.
Extracts all stores from the test holdout with telemetry in 2024 and 2025.
Computes:
- Power BI Actuals for 2024
- Power BI Actuals for 2025
- Power BI Combined 2024-2025 Mean
- Stacking Ensemble and Neural Network Model Predictions
- Yearly Accuracy Scorecards
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

def build_2024_2025_audit():
    parquet_path = "models/preprocessing/test_holdout.parquet"
    pred_path = "reports/holdout_store_predictions_331.csv"
    output_csv = "reports/holdout_stores_2024_2025_audit.csv"
    
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    df['dt'] = pd.to_datetime(df['GIORNO'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    if df['dt'].isna().sum() > len(df) * 0.5:
        df['dt'] = pd.to_datetime(df['GIORNO'], errors='coerce')
    df['year'] = df['dt'].dt.year
    
    if 'DataPrimaAttivazione' in df.columns:
        df['act_dt'] = pd.to_datetime(df['DataPrimaAttivazione'], errors='coerce')
        df['activation_year'] = df['act_dt'].dt.year
    else:
        df['activation_year'] = np.nan

    # Load previously computed predictions
    df_preds = pd.read_csv(pred_path)
    preds_dict = df_preds.set_index("PF").to_dict(orient="index")
    
    # Filter 2024 and 2025 telemetry
    df_24_25 = df[df['year'].isin([2024, 2025])].copy()
    pfs = df_24_25['PF'].unique()
    print(f"Total unique stores with 2024-2025 telemetry: {len(pfs)}")
    
    # Helpers
    def first_val(s):
        v = s.dropna()
        return v.iloc[0] if not v.empty else "UNKNOWN"
        
    records = []
    for pf in pfs:
        sub = df_24_25[df_24_25['PF'] == pf]
        sub_24 = sub[sub['year'] == 2024]
        sub_25 = sub[sub['year'] == 2025]
        
        desc = first_val(sub['Descrizione'])
        store_name = first_val(sub['CLIENTE'])
        insegna = first_val(sub['nielsen_insegna'])
        gruppo = first_val(sub['nielsen_gruppo'])
        comune = first_val(sub['Comune'])
        provincia = first_val(sub['Provincia'])
        regione = first_val(sub['Regione'])
        machine = first_val(sub['TipoMacchina'])
        act_yr = sub['activation_year'].dropna().iloc[0] if not sub['activation_year'].dropna().empty else "N/A"
        
        # Telemetry actuals
        b_24 = round(float(sub_24['BOTTIGLIE'].mean()), 1) if not sub_24.empty else np.nan
        b_25 = round(float(sub_25['BOTTIGLIE'].mean()), 1) if not sub_25.empty else np.nan
        b_avg = round(float(sub['BOTTIGLIE'].mean()), 1)
        
        p_24 = round(float(sub_24['PRESENZE'].mean()), 1) if not sub_24.empty else np.nan
        p_25 = round(float(sub_25['PRESENZE'].mean()), 1) if not sub_25.empty else np.nan
        p_avg = round(float(sub['PRESENZE'].mean()), 1)
        
        days_24 = int(len(sub_24))
        days_25 = int(len(sub_25))
        days_tot = int(len(sub))
        
        # Model predictions
        p_info = preds_dict.get(pf, {})
        pred_b = p_info.get("ens_pred_mean_b", np.nan)
        pred_p = p_info.get("ens_pred_mean_p", np.nan)
        
        # Accuracy calculations vs Power BI Actuals
        # 1. Vs Combined 2024-2025 Average
        diff_avg = round(pred_b - b_avg, 1) if pd.notna(pred_b) and pd.notna(b_avg) else np.nan
        pct_err_avg = round(abs(diff_avg) / b_avg * 100.0, 1) if b_avg > 0 else np.nan
        acc_avg = round(max(0.0, 100.0 - pct_err_avg), 1) if pd.notna(pct_err_avg) else np.nan
        
        # 2. Vs 2024 Actual
        diff_24 = round(pred_b - b_24, 1) if pd.notna(pred_b) and pd.notna(b_24) else np.nan
        pct_err_24 = round(abs(diff_24) / b_24 * 100.0, 1) if pd.notna(b_24) and b_24 > 0 else np.nan
        acc_24 = round(max(0.0, 100.0 - pct_err_24), 1) if pd.notna(pct_err_24) else np.nan

        # 3. Vs 2025 Actual
        diff_25 = round(pred_b - b_25, 1) if pd.notna(pred_b) and pd.notna(b_25) else np.nan
        pct_err_25 = round(abs(diff_25) / b_25 * 100.0, 1) if pd.notna(b_25) and b_25 > 0 else np.nan
        acc_25 = round(max(0.0, 100.0 - pct_err_25), 1) if pd.notna(pct_err_25) else np.nan
        
        # Volume Tier
        if b_avg >= 1200:
            tier = "Tier 1: Exceptional (> 1,200 b/d)"
        elif b_avg >= 600:
            tier = "Tier 2: High (600 - 1,200 b/d)"
        elif b_avg >= 300:
            tier = "Tier 3: Moderate (300 - 600 b/d)"
        else:
            tier = "Tier 4: Low (< 300 b/d)"
            
        records.append({
            "Descrizione": desc,
            "PF": pf,
            "comune": comune,
            "provincia": provincia,
            "regione": regione,
            "machine_type": machine,
            "store_name": store_name,
            "insegna": insegna,
            "gruppo": gruppo,
            "activation_year": act_yr,
            "days_2024": days_24,
            "days_2025": days_25,
            "total_days_24_25": days_tot,
            "powerbi_actual_2024_b": b_24,
            "powerbi_actual_2025_b": b_25,
            "powerbi_actual_24_25_avg_b": b_avg,
            "ai_predicted_b": pred_b,
            "variance_vs_avg_b": diff_avg,
            "accuracy_vs_24_25_avg_pct": acc_avg,
            "accuracy_vs_2024_pct": acc_24,
            "accuracy_vs_2025_pct": acc_25,
            "powerbi_actual_2024_p": p_24,
            "powerbi_actual_2025_p": p_25,
            "powerbi_actual_24_25_avg_p": p_avg,
            "ai_predicted_p": pred_p,
            "volume_tier": tier
        })
        
    res_df = pd.DataFrame(records)
    res_df = res_df.sort_values(by="powerbi_actual_24_25_avg_b", ascending=False)
    res_df.to_csv(output_csv, index=False)
    print(f"Successfully saved {len(res_df)} stores to {output_csv}!")
    
    # Print preview
    print("\n--- SAMPLE PREVIEW: TOP 2024/2025 STORES FOR POWER BI VERIFICATION ---")
    cols_preview = ["Descrizione", "comune", "machine_type", "powerbi_actual_2024_b", "powerbi_actual_2025_b", "ai_predicted_b", "accuracy_vs_24_25_avg_pct"]
    print(res_df[cols_preview].head(15).to_string(index=False))

if __name__ == "__main__":
    build_2024_2025_audit()
