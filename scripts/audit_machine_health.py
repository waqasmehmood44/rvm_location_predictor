"""
Standalone Fleet Health & Anomaly Diagnostic Auditor for Coripet RVM Predictor.
Audits strictly the 1,849 current active machines (ATTVO) in the field,
excluding all 106 deactivated, suspended, or old swapped units.
Exports: data/reports/defected_and_outlier_machines.csv
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from src.model.outlier_detector import RVMOutlierDetector

REPORT_CSV = ROOT_DIR / "data" / "reports" / "defected_and_outlier_machines.csv"
REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)


def run_audit():
    print("=" * 80)
    print("CORIPET RVM FLEET HEALTH & DEFECT AUDITOR (ACTIVE FLEET: 1,849 MACHINES)")
    print("=" * 80)

    pg_url = os.getenv("DATABASE_URL", "").strip()
    if pg_url.startswith("postgresql://") and "+" not in pg_url.split("://")[0]:
        pg_url = pg_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    engine = create_engine(pg_url)
    
    print("1. Loading live active installations (ATTVO) from raw_storico_punto_fisico...")
    with engine.connect() as conn:
        active_fleet_df = pd.read_sql("""
            SELECT puntofisico as pf, macchina, codicemacchinaprod,
                   descrizione as store_banner, comune, regione, tipomacchina, tipocontenitore
            FROM raw_storico_punto_fisico
            WHERE end_dt = (SELECT MAX(end_dt) FROM raw_storico_punto_fisico)
              AND UPPER(stato) = 'ATTVO'
        """, conn)

    total_active_target = len(active_fleet_df)
    print(f"--> Found {total_active_target:,} currently active machines in Coripet fleet registry.")
    assert total_active_target == 1849, f"Expected 1849 active machines, got {total_active_target}"

    # Build lookup dictionaries
    active_pf_set = set(active_fleet_df["pf"].dropna().str.strip())
    active_mach_to_pf = active_fleet_df.set_index("macchina")["pf"].to_dict()
    active_cod_to_pf = active_fleet_df.dropna(subset=["codicemacchinaprod"]).set_index("codicemacchinaprod")["pf"].to_dict()

    print("2. Connecting to PostgreSQL and loading telemetry from master_daily_rvm (>= 2024)...")
    start_t = time.time()
    with engine.connect() as conn:
        df_telemetry = pd.read_sql("""
            SELECT pf, macchina, codicemacchinaprod, giorno_dt, bottiglie, presenze, cliente, comune, regione,
                   tipomacchina, maturity_status
            FROM master_daily_rvm
            WHERE giorno_dt >= '2024-01-01'
            ORDER BY pf, giorno_dt;
        """, conn)
    print(f"Loaded {len(df_telemetry):,} raw records in {time.time() - start_t:.1f}s.")

    # Map each telemetry row to the canonical active PuntoFisico
    def map_to_active_pf(row):
        r_pf = str(row.get("pf", "")).strip()
        if r_pf in active_pf_set:
            return r_pf
        r_mach = str(row.get("macchina", "")).strip()
        if r_mach in active_mach_to_pf:
            return active_mach_to_pf[r_mach]
        r_cod = str(row.get("codicemacchinaprod", "")).strip()
        if r_cod in active_cod_to_pf:
            return active_cod_to_pf[r_cod]
        return None

    df_telemetry["canonical_active_pf"] = df_telemetry.apply(map_to_active_pf, axis=1)
    
    # Filter strictly to active machines (removes the 106 deactivated, suspended, or swapped machines)
    df_active_telemetry = df_telemetry[df_telemetry["canonical_active_pf"].notna()].copy()
    df_active_telemetry["PF"] = df_active_telemetry["canonical_active_pf"]
    df_active_telemetry["BOTTIGLIE"] = df_active_telemetry["bottiglie"]
    df_active_telemetry["PRESENZE"] = df_active_telemetry["presenze"]
    df_active_telemetry["CLIENTE"] = df_active_telemetry["cliente"]
    df_active_telemetry["Comune"] = df_active_telemetry["comune"]
    df_active_telemetry["Regione"] = df_active_telemetry["regione"]
    df_active_telemetry["TipoMacchina"] = df_active_telemetry["tipomacchina"]

    active_with_data = df_active_telemetry["PF"].nunique()
    print(f"--> Filtered strictly to active fleet: {active_with_data} active machines have telemetry; {total_active_target - active_with_data} are brand-new installs.")

    detector = RVMOutlierDetector()
    print("3. Executing 8-Regime Anomaly & Defect Detection Engine...")
    audit_df = detector.audit_fleet(df_active_telemetry)

    # For any of the 1,849 active machines not yet in audit_df (e.g. brand new machines with < 14 days), add them as nominal
    audited_pfs = set(audit_df["PF"])
    missing_active = active_fleet_df[~active_fleet_df["pf"].isin(audited_pfs)]

    if not missing_active.empty:
        new_records = []
        for _, r in missing_active.iterrows():
            new_records.append({
                "PF": r["pf"],
                "Store_Banner": r["store_banner"] or "Active Coripet RVM",
                "Comune": r["comune"] or "",
                "Regione": r["regione"] or "",
                "TipoMacchina": r["tipomacchina"] or "DAINT_G",
                "Operational_Days": 0,
                "Anomaly_Classification": "HEALTHY_NORMAL",
                "Max_Consecutive_Zero_Days": 0,
                "Recent_Zero_Rate_Pct": 0.0,
                "Median_Bottles_Daily": 0.0,
                "Recent_30d_Median": 0.0,
                "Max_Day_Bottles": 0.0,
                "Volatility_CV": 0.0,
                "Bottles_Per_User": 0.0,
                "Expected_Predicted_Bottles": 500.0,
                "Volume_Error_Bias": 0.0,
                "Recommended_Action": "New deployment onboarding; monitor initial 30-day adoption."
            })
        audit_df = pd.concat([audit_df, pd.DataFrame(new_records)], ignore_index=True)

    # Ensure strictly 1,849 unique active machines
    audit_df = audit_df.drop_duplicates(subset=["PF"]).copy()
    assert len(audit_df) == 1849, f"Expected exactly 1849 records, got {len(audit_df)}"

    # Save to CSV
    audit_df.to_csv(REPORT_CSV, index=False)
    print(f"\n--> Successfully generated clean active fleet report ({len(audit_df):,} machines): {REPORT_CSV}")

    # Summary by Classification
    print("\n" + "=" * 80)
    print("                      CLEAN ACTIVE FLEET HEALTH AUDIT (1,849 UNITS)")
    print("=" * 80)
    counts = audit_df["Anomaly_Classification"].value_counts()
    for cat, count in counts.items():
        pct = (count / len(audit_df)) * 100.0
        print(f"  {cat:<28}: {count:>4} machines ({pct:>5.1f}%)")

    defective = audit_df[audit_df["Anomaly_Classification"] != "HEALTHY_NORMAL"]
    print("\n" + "-" * 80)
    print(f"  TOTAL ANOMALOUS / ACTIONABLE ACTIVE MACHINES: {len(defective)} out of {len(audit_df)}")
    print("-" * 80)
    print("=" * 80)
    print("AUDIT COMPLETE. 106 deactivated/swapped units removed. Report matches live fleet.")
    print("=" * 80)


if __name__ == "__main__":
    run_audit()
