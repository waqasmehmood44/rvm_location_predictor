"""
Comprehensive feature audit and target analysis script for Coripet RVM Predictor.
Analyzes modeling_daily_rvm.parquet and raw tables for feature inventory, target distributions,
machine types, Nielsen features, macro features, and temporal aspects.
Generates reports/feature_dictionary.csv and reports/target_stats_summary.json.
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import json
import os

def run_audit():
    print("Loading data/processed/modeling_daily_rvm.parquet...")
    df = pd.read_parquet("data/processed/modeling_daily_rvm.parquet")
    n_rows, n_cols = df.shape
    n_pfs = df["PF"].nunique()
    print(f"Dataset shape: {n_rows:,} rows x {n_cols} columns across {n_pfs:,} unique PFs")

    # =========================================================================
    # 1. Complete Column Audit for Feature Inventory
    # =========================================================================
    col_audit = []
    for col in df.columns:
        s = df[col]
        missing_count = s.isna().sum()
        missing_pct = (missing_count / n_rows) * 100.0
        n_unique = s.nunique(dropna=True)
        dtype_str = str(s.dtype)
        sample_vals = s.dropna().unique()[:3].tolist()
        sample_str = str(sample_vals)[:60]
        
        col_audit.append({
            "column": col,
            "dtype": dtype_str,
            "missing_count": int(missing_count),
            "missing_pct": round(missing_pct, 2),
            "n_unique": int(n_unique),
            "sample": sample_str
        })
    
    audit_df = pd.DataFrame(col_audit)
    print("\n--- Column Audit Overview ---")
    print(audit_df.to_string())

    # =========================================================================
    # 2. Target Distribution Analysis (BOTTIGLIE & PRESENZE)
    # =========================================================================
    print("\n" + "="*80)
    print("TARGET DISTRIBUTION ANALYSIS")
    print("="*80)
    target_stats = {}
    for t in ["BOTTIGLIE", "PRESENZE"]:
        s = df[t].astype(float)
        p1 = float(np.percentile(s, 1))
        p5 = float(np.percentile(s, 5))
        p25 = float(np.percentile(s, 25))
        p50 = float(np.percentile(s, 50))
        p75 = float(np.percentile(s, 75))
        p90 = float(np.percentile(s, 90))
        p95 = float(np.percentile(s, 95))
        p99 = float(np.percentile(s, 99))
        p99_9 = float(np.percentile(s, 99.9))
        
        mean_val = float(s.mean())
        median_val = float(s.median())
        std_val = float(s.std())
        min_val = float(s.min())
        max_val = float(s.max())
        skew_val = float(stats.skew(s))
        kurt_val = float(stats.kurtosis(s))
        zero_count = int((s == 0).sum())
        zero_pct = float(zero_count / n_rows * 100.0)

        # Log1p stats
        log_s = np.log1p(s)
        log_skew = float(stats.skew(log_s))
        log_kurt = float(stats.kurtosis(log_s))
        log_mean = float(log_s.mean())
        log_std = float(log_s.std())
        
        target_stats[t] = {
            "mean": round(mean_val, 2),
            "median": round(median_val, 2),
            "std": round(std_val, 2),
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "p1": round(p1, 2),
            "p5": round(p5, 2),
            "p25": round(p25, 2),
            "p50": round(p50, 2),
            "p75": round(p75, 2),
            "p90": round(p90, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "p99_9": round(p99_9, 2),
            "zero_count": zero_count,
            "zero_pct": round(zero_pct, 3),
            "skewness": round(skew_val, 2),
            "kurtosis": round(kurt_val, 2),
            "log1p_mean": round(log_mean, 2),
            "log1p_std": round(log_std, 2),
            "log1p_skewness": round(log_skew, 2),
            "log1p_kurtosis": round(log_kurt, 2),
        }
        print(f"\n--- Target: {t} ---")
        for k, v in target_stats[t].items():
            print(f"  {k}: {v}")

    # Correlation between targets
    corr = df["BOTTIGLIE"].corr(df["PRESENZE"])
    spearman_corr = df["BOTTIGLIE"].corr(df["PRESENZE"], method="spearman")
    print(f"\nCorrelation between BOTTIGLIE and PRESENZE: Pearson={corr:.4f}, Spearman={spearman_corr:.4f}")

    # Bottles per presence ratio
    df_nonzero = df[df["PRESENZE"] > 0]
    bpp = df_nonzero["BOTTIGLIE"] / df_nonzero["PRESENZE"]
    print(f"Bottles per Presence (when PRESENZE > 0): Median={bpp.median():.2f}, Mean={bpp.mean():.2f}, 90%={bpp.quantile(0.9):.2f}")

    # =========================================================================
    # 3. Machine-Type Variables Audit
    # =========================================================================
    print("\n" + "="*80)
    print("MACHINE VARIABLES AUDIT")
    print("="*80)
    machine_cols = ["TipoMacchina", "Macchina", "CodiceMacchinaProd", "CODICEMACCHINA", "CODICEMACCHINAPROD", "Fornitore"]
    for mc in machine_cols:
        if mc in df.columns:
            vc = df[mc].value_counts(dropna=False)
            print(f"\nColumn: {mc} (Unique: {df[mc].nunique()}, Missing: {df[mc].isna().sum()})")
            print(vc.head(10))

    if "TipoMacchina" in df.columns and "Fornitore" in df.columns:
        print("\nCross-tabulation: TipoMacchina vs Fornitore:")
        print(pd.crosstab(df["TipoMacchina"].fillna("MISSING"), df["Fornitore"].fillna("MISSING"), margins=True))

    if "TipoMacchina" in df.columns:
        print("\nTarget performance by TipoMacchina:")
        tm_perf = df.groupby("TipoMacchina", observed=False).agg(
            observations=("BOTTIGLIE", "count"),
            unique_pfs=("PF", "nunique"),
            bottiglie_mean=("BOTTIGLIE", "mean"),
            bottiglie_median=("BOTTIGLIE", "median"),
            presenze_mean=("PRESENZE", "mean"),
            presenze_median=("PRESENZE", "median"),
        ).round(1)
        print(tm_perf)

    # =========================================================================
    # 4. Location Features Audit (Geographic & Municipality)
    # =========================================================================
    print("\n" + "="*80)
    print("LOCATION FEATURES AUDIT")
    print("="*80)
    geo_cols = ["Latitudine", "Longitudine", "Cap", "Comune", "Provincia", "Regione", "population", "surface", "PIL", "population_density", "PIL_per_capita"]
    for gc in geo_cols:
        if gc in df.columns:
            s = df[gc]
            is_num = pd.api.types.is_numeric_dtype(s.dtype)
            min_v = f"{s.min():.2f}" if is_num and not s.empty else str(s.dropna().min() if not s.dropna().empty else "N/A")
            max_v = f"{s.max():.2f}" if is_num and not s.empty else str(s.dropna().max() if not s.dropna().empty else "N/A")
            print(f"{gc}: missing={s.isna().sum()} ({s.isna().mean()*100:.2f}%), unique={s.nunique()}, min={min_v}, max={max_v}")

    # =========================================================================
    # 5. Nielsen Store Features Audit
    # =========================================================================
    print("\n" + "="*80)
    print("NIELSEN FEATURES AUDIT")
    print("="*80)
    nielsen_cols = [c for c in df.columns if c.startswith("nielsen_")]
    for nc in nielsen_cols:
        s = df[nc]
        print(f"{nc}: missing={s.isna().sum()} ({s.isna().mean()*100:.2f}%), unique={s.nunique()}")
        if s.nunique() <= 10:
            print(f"  values: {s.value_counts(dropna=False).to_dict()}")

    # =========================================================================
    # 6. Temporal Features & Stationarity
    # =========================================================================
    print("\n" + "="*80)
    print("TEMPORAL FEATURES AUDIT")
    print("="*80)
    df["dt"] = pd.to_datetime(df["GIORNO"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df["year"] = df["dt"].dt.year
    df["month"] = df["dt"].dt.month
    df["dayofweek"] = df["dt"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    
    print("\nTarget performance by Day of Week (0=Mon, 6=Sun):")
    print(df.groupby("dayofweek").agg(
        rows=("BOTTIGLIE", "count"),
        bottiglie_mean=("BOTTIGLIE", "mean"),
        presenze_mean=("PRESENZE", "mean")
    ).round(1))

    print("\nTarget performance by Month:")
    print(df.groupby("month").agg(
        rows=("BOTTIGLIE", "count"),
        bottiglie_mean=("BOTTIGLIE", "mean"),
        presenze_mean=("PRESENZE", "mean")
    ).round(1))

    print("\nTarget performance by Year:")
    print(df.groupby("year").agg(
        rows=("BOTTIGLIE", "count"),
        unique_pfs=("PF", "nunique"),
        bottiglie_mean=("BOTTIGLIE", "mean"),
        presenze_mean=("PRESENZE", "mean")
    ).round(1))

    # Save target summary to json for report inclusion
    os.makedirs("reports", exist_ok=True)
    with open("reports/target_stats_summary.json", "w") as f:
        json.dump(target_stats, f, indent=2)
    print("\nSaved target stats summary to reports/target_stats_summary.json")

    # =========================================================================
    # 7. Generate Complete Feature Dictionary
    # =========================================================================
    print("\n" + "="*80)
    print("GENERATING FEATURE DICTIONARY")
    print("="*80)
    
    # Feature classification metadata
    # We classify every column in modeling_daily_rvm.parquet plus potential inputs
    inventory = []
    
    # Column metadata dictionary: (description, pre_install, allowed, leakage, reason)
    meta = {
        "PF": ("Physical point identifier", "YES", "NO", "YES", "High-cardinality ID; leaks specific historical entity, prevents generalization"),
        "GIORNO": ("Observation date timestamp", "NO", "NO", "NO", "Calendar timestamp; used to extract time/seasonality, but raw timestamp not used directly for static new location"),
        "BOTTIGLIE": ("Total bottles collected in day (Target 1)", "NO", "NO", "YES", "Direct target variable; strict leakage if used as predictor"),
        "PRESENZE": ("Total unique customer sessions/users in day (Target 2)", "NO", "NO", "YES", "Direct target variable; strict leakage if used as predictor"),
        "Indirizzo": ("Store street address text", "YES", "NO", "NO", "Raw unparsed text; used as input for geocoding, not directly fed as unstructured string to neural net"),
        "Cap": ("Postal code (5 digits)", "YES", "YES", "NO", "Valid geographic location identifier and validation field for new location"),
        "Comune": ("Municipality name", "YES", "YES", "NO", "Core geographic grouping; links store to administrative and macro-demographic data"),
        "Provincia": ("Province code (e.g. MI, RM)", "YES", "YES", "NO", "Core geographic grouping; regional clustering"),
        "Regione": ("Administrative Region (e.g. Lombardia)", "YES", "YES", "NO", "Macro geographic area; captures regional recycling habits and climate"),
        "Latitudine": ("WGS84 Latitude coordinate", "YES", "YES", "NO", "Core continuous spatial coordinate derived from geocoding address"),
        "Longitudine": ("WGS84 Longitude coordinate", "YES", "YES", "NO", "Core continuous spatial coordinate derived from geocoding address"),
        "TipoMacchina": ("Hardware machine model category", "YES", "YES", "NO", "Core hardware configuration feature; captures machine throughput and size"),
        "Macchina": ("Machine internal asset ID", "NO", "NO", "YES", "Physical machine hardware serial/identifier; memorizes individual machine instances"),
        "CodiceMacchinaProd": ("Manufacturer serial number", "NO", "NO", "YES", "Hardware asset serial; memorizes individual machine instances"),
        "CODICEMACCHINA": ("Machine code from master table", "NO", "NO", "YES", "Physical machine hardware serial/identifier; memorizes individual machine instances"),
        "CODICEMACCHINAPROD": ("Manufacturer serial code from master", "NO", "NO", "YES", "Hardware asset serial; memorizes individual machine instances"),
        "Fornitore": ("Hardware manufacturer / supplier", "YES", "YES", "NO", "Supplier/OEM category (DAINT, RIPET, EURVEN, DENTISTECH); perfectly correlated with TipoMacchina"),
        "TipologiaPuntoFisico": ("Physical point type category", "YES", "NO", "NO", "Business eligibility filter (100% PUNTOVENDITA); zero variance in modeling population"),
        "Attributo": ("Machine incentive configuration (STANDARD/PROMO)", "YES", "NO", "NO", "Business eligibility filter (100% STANDARD); zero variance in modeling population"),
        "DataPrimaAttivazione": ("First activation date of physical point", "NO", "NO", "NO", "Ramp-up anchor date; not known pre-installation for a new prospective location"),
        "DataInstallazione": ("Installation date of machine", "NO", "NO", "NO", "Operational logistics date; post-installation timestamp not relevant for prospective location"),
        "DataRimozione": ("Removal/decommission date", "NO", "NO", "YES", "Future operational event; post-installation leakage"),
        "DataInizioRange": ("Historical version interval start date", "NO", "NO", "NO", "Internal versioning tracking timestamp"),
        "DataFineRange": ("Historical version interval end date", "NO", "NO", "NO", "Internal versioning tracking timestamp"),
        "StoricoPuntoFisico_Id": ("Historical record surrogate primary key", "NO", "NO", "YES", "Arbitrary database ID; arbitrary numeric noise"),
        "IdInstallazione": ("Installation event surrogate key", "NO", "NO", "YES", "Arbitrary operational database key"),
        "Denominazione": ("Physical point operational store name", "YES", "YES", "NO", "Store name; used for retail group matching and brand embedding"),
        "Stato": ("Physical point active/inactive status", "NO", "NO", "NO", "Operational status filter; 100% Attivo in modeling population"),
        "Installato": ("Installation status flag", "NO", "NO", "NO", "Operational status; 100% true for operational points"),
        "matricola": ("Manufacturer hardware serial number", "NO", "NO", "YES", "Machine hardware serial; leaks specific machine identity"),
        "progressivo": ("Incremental machine sequence number", "NO", "NO", "NO", "Internal installation counter"),
        "is_active_version": ("Flag for active historical version", "NO", "NO", "NO", "Internal tracking flag"),
        "is_current_active_pf": ("Active as of latest extract flag", "NO", "NO", "NO", "Operational status flag"),
        "nielsen_matched": ("Binary indicator of Nielsen dataset match", "YES", "YES", "NO", "Missingness indicator: flags whether store was enriched with Nielsen attributes"),
        "nielsen_match_type": ("Method of Nielsen match (exact/partial)", "YES", "NO", "NO", "Audit diagnostic field; internal pipeline detail"),
        "nielsen_codice_punto_vendita": ("Nielsen retail point ID", "NO", "NO", "YES", "Nielsen database ID; entity identifier"),
        "nielsen_insegna": ("Nielsen store retail banner / brand", "YES", "YES", "NO", "Standardized retail chain banner (e.g. Conad, Coop, Esselunga)"),
        "nielsen_ragione_sociale": ("Corporate entity legal name", "YES", "NO", "NO", "High cardinality text string; redundant with Insegna"),
        "nielsen_indirizzo": ("Nielsen store street address", "YES", "NO", "NO", "Redundant raw address text"),
        "nielsen_localita": ("Nielsen locality name", "YES", "NO", "NO", "Redundant municipality text"),
        "nielsen_provincia": ("Nielsen province code", "YES", "NO", "NO", "Redundant province code"),
        "nielsen_cap": ("Nielsen postal code", "YES", "NO", "NO", "Redundant postal code"),
        "nielsen_tipo_negozio": ("Store format classification (Iper/Super/etc.)", "YES", "YES", "NO", "Store retail typology: Hypermarket, Supermarket, Superette, etc."),
        "nielsen_superficie_vendita": ("Store sales floor area (sqm)", "YES", "YES", "NO", "Continuous store capacity measure; strong proxy for foot traffic"),
        "nielsen_numero_casse": ("Number of checkout cash registers", "YES", "YES", "NO", "Discrete store capacity measure; strong proxy for customer transaction volume"),
        "nielsen_addetti": ("Number of store employees", "YES", "YES", "NO", "Discrete store scale measure; proxy for operational scale"),
        "nielsen_parcheggio": ("Store dedicated parking availability flag", "YES", "YES", "NO", "Store accessibility feature: customer convenience for bringing plastic bottles"),
        "nielsen_centro_commerciale": ("Flag if store is inside shopping mall", "YES", "YES", "NO", "Location context feature: mall environment vs standalone street store"),
        "nielsen_gruppo": ("Retail parent group / buying alliance", "YES", "YES", "NO", "Macro retail group (e.g. Selex, Coop Italia, Conad)"),
        "population": ("Municipality total population", "YES", "YES", "NO", "Core demographic feature: local population catchment size"),
        "surface": ("Municipality surface area (sq km)", "YES", "YES", "NO", "Core geographic feature: municipality physical footprint"),
        "PIL": ("Municipality aggregate Gross Domestic Product (€)", "YES", "YES", "NO", "Core macroeconomic feature: municipality economic output"),
        "population_density": ("Population density (inhabitants / sq km)", "YES", "YES", "NO", "Core urbanicity feature: rural vs suburban vs dense urban fabric"),
        "PIL_per_capita": ("Gross Domestic Product per inhabitant (€/capita)", "YES", "YES", "NO", "Core macroeconomic feature: purchasing power and wealth proxy"),
        "municipality_match_status": ("Status of ISTAT municipality match", "YES", "NO", "NO", "Data pipeline audit diagnostic flag"),
        "puntovendita_eligible": ("Retail eligibility filter flag", "NO", "NO", "NO", "Constant = 1 in modeling dataset"),
        "active_as_of_2026_08_31": ("PF active on 31/08/2026 filter flag", "NO", "NO", "NO", "Constant = 1 in modeling dataset"),
        "daily_standard_eligible": ("Daily standard attribute filter flag", "NO", "NO", "NO", "Constant = 1 in modeling dataset"),
        "maturity_date": ("Date 3 months post activation", "NO", "NO", "NO", "Ramp-up filtering date; not available for prospective location"),
        "days_since_activation": ("Days elapsed since PF activation", "NO", "NO", "NO", "Age of installation; not available for prospective new location"),
        "months_since_activation": ("Months elapsed since PF activation", "NO", "NO", "NO", "Age of installation; not available for prospective new location"),
        "maturity_status": ("Maturity eligibility classification", "NO", "NO", "NO", "Constant = 'MATURE' in modeling dataset"),
        "modeling_eligible": ("Final modeling eligibility flag", "NO", "NO", "NO", "Constant = 1 in modeling dataset"),
    }

    for col in df.columns:
        s = df[col]
        missing_count = s.isna().sum()
        missing_pct = round((missing_count / n_rows) * 100.0, 2)
        n_unique = s.nunique(dropna=True)
        dtype_str = str(s.dtype)
        
        if col in meta:
            desc, pre_inst, allowed, leakage, reason = meta[col]
        else:
            desc = "Derived or metadata column"
            pre_inst = "NO"
            allowed = "NO"
            leakage = "NO"
            reason = "Pipeline auxiliary variable"
            
        inventory.append({
            "Column": col,
            "Source": "modeling_daily_rvm / integrated",
            "Description": desc,
            "Data type": dtype_str,
            "Missing %": missing_pct,
            "Unique values": n_unique,
            "Pre-installation available?": pre_inst,
            "Allowed for NEW-LOCATION model?": allowed,
            "Potential leakage?": leakage,
            "Reason": reason
        })

    inv_df = pd.DataFrame(inventory)
    inv_df.to_csv("reports/feature_dictionary.csv", index=False)
    print(f"Saved complete feature inventory ({len(inv_df)} columns) to reports/feature_dictionary.csv")

if __name__ == "__main__":
    run_audit()
