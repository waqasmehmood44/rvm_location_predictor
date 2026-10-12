"""
Script to generate reports/feature_dictionary.csv with comprehensive metadata for all candidate features.
"""
import pandas as pd
import numpy as np

def generate_dictionary():
    df = pd.read_parquet("data/processed/modeling_daily_rvm.parquet")
    n_rows = len(df)
    
    # Precise metadata mapping for each column in modeling_daily_rvm
    col_meta = {
        "CLIENTE": {
            "source": "daily_bottles.csv",
            "desc": "Customer / retail brand partner name in operational log",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "Proxy for retail chain identity; can be used if standardized/matched to Store Name"
        },
        "GIORNO": {
            "source": "daily_bottles.csv",
            "desc": "Date of daily observation (DD/MM/YYYY)",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Timestamp; not a static property of a new store. Used only to extract calendar components for training conditioning"
        },
        "BOTTIGLIE": {
            "source": "daily_bottles.csv",
            "desc": "Target 1: Total beverage bottles collected by machine in that day",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "YES",
            "reason": "PRIMARY TARGET; strictly forbidden as a predictor feature for new location model"
        },
        "PRESENZE": {
            "source": "daily_bottles.csv",
            "desc": "Target 2: Total unique customer recycling sessions in that day",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "YES",
            "reason": "PRIMARY TARGET; strictly forbidden as a predictor feature for new location model"
        },
        "PF": {
            "source": "daily_bottles.csv",
            "desc": "Physical point unique identifier (e.g. PF000123)",
            "pre_inst": "YES",
            "allowed": "NO",
            "leakage": "YES",
            "reason": "High-cardinality database entity ID; would cause neural net to memorize historical stores rather than learning generalizable patterns"
        },
        "CODICEMACCHINAPROD": {
            "source": "daily_bottles.csv",
            "desc": "Manufacturer physical serial number from operational log",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "YES",
            "reason": "Hardware asset serial; memorizes individual machine instances (entity leakage)"
        },
        "CODICEMACCHINA": {
            "source": "daily_bottles.csv",
            "desc": "Internal Coripet machine asset ID from operational log",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "YES",
            "reason": "Hardware asset serial; memorizes individual machine instances (entity leakage)"
        },
        "REFCORIPET": {
            "source": "daily_bottles.csv",
            "desc": "Coripet commercial account manager reference code",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Internal organizational administrative code; irrelevant to store foot traffic"
        },
        "giorno_dt": {
            "source": "processed pipeline",
            "desc": "Parsed datetime timestamp of GIORNO",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Internal parsed timestamp; used only for feature extraction"
        },
        "PuntoFisico": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Physical point identifier in historical table (identical to PF)",
            "pre_inst": "YES",
            "allowed": "NO",
            "leakage": "YES",
            "reason": "Redundant primary key; identical to PF"
        },
        "Indirizzo": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Store street address string (e.g. Via Roma 12)",
            "pre_inst": "YES",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Raw unstandardized text; used by geocoding pipeline to produce lat/lon and municipality, but not fed directly to neural net"
        },
        "Cap": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Italian 5-digit postal code (CAP)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "Core location feature; spatial clustering and administrative routing"
        },
        "Comune": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Municipality name (e.g. Milano, Roma)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "Core location feature; links store to ISTAT census demographics"
        },
        "Provincia": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Two-letter province code (e.g. MI, RM, NA)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "Core location feature; intermediate administrative grouping"
        },
        "Regione": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Administrative region (e.g. Lombardia, Lazio, Sicilia)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "Core location feature; macro-geographic clustering capturing regional recycling propensity"
        },
        "Nazione": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Country name (ITALIA)",
            "pre_inst": "YES",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Constant value ('ITALIA'); zero variance"
        },
        "Latitudine": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "WGS84 Latitude coordinate (decimal degrees)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "Core spatial coordinate; allows spatial neural embeddings and distance calculations"
        },
        "Longitudine": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "WGS84 Longitude coordinate (decimal degrees)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "Core spatial coordinate; allows spatial neural embeddings and distance calculations"
        },
        "Stato": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Operational status in historical table ('Attivo')",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Operational status; 100% Attivo in certified population (zero variance)"
        },
        "TipologiaPuntoFisico": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Type of physical installation ('PUNTOVENDITA')",
            "pre_inst": "YES",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Business eligibility gate; 100% PUNTOVENDITA in certified population (zero variance)"
        },
        "Attributo": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "RVM machine incentive configuration ('STANDARD')",
            "pre_inst": "YES",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Business eligibility gate; 100% STANDARD in certified population (zero variance)"
        },
        "Cooperativa": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Retail cooperative or consortium name",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "Optional retail organization grouping"
        },
        "Filiera": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Coripet supply chain operational routing code",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Internal logistics routing; not known pre-installation for prospective store"
        },
        "Descrizione": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Location description inside store (e.g. Ingresso, Parcheggio)",
            "pre_inst": "YES",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Unstandardized free text; high missingness/inconsistency"
        },
        "TipoContenitore": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Internal waste collection receptacle type (Big Bag, Roll)",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Post-installation logistics configuration; not a store driver"
        },
        "ReferenteCommerciale": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Coripet commercial sales representative name",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "97.5% missing; internal administrative personnel field"
        },
        "ReferenteCoripet": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Coripet operations representative code",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Internal administrative personnel field"
        },
        "Comodatario": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Legal corporate name of store holding bailment contract",
            "pre_inst": "YES",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "High-cardinality corporate text; redundant with store name"
        },
        "NomeFiliera": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Descriptive name of logistics recycling supply chain",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Internal logistics network label"
        },
        "Raggruppamento": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Commercial grouping of stores",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "Optional retail chain grouping if available"
        },
        "Macchina": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Machine serial / asset tag in historical configuration",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "YES",
            "reason": "Hardware asset serial; memorizes individual machine instances (entity leakage)"
        },
        "CodiceMacchinaProd": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Manufacturer serial number in historical configuration",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "YES",
            "reason": "Hardware asset serial; memorizes individual machine instances (entity leakage)"
        },
        "TipoMacchina": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Hardware model type / family (e.g. DAINT_G, Ripet_C-front)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "CORE HARDWARE FEATURE; represents machine physical footprint, bin volume, and intake speed"
        },
        "Fornitore": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Hardware OEM vendor (DAINT, RIPET, EURVEN, DENTISTECH)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "Hardware manufacturer category; directly derived from TipoMacchina"
        },
        "DataInstallazione": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "Physical installation date of machine",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Operational date; not available for prospective pre-installation location"
        },
        "DataPrimaAttivazione": {
            "source": "CaricaViewStoricoPuntoFisico.csv",
            "desc": "First operational turn-on date of physical point",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Operational ramp-up anchor date; not available for prospective location"
        },
        "historical_match_status": {
            "source": "processed pipeline",
            "desc": "Audit status of historical version join ('MATCHED')",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Pipeline audit diagnostic flag"
        },
        "nielsen_match_status": {
            "source": "processed pipeline",
            "desc": "Nielsen enrichment status (MATCHED, UNMATCHED, AMBIGUOUS)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "Missingness indicator flag; crucial for handling optional Nielsen enrichment"
        },
        "nielsen_match_method": {
            "source": "processed pipeline",
            "desc": "Heuristic match method used in Nielsen integration",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Pipeline audit diagnostic flag"
        },
        "nielsen_match_confidence": {
            "source": "processed pipeline",
            "desc": "Confidence tier of Nielsen match (HIGH, MEDIUM, LOW, NONE)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "Data quality weight / confidence indicator for store features"
        },
        "nielsen_country_shop_id": {
            "source": "filenielsen_1203.csv",
            "desc": "Nielsen census unique store ID",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "YES",
            "reason": "Nielsen database primary key; entity identifier"
        },
        "nielsen_tipo_negozio": {
            "source": "filenielsen_1203.csv",
            "desc": "Nielsen store format (IPR=Hyper, SUP=Super, LIS=Superette, DIS=Discount)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "OPTIONAL ENRICHMENT; strong predictor of store customer volume and shopping format"
        },
        "nielsen_superficie_vendita": {
            "source": "filenielsen_1203.csv",
            "desc": "Store sales floor surface area in square meters",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "OPTIONAL ENRICHMENT; continuous capacity proxy directly correlated with foot traffic"
        },
        "nielsen_numero_casse": {
            "source": "filenielsen_1203.csv",
            "desc": "Total number of checkout cash registers in store",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "OPTIONAL ENRICHMENT; checkout count is an industry standard proxy for daily shoppers"
        },
        "nielsen_insegna": {
            "source": "filenielsen_1203.csv",
            "desc": "Standardized retail store banner (e.g. Conad, Coop, Esselunga)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "OPTIONAL ENRICHMENT; brand identity captures customer loyalty and promotional affinity"
        },
        "nielsen_gruppo": {
            "source": "filenielsen_1203.csv",
            "desc": "Retail parent alliance / buying consortium (e.g. Selex, Coop, Conad)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "OPTIONAL ENRICHMENT; macro corporate alliance grouping"
        },
        "nielsen_area": {
            "source": "filenielsen_1203.csv",
            "desc": "Nielsen macro geographic area code (1=NW, 2=NE, 3=Center, 4=South)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "OPTIONAL ENRICHMENT; macro-regional market segmentation"
        },
        "population": {
            "source": "population.csv",
            "desc": "Municipality total resident population (ISTAT)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "CORE DEMOGRAPHIC FEATURE; municipality catchment population"
        },
        "surface": {
            "source": "surface.csv",
            "desc": "Municipality geographic surface area in square kilometers",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "CORE GEOGRAPHIC FEATURE; physical size of municipality"
        },
        "PIL": {
            "source": "PIL.csv",
            "desc": "Municipality aggregate Gross Domestic Product in Euros",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "CORE ECONOMIC FEATURE; overall economic output of the local market"
        },
        "population_density": {
            "source": "integrated pipeline",
            "desc": "Population density (residents per sq km)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "CORE URBANICITY FEATURE; distinguishes dense urban centers from rural/suburban locations"
        },
        "PIL_per_capita": {
            "source": "integrated pipeline",
            "desc": "Gross Domestic Product per resident (EUR / capita)",
            "pre_inst": "YES",
            "allowed": "YES",
            "leakage": "NO",
            "reason": "CORE WEALTH FEATURE; local purchasing power proxy"
        },
        "municipality_match_status": {
            "source": "processed pipeline",
            "desc": "Audit match status for municipality demographics",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Pipeline audit diagnostic flag"
        },
        "puntovendita_eligible": {
            "source": "processed pipeline",
            "desc": "Eligibility flag for PUNTOVENDITA filter (100% = 1)",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Filter indicator; constant = 1"
        },
        "active_as_of_2026_08_31": {
            "source": "processed pipeline",
            "desc": "Eligibility flag for active status as of 31/08/2026 (100% = 1)",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Filter indicator; constant = 1"
        },
        "daily_standard_eligible": {
            "source": "processed pipeline",
            "desc": "Eligibility flag for STANDARD attribute (100% = 1)",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Filter indicator; constant = 1"
        },
        "maturity_date": {
            "source": "processed pipeline",
            "desc": "Date 3 calendar months after DataPrimaAttivazione",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Operational ramp-up anchor date; not available for new location"
        },
        "days_since_activation": {
            "source": "processed pipeline",
            "desc": "Days elapsed since machine first activation",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Installation lifecycle age; not available for prospective new store"
        },
        "months_since_activation": {
            "source": "processed pipeline",
            "desc": "Months elapsed since machine first activation",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Installation lifecycle age; not available for prospective new store"
        },
        "maturity_status": {
            "source": "processed pipeline",
            "desc": "Maturity status flag (100% = 'MATURE')",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Filter indicator; constant = 'MATURE'"
        },
        "modeling_eligible": {
            "source": "processed pipeline",
            "desc": "Final modeling eligibility flag (100% = 1)",
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Filter indicator; constant = 1"
        }
    }
    
    # Also candidate features from raw filenielsen_1203 not yet merged into modeling_daily_rvm
    additional_candidates = [
        {
            "Column": "PRESENZA POSTEGGIO (nielsen_parcheggio)",
            "Source": "filenielsen_1203.csv",
            "Description": "Binary indicator for dedicated store parking lot (1=Yes, 0/null=No)",
            "Data type": "int64 / binary",
            "Missing %": 58.2,
            "Unique values": 2,
            "Pre-installation available?": "YES",
            "Allowed for NEW-LOCATION model?": "YES",
            "Potential leakage?": "NO",
            "Reason": "OPTIONAL ENRICHMENT; customer convenience factor: stores with parking facilitate carrying large bags of bottles"
        },
        {
            "Column": "PUNTO VENDITA IN CENTRO COMMERCIALE (nielsen_centro_commerciale)",
            "Source": "filenielsen_1203.csv",
            "Description": "Binary indicator if store is located inside an enclosed shopping mall",
            "Data type": "int64 / binary",
            "Missing %": 58.2,
            "Unique values": 2,
            "Pre-installation available?": "YES",
            "Allowed for NEW-LOCATION model?": "YES",
            "Potential leakage?": "NO",
            "Reason": "OPTIONAL ENRICHMENT; context factor: shopping malls exhibit high weekend foot traffic and different visitation patterns"
        },
        {
            "Column": "FASCIA ADDETTI (nielsen_addetti)",
            "Source": "filenielsen_1203.csv",
            "Description": "Categorical employee headcount bracket (A: 1-5, B: 6-15, C: 16-30, D: 31-50, E: >50)",
            "Data type": "str / category",
            "Missing %": 58.2,
            "Unique values": 6,
            "Pre-installation available?": "YES",
            "Allowed for NEW-LOCATION model?": "YES",
            "Potential leakage?": "NO",
            "Reason": "OPTIONAL ENRICHMENT; operational scale proxy"
        },
        {
            "Column": "day_of_week",
            "Source": "derived from GIORNO",
            "Description": "Day of the week (0=Monday to 6=Sunday)",
            "Data type": "int64 / cyclical",
            "Missing %": 0.0,
            "Unique values": 7,
            "Pre-installation available?": "NO",
            "Allowed for NEW-LOCATION model?": "NO",
            "Potential leakage?": "NO",
            "Reason": "Training conditioning feature; not a static location feature. Used to marginalize/average over the week to compute Model A baseline expectation"
        },
        {
            "Column": "month_of_year",
            "Source": "derived from GIORNO",
            "Description": "Month of the year (1=January to 12=December)",
            "Data type": "int64 / cyclical",
            "Missing %": 0.0,
            "Unique values": 12,
            "Pre-installation available?": "NO",
            "Allowed for NEW-LOCATION model?": "NO",
            "Potential leakage?": "NO",
            "Reason": "Training conditioning feature; captures summer beverage consumption peaks. Marginalized across 12 months for annual expected potential"
        },
        {
            "Column": "is_weekend",
            "Source": "derived from GIORNO",
            "Description": "Indicator for Saturday or Sunday",
            "Data type": "int64 / binary",
            "Missing %": 0.0,
            "Unique values": 2,
            "Pre-installation available?": "NO",
            "Allowed for NEW-LOCATION model?": "NO",
            "Potential leakage?": "NO",
            "Reason": "Training conditioning feature"
        },
        {
            "Column": "is_italian_holiday",
            "Source": "derived from GIORNO",
            "Description": "National Italian public holidays (Christmas, Easter, Ferragosto, etc.)",
            "Data type": "int64 / binary",
            "Missing %": 0.0,
            "Unique values": 2,
            "Pre-installation available?": "NO",
            "Allowed for NEW-LOCATION model?": "NO",
            "Potential leakage?": "NO",
            "Reason": "Training conditioning feature; accounts for store closure days"
        },
        {
            "Column": "User_Input_Store_Name",
            "Source": "Application User Input",
            "Description": "Store banner / commercial name entered by user (e.g. Conad Superstore)",
            "Data type": "str",
            "Missing %": 0.0,
            "Unique values": "Dynamic",
            "Pre-installation available?": "YES",
            "Allowed for NEW-LOCATION model?": "YES",
            "Potential leakage?": "NO",
            "Reason": "APPLICATION INPUT 1; parsed to extract standardized retail brand/insegna"
        },
        {
            "Column": "User_Input_Address",
            "Source": "Application User Input",
            "Description": "Street address entered by user (e.g. Via Roma 15)",
            "Data type": "str",
            "Missing %": 0.0,
            "Unique values": "Dynamic",
            "Pre-installation available?": "YES",
            "Allowed for NEW-LOCATION model?": "NO",
            "Potential leakage?": "NO",
            "Reason": "APPLICATION INPUT 2; passed to geocoding API to resolve coordinates and municipality"
        },
        {
            "Column": "User_Input_CAP",
            "Source": "Application User Input",
            "Description": "5-digit postal code entered by user (e.g. 20121)",
            "Data type": "str / int",
            "Missing %": 0.0,
            "Unique values": "Dynamic",
            "Pre-installation available?": "YES",
            "Allowed for NEW-LOCATION model?": "YES",
            "Potential leakage?": "NO",
            "Reason": "APPLICATION INPUT 3; validates geocoded locality and provides spatial postal code"
        },
        {
            "Column": "User_Input_Machine_Type",
            "Source": "Application User Input",
            "Description": "RVM hardware model selected by user (e.g. DAINT_G, Ripet_C-front)",
            "Data type": "str / category",
            "Missing %": 0.0,
            "Unique values": 14,
            "Pre-installation available?": "YES",
            "Allowed for NEW-LOCATION model?": "YES",
            "Potential leakage?": "NO",
            "Reason": "APPLICATION INPUT 4; maps directly to TipoMacchina hardware profile"
        }
    ]

    rows = []
    for col in df.columns:
        s = df[col]
        missing_count = s.isna().sum()
        missing_pct = round((missing_count / n_rows) * 100.0, 2)
        n_unique = s.nunique(dropna=True)
        dtype_str = str(s.dtype)
        
        info = col_meta.get(col, {
            "source": "integrated",
            "desc": col,
            "pre_inst": "NO",
            "allowed": "NO",
            "leakage": "NO",
            "reason": "Pipeline field"
        })
        
        rows.append({
            "Column": col,
            "Source": info["source"],
            "Description": info["desc"],
            "Data type": dtype_str,
            "Missing %": missing_pct,
            "Unique values": n_unique,
            "Pre-installation available?": info["pre_inst"],
            "Allowed for NEW-LOCATION model?": info["allowed"],
            "Potential leakage?": info["leakage"],
            "Reason": info["reason"]
        })
        
    for ac in additional_candidates:
        rows.append(ac)
        
    inv_df = pd.DataFrame(rows)
    inv_df.to_csv("reports/feature_dictionary.csv", index=False)
    print(f"Successfully generated reports/feature_dictionary.csv with {len(inv_df)} candidate variables.")

if __name__ == "__main__":
    generate_dictionary()
