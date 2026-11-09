import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import pandas as pd
from src.ui.inspection_history import fetch_inspection_history

df = fetch_inspection_history(limit=1000)
print(f"Fetched {len(df)} records from database.")

sql_lines = [
    "-- ==============================================================================",
    "-- MariaDB Data Dump for `machine_prediction_history`",
    "-- Target Database: coripet_rvm",
    f"-- Total Records: {len(df)}",
    "-- ==============================================================================",
    "",
    "USE `coripet_rvm`;",
    "",
    "SET FOREIGN_KEY_CHECKS=0;",
    "SET NAMES utf8mb4;",
    "",
    "CREATE TABLE IF NOT EXISTS `machine_prediction_history` (",
    "    `id` INT AUTO_INCREMENT PRIMARY KEY,",
    "    `punto_fisico` VARCHAR(64) NOT NULL,",
    "    `status_source` VARCHAR(20) NOT NULL,",
    "    `store_name` VARCHAR(255) DEFAULT NULL,",
    "    `address` TEXT DEFAULT NULL,",
    "    `cap` VARCHAR(20) DEFAULT NULL,",
    "    `comune` VARCHAR(128) DEFAULT NULL,",
    "    `provincia` VARCHAR(32) DEFAULT NULL,",
    "    `regione` VARCHAR(64) DEFAULT NULL,",
    "    `machine_type` VARCHAR(64) DEFAULT NULL,",
    "    `filiera` VARCHAR(128) DEFAULT NULL,",
    "    `predicted_daily_bottles` FLOAT DEFAULT 0.0,",
    "    `predicted_monthly_bottles` BIGINT DEFAULT 0,",
    "    `predicted_annual_bottles` BIGINT DEFAULT 0,",
    "    `predicted_daily_visits` FLOAT DEFAULT 0.0,",
    "    `bottles_per_user` FLOAT DEFAULT 0.0,",
    "    `volume_tier` VARCHAR(64) DEFAULT NULL,",
    "    `p10_floor` FLOAT DEFAULT 0.0,",
    "    `p90_peak` FLOAT DEFAULT 0.0,",
    "    `confidence_score` FLOAT DEFAULT 0.0,",
    "    `sales_area_sqm` FLOAT DEFAULT 0.0,",
    "    `cash_registers` FLOAT DEFAULT 0.0,",
    "    `nielsen_format` VARCHAR(64) DEFAULT NULL,",
    "    `reward_type` VARCHAR(128) DEFAULT NULL,",
    "    `reward_multiplier` FLOAT DEFAULT 1.0,",
    "    `nearest_coripet_km` FLOAT DEFAULT 0.0,",
    "    `seasonality_json` LONGTEXT DEFAULT NULL,",
    "    `full_prediction_json` LONGTEXT DEFAULT NULL,",
    "    `inspection_date` DATE NOT NULL,",
    "    `inspection_timestamp` DATETIME NOT NULL,",
    "    INDEX `idx_hist_pf` (`punto_fisico`),",
    "    INDEX `idx_hist_date` (`inspection_date`),",
    "    INDEX `idx_hist_status` (`status_source`),",
    "    UNIQUE KEY `uq_pf_inspection_date` (`punto_fisico`, `inspection_date`)",
    ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;",
    "",
    "INSERT INTO `machine_prediction_history` (",
    "    `id`, `punto_fisico`, `status_source`, `store_name`, `address`, `cap`, `comune`, `provincia`, `regione`,",
    "    `machine_type`, `filiera`, `predicted_daily_bottles`, `predicted_monthly_bottles`, `predicted_annual_bottles`,",
    "    `predicted_daily_visits`, `bottles_per_user`, `volume_tier`, `p10_floor`, `p90_peak`, `confidence_score`,",
    "    `sales_area_sqm`, `cash_registers`, `nielsen_format`, `reward_type`, `reward_multiplier`, `nearest_coripet_km`,",
    "    `seasonality_json`, `full_prediction_json`, `inspection_date`, `inspection_timestamp`",
    ") VALUES"
]

def esc(val):
    if val is None or (isinstance(val, float) and (val != val)):
        return "NULL"
    s = str(val).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{s}'"

val_rows = []
for _, r in df.sort_values(by="id").iterrows():
    r_id = int(r["id"])
    pf = esc(r["punto_fisico"])
    st = esc(r["status_source"])
    sn = esc(r["store_name"])
    addr = esc(r["address"])
    cap = esc(r["cap"])
    com = esc(r["comune"])
    pr = esc(r["provincia"])
    reg = esc(r["regione"])
    mt = esc(r["machine_type"])
    fil = esc(r["filiera"])
    pdb = float(r.get("predicted_daily_bottles") or 0.0)
    pmb = int(r.get("predicted_monthly_bottles") or 0)
    pab = int(r.get("predicted_annual_bottles") or 0)
    pdv = float(r.get("predicted_daily_visits") or 0.0)
    bpu = float(r.get("bottles_per_user") or 0.0)
    vt = esc(r["volume_tier"])
    p10 = float(r.get("p10_floor") or 0.0)
    p90 = float(r.get("p90_peak") or 0.0)
    cs = float(r.get("confidence_score") or 0.0)
    sqm = float(r.get("sales_area_sqm") or 0.0)
    cr = float(r.get("cash_registers") or 0.0)
    nf = esc(r["nielsen_format"])
    rt = esc(r["reward_type"])
    rm = float(r.get("reward_multiplier") or 1.0)
    nck = float(r.get("nearest_coripet_km") or 0.0)
    sj = esc(r.get("seasonality_json"))
    fj = esc(r.get("full_prediction_json"))
    idate = esc(str(r["inspection_date"]))
    its = esc(str(r["inspection_timestamp"]))

    val_rows.append(f"({r_id}, {pf}, {st}, {sn}, {addr}, {cap}, {com}, {pr}, {reg}, {mt}, {fil}, {pdb}, {pmb}, {pab}, {pdv}, {bpu}, {vt}, {p10}, {p90}, {cs}, {sqm}, {cr}, {nf}, {rt}, {rm}, {nck}, {sj}, {fj}, {idate}, {its})")

if val_rows:
    sql_lines.append(",\n".join(val_rows) + " ON DUPLICATE KEY UPDATE predicted_daily_bottles=VALUES(predicted_daily_bottles);")
else:
    sql_lines.append("-- No records to insert")

sql_lines.append("")
sql_lines.append("SET FOREIGN_KEY_CHECKS=1;")

content = "\n".join(sql_lines)
with open("deploy/simple_mariadb_import.sql", "w", encoding="utf-8") as f:
    f.write(content)
with open("deploy/machine_prediction_history_dump.sql", "w", encoding="utf-8") as f:
    f.write(content)

print("Export completed successfully.")
