"""
Exports PostgreSQL database tables to a MariaDB/MySQL compatible compressed SQL dump.
Compatible with Plesk MariaDB 10.6+ and phpMyAdmin.
Output: coripet_rvm_mariadb.sql.gz
"""
import os
import sys
import gzip
import time
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

OUTPUT_PATH = ROOT_DIR / "coripet_rvm_mariadb.sql.gz"


def escape_val(v):
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (v != v):  # NaN
            return "NULL"
        return str(v)
    if isinstance(v, bool):
        return "1" if v else "0"
    s = str(v).replace("\\", "\\\\").replace("'", "''")
    return f"'{s}'"


def export_mariadb():
    print("=" * 70)
    print("EXPORTING CORIPET DATABASE TO MARIADB / MYSQL COMPATIBLE SQL DUMP")
    print("=" * 70)

    pg_url = os.getenv("DATABASE_URL", "postgresql://postgres:Coripet123*@localhost:5432/coripet_rvm")
    if "+" in pg_url:
        pg_url = pg_url.replace("postgresql+psycopg2://", "postgresql://")

    pg_conn = psycopg2.connect(pg_url)

    t_start = time.time()
    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as out:
        out.write("-- Coripet RVM MariaDB / MySQL Database Dump\n")
        out.write("-- Compatible with Plesk MariaDB 10.6+ & phpMyAdmin\n\n")
        out.write("SET FOREIGN_KEY_CHECKS=0;\n")
        out.write("SET SQL_MODE='NO_AUTO_VALUE_ON_ZERO';\n")
        out.write("SET NAMES utf8mb4;\n\n")

        # ---------------------------------------------------------
        # Table 1: raw_storico_punto_fisico
        # ---------------------------------------------------------
        print("\n1. Exporting raw_storico_punto_fisico (Fleet Registry)...")
        t0 = time.time()
        out.write("DROP TABLE IF EXISTS `raw_storico_punto_fisico`;\n")
        out.write("""CREATE TABLE `raw_storico_punto_fisico` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `puntofisico` VARCHAR(64) DEFAULT NULL,
  `datainiziorange` VARCHAR(64) DEFAULT NULL,
  `datafinerange` VARCHAR(64) DEFAULT NULL,
  `start_dt` DATE DEFAULT NULL,
  `end_dt` DATE DEFAULT NULL,
  `tipologiapuntofisico` VARCHAR(64) DEFAULT NULL,
  `nascondiapp` VARCHAR(16) DEFAULT NULL,
  `indirizzo` TEXT DEFAULT NULL,
  `cap` VARCHAR(16) DEFAULT NULL,
  `comune` VARCHAR(128) DEFAULT NULL,
  `provincia` VARCHAR(16) DEFAULT NULL,
  `regione` VARCHAR(64) DEFAULT NULL,
  `nazione` VARCHAR(32) DEFAULT NULL,
  `latitudine` DOUBLE DEFAULT NULL,
  `longitudine` DOUBLE DEFAULT NULL,
  `stato` VARCHAR(32) DEFAULT NULL,
  `attributo` VARCHAR(64) DEFAULT NULL,
  `cooperativa` VARCHAR(128) DEFAULT NULL,
  `filiera` VARCHAR(128) DEFAULT NULL,
  `descrizione` TEXT DEFAULT NULL,
  `tipocontenitore` VARCHAR(64) DEFAULT NULL,
  `referentecommerciale` VARCHAR(128) DEFAULT NULL,
  `referentecoripet` VARCHAR(128) DEFAULT NULL,
  `comodatario` VARCHAR(128) DEFAULT NULL,
  `nomefiliera` VARCHAR(128) DEFAULT NULL,
  `raggruppamento` VARCHAR(128) DEFAULT NULL,
  `macchina` VARCHAR(64) DEFAULT NULL,
  `codicemacchinaprod` VARCHAR(64) DEFAULT NULL,
  `tipomacchina` VARCHAR(128) DEFAULT NULL,
  `fornitore` VARCHAR(128) DEFAULT NULL,
  `datainstallazione` VARCHAR(64) DEFAULT NULL,
  `dataprimaattivazione` VARCHAR(64) DEFAULT NULL,
  `macchina_clean` VARCHAR(64) DEFAULT NULL,
  `codicemacchinaprod_clean` VARCHAR(64) DEFAULT NULL,
  `created_at` DATETIME DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_storico_pf` (`puntofisico`),
  KEY `idx_storico_macchina` (`macchina`),
  KEY `idx_storico_stato` (`stato`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;\n\n""")

        cur = pg_conn.cursor()
        cur.execute("SELECT * FROM raw_storico_punto_fisico ORDER BY id;")
        col_names = [desc[0] for desc in cur.description]
        col_str = ", ".join([f"`{c}`" for c in col_names])

        rows = cur.fetchall()
        if rows:
            for i in range(0, len(rows), 500):
                chunk = rows[i:i + 500]
                val_lines = []
                for r in chunk:
                    val_lines.append("(" + ", ".join(escape_val(v) for v in r) + ")")
                out.write(f"INSERT INTO `raw_storico_punto_fisico` ({col_str}) VALUES\n" + ",\n".join(val_lines) + ";\n")
        print(f"--> Exported {len(rows):,} rows in {time.time() - t0:.2f}s")

        # ---------------------------------------------------------
        # Table 2: master_daily_rvm
        # ---------------------------------------------------------
        print("\n2. Exporting master_daily_rvm (1.77M Telemetry Records)...")
        t0 = time.time()
        out.write("\nDROP TABLE IF EXISTS `master_daily_rvm`;\n")
        out.write("""CREATE TABLE `master_daily_rvm` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `cliente` VARCHAR(128) DEFAULT NULL,
  `giorno` VARCHAR(64) DEFAULT NULL,
  `bottiglie` DOUBLE DEFAULT NULL,
  `presenze` DOUBLE DEFAULT NULL,
  `pf` VARCHAR(64) DEFAULT NULL,
  `codicemacchinaprod` VARCHAR(64) DEFAULT NULL,
  `codicemacchina` VARCHAR(64) DEFAULT NULL,
  `refcoripet` VARCHAR(128) DEFAULT NULL,
  `giorno_dt` DATE DEFAULT NULL,
  `puntofisico` VARCHAR(64) DEFAULT NULL,
  `indirizzo` TEXT DEFAULT NULL,
  `cap` VARCHAR(16) DEFAULT NULL,
  `comune` VARCHAR(128) DEFAULT NULL,
  `provincia` VARCHAR(16) DEFAULT NULL,
  `regione` VARCHAR(64) DEFAULT NULL,
  `nazione` VARCHAR(32) DEFAULT NULL,
  `latitudine` DOUBLE DEFAULT NULL,
  `longitudine` DOUBLE DEFAULT NULL,
  `stato` VARCHAR(32) DEFAULT NULL,
  `tipologiapuntofisico` VARCHAR(64) DEFAULT NULL,
  `attributo` VARCHAR(64) DEFAULT NULL,
  `cooperativa` VARCHAR(128) DEFAULT NULL,
  `filiera` VARCHAR(128) DEFAULT NULL,
  `descrizione` TEXT DEFAULT NULL,
  `tipocontenitore` VARCHAR(64) DEFAULT NULL,
  `referentecommerciale` VARCHAR(128) DEFAULT NULL,
  `referentecoripet` VARCHAR(128) DEFAULT NULL,
  `comodatario` VARCHAR(128) DEFAULT NULL,
  `nomefiliera` VARCHAR(128) DEFAULT NULL,
  `raggruppamento` VARCHAR(128) DEFAULT NULL,
  `macchina` VARCHAR(64) DEFAULT NULL,
  `codice_macchina_prod` VARCHAR(64) DEFAULT NULL,
  `tipomacchina` VARCHAR(64) DEFAULT NULL,
  `fornitore` VARCHAR(64) DEFAULT NULL,
  `datainstallazione` VARCHAR(64) DEFAULT NULL,
  `dataprimaattivazione` VARCHAR(64) DEFAULT NULL,
  `historical_match_status` VARCHAR(32) DEFAULT NULL,
  `nielsen_match_status` VARCHAR(32) DEFAULT NULL,
  `nielsen_match_method` VARCHAR(64) DEFAULT NULL,
  `nielsen_match_confidence` VARCHAR(32) DEFAULT NULL,
  `nielsen_country_shop_id` DOUBLE DEFAULT NULL,
  `nielsen_tipo_negozio` VARCHAR(64) DEFAULT NULL,
  `nielsen_superficie_vendita` DOUBLE DEFAULT NULL,
  `nielsen_numero_casse` DOUBLE DEFAULT NULL,
  `nielsen_insegna` VARCHAR(128) DEFAULT NULL,
  `nielsen_gruppo` VARCHAR(128) DEFAULT NULL,
  `nielsen_area` DOUBLE DEFAULT NULL,
  `population` DOUBLE DEFAULT NULL,
  `surface` DOUBLE DEFAULT NULL,
  `pil` DOUBLE DEFAULT NULL,
  `population_density` DOUBLE DEFAULT NULL,
  `pil_per_capita` DOUBLE DEFAULT NULL,
  `municipality_match_status` VARCHAR(32) DEFAULT NULL,
  `puntovendita_eligible` INT DEFAULT NULL,
  `active_as_of_2026_08_31` INT DEFAULT NULL,
  `daily_standard_eligible` INT DEFAULT NULL,
  `maturity_date` VARCHAR(64) DEFAULT NULL,
  `days_since_activation` INT DEFAULT NULL,
  `months_since_activation` DOUBLE DEFAULT NULL,
  `maturity_status` VARCHAR(32) DEFAULT NULL,
  `modeling_eligible` INT DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_master_pf` (`pf`),
  KEY `idx_master_macchina` (`macchina`),
  KEY `idx_master_codprod` (`codicemacchinaprod`),
  KEY `idx_master_giorno_dt` (`giorno_dt`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;\n\n""")

        cur.execute("SELECT * FROM master_daily_rvm LIMIT 1;")
        col_names = [desc[0] for desc in cur.description]
        col_str = ", ".join([f"`{c}`" for c in col_names])

        cur_stream = pg_conn.cursor(name="master_stream")
        cur_stream.itersize = 25000
        cur_stream.execute("SELECT * FROM master_daily_rvm;")

        total = 0
        batch = []
        batch_size = 1000
        for row in cur_stream:
            batch.append("(" + ", ".join(escape_val(v) for v in row) + ")")
            total += 1
            if len(batch) >= batch_size:
                out.write(f"INSERT INTO `master_daily_rvm` ({col_str}) VALUES\n" + ",\n".join(batch) + ";\n")
                batch = []
                if total % 100000 == 0:
                    print(f"--> Exported {total:,} rows...")

        if batch:
            out.write(f"INSERT INTO `master_daily_rvm` ({col_str}) VALUES\n" + ",\n".join(batch) + ";\n")

        print(f"--> Finished master_daily_rvm ({total:,} rows) in {time.time() - t0:.2f}s")

        out.write("\nSET FOREIGN_KEY_CHECKS=1;\n")

    pg_conn.close()
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print("\n" + "=" * 70)
    print(f"SUCCESS! MariaDB SQL dump generated: {OUTPUT_PATH}")
    print(f"Dump File Size: {size_mb:.2f} MB")
    print(f"Total Export Duration: {time.time() - t_start:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    export_mariadb()
