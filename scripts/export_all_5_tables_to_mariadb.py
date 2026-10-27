"""
Exports ALL 5 tables to a single complete MariaDB/MySQL compressed dump:
1. raw_storico_punto_fisico (5,735 rows)
2. master_daily_rvm (1,770,755 rows)
3. raw_daily_conferimenti (1,986,890 rows)
4. training_master_daily (20,314 rows)
5. test_holdout (111,215 rows)

Output: coripet_rvm_complete_all_5_tables.sql.gz
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

OUTPUT_PATH = ROOT_DIR / "coripet_rvm_complete_all_5_tables.sql.gz"


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


def get_mariadb_type(pg_type, col_name):
    pg_type = pg_type.lower()
    if "bigint" in pg_type:
        return "BIGINT"
    if "int" in pg_type:
        return "INT"
    if "double" in pg_type or "float" in pg_type or "real" in pg_type or "numeric" in pg_type:
        return "DOUBLE"
    if "date" in pg_type:
        return "DATE"
    if "timestamp" in pg_type:
        return "DATETIME"
    if "bool" in pg_type:
        return "TINYINT(1)"
    if "text" in pg_type or col_name in ["indirizzo", "descrizione"]:
        return "TEXT"
    return "VARCHAR(255)"


def export_all():
    print("=" * 70)
    print("EXPORTING ALL 5 TABLES TO SINGLE COMPLETE MARIADB DUMP")
    print("=" * 70)

    pg_url = os.getenv("DATABASE_URL", "postgresql://postgres:Coripet123*@localhost:5432/coripet_rvm")
    if "+" in pg_url:
        pg_url = pg_url.replace("postgresql+psycopg2://", "postgresql://")

    pg_conn = psycopg2.connect(pg_url)
    cur = pg_conn.cursor()

    tables = [
        "raw_storico_punto_fisico",
        "master_daily_rvm",
        "raw_daily_conferimenti",
        "training_master_daily",
        "test_holdout"
    ]

    t_start = time.time()
    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as out:
        out.write("-- Coripet RVM Complete All 5 Tables MariaDB Dump\n")
        out.write("-- Compatible with Plesk MariaDB 10.6+ & phpMyAdmin\n\n")
        out.write("SET FOREIGN_KEY_CHECKS=0;\n")
        out.write("SET SQL_MODE='NO_AUTO_VALUE_ON_ZERO';\n")
        out.write("SET NAMES utf8mb4;\n\n")

        for table in tables:
            t0 = time.time()
            print(f"\nProcessing `{table}`...")

            # Get columns from PostgreSQL
            cur.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table}' 
                ORDER BY ordinal_position;
            """)
            col_info = cur.fetchall()
            col_names = [c[0] for c in col_info]

            # Build CREATE TABLE
            out.write(f"\nDROP TABLE IF EXISTS `{table}`;\n")
            out.write(f"CREATE TABLE `{table}` (\n")
            defs = []
            has_id = False
            for col, dtype in col_info:
                m_type = get_mariadb_type(dtype, col)
                if col == "id":
                    has_id = True
                    defs.append(f"  `id` {m_type} NOT NULL AUTO_INCREMENT")
                else:
                    defs.append(f"  `{col}` {m_type} DEFAULT NULL")

            if has_id:
                defs.append("  PRIMARY KEY (`id`)")

            # Add useful indexes
            if table == "raw_storico_punto_fisico":
                defs.append("  KEY `idx_storico_pf` (`puntofisico`)")
                defs.append("  KEY `idx_storico_macchina` (`macchina`)")
            elif table == "master_daily_rvm":
                defs.append("  KEY `idx_master_pf` (`pf`)")
                defs.append("  KEY `idx_master_giorno_dt` (`giorno_dt`)")
                defs.append("  KEY `idx_master_macchina` (`macchina`)")
                defs.append("  KEY `idx_master_codicemacchinaprod` (`codicemacchinaprod`)")
                defs.append("  KEY `idx_master_modeling_eligible` (`modeling_eligible`)")
            elif table == "raw_daily_conferimenti":
                defs.append("  KEY `idx_conferimenti_pf` (`pf`)")
                defs.append("  KEY `idx_conferimenti_giorno_dt` (`giorno_dt`)")
                defs.append("  KEY `idx_conferimenti_macchina` (`codicemacchina`)")
            elif table == "training_master_daily":
                defs.append("  KEY `idx_train_pf` (`pf`)")
                defs.append("  KEY `idx_train_giorno_dt` (`giorno_dt`)")
            elif table == "test_holdout":
                defs.append("  KEY `idx_test_pf` (`pf`)")
                defs.append("  KEY `idx_test_giorno_dt` (`giorno_dt`)")

            out.write(",\n".join(defs))
            out.write("\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;\n\n")

            # Stream rows in chunks
            col_str = ", ".join([f"`{c}`" for c in col_names])
            select_cols = ", ".join([f'"{c}"' for c in col_names])
            cur.execute(f'SELECT {select_cols} FROM "{table}";')

            row_count = 0
            while True:
                rows = cur.fetchmany(1000)
                if not rows:
                    break
                row_count += len(rows)
                val_lines = []
                for r in rows:
                    val_lines.append("(" + ", ".join(escape_val(v) for v in r) + ")")
                out.write(f"INSERT INTO `{table}` ({col_str}) VALUES\n" + ",\n".join(val_lines) + ";\n")
                if row_count % 100000 == 0:
                    print(f"  ... written {row_count:,} rows")

            print(f"--> Exported `{table}`: {row_count:,} rows in {time.time() - t0:.1f}s")

        out.write("\nSET FOREIGN_KEY_CHECKS=1;\n")

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print("=" * 70)
    print(f"EXPORT COMPLETE in {time.time() - t_start:.1f}s")
    print(f"Output: {OUTPUT_PATH} ({size_mb:.2f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    export_all()
