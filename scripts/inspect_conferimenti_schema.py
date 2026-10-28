"""
Script to inspect PostgreSQL raw_daily_conferimenti schema and constraints.
"""
import os
import dotenv
from sqlalchemy import create_engine, text

dotenv.load_dotenv()
pg_url = os.getenv("DATABASE_URL")
engine = create_engine(pg_url)

with engine.connect() as conn:
    print("Columns:")
    cols = conn.execute(text("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'raw_daily_conferimenti'
        ORDER BY ordinal_position;
    """)).fetchall()
    for col in cols:
        print(" ", col)

    print("\nConstraints:")
    cons = conn.execute(text("""
        SELECT conname, pg_get_constraintdef(c.oid)
        FROM pg_constraint c
        WHERE conrelid = 'raw_daily_conferimenti'::regclass;
    """)).fetchall()
    for con in cons:
        print(" ", con)

    print("\nIndexes:")
    idxs = conn.execute(text("""
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = 'raw_daily_conferimenti';
    """)).fetchall()
    for idx in idxs:
        print(" ", idx)
