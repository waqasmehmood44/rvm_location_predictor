-- ==============================================================================
-- Coripet RVM Predictor: Hidden Machines Tables for PostgreSQL
-- Compatible with PostgreSQL 14+, 16+, and pgAdmin
-- ==============================================================================

-- 1. Hidden SPL Machines Table
CREATE TABLE IF NOT EXISTS hidden_spl_machines (
    punto_fisico VARCHAR(64) PRIMARY KEY,
    store_name TEXT,
    address TEXT,
    cap VARCHAR(16),
    comune VARCHAR(128),
    provincia VARCHAR(16),
    regione VARCHAR(64),
    machine_type VARCHAR(64),
    predicted_bottles REAL DEFAULT 0.0,
    predicted_month INTEGER DEFAULT 0,
    volume_tier VARCHAR(64),
    reward_type VARCHAR(128),
    dt_fine_sop VARCHAR(32),
    hidden_at VARCHAR(32)
);

-- 2. Hidden POT Machines Table
CREATE TABLE IF NOT EXISTS hidden_pot_machines (
    punto_fisico VARCHAR(64) PRIMARY KEY,
    store_name TEXT,
    address TEXT,
    cap VARCHAR(16),
    comune VARCHAR(128),
    provincia VARCHAR(16),
    regione VARCHAR(64),
    machine_type VARCHAR(64),
    predicted_bottles REAL DEFAULT 0.0,
    predicted_month INTEGER DEFAULT 0,
    volume_tier VARCHAR(64),
    reward_type VARCHAR(128),
    dt_fine_sop VARCHAR(32),
    hidden_at VARCHAR(32)
);
