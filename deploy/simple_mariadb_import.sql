-- ==============================================================================
-- Simple MariaDB Import Script for `machine_prediction_history`
-- Target Database: coripet_rvm
-- ==============================================================================

USE `coripet_rvm`;

SET FOREIGN_KEY_CHECKS=0;
SET NAMES utf8mb4;

-- 1. Create Table
DROP TABLE IF EXISTS `machine_prediction_history`;

CREATE TABLE `machine_prediction_history` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `punto_fisico` VARCHAR(64) NOT NULL,
    `status_source` VARCHAR(20) NOT NULL,
    `store_name` VARCHAR(255) DEFAULT NULL,
    `address` TEXT DEFAULT NULL,
    `cap` VARCHAR(20) DEFAULT NULL,
    `comune` VARCHAR(128) DEFAULT NULL,
    `provincia` VARCHAR(32) DEFAULT NULL,
    `regione` VARCHAR(64) DEFAULT NULL,
    `machine_type` VARCHAR(64) DEFAULT NULL,
    `filiera` VARCHAR(128) DEFAULT NULL,
    `predicted_daily_bottles` FLOAT DEFAULT 0.0,
    `predicted_monthly_bottles` BIGINT DEFAULT 0,
    `predicted_annual_bottles` BIGINT DEFAULT 0,
    `predicted_daily_visits` FLOAT DEFAULT 0.0,
    `bottles_per_user` FLOAT DEFAULT 0.0,
    `volume_tier` VARCHAR(64) DEFAULT NULL,
    `p10_floor` FLOAT DEFAULT 0.0,
    `p90_peak` FLOAT DEFAULT 0.0,
    `confidence_score` FLOAT DEFAULT 0.0,
    `sales_area_sqm` FLOAT DEFAULT 0.0,
    `cash_registers` FLOAT DEFAULT 0.0,
    `nielsen_format` VARCHAR(64) DEFAULT NULL,
    `reward_type` VARCHAR(128) DEFAULT NULL,
    `reward_multiplier` FLOAT DEFAULT 1.0,
    `nearest_coripet_km` FLOAT DEFAULT 0.0,
    `seasonality_json` LONGTEXT DEFAULT NULL,
    `full_prediction_json` LONGTEXT DEFAULT NULL,
    `inspection_date` DATE NOT NULL,
    `inspection_timestamp` DATETIME NOT NULL,
    INDEX `idx_hist_pf` (`punto_fisico`),
    INDEX `idx_hist_date` (`inspection_date`),
    INDEX `idx_hist_status` (`status_source`),
    UNIQUE KEY `uq_pf_inspection_date` (`punto_fisico`, `inspection_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. Insert Initial Seed / Sample Records
INSERT INTO `machine_prediction_history` (
    `id`, `punto_fisico`, `status_source`, `store_name`, `address`, `cap`, `comune`, `provincia`, `regione`,
    `machine_type`, `filiera`, `predicted_daily_bottles`, `predicted_monthly_bottles`, `predicted_annual_bottles`,
    `predicted_daily_visits`, `bottles_per_user`, `volume_tier`, `p10_floor`, `p90_peak`, `confidence_score`,
    `sales_area_sqm`, `cash_registers`, `nielsen_format`, `reward_type`, `reward_multiplier`, `nearest_coripet_km`,
    `seasonality_json`, `full_prediction_json`, `inspection_date`, `inspection_timestamp`
) VALUES 
(
    1, 'PF00003087', 'POT', 'Conad Castelfidardo - Via Musone 2', 'Via Musone 2', '60022', 'Castelfidardo', 'AN', 'Marche',
    'DAINT_G', 'CONAD', 1463.6, 44493, 534214, 52.0, 28.1, 'ALTO PRENDENTE', 1054.0, 1976.0, 0.85,
    2576.0, 16.0, 'SUP', 'Emissione Buono sconto', 1.0, 3.7,
    '[]', '{}', '2026-11-04', '2026-11-04 09:52:30'
),
(
    2, 'PF00003059', 'POT', 'Sì con Te Fabriano - Viale Campo Sportivo', 'Viale Campo Sportivo, 29', '60044', 'Fabriano', 'AN', 'Marche',
    'DAINT_G', 'Sì con Te', 124.3, 3779, 45370, 6.4, 19.4, 'BASSO PRENDENTE', 89.5, 167.8, 0.85,
    750.0, 4.0, 'SUP', 'Emissione Buono sconto', 1.0, 2.02,
    '[]', '{}', '2026-11-04', '2026-11-04 10:00:00'
),
(
    3, 'PF00002997', 'SPL', 'Coop Barano d\'Ischia - Via Croce Testaccio', 'Via Croce Testaccio', '80070', 'Barano d\'Ischia', 'NA', 'Campania',
    'Ripet_C-front', 'COOP', 357.5, 10868, 130488, 18.0, 19.9, 'MEDIO PRENDENTE', 257.0, 482.0, 0.85,
    950.0, 6.0, 'SUP', 'Punti Spesa Coop', 1.15, 4.5,
    '[]', '{}', '2026-11-04', '2026-11-04 10:05:00'
);

SET FOREIGN_KEY_CHECKS=1;
