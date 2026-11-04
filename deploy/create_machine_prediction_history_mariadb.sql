-- ==============================================================================
-- Coripet RVM Predictor: Machine Prediction & Inspection History Table
-- Compatible with Plesk MariaDB 10.5+, MySQL 8+, and phpMyAdmin Direct Import
-- Database: coripet_rvm
-- ==============================================================================

SET FOREIGN_KEY_CHECKS=0;
SET NAMES utf8mb4;

-- Table structure for `machine_prediction_history`
CREATE TABLE IF NOT EXISTS `machine_prediction_history` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `punto_fisico` VARCHAR(64) NOT NULL COMMENT 'Coripet Physical Machine ID (e.g. PF00003087)',
    `status_source` VARCHAR(20) NOT NULL COMMENT 'SPL or POT machine registry',
    `store_name` VARCHAR(255) DEFAULT NULL COMMENT 'Retail store or site name',
    `address` TEXT DEFAULT NULL COMMENT 'Street address',
    `cap` VARCHAR(20) DEFAULT NULL COMMENT 'Postal code',
    `comune` VARCHAR(128) DEFAULT NULL COMMENT 'Municipality / City',
    `provincia` VARCHAR(32) DEFAULT NULL COMMENT 'Province code (e.g. AN, MI)',
    `regione` VARCHAR(64) DEFAULT NULL COMMENT 'Italian administrative region',
    `machine_type` VARCHAR(64) DEFAULT NULL COMMENT 'Hardware model (e.g. DAINT_G, Ripet_C-front)',
    `filiera` VARCHAR(128) DEFAULT NULL COMMENT 'Retail partner brand (e.g. CONAD, COOP)',
    `predicted_daily_bottles` FLOAT DEFAULT 0.0 COMMENT 'Forecasted intake (bottles / day)',
    `predicted_monthly_bottles` BIGINT DEFAULT 0 COMMENT 'Forecasted monthly target volume',
    `predicted_annual_bottles` BIGINT DEFAULT 0 COMMENT 'Forecasted annual volume',
    `predicted_daily_visits` FLOAT DEFAULT 0.0 COMMENT 'Expected daily unique user visits',
    `bottles_per_user` FLOAT DEFAULT 0.0 COMMENT 'Average bottle count per user visit',
    `volume_tier` VARCHAR(64) DEFAULT NULL COMMENT 'Classification: SUPER, ALTO, MEDIO, or BASSO PRENDENTE',
    `p10_floor` FLOAT DEFAULT 0.0 COMMENT '10th percentile conservative volume floor',
    `p90_peak` FLOAT DEFAULT 0.0 COMMENT '90th percentile peak upside capacity',
    `confidence_score` FLOAT DEFAULT 0.0 COMMENT 'Model statistical confidence metric (0.0 - 1.0)',
    `sales_area_sqm` FLOAT DEFAULT 0.0 COMMENT 'Nielsen store sales area in square meters',
    `cash_registers` FLOAT DEFAULT 0.0 COMMENT 'Number of cash tills in store',
    `nielsen_format` VARCHAR(64) DEFAULT NULL COMMENT 'Store Nielsen format (SUP, HYPER, DISCOUNT)',
    `reward_type` VARCHAR(128) DEFAULT NULL COMMENT 'Active contract incentive mechanism',
    `reward_multiplier` FLOAT DEFAULT 1.0 COMMENT 'Model multiplier applied for reward engagement',
    `nearest_coripet_km` FLOAT DEFAULT 0.0 COMMENT 'Distance to nearest operating Coripet RVM (km)',
    `seasonality_json` LONGTEXT DEFAULT NULL COMMENT 'Full 15-month operational profile (3M ramp + 12M seasonality)',
    `full_prediction_json` LONGTEXT DEFAULT NULL COMMENT 'Complete lossless raw JSON payload from predictor',
    `inspection_date` DATE NOT NULL COMMENT 'Calendar date when inspection occurred',
    `inspection_timestamp` DATETIME NOT NULL COMMENT 'Exact timestamp of inspection',
    INDEX `idx_hist_pf` (`punto_fisico`),
    INDEX `idx_hist_date` (`inspection_date`),
    INDEX `idx_hist_status` (`status_source`),
    UNIQUE KEY `uq_pf_inspection_date` (`punto_fisico`, `inspection_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

SET FOREIGN_KEY_CHECKS=1;

