-- ==============================================================================
-- Coripet RVM Predictor: Hidden Machines Tables for MariaDB / MySQL
-- Compatible with Plesk MariaDB 10.6+, MySQL 8+, and phpMyAdmin
-- ==============================================================================

SET FOREIGN_KEY_CHECKS=0;
SET NAMES utf8mb4;

-- 1. Hidden SPL Machines Table
CREATE TABLE IF NOT EXISTS `hidden_spl_machines` (
    `punto_fisico` VARCHAR(64) NOT NULL,
    `store_name` TEXT DEFAULT NULL,
    `address` TEXT DEFAULT NULL,
    `cap` VARCHAR(16) DEFAULT NULL,
    `comune` VARCHAR(128) DEFAULT NULL,
    `provincia` VARCHAR(16) DEFAULT NULL,
    `regione` VARCHAR(64) DEFAULT NULL,
    `machine_type` VARCHAR(64) DEFAULT NULL,
    `predicted_bottles` FLOAT DEFAULT 0.0,
    `predicted_month` INT DEFAULT 0,
    `volume_tier` VARCHAR(64) DEFAULT NULL,
    `reward_type` VARCHAR(128) DEFAULT NULL,
    `dt_fine_sop` VARCHAR(32) DEFAULT NULL,
    `hidden_at` VARCHAR(32) DEFAULT NULL,
    PRIMARY KEY (`punto_fisico`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. Hidden POT Machines Table
CREATE TABLE IF NOT EXISTS `hidden_pot_machines` (
    `punto_fisico` VARCHAR(64) NOT NULL,
    `store_name` TEXT DEFAULT NULL,
    `address` TEXT DEFAULT NULL,
    `cap` VARCHAR(16) DEFAULT NULL,
    `comune` VARCHAR(128) DEFAULT NULL,
    `provincia` VARCHAR(16) DEFAULT NULL,
    `regione` VARCHAR(64) DEFAULT NULL,
    `machine_type` VARCHAR(64) DEFAULT NULL,
    `predicted_bottles` FLOAT DEFAULT 0.0,
    `predicted_month` INT DEFAULT 0,
    `volume_tier` VARCHAR(64) DEFAULT NULL,
    `reward_type` VARCHAR(128) DEFAULT NULL,
    `dt_fine_sop` VARCHAR(32) DEFAULT NULL,
    `hidden_at` VARCHAR(32) DEFAULT NULL,
    PRIMARY KEY (`punto_fisico`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

SET FOREIGN_KEY_CHECKS=1;
