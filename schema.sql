-- SkillMap MySQL schema (MySQL 8 / MariaDB 10.5+). Created by load_db.py or by: mysql < schema.sql
CREATE DATABASE IF NOT EXISTS skillmap CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE skillmap;

DROP VIEW IF EXISTS v_series;
DROP TABLE IF EXISTS fact_history;
DROP TABLE IF EXISTS fact_series;
DROP TABLE IF EXISTS dim_district;
DROP TABLE IF EXISTS dim_skill;
DROP TABLE IF EXISTS dim_state;
DROP TABLE IF EXISTS kb_document;
DROP TABLE IF EXISTS app_meta;

CREATE TABLE dim_state (
  state_id INT AUTO_INCREMENT PRIMARY KEY,
  state_name VARCHAR(100) NOT NULL UNIQUE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE dim_district (
  district_id INT AUTO_INCREMENT PRIMARY KEY,
  state_id INT NOT NULL,
  district_name VARCHAR(120) NOT NULL,
  INDEX ix_district_name (district_name),
  CONSTRAINT fk_district_state FOREIGN KEY (state_id) REFERENCES dim_state(state_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE dim_skill (
  skill_id INT AUTO_INCREMENT PRIMARY KEY,
  skill_name VARCHAR(150) NOT NULL UNIQUE,
  sector VARCHAR(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE fact_series (
  series_id VARCHAR(190) NOT NULL PRIMARY KEY,
  state_id INT NOT NULL,
  district_id INT NOT NULL,
  skill_id INT NOT NULL,
  `current_demand` INT NULL,
  `forecast_3m` INT NULL,
  `forecast_6m` INT NULL,
  `forecast_12m` INT NULL,
  `demand_growth` DOUBLE NULL,
  `confidence` DOUBLE NULL,
  `low_volume_flag` INT NULL,
  `demand_trend` VARCHAR(64) NULL,
  `forecast_3m_low` INT NULL,
  `forecast_3m_high` INT NULL,
  `forecast_6m_low` INT NULL,
  `forecast_6m_high` INT NULL,
  `forecast_12m_low` INT NULL,
  `forecast_12m_high` INT NULL,
  `supply_3m` INT NULL,
  `gap_3m` INT NULL,
  `gap_index_3m` DOUBLE NULL,
  `risk_3m` VARCHAR(64) NULL,
  `direction_3m` VARCHAR(64) NULL,
  `supply_6m` INT NULL,
  `gap_6m` INT NULL,
  `gap_index_6m` DOUBLE NULL,
  `risk_6m` VARCHAR(64) NULL,
  `direction_6m` VARCHAR(64) NULL,
  `supply_12m` INT NULL,
  `gap_12m` INT NULL,
  `gap_index_12m` DOUBLE NULL,
  `risk_12m` VARCHAR(64) NULL,
  `direction_12m` VARCHAR(64) NULL,
  `skill_gap` INT NULL,
  `risk` VARCHAR(64) NULL,
  `direction` VARCHAR(64) NULL,
  `gap_index` DOUBLE NULL,
  `ml_risk` VARCHAR(64) NULL,
  `ml_prob_low` DOUBLE NULL,
  `ml_prob_medium` DOUBLE NULL,
  `ml_prob_high` DOUBLE NULL,
  `ml_prob_critical` DOUBLE NULL,
  `ml_confidence` DOUBLE NULL,
  `training_capacity` INT NULL,
  `completion_rate` DOUBLE NULL,
  `seat_utilisation` DOUBLE NULL,
  `placement_rate` DOUBLE NULL,
  `training_centres` INT NULL,
  `current_supply` INT NULL,
  `current_supply_3m_avg` DOUBLE NULL,
  `supply_confidence` DOUBLE NULL,
  `recommendation` VARCHAR(64) NULL,
  `seat_change` INT NULL,
  `priority_score` DOUBLE NULL,
  `priority_rank` INT NULL,
  `priority` VARCHAR(64) NULL,
  `recommendation_detail` VARCHAR(1000) NULL,
  `supply_drivers_12m` VARCHAR(1000) NULL,
  `risk_drivers` VARCHAR(1000) NULL,
  `supply_3m_low` INT NULL,
  `supply_3m_high` INT NULL,
  `supply_confidence_3m` DOUBLE NULL,
  `supply_6m_low` INT NULL,
  `supply_6m_high` INT NULL,
  `supply_confidence_6m` DOUBLE NULL,
  `supply_12m_low` INT NULL,
  `supply_12m_high` INT NULL,
  `supply_confidence_12m` DOUBLE NULL,
  `supply_model_used` VARCHAR(1000) NULL,
  `annual_seats` INT NULL,
  INDEX ix_fs_state (state_id), INDEX ix_fs_district (district_id), INDEX ix_fs_skill (skill_id),
  INDEX ix_fs_risk (risk), INDEX ix_fs_rank (priority_rank)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE fact_history (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  series_id VARCHAR(190) NOT NULL,
  month_date DATE NOT NULL,
  demand_raw DOUBLE NULL, demand_const DOUBLE NULL,
  supply_certified DOUBLE NULL, supply_enrolled DOUBLE NULL, supply_placed DOUBLE NULL,
  INDEX ix_fh_series (series_id), INDEX ix_fh_month (month_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE kb_document (
  doc_id INT AUTO_INCREMENT PRIMARY KEY,
  category VARCHAR(40) NOT NULL,
  title VARCHAR(200) NOT NULL,
  content TEXT NOT NULL,
  keywords TEXT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE app_meta (
  meta_key VARCHAR(60) PRIMARY KEY,
  meta_value LONGTEXT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS chat_log (
  chat_id BIGINT AUTO_INCREMENT PRIMARY KEY,
  session_id VARCHAR(64) NOT NULL,
  question TEXT NOT NULL,
  intent VARCHAR(40) NULL,
  answer MEDIUMTEXT NULL,
  used_llm TINYINT(1) NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX ix_chat_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE VIEW v_series AS
SELECT f.*, s.state_name AS state, d.district_name AS district, k.skill_name AS skill, k.sector AS sector
FROM fact_series f
JOIN dim_state s ON s.state_id=f.state_id
JOIN dim_district d ON d.district_id=f.district_id
JOIN dim_skill k ON k.skill_id=f.skill_id;

CREATE TABLE IF NOT EXISTS chat_conversation (
  conv_id VARCHAR(32) PRIMARY KEY,
  title VARCHAR(200) NOT NULL,
  day_key VARCHAR(12) NULL,
  updated_label VARCHAR(40) NULL,
  msgs LONGTEXT NOT NULL,
  updated_ms BIGINT NOT NULL,
  INDEX ix_conv_updated (updated_ms)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;