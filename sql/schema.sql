-- Credence AI MySQL schema (generated from app/models.py by scripts/init_db.py)
CREATE DATABASE IF NOT EXISTS `credence_ai` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `credence_ai`;

CREATE TABLE prediction_batches (
	id INTEGER NOT NULL AUTO_INCREMENT, 
	name VARCHAR(120), 
	total_customers INTEGER NOT NULL, 
	predicted_defaults INTEGER NOT NULL, 
	average_default_probability DOUBLE NOT NULL, 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	PRIMARY KEY (id)
)ENGINE=InnoDB CHARSET=utf8mb4;


CREATE TABLE predictions (
	id INTEGER NOT NULL AUTO_INCREMENT, 
	batch_id INTEGER, 
	customer_ref VARCHAR(64), 
	`LIMIT_BAL` BIGINT NOT NULL, 
	`SEX` SMALLINT NOT NULL, 
	`EDUCATION` SMALLINT NOT NULL, 
	`MARRIAGE` SMALLINT NOT NULL, 
	`AGE` SMALLINT NOT NULL, 
	`PAY_0` SMALLINT NOT NULL, 
	`PAY_2` SMALLINT NOT NULL, 
	`PAY_3` SMALLINT NOT NULL, 
	`PAY_4` SMALLINT NOT NULL, 
	`PAY_5` SMALLINT NOT NULL, 
	`PAY_6` SMALLINT NOT NULL, 
	`BILL_AMT1` BIGINT NOT NULL, 
	`BILL_AMT2` BIGINT NOT NULL, 
	`BILL_AMT3` BIGINT NOT NULL, 
	`BILL_AMT4` BIGINT NOT NULL, 
	`BILL_AMT5` BIGINT NOT NULL, 
	`BILL_AMT6` BIGINT NOT NULL, 
	`PAY_AMT1` BIGINT NOT NULL, 
	`PAY_AMT2` BIGINT NOT NULL, 
	`PAY_AMT3` BIGINT NOT NULL, 
	`PAY_AMT4` BIGINT NOT NULL, 
	`PAY_AMT5` BIGINT NOT NULL, 
	`PAY_AMT6` BIGINT NOT NULL, 
	default_probability DOUBLE NOT NULL, 
	predicted_default BOOL NOT NULL, 
	risk_band ENUM('LOW','MEDIUM','HIGH') NOT NULL, 
	decision_threshold DOUBLE NOT NULL, 
	model_name VARCHAR(64) NOT NULL, 
	model_version VARCHAR(16) NOT NULL, 
	imputed_features VARCHAR(300), 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	PRIMARY KEY (id), 
	FOREIGN KEY(batch_id) REFERENCES prediction_batches (id) ON DELETE SET NULL
)ENGINE=InnoDB CHARSET=utf8mb4;

CREATE INDEX ix_predictions_batch_id ON predictions (batch_id);
CREATE INDEX ix_predictions_created_at ON predictions (created_at);
CREATE INDEX ix_predictions_customer_ref ON predictions (customer_ref);
CREATE INDEX ix_predictions_risk_band ON predictions (risk_band);
