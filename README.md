# Vendor Performance Analysis

> **Enterprise-grade data analytics pipeline** for vendor performance evaluation, inventory optimization, and ML-ready feature engineering — built with Python, MySQL, and production-quality engineering practices.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-8.0+-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production-brightgreen?style=for-the-badge)

---

## Overview

This project implements an end-to-end data analytics pipeline that ingests, validates, transforms, and analyzes vendor performance data from a multi-store retail/distribution operation processing **15.6 million+ transactional records** across 6 source tables.

The pipeline is designed with clear separation of concerns and extensibility for downstream **Machine Learning / Deep Learning** applications.

### What This Project Demonstrates

| Capability | Implementation |
|---|---|
| **ETL Pipeline** | Enterprise CSV→MySQL loader with connection pooling, batching, and error handling |
| **Real-Time Ingestion** | Multi-source pipeline (API, FTP, SFTP, S3, DB) with scheduling and alerting |
| **Data Quality** | Automated validation framework with schema, range, null, and outlier checks |
| **Analytical SQL** | CTE-based vendor summary generation joining purchases, sales, pricing, and freight |
| **EDA & Statistical Analysis** | Jupyter notebooks with distribution analysis, outlier detection, and correlation matrices |
| **Feature Engineering** | 17+ ML-ready features across profitability, pricing, inventory, and vendor dimensions |
| **Configuration Management** | Centralized env-based config with no hardcoded credentials |
| **Testing** | Pytest suite for data quality and feature engineering validation |
| **Logging & Observability** | Rotating file logs, structured output, execution timing decorators |

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │            DATA SOURCES                 │
                    │  CSV │ API │ FTP │ SFTP │ S3 │ Database │
                    └─────────────┬───────────────────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   INGESTION LAYER            │
                    │  mysql_database_loader.py    │
                    │  real_time_data_pipeline.py  │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   VALIDATION LAYER           │
                    │  data_quality.py             │
                    │  Schema │ Nulls │ Ranges │   │
                    │  Outliers │ Duplicates       │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   TRANSFORMATION LAYER       │
                    │  get_vendor_summary.py       │
                    │  CTE joins │ Cleaning │      │
                    │  Derived metrics              │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   ANALYSIS LAYER             │
                    │  eda.ipynb                   │
                    │  vendor_performance_         │
                    │  analysis.ipynb              │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   ML FEATURE LAYER           │
                    │  feature_engineering.py      │
                    │  17+ features → ml_feature_  │
                    │  store table                  │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   ML/DL MODELS (Future)      │
                    │  Demand Forecasting          │
                    │  Vendor Scoring               │
                    │  Price Optimization           │
                    └─────────────────────────────┘
```

---

## Project Structure

```
Vendor-Performance-Analysis/
├── config.py                         # Centralized configuration (.env loader)
├── db_connect.py                     # MySQL connection module (SQLAlchemy)
├── mysql_database_loader.py          # Enterprise CSV → MySQL loader
├── real_time_data_pipeline.py        # Multi-source ETL pipeline with scheduling
├── get_vendor_summary.py             # Vendor summary generator (CTE SQL + cleaning)
├── data_quality.py                   # Data quality validation framework
├── feature_engineering.py            # ML feature engineering (17+ features)
│
├── eda.ipynb                         # Exploratory Data Analysis notebook
├── vendor_performance_analysis.ipynb # Statistical analysis & visualization
│
├── utils/
│   ├── __init__.py
│   ├── logging_config.py             # Rotating file + console logging
│   └── decorators.py                 # @timer, @retry, @validate_dataframe
│
├── tests/
│   ├── __init__.py
│   ├── test_data_quality.py          # DQ validation tests (synthetic data)
│   └── test_feature_engineering.py   # Feature computation tests
│
├── data/                             # Raw CSV data (gitignored)
│   ├── sales.csv                     # 12.8M records — point-of-sale transactions
│   ├── purchases.csv                 # 2.4M records — vendor purchase orders
│   ├── purchase_prices.csv           # 12K records — product pricing catalog
│   ├── begin_inventory.csv           # 207K records — period-start inventory
│   ├── end_inventory.csv             # 224K records — period-end inventory
│   └── vendor_invoice.csv            # 5.5K records — vendor invoices with freight
│
├── docs/
│   └── architecture.md               # Detailed architecture documentation
│
├── logs/                             # Pipeline execution logs (gitignored)
├── output/                           # Generated reports & exports
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variable template
└── .gitignore                        # Comprehensive gitignore
```

---

## Data Model

The pipeline processes 6 interrelated tables from a retail/distribution operation:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│    purchases     │     │  purchase_prices  │     │  vendor_invoice  │
│──────────────────│     │──────────────────│     │──────────────────│
│ VendorNumber  ◄──┼─────┤ VendorNumber     │     │ VendorNumber  ◄──┤
│ Brand         ◄──┼──┐  │ Brand            │     │ Freight          │
│ PurchasePrice    │  │  │ Price (retail)   │     │ InvoiceDate      │
│ Quantity         │  │  │ Volume           │     │ PayDate          │
│ Dollars          │  │  └──────────────────┘     └──────────────────┘
└──────────────────┘  │
                      │  ┌──────────────────┐     ┌──────────────────┐
                      │  │      sales       │     │  begin/end       │
                      │  │──────────────────│     │  inventory       │
                      └──┤ Brand            │     │──────────────────│
                         │ VendorNo         │     │ Store            │
                         │ SalesQuantity    │     │ Brand            │
                         │ SalesDollars     │     │ onHand           │
                         │ ExciseTax        │     │ Price            │
                         └──────────────────┘     └──────────────────┘

                    ▼ CTE JOIN + Aggregation ▼

                   ┌─────────────────────────┐
                   │  vendor_sales_summary   │
                   │─────────────────────────│
                   │ VendorNumber            │
                   │ TotalPurchaseDollars    │
                   │ TotalSalesDollars       │
                   │ GrossProfit             │
                   │ ProfitMargin            │
                   │ StockTurnover           │
                   │ FreightToSalesRatio     │
                   │ PriceMarkup             │
                   └────────────┬────────────┘
                                │
                   ┌────────────▼────────────┐
                   │    ml_feature_store     │
                   │─────────────────────────│
                   │ feat_profit_per_unit    │
                   │ feat_price_gap_pct      │
                   │ feat_sell_through_rate  │
                   │ feat_vendor_brand_count │
                   │ feat_log_sales          │
                   │ ... (17+ features)      │
                   └─────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- MySQL 8.0+ (with a database created)
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Vendor-Performance-Analysis.git
cd Vendor-Performance-Analysis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your MySQL credentials

# 5. Place CSV data files in ./data/
```

### Run the Pipeline

```bash
# Step 1: Load CSV data into MySQL
python mysql_database_loader.py

# Step 2: Generate vendor summary table
python get_vendor_summary.py

# Step 3: Run data quality checks
python data_quality.py --profile vendor_summary --output output/dq_report.json

# Step 4: Build ML features
python feature_engineering.py --save-db

# Step 5: Run tests
python -m pytest tests/ -v
```

### Start the Real-Time Pipeline

```bash
# Starts the scheduled ETL pipeline with all configured data sources
python real_time_data_pipeline.py
# Press Ctrl+C to stop
```

---

## Key Components

### 1. CSV to MySQL Loader (`mysql_database_loader.py`)

Enterprise-grade data ingestion with:
- **Connection pooling** for concurrent operations
- **Batch inserts** (configurable batch size) for memory efficiency
- **Automatic schema inference** from DataFrame dtypes → MySQL types
- **Detailed logging** and JSON report generation
- Successfully loaded **15.6M rows** across 6 tables

### 2. Real-Time Data Pipeline (`real_time_data_pipeline.py`)

Multi-source ETL orchestrator supporting:
- **5 source types**: REST API, FTP, SFTP, Local Files, Database
- **3 sync modes**: Full replacement, Incremental (timestamp-based), Upsert
- **Automated scheduling** via `schedule` library
- **Email alerting** on sync failures
- **Metadata tracking** tables for sync history and data quality metrics

### 3. Vendor Summary Generator (`get_vendor_summary.py`)

Analytical SQL engine that:
- Joins purchases, sales, pricing, and freight via **CTE-based SQL**
- Computes **6 derived business metrics** (GrossProfit, ProfitMargin, StockTurnover, etc.)
- Supports CLI with `--dry-run`, `--output csv/db/both` options
- Includes retry logic for database resilience

### 4. Data Quality Framework (`data_quality.py`)

Configurable validation with **5 check categories**:
- Schema completeness (required columns)
- Null percentage thresholds & non-nullable constraints
- Business-rule range validation
- Statistical outlier detection (z-score based)
- Data type verification

### 5. Feature Engineering (`feature_engineering.py`)

ML-ready feature store generating **17+ features** across 7 layers:

| Layer | Features | Purpose |
|---|---|---|
| Profitability | profit_per_unit, profit_tier, is_profitable | Target variables & margin analysis |
| Volume | volume_category, demand_velocity | Product sizing & demand signals |
| Pricing | price_gap, price_competitiveness, excise_burden | Pricing strategy inputs |
| Inventory | sell_through_rate, overstock_flag, revenue_per_purchase_dollar | Stock optimization |
| Vendor Aggregates | vendor_brand_count, vendor_revenue_share | Vendor portfolio analysis |
| Statistical | log transforms, z-scores | Distribution normalization |
| Categorical | vendor_frequency encoding | Model-ready encodings |

---

## ML/DL Foundation

This project is designed as the **data engineering foundation** for downstream machine learning applications:

### Planned ML Extensions

| Model | Objective | Key Features Used |
|---|---|---|
| **Demand Forecasting** | Predict future sales quantities per brand | feat_demand_velocity, feat_sell_through_rate, feat_log_sales |
| **Vendor Scoring** | Rank vendors by overall performance | feat_vendor_avg_margin, feat_vendor_brand_count, feat_profit_tier |
| **Price Optimization** | Recommend optimal pricing | feat_price_gap_pct, feat_price_competitiveness, feat_excise_burden |
| **Inventory Anomaly Detection** | Flag unusual stock patterns | feat_overstock_flag, feat_quantity_ratio, feat_zscore_profit_margin |

### Feature Store Design

The `feat_` prefix convention enables easy model consumption:
```python
from feature_engineering import FeatureEngineer

fe = FeatureEngineer()
features_df = fe.build_features(vendor_summary_df)

# Extract only model-ready numeric features
X = fe.get_numeric_features(features_df)
y = features_df["feat_is_profitable"]  # Binary classification target
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Database | MySQL 8.0+ (InnoDB) |
| ORM/Driver | SQLAlchemy 2.0, mysql-connector-python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Statistics | SciPy |
| Pipeline Scheduling | schedule |
| Remote Sources | requests, paramiko (SFTP), ftplib |
| File Formats | CSV, JSON, Parquet (PyArrow) |
| Configuration | python-dotenv |
| Testing | pytest |
| Logging | logging (RotatingFileHandler) |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_data_quality.py -v
python -m pytest tests/test_feature_engineering.py -v
```

Tests use **synthetic data** and require no database connection.

---

## Configuration

All settings are loaded from environment variables. Copy `.env.example` to `.env` and configure:

```bash
# Required
MYSQL_HOST=localhost
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=vendor_performance

# Optional
PIPELINE_BATCH_SIZE=1000
PIPELINE_LOG_LEVEL=INFO
```

See [.env.example](.env.example) for the complete list of configurable parameters.
