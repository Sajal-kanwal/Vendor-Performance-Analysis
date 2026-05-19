# Architecture Documentation

## System Architecture

This document describes the technical architecture of the Vendor Performance Analysis pipeline.

---

## Component Overview

### Layer 1: Data Ingestion

The ingestion layer handles extracting data from multiple source types and loading it into MySQL.

#### `mysql_database_loader.py` — Batch CSV Loader

**Purpose**: One-time or periodic bulk loading of CSV files into MySQL tables.

**Design Decisions**:
- Uses MySQL connection pooling (`pool_size=5`) for concurrent table creation and data insertion
- Batch insert strategy (`batch_size=1000`) prevents memory overflow on large files (sales.csv = 1.5GB)
- Automatic MySQL type inference from pandas dtypes — handles int, float, datetime, varchar, text
- Table names are sanitized from filenames (lowercase, alphanumeric + underscore)
- Auto-incremented `id` primary key added to every table for referential integrity

**Performance**: Successfully loaded 15.6M rows across 6 tables with 100% success rate.

#### `real_time_data_pipeline.py` — Multi-Source ETL Pipeline

**Purpose**: Scheduled, automated data synchronization from heterogeneous sources.

**Architecture Pattern**: Strategy pattern via `DataExtractor` abstract base class with concrete implementations for each source type.

```
DataExtractor (ABC)
├── APIExtractor      — REST API with auth/headers/pagination
├── FTPExtractor      — FTP server file download
├── SFTPExtractor     — Secure FTP via paramiko
├── LocalFileExtractor — Local filesystem monitoring
└── DatabaseExtractor — Cross-database replication via SQL queries
```

**Sync Modes**:
| Mode | Behavior | Use Case |
|---|---|---|
| `full` | TRUNCATE + INSERT all rows | Small reference tables |
| `incremental` | INSERT only rows newer than last sync | Large transactional tables |
| `upsert` | INSERT ON DUPLICATE KEY UPDATE | Master data with updates |

**Metadata Tracking**: Two internal tables track pipeline health:
- `pipeline_sync_history` — per-sync audit log (start/end time, row count, status)
- `pipeline_data_quality` — data integrity checksums and null/duplicate counts

---

### Layer 2: Data Validation

#### `data_quality.py` — Validation Framework

**Purpose**: Automated, profile-based data quality checks.

**Design**: Uses a declarative validation profile pattern — each table has a JSON-like configuration defining expected schema, ranges, and thresholds.

**Check Categories**:
1. **Schema** — Required columns present
2. **Integrity** — Null percentages, non-nullable constraints, duplicate detection
3. **Range** — Business-rule value bounds (e.g., PurchasePrice ∈ [0, 10000])
4. **Statistical** — Z-score outlier detection (threshold configurable)
5. **Type** — Numeric columns actually stored as numeric dtype

**Output**: Structured JSON report with pass/fail per check, violation counts, and summary statistics.

---

### Layer 3: Data Transformation

#### `get_vendor_summary.py` — Analytical Core

**Purpose**: Creates the denormalized `vendor_sales_summary` table that drives all analysis.

**SQL Strategy**: Uses Common Table Expressions (CTEs) for readability and maintainability:

```sql
WITH FreightSummary   AS (...),   -- Vendor freight totals
     PurchaseSummary  AS (...),   -- Vendor×Brand purchase aggregates
     SalesSummary     AS (...)    -- Vendor×Brand sales aggregates
SELECT ... FROM PurchaseSummary
LEFT JOIN SalesSummary ON vendor+brand
LEFT JOIN FreightSummary ON vendor
```

**Derived Metrics**:
| Metric | Formula | Business Meaning |
|---|---|---|
| GrossProfit | Sales$ − Purchase$ | Raw profitability |
| ProfitMargin | (Profit / Sales$) × 100 | Percentage margin |
| StockTurnover | SalesQty / PurchaseQty | Inventory velocity |
| FreightToSalesRatio | (Freight / Sales$) × 100 | Logistics cost burden |
| PriceMarkup | ((Retail − Cost) / Cost) × 100 | Pricing power |

---

### Layer 4: Analysis

#### Jupyter Notebooks

- `eda.ipynb` — Exploratory analysis: table profiling, data types, null patterns, initial summary creation
- `vendor_performance_analysis.ipynb` — Deep analysis: distributions (histograms, box plots), outlier detection (IQR + z-scores), log transformations, correlation matrices, normality tests, margin tier segmentation

---

### Layer 5: ML Feature Engineering

#### `feature_engineering.py` — Feature Store

**Purpose**: Transform business data into model-ready features with clear semantic meaning.

**Design Principles**:
- All engineered features prefixed with `feat_` for namespace separation
- Pure pandas transformations — no database dependency during computation
- Each feature includes a docstring explaining its business meaning
- Division-by-zero guards on all ratio computations

**Feature Layers**:

```
Input: vendor_sales_summary (14 columns)
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
Profitability  Volume    Pricing      ← Business features
    │           │           │
    ▼           ▼           ▼
Inventory   Vendor Agg  Statistical  ← Derived features
    │           │           │
    ▼           ▼           ▼
           Categorical               ← Encoding features
                │
Output: ml_feature_store (31+ columns)
```

---

## Cross-Cutting Concerns

### Configuration (`config.py`)
- Single source of truth for all settings
- Reads from `.env` files with `python-dotenv`
- Frozen dataclasses prevent accidental mutation
- Cached singleton pattern (`get_config()`)

### Logging (`utils/logging_config.py`)
- Rotating file handlers (10MB max, 5 backups)
- Consistent format across all modules
- `@timer` decorator for performance monitoring

### Error Handling (`utils/decorators.py`)
- `@retry` with exponential backoff for transient DB failures
- `@validate_dataframe` for input schema enforcement
- All database operations wrapped in try/except with rollback

---

## Data Flow Summary

```
CSV Files → mysql_database_loader → MySQL Tables
                                        │
MySQL Tables → data_quality.py → Validation Report
                                        │
MySQL Tables → get_vendor_summary.py → vendor_sales_summary
                                        │
vendor_sales_summary → eda.ipynb → Analysis Outputs
                                        │
vendor_sales_summary → feature_engineering.py → ml_feature_store
                                        │
ml_feature_store → [Future ML/DL Models]
```
