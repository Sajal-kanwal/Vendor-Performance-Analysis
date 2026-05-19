"""
Vendor Summary Generator
=========================
Production-grade pipeline step that creates an aggregated vendor performance
summary by joining purchases, sales, purchase prices, and freight data.

This is the analytical core of the project — it transforms raw transactional
tables into a single denormalized summary table (``vendor_sales_summary``)
that drives all downstream analysis, dashboards, and ML feature engineering.

SQL Logic (CTE-based):
    1. FreightSummary   — total freight cost per vendor
    2. PurchaseSummary  — aggregated purchase quantities & dollars per vendor-brand
    3. SalesSummary     — aggregated sales metrics per vendor-brand
    4. Final JOIN       — left-joins all CTEs and orders by purchase volume

Derived Columns:
    - GrossProfit         = TotalSalesDollars − TotalPurchaseDollars
    - ProfitMargin (%)    = (GrossProfit / TotalSalesDollars) × 100
    - StockTurnover       = TotalSalesQuantity / TotalPurchaseQuantity
    - SalesToPurchaseRatio = TotalSalesDollars / TotalPurchaseDollars
    - FreightToSalesRatio = FreightCost / TotalSalesDollars
    - PriceMarkup (%)     = ((ActualPrice − PurchasePrice) / PurchasePrice) × 100

Usage:
    python get_vendor_summary.py                    # default: generate + save
    python get_vendor_summary.py --dry-run          # generate without saving
    python get_vendor_summary.py --output csv       # export to CSV
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from config import get_config
from utils.logging_config import setup_logger
from utils.decorators import timer, retry

logger = setup_logger(__name__)


# ────────────────────────────────────────────────
# SQL: Core Vendor Summary Query
# ────────────────────────────────────────────────

VENDOR_SUMMARY_SQL = """
WITH FreightSummary AS (
    SELECT
        VendorNumber,
        SUM(Freight) AS FreightCost
    FROM vendor_invoice
    GROUP BY VendorNumber
),

PurchaseSummary AS (
    SELECT
        p.VendorNumber,
        p.VendorName,
        p.Brand,
        p.Description,
        p.PurchasePrice,
        pp.Price          AS ActualPrice,
        pp.Volume,
        SUM(p.Quantity)   AS TotalPurchaseQuantity,
        SUM(p.Dollars)    AS TotalPurchaseDollars
    FROM purchases p
    JOIN purchase_prices pp
        ON p.Brand = pp.Brand
    WHERE p.PurchasePrice > 0
    GROUP BY
        p.VendorNumber, p.VendorName, p.Brand,
        p.Description, p.PurchasePrice, pp.Price, pp.Volume
),

SalesSummary AS (
    SELECT
        VendorNo,
        Brand,
        SUM(SalesQuantity)  AS TotalSalesQuantity,
        SUM(SalesDollars)   AS TotalSalesDollars,
        SUM(SalesPrice)     AS TotalSalesPrice,
        SUM(ExciseTax)      AS TotalExciseTax
    FROM sales
    GROUP BY VendorNo, Brand
)

SELECT
    ps.VendorNumber,
    ps.VendorName,
    ps.Brand,
    ps.Description,
    ps.PurchasePrice,
    ps.ActualPrice,
    ps.Volume,
    ps.TotalPurchaseQuantity,
    ps.TotalPurchaseDollars,
    ss.TotalSalesQuantity,
    ss.TotalSalesDollars,
    ss.TotalSalesPrice,
    ss.TotalExciseTax,
    fs.FreightCost
FROM PurchaseSummary ps
LEFT JOIN SalesSummary ss
    ON ps.VendorNumber = ss.VendorNo
    AND ps.Brand = ss.Brand
LEFT JOIN FreightSummary fs
    ON ps.VendorNumber = fs.VendorNumber
ORDER BY ps.TotalPurchaseDollars DESC
"""


# ────────────────────────────────────────────────
# Data Cleaning
# ────────────────────────────────────────────────

def clean_vendor_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and type-cast the raw vendor summary DataFrame.

    Steps:
        1. Convert Volume to float
        2. Fill missing numeric values with 0
        3. Strip whitespace from categorical columns
        4. Compute derived financial metrics
    """
    logger.info(f"Cleaning vendor summary — {len(df)} rows")

    # Type coercion
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(float)

    # Null handling — fill numeric columns with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Whitespace cleanup
    for col in ["VendorName", "Description"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # ── Derived Metrics ──────────────────────────────

    # Core profitability
    df["GrossProfit"] = df["TotalSalesDollars"] - df["TotalPurchaseDollars"]

    df["ProfitMargin"] = np.where(
        df["TotalSalesDollars"] != 0,
        (df["GrossProfit"] / df["TotalSalesDollars"]) * 100,
        0.0,
    )

    # Inventory efficiency
    df["StockTurnover"] = np.where(
        df["TotalPurchaseQuantity"] != 0,
        df["TotalSalesQuantity"] / df["TotalPurchaseQuantity"],
        0.0,
    )

    df["SalesToPurchaseRatio"] = np.where(
        df["TotalPurchaseDollars"] != 0,
        df["TotalSalesDollars"] / df["TotalPurchaseDollars"],
        0.0,
    )

    # Freight analysis
    df["FreightToSalesRatio"] = np.where(
        df["TotalSalesDollars"] != 0,
        (df["FreightCost"] / df["TotalSalesDollars"]) * 100,
        0.0,
    )

    # Pricing analysis
    df["PriceMarkup"] = np.where(
        df["PurchasePrice"] != 0,
        ((df["ActualPrice"] - df["PurchasePrice"]) / df["PurchasePrice"]) * 100,
        0.0,
    )

    logger.info(
        f"Cleaning complete — {len(df)} rows, "
        f"{len(df.columns)} columns (including derived metrics)"
    )
    return df


# ────────────────────────────────────────────────
# Database Operations
# ────────────────────────────────────────────────

@timer
@retry(max_attempts=3, delay=2.0, exceptions=(Exception,))
def fetch_vendor_summary(engine) -> pd.DataFrame:
    """Execute the vendor summary CTE query against MySQL."""
    logger.info("Executing vendor summary SQL query...")
    with engine.connect() as conn:
        df = pd.read_sql(VENDOR_SUMMARY_SQL, conn)
    logger.info(f"Query returned {len(df):,} rows")
    return df


@timer
@retry(max_attempts=3, delay=2.0, exceptions=(Exception,))
def save_to_database(df: pd.DataFrame, engine, table_name: str = "vendor_sales_summary"):
    """Write the cleaned summary DataFrame to MySQL."""
    logger.info(f"Saving {len(df):,} rows to table '{table_name}'...")
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    logger.info(f"Table '{table_name}' saved successfully")


def save_to_csv(df: pd.DataFrame, output_dir: Path, filename: str = None):
    """Export summary to CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vendor_sales_summary_{timestamp}.csv"
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    logger.info(f"Summary exported to {filepath}")
    return filepath


# ────────────────────────────────────────────────
# CLI Entry Point
# ────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate vendor performance summary from transactional data."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate summary without saving to database.",
    )
    parser.add_argument(
        "--output",
        choices=["db", "csv", "both"],
        default="db",
        help="Output destination: 'db' (MySQL), 'csv', or 'both'. Default: db.",
    )
    parser.add_argument(
        "--table",
        default="vendor_sales_summary",
        help="Target MySQL table name. Default: vendor_sales_summary.",
    )
    return parser.parse_args()


def main():
    """Main execution pipeline."""
    args = parse_args()
    cfg = get_config()

    logger.info("=" * 60)
    logger.info("VENDOR SUMMARY GENERATOR — START")
    logger.info("=" * 60)

    # Connect to database
    from db_connect import get_engine
    engine = get_engine()

    # Step 1: Extract
    raw_df = fetch_vendor_summary(engine)

    # Step 2: Transform
    clean_df = clean_vendor_data(raw_df)

    # Step 3: Load
    if args.dry_run:
        logger.info("[DRY RUN] Skipping database save")
        print(clean_df.head(10).to_string())
    else:
        if args.output in ("db", "both"):
            save_to_database(clean_df, engine, args.table)

        if args.output in ("csv", "both"):
            save_to_csv(clean_df, cfg.paths.output_dir)

    logger.info("=" * 60)
    logger.info("VENDOR SUMMARY GENERATOR — COMPLETE")
    logger.info(f"  Rows processed: {len(clean_df):,}")
    logger.info(f"  Columns: {list(clean_df.columns)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
