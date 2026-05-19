"""
ML Feature Engineering Module
==============================
Transforms the vendor sales summary into a machine-learning-ready feature
matrix. This module is the bridge between traditional analytics and
downstream ML/Deep Learning models.

Feature Categories:
    1. Profitability   — margins, markup tiers, profit per unit
    2. Volume          — quantity buckets, sales velocity
    3. Pricing         — price gap analysis, elasticity proxies
    4. Inventory       — turnover efficiency, sell-through ratios
    5. Vendor-Level    — aggregated vendor scores, concentration risk
    6. Statistical     — log transforms, z-scores for normalization

Design Principles:
    - Pure pandas transforms (no DB dependency for feature computation)
    - Each feature method is independent and idempotent
    - Clear docstrings explaining business meaning of each feature
    - Ready for scikit-learn, XGBoost, or PyTorch consumption

Usage:
    from feature_engineering import FeatureEngineer

    fe = FeatureEngineer()
    features_df = fe.build_features(vendor_summary_df)
    fe.save_feature_matrix(features_df, engine, "ml_feature_store")
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from utils.logging_config import setup_logger
from utils.decorators import timer, validate_dataframe

logger = setup_logger(__name__)


class FeatureEngineer:
    """
    Builds ML-ready features from the vendor sales summary table.

    The output DataFrame retains all original columns and appends
    engineered features with a ``feat_`` prefix for easy identification.
    """

    # Columns expected in the input DataFrame
    REQUIRED_COLUMNS = [
        "VendorNumber", "VendorName", "Brand",
        "PurchasePrice", "ActualPrice", "Volume",
        "TotalPurchaseQuantity", "TotalPurchaseDollars",
        "TotalSalesQuantity", "TotalSalesDollars",
        "GrossProfit", "ProfitMargin",
    ]

    @timer
    @validate_dataframe(required_columns=REQUIRED_COLUMNS, min_rows=1)
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the full feature engineering pipeline.

        Args:
            df: Cleaned vendor sales summary DataFrame.

        Returns:
            DataFrame with all original + engineered feature columns.
        """
        logger.info(f"Building features for {len(df):,} rows...")

        df = df.copy()

        # Layer 1: Profitability features
        df = self._profitability_features(df)

        # Layer 2: Volume & quantity features
        df = self._volume_features(df)

        # Layer 3: Pricing features
        df = self._pricing_features(df)

        # Layer 4: Inventory efficiency features
        df = self._inventory_features(df)

        # Layer 5: Vendor-level aggregate features
        df = self._vendor_aggregate_features(df)

        # Layer 6: Statistical normalization features
        df = self._statistical_features(df)

        # Layer 7: Categorical encodings
        df = self._categorical_features(df)

        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        logger.info(f"Feature engineering complete — {len(feat_cols)} new features created")

        return df

    # ────────────────────────────────────────────────
    # Layer 1: Profitability
    # ────────────────────────────────────────────────

    def _profitability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Profitability-derived features.

        - feat_profit_per_unit: Gross profit divided by quantity sold.
          Business meaning: How much profit each unit sale generates.

        - feat_profit_tier: Categorical bucketing of profit margin into
          'loss', 'low', 'medium', 'high' tiers for classification tasks.

        - feat_net_profit_after_freight: Gross profit minus freight costs.
          Business meaning: True bottom-line profitability per vendor-brand.

        - feat_is_profitable: Binary flag (1/0) indicating positive gross profit.
          Useful as a classification target variable.
        """
        df["feat_profit_per_unit"] = np.where(
            df["TotalSalesQuantity"] > 0,
            df["GrossProfit"] / df["TotalSalesQuantity"],
            0.0,
        )

        df["feat_profit_tier"] = pd.cut(
            df["ProfitMargin"],
            bins=[-np.inf, 0, 15, 35, np.inf],
            labels=["loss", "low", "medium", "high"],
        )

        freight = df.get("FreightCost", pd.Series(0, index=df.index))
        df["feat_net_profit_after_freight"] = df["GrossProfit"] - freight

        df["feat_is_profitable"] = (df["GrossProfit"] > 0).astype(int)

        return df

    # ────────────────────────────────────────────────
    # Layer 2: Volume & Quantity
    # ────────────────────────────────────────────────

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume and quantity-based features.

        - feat_volume_category: Standard bottle-size grouping
          (Mini ≤200ml, Half ≤500ml, Standard ≤750ml, Large ≤1500ml, Magnum >1500ml).

        - feat_quantity_ratio: Ratio of sales quantity to purchase quantity.
          Values > 1 indicate demand exceeding supply.

        - feat_demand_velocity: Sales dollars per unit volume (ml).
          Business meaning: Revenue density — how much revenue each ml generates.
        """
        def categorize_volume(vol):
            if vol <= 200:
                return "mini"
            elif vol <= 500:
                return "half"
            elif vol <= 750:
                return "standard"
            elif vol <= 1500:
                return "large"
            else:
                return "magnum"

        df["feat_volume_category"] = df["Volume"].apply(categorize_volume)

        df["feat_quantity_ratio"] = np.where(
            df["TotalPurchaseQuantity"] > 0,
            df["TotalSalesQuantity"] / df["TotalPurchaseQuantity"],
            0.0,
        )

        df["feat_demand_velocity"] = np.where(
            df["Volume"] > 0,
            df["TotalSalesDollars"] / df["Volume"],
            0.0,
        )

        return df

    # ────────────────────────────────────────────────
    # Layer 3: Pricing
    # ────────────────────────────────────────────────

    def _pricing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pricing analysis features.

        - feat_price_gap: Absolute difference between retail and purchase price.
          Business meaning: Raw dollar margin per unit before volume effects.

        - feat_price_gap_pct: Price gap as a percentage of purchase price.
          Business meaning: Relative markup — normalizes across price ranges.

        - feat_price_competitiveness: Ratio of purchase price to retail price.
          Values close to 1 = thin margins; close to 0 = high markup.

        - feat_excise_burden: Excise tax as percentage of total sales.
          Business meaning: Tax load impact on effective pricing.
        """
        df["feat_price_gap"] = df["ActualPrice"] - df["PurchasePrice"]

        df["feat_price_gap_pct"] = np.where(
            df["PurchasePrice"] > 0,
            (df["feat_price_gap"] / df["PurchasePrice"]) * 100,
            0.0,
        )

        df["feat_price_competitiveness"] = np.where(
            df["ActualPrice"] > 0,
            df["PurchasePrice"] / df["ActualPrice"],
            0.0,
        )

        excise = df.get("TotalExciseTax", pd.Series(0, index=df.index))
        df["feat_excise_burden"] = np.where(
            df["TotalSalesDollars"] > 0,
            (excise / df["TotalSalesDollars"]) * 100,
            0.0,
        )

        return df

    # ────────────────────────────────────────────────
    # Layer 4: Inventory Efficiency
    # ────────────────────────────────────────────────

    def _inventory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inventory management features.

        - feat_sell_through_rate: Percentage of purchased quantity that was sold.
          Values < 100% indicate leftover inventory; > 100% means understocking.

        - feat_overstock_flag: Binary flag for items where sales < 50% of purchases.
          Business meaning: Identifies slow-moving or dead stock.

        - feat_revenue_per_purchase_dollar: Revenue generated per dollar invested.
          Business meaning: Return on inventory investment (ROII).
        """
        df["feat_sell_through_rate"] = np.where(
            df["TotalPurchaseQuantity"] > 0,
            (df["TotalSalesQuantity"] / df["TotalPurchaseQuantity"]) * 100,
            0.0,
        )

        df["feat_overstock_flag"] = (df["feat_sell_through_rate"] < 50).astype(int)

        df["feat_revenue_per_purchase_dollar"] = np.where(
            df["TotalPurchaseDollars"] > 0,
            df["TotalSalesDollars"] / df["TotalPurchaseDollars"],
            0.0,
        )

        return df

    # ────────────────────────────────────────────────
    # Layer 5: Vendor-Level Aggregates
    # ────────────────────────────────────────────────

    def _vendor_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vendor-level aggregated features (group-by vendor, then merge back).

        - feat_vendor_brand_count: Number of distinct brands per vendor.
          Business meaning: Vendor portfolio diversity.

        - feat_vendor_total_revenue: Total revenue across all brands for the vendor.
          Business meaning: Vendor's overall scale of business.

        - feat_vendor_avg_margin: Average profit margin across vendor's brands.
          Business meaning: Overall vendor profitability profile.

        - feat_vendor_revenue_share: This brand's share of the vendor's total revenue.
          Business meaning: Concentration risk — dependency on single products.
        """
        vendor_stats = df.groupby("VendorNumber").agg(
            vendor_brand_count=("Brand", "nunique"),
            vendor_total_revenue=("TotalSalesDollars", "sum"),
            vendor_avg_margin=("ProfitMargin", "mean"),
        ).reset_index()

        df = df.merge(
            vendor_stats,
            on="VendorNumber",
            how="left",
            suffixes=("", "_vendor"),
        )

        df.rename(columns={
            "vendor_brand_count": "feat_vendor_brand_count",
            "vendor_total_revenue": "feat_vendor_total_revenue",
            "vendor_avg_margin": "feat_vendor_avg_margin",
        }, inplace=True)

        df["feat_vendor_revenue_share"] = np.where(
            df["feat_vendor_total_revenue"] > 0,
            (df["TotalSalesDollars"] / df["feat_vendor_total_revenue"]) * 100,
            0.0,
        )

        return df

    # ────────────────────────────────────────────────
    # Layer 6: Statistical / Normalization
    # ────────────────────────────────────────────────

    def _statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Statistical transformations for ML model consumption.

        - feat_log_sales: Log-transformed total sales dollars.
          Reduces right-skew common in financial data.

        - feat_log_purchases: Log-transformed total purchase dollars.

        - feat_log_profit: Log-transformed absolute gross profit (sign preserved).

        - feat_zscore_profit_margin: Z-scored profit margin for standardization.
        """
        df["feat_log_sales"] = np.log1p(df["TotalSalesDollars"].clip(lower=0))
        df["feat_log_purchases"] = np.log1p(df["TotalPurchaseDollars"].clip(lower=0))

        # Sign-preserving log: log(|x|+1) * sign(x)
        df["feat_log_profit"] = (
            np.log1p(df["GrossProfit"].abs()) * np.sign(df["GrossProfit"])
        )

        mean = df["ProfitMargin"].mean()
        std = df["ProfitMargin"].std()
        df["feat_zscore_profit_margin"] = np.where(
            std > 0, (df["ProfitMargin"] - mean) / std, 0.0
        )

        return df

    # ────────────────────────────────────────────────
    # Layer 7: Categorical Encodings
    # ────────────────────────────────────────────────

    def _categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables for model consumption.

        - feat_vendor_encoded: Frequency-based encoding of VendorNumber.
          Higher values = more product lines from that vendor.
        """
        vendor_freq = df["VendorNumber"].value_counts(normalize=True)
        df["feat_vendor_frequency"] = df["VendorNumber"].map(vendor_freq)

        return df

    # ────────────────────────────────────────────────
    # Export
    # ────────────────────────────────────────────────

    @staticmethod
    def save_feature_matrix(df: pd.DataFrame, engine, table_name: str = "ml_feature_store"):
        """Save the feature matrix to a MySQL table."""
        logger.info(f"Saving feature matrix ({len(df):,} rows) to '{table_name}'...")
        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
        logger.info(f"Feature matrix saved to '{table_name}'")

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> list:
        """Return only the engineered feature column names."""
        return [c for c in df.columns if c.startswith("feat_")]

    @staticmethod
    def get_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
        """Return only numeric engineered features (ready for model input)."""
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        return df[feat_cols].select_dtypes(include=[np.number])


# ────────────────────────────────────────────────
# CLI Entry Point
# ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ML features from vendor summary.")
    parser.add_argument("--source", default="db", choices=["db", "csv"])
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--save-db", action="store_true", help="Save to ML feature store table.")
    parser.add_argument("--save-csv", default=None, help="Export features to CSV.")
    args = parser.parse_args()

    # Load data
    if args.source == "csv" and args.csv_path:
        df = pd.read_csv(args.csv_path)
    else:
        from db_connect import get_engine
        engine = get_engine()
        df = pd.read_sql("SELECT * FROM vendor_sales_summary", engine)

    # Build features
    fe = FeatureEngineer()
    features_df = fe.build_features(df)

    # Summary
    feat_cols = fe.get_feature_columns(features_df)
    print(f"\nTotal features: {len(feat_cols)}")
    print(f"Feature columns: {feat_cols}")
    print(f"\nSample (first 5 rows, features only):")
    print(features_df[feat_cols].head().to_string())

    # Save
    if args.save_db:
        from db_connect import get_engine
        engine = get_engine()
        fe.save_feature_matrix(features_df, engine)

    if args.save_csv:
        features_df.to_csv(args.save_csv, index=False)
        print(f"\nFeatures exported to {args.save_csv}")
