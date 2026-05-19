"""
Unit Tests for Feature Engineering Module
==========================================
Validates feature computations with known inputs and expected outputs.
No database connection required — all tests use synthetic DataFrames.
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_engineering import FeatureEngineer


# ────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────

@pytest.fixture
def fe():
    """Feature engineer instance."""
    return FeatureEngineer()


@pytest.fixture
def sample_df():
    """Minimal valid DataFrame for feature engineering."""
    return pd.DataFrame({
        "VendorNumber": [100, 100, 200],
        "VendorName": ["Vendor_A", "Vendor_A", "Vendor_B"],
        "Brand": [1, 2, 3],
        "Description": ["Prod_1", "Prod_2", "Prod_3"],
        "PurchasePrice": [10.0, 20.0, 30.0],
        "ActualPrice": [15.0, 35.0, 40.0],
        "Volume": [750, 375, 1750],
        "TotalPurchaseQuantity": [100, 200, 300],
        "TotalPurchaseDollars": [1000.0, 4000.0, 9000.0],
        "TotalSalesQuantity": [90, 250, 280],
        "TotalSalesDollars": [1350.0, 8750.0, 11200.0],
        "TotalSalesPrice": [15.0, 35.0, 40.0],
        "TotalExciseTax": [10.0, 50.0, 100.0],
        "FreightCost": [5.0, 20.0, 50.0],
        "GrossProfit": [350.0, 4750.0, 2200.0],
        "ProfitMargin": [25.93, 54.29, 19.64],
        "StockTurnover": [0.9, 1.25, 0.93],
        "SalesToPurchaseRatio": [1.35, 2.19, 1.24],
    })


@pytest.fixture
def zero_df():
    """DataFrame with zeros to test division-by-zero guards."""
    return pd.DataFrame({
        "VendorNumber": [1],
        "VendorName": ["Zero_Vendor"],
        "Brand": [999],
        "Description": ["Zero_Product"],
        "PurchasePrice": [0.0],
        "ActualPrice": [0.0],
        "Volume": [0],
        "TotalPurchaseQuantity": [0],
        "TotalPurchaseDollars": [0.0],
        "TotalSalesQuantity": [0],
        "TotalSalesDollars": [0.0],
        "TotalSalesPrice": [0.0],
        "TotalExciseTax": [0.0],
        "FreightCost": [0.0],
        "GrossProfit": [0.0],
        "ProfitMargin": [0.0],
        "StockTurnover": [0.0],
        "SalesToPurchaseRatio": [0.0],
    })


# ────────────────────────────────────────────────
# Tests: Feature Pipeline
# ────────────────────────────────────────────────

class TestBuildFeatures:
    def test_returns_dataframe(self, fe, sample_df):
        result = fe.build_features(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_creates_feat_columns(self, fe, sample_df):
        result = fe.build_features(sample_df)
        feat_cols = [c for c in result.columns if c.startswith("feat_")]
        assert len(feat_cols) >= 15  # We expect at least 15 engineered features

    def test_preserves_original_columns(self, fe, sample_df):
        original_cols = set(sample_df.columns)
        result = fe.build_features(sample_df)
        for col in original_cols:
            assert col in result.columns, f"Original column '{col}' was dropped"

    def test_row_count_preserved(self, fe, sample_df):
        result = fe.build_features(sample_df)
        assert len(result) == len(sample_df)

    def test_no_nans_in_numeric_features(self, fe, sample_df):
        result = fe.build_features(sample_df)
        numeric_feats = fe.get_numeric_features(result)
        nan_counts = numeric_feats.isna().sum()
        cols_with_nans = nan_counts[nan_counts > 0]
        assert len(cols_with_nans) == 0, f"NaN found in: {cols_with_nans.to_dict()}"


# ────────────────────────────────────────────────
# Tests: Profitability Features
# ────────────────────────────────────────────────

class TestProfitabilityFeatures:
    def test_profit_per_unit(self, fe, sample_df):
        result = fe.build_features(sample_df)
        # Row 0: GrossProfit=350, TotalSalesQuantity=90 → 350/90 ≈ 3.889
        expected = 350.0 / 90.0
        assert abs(result.loc[0, "feat_profit_per_unit"] - expected) < 0.01

    def test_is_profitable_flag(self, fe, sample_df):
        result = fe.build_features(sample_df)
        # All rows have positive GrossProfit
        assert result["feat_is_profitable"].sum() == len(sample_df)

    def test_profit_tier_categories(self, fe, sample_df):
        result = fe.build_features(sample_df)
        valid_tiers = {"loss", "low", "medium", "high"}
        actual_tiers = set(result["feat_profit_tier"].dropna().astype(str))
        assert actual_tiers.issubset(valid_tiers)


# ────────────────────────────────────────────────
# Tests: Volume Features
# ────────────────────────────────────────────────

class TestVolumeFeatures:
    def test_volume_category(self, fe, sample_df):
        result = fe.build_features(sample_df)
        # 750 → standard, 375 → half, 1750 → magnum
        assert result.loc[0, "feat_volume_category"] == "standard"
        assert result.loc[1, "feat_volume_category"] == "half"
        assert result.loc[2, "feat_volume_category"] == "magnum"

    def test_quantity_ratio(self, fe, sample_df):
        result = fe.build_features(sample_df)
        # Row 0: 90/100 = 0.9
        assert abs(result.loc[0, "feat_quantity_ratio"] - 0.9) < 0.01


# ────────────────────────────────────────────────
# Tests: Pricing Features
# ────────────────────────────────────────────────

class TestPricingFeatures:
    def test_price_gap(self, fe, sample_df):
        result = fe.build_features(sample_df)
        # Row 0: ActualPrice(15) - PurchasePrice(10) = 5
        assert abs(result.loc[0, "feat_price_gap"] - 5.0) < 0.01

    def test_price_gap_pct(self, fe, sample_df):
        result = fe.build_features(sample_df)
        # Row 0: (5/10)*100 = 50%
        assert abs(result.loc[0, "feat_price_gap_pct"] - 50.0) < 0.01


# ────────────────────────────────────────────────
# Tests: Inventory Features
# ────────────────────────────────────────────────

class TestInventoryFeatures:
    def test_sell_through_rate(self, fe, sample_df):
        result = fe.build_features(sample_df)
        # Row 0: (90/100)*100 = 90%
        assert abs(result.loc[0, "feat_sell_through_rate"] - 90.0) < 0.01

    def test_overstock_flag(self, fe, sample_df):
        result = fe.build_features(sample_df)
        # No row has sell-through < 50%, so all should be 0
        assert result["feat_overstock_flag"].sum() == 0


# ────────────────────────────────────────────────
# Tests: Vendor Aggregate Features
# ────────────────────────────────────────────────

class TestVendorAggregateFeatures:
    def test_vendor_brand_count(self, fe, sample_df):
        result = fe.build_features(sample_df)
        # Vendor 100 has brands 1, 2 → count = 2
        v100_rows = result[result["VendorNumber"] == 100]
        assert v100_rows["feat_vendor_brand_count"].iloc[0] == 2

    def test_vendor_revenue_share(self, fe, sample_df):
        result = fe.build_features(sample_df)
        # Check shares sum to ~100% within each vendor
        for vendor in result["VendorNumber"].unique():
            vendor_rows = result[result["VendorNumber"] == vendor]
            total_share = vendor_rows["feat_vendor_revenue_share"].sum()
            assert abs(total_share - 100.0) < 0.1


# ────────────────────────────────────────────────
# Tests: Zero-Division Safety
# ────────────────────────────────────────────────

class TestZeroDivisionSafety:
    def test_zero_values_no_errors(self, fe, zero_df):
        """Feature engineering should handle all-zero data without errors."""
        result = fe.build_features(zero_df)
        assert len(result) == 1

    def test_zero_values_no_inf(self, fe, zero_df):
        """No infinite values should appear from division by zero."""
        result = fe.build_features(zero_df)
        numeric_feats = fe.get_numeric_features(result)
        inf_count = np.isinf(numeric_feats.values).sum()
        assert inf_count == 0, "Infinite values found in features"


# ────────────────────────────────────────────────
# Tests: Utility Methods
# ────────────────────────────────────────────────

class TestUtilityMethods:
    def test_get_feature_columns(self, fe, sample_df):
        result = fe.build_features(sample_df)
        feat_cols = fe.get_feature_columns(result)
        assert all(c.startswith("feat_") for c in feat_cols)

    def test_get_numeric_features(self, fe, sample_df):
        result = fe.build_features(sample_df)
        numeric = fe.get_numeric_features(result)
        assert all(pd.api.types.is_numeric_dtype(numeric[c]) for c in numeric.columns)


# ────────────────────────────────────────────────
# Tests: Input Validation
# ────────────────────────────────────────────────

class TestInputValidation:
    def test_missing_required_column_raises(self, fe):
        df = pd.DataFrame({"VendorNumber": [1], "Brand": [10]})
        with pytest.raises(ValueError, match="missing columns"):
            fe.build_features(df)

    def test_empty_dataframe_raises(self, fe):
        df = pd.DataFrame(columns=FeatureEngineer.REQUIRED_COLUMNS)
        with pytest.raises(ValueError, match="0 rows"):
            fe.build_features(df)
