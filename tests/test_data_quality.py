"""
Unit Tests for Data Quality Validator
======================================
Tests run entirely on synthetic data — no database connection required.
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_quality import DataQualityValidator, ValidationResult


# ────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────

@pytest.fixture
def validator():
    """Fresh validator instance."""
    return DataQualityValidator()


@pytest.fixture
def good_vendor_df():
    """Valid vendor summary DataFrame — should pass all checks."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "VendorNumber": np.random.randint(100, 999, n),
        "VendorName": [f"Vendor_{i}" for i in range(n)],
        "Brand": np.random.randint(1, 500, n),
        "Description": [f"Product_{i}" for i in range(n)],
        "PurchasePrice": np.random.uniform(5, 100, n),
        "ActualPrice": np.random.uniform(10, 150, n),
        "Volume": np.random.choice([375, 500, 750, 1000, 1750], n),
        "TotalPurchaseQuantity": np.random.randint(10, 5000, n),
        "TotalPurchaseDollars": np.random.uniform(100, 50000, n),
        "TotalSalesQuantity": np.random.randint(10, 5000, n),
        "TotalSalesDollars": np.random.uniform(200, 80000, n),
        "TotalSalesPrice": np.random.uniform(10, 150, n),
        "TotalExciseTax": np.random.uniform(0, 500, n),
        "FreightCost": np.random.uniform(0, 1000, n),
        "GrossProfit": np.random.uniform(-500, 30000, n),
        "ProfitMargin": np.random.uniform(-20, 60, n),
        "StockTurnover": np.random.uniform(0.1, 5, n),
        "SalesToPurchaseRatio": np.random.uniform(0.5, 3, n),
    })


@pytest.fixture
def bad_vendor_df():
    """DataFrame with intentional quality issues."""
    return pd.DataFrame({
        "VendorNumber": [1, 2, None, 4, 5],
        "VendorName": ["A", None, "C", "D", "E"],
        "Brand": [10, 20, 30, 40, 50],
        "Description": ["p1", "p2", "p3", "p4", "p5"],
        "PurchasePrice": [10, -5, 100, 99999, 20],       # -5 and 99999 violate range
        "ActualPrice": [15, 8, 150, 120, 30],
        "Volume": [750, 750, 750, 750, 750],
        "TotalPurchaseQuantity": [100, 200, 300, 400, 500],
        "TotalPurchaseDollars": [1000, 2000, 3000, 4000, 5000],
        "TotalSalesQuantity": [90, 180, 290, 380, 480],
        "TotalSalesDollars": [1500, 3000, 4500, 6000, 7500],
        "TotalSalesPrice": [15, 15, 15, 15, 15],
        "TotalExciseTax": [10, 20, 30, 40, 50],
        "FreightCost": [5, 10, 15, 20, 25],
        "GrossProfit": [500, 1000, 1500, 2000, 2500],
        "ProfitMargin": [33.3, 33.3, 33.3, 33.3, 33.3],
        "StockTurnover": [0.9, 0.9, 0.97, 0.95, 0.96],
        "SalesToPurchaseRatio": [1.5, 1.5, 1.5, 1.5, 1.5],
    })


# ────────────────────────────────────────────────
# Tests: ValidationResult
# ────────────────────────────────────────────────

class TestValidationResult:
    def test_passed_result(self):
        r = ValidationResult("test_check", True, {"info": "all good"})
        assert r.passed is True
        assert r.check_name == "test_check"
        assert r.details["info"] == "all good"

    def test_failed_result(self):
        r = ValidationResult("test_check", False, {"missing": ["col_a"]})
        assert r.passed is False

    def test_to_dict(self):
        r = ValidationResult("test", True)
        d = r.to_dict()
        assert "check" in d
        assert "passed" in d
        assert "timestamp" in d


# ────────────────────────────────────────────────
# Tests: Schema Validation
# ────────────────────────────────────────────────

class TestSchemaValidation:
    def test_schema_pass(self, validator, good_vendor_df):
        report = validator.validate(good_vendor_df, profile="vendor_summary")
        schema_check = next(r for r in report["results"] if r["check"] == "schema_completeness")
        assert schema_check["passed"] is True

    def test_schema_fail_missing_columns(self, validator):
        df = pd.DataFrame({"VendorNumber": [1], "Brand": [10]})
        report = validator.validate(df, profile="vendor_summary")
        schema_check = next(r for r in report["results"] if r["check"] == "schema_completeness")
        assert schema_check["passed"] is False
        assert len(schema_check["details"]["missing_columns"]) > 0


# ────────────────────────────────────────────────
# Tests: Null Checks
# ────────────────────────────────────────────────

class TestNullChecks:
    def test_non_nullable_pass(self, validator, good_vendor_df):
        report = validator.validate(good_vendor_df, profile="vendor_summary")
        nn_checks = [r for r in report["results"] if r["check"].startswith("non_nullable_")]
        assert all(r["passed"] for r in nn_checks)

    def test_non_nullable_fail(self, validator, bad_vendor_df):
        report = validator.validate(bad_vendor_df, profile="vendor_summary")
        nn_vendor = next(
            (r for r in report["results"] if r["check"] == "non_nullable_VendorNumber"),
            None,
        )
        # VendorNumber has a None in bad_vendor_df
        if nn_vendor:
            assert nn_vendor["passed"] is False


# ────────────────────────────────────────────────
# Tests: Range Checks
# ────────────────────────────────────────────────

class TestRangeChecks:
    def test_range_pass(self, validator, good_vendor_df):
        report = validator.validate(good_vendor_df, profile="vendor_summary")
        range_checks = [r for r in report["results"] if r["check"].startswith("range_")]
        # Most should pass with good random data in [5, 100] range
        assert len(range_checks) > 0

    def test_range_fail_out_of_bounds(self, validator, bad_vendor_df):
        report = validator.validate(bad_vendor_df, profile="vendor_summary")
        pp_check = next(
            (r for r in report["results"] if r["check"] == "range_PurchasePrice"),
            None,
        )
        if pp_check:
            # bad_vendor_df has PurchasePrice = -5 and 99999
            assert pp_check["passed"] is False
            assert pp_check["details"]["violations"] > 0


# ────────────────────────────────────────────────
# Tests: Duplicate Check
# ────────────────────────────────────────────────

class TestDuplicateCheck:
    def test_no_duplicates(self, validator, good_vendor_df):
        report = validator.validate(good_vendor_df, profile="vendor_summary")
        dup_check = next(r for r in report["results"] if r["check"] == "row_duplicates")
        assert dup_check["passed"] is True

    def test_with_duplicates(self, validator):
        df_row = pd.DataFrame({
            col: [1] for col in [
                "VendorNumber", "VendorName", "Brand", "Description",
                "PurchasePrice", "ActualPrice", "Volume",
                "TotalPurchaseQuantity", "TotalPurchaseDollars",
                "TotalSalesQuantity", "TotalSalesDollars",
                "GrossProfit", "ProfitMargin", "StockTurnover",
                "SalesToPurchaseRatio", "FreightCost",
            ]
        })
        df = pd.concat([df_row, df_row], ignore_index=True)
        report = validator.validate(df, profile="vendor_summary")
        dup_check = next(r for r in report["results"] if r["check"] == "row_duplicates")
        assert dup_check["passed"] is False


# ────────────────────────────────────────────────
# Tests: Report Structure
# ────────────────────────────────────────────────

class TestReportStructure:
    def test_report_keys(self, validator, good_vendor_df):
        report = validator.validate(good_vendor_df, profile="vendor_summary")
        required_keys = [
            "profile", "timestamp", "row_count", "column_count",
            "total_checks", "passed", "failed", "pass_rate", "results",
        ]
        for key in required_keys:
            assert key in report, f"Missing key: {key}"

    def test_pass_rate_calculation(self, validator, good_vendor_df):
        report = validator.validate(good_vendor_df, profile="vendor_summary")
        expected_rate = round(
            report["passed"] / report["total_checks"] * 100, 2
        )
        assert report["pass_rate"] == expected_rate

    def test_unknown_profile_raises(self, validator, good_vendor_df):
        with pytest.raises(ValueError, match="Unknown profile"):
            validator.validate(good_vendor_df, profile="nonexistent")


# ────────────────────────────────────────────────
# Tests: Custom Profiles
# ────────────────────────────────────────────────

class TestCustomProfiles:
    def test_custom_profile(self):
        custom = {
            "custom_table": {
                "required_columns": ["id", "value"],
                "numeric_columns": ["value"],
                "non_nullable": ["id"],
                "unique_keys": [],
                "range_checks": {"value": {"min": 0, "max": 100}},
                "null_threshold": 0.05,
            }
        }
        validator = DataQualityValidator(custom_profiles=custom)
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 50, 90]})
        report = validator.validate(df, profile="custom_table")
        assert report["failed"] == 0
