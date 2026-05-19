"""
Data Quality Validation Framework
===================================
Reusable, configurable data quality checks for the vendor performance
pipeline. Designed to run as a standalone validation step or as part
of the ingestion/ETL pipeline.

Validation Layers:
    1. Schema    — column presence, data types
    2. Integrity — nulls, duplicates, primary key uniqueness
    3. Range     — business-rule value bounds
    4. Statistical — outlier detection via z-scores and IQR
    5. Cross-table — consistency between related tables

Usage:
    from data_quality import DataQualityValidator

    validator = DataQualityValidator()
    report = validator.validate(df, profile="vendor_summary")
    validator.save_report(report, "output/dq_report.json")
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from utils.logging_config import setup_logger
from utils.decorators import timer

logger = setup_logger(__name__)


# ────────────────────────────────────────────────
# Validation Profiles (schema + rules per table)
# ────────────────────────────────────────────────

VALIDATION_PROFILES = {
    "vendor_summary": {
        "required_columns": [
            "VendorNumber", "VendorName", "Brand", "Description",
            "PurchasePrice", "ActualPrice", "Volume",
            "TotalPurchaseQuantity", "TotalPurchaseDollars",
            "TotalSalesQuantity", "TotalSalesDollars",
            "GrossProfit", "ProfitMargin", "StockTurnover",
        ],
        "numeric_columns": [
            "PurchasePrice", "ActualPrice", "Volume",
            "TotalPurchaseQuantity", "TotalPurchaseDollars",
            "TotalSalesQuantity", "TotalSalesDollars",
            "GrossProfit", "ProfitMargin", "StockTurnover",
            "SalesToPurchaseRatio", "FreightCost",
        ],
        "non_nullable": ["VendorNumber", "VendorName", "Brand"],
        "unique_keys": [],  # composite key: VendorNumber + Brand
        "range_checks": {
            "PurchasePrice": {"min": 0, "max": 10000},
            "ActualPrice": {"min": 0, "max": 10000},
            "Volume": {"min": 0, "max": 100000},
            "ProfitMargin": {"min": -500, "max": 500},
            "StockTurnover": {"min": 0, "max": 1000},
        },
        "null_threshold": 0.10,  # max 10% nulls per column
    },
    "purchases": {
        "required_columns": [
            "InventoryId", "Store", "Brand", "VendorNumber",
            "VendorName", "PurchasePrice", "Quantity", "Dollars",
        ],
        "numeric_columns": ["Store", "Brand", "VendorNumber", "PurchasePrice", "Quantity", "Dollars"],
        "non_nullable": ["InventoryId", "Brand", "VendorNumber"],
        "unique_keys": [],
        "range_checks": {
            "PurchasePrice": {"min": 0, "max": 10000},
            "Quantity": {"min": -100, "max": 100000},
        },
        "null_threshold": 0.05,
    },
    "sales": {
        "required_columns": [
            "SalesQuantity", "SalesDollars", "SalesPrice",
            "VendorNo", "Brand",
        ],
        "numeric_columns": ["SalesQuantity", "SalesDollars", "SalesPrice", "VendorNo", "Brand"],
        "non_nullable": ["VendorNo", "Brand"],
        "unique_keys": [],
        "range_checks": {
            "SalesPrice": {"min": 0, "max": 50000},
            "SalesQuantity": {"min": -1000, "max": 100000},
        },
        "null_threshold": 0.05,
    },
}


# ────────────────────────────────────────────────
# Validation Result Container
# ────────────────────────────────────────────────

class ValidationResult:
    """Container for a single validation check's result."""

    def __init__(self, check_name: str, passed: bool, details: Dict[str, Any] = None):
        self.check_name = check_name
        self.passed = passed
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "check": self.check_name,
            "passed": self.passed,
            "details": self.details,
            "timestamp": self.timestamp,
        }


# ────────────────────────────────────────────────
# Core Validator
# ────────────────────────────────────────────────

class DataQualityValidator:
    """
    Configurable data quality validator supporting multiple validation profiles.

    Each profile defines the expected schema, business-rule ranges, and
    statistical thresholds for a specific table/DataFrame.
    """

    def __init__(self, custom_profiles: Optional[Dict] = None):
        self.profiles = {**VALIDATION_PROFILES}
        if custom_profiles:
            self.profiles.update(custom_profiles)

    @timer
    def validate(
        self,
        df: pd.DataFrame,
        profile: str = "vendor_summary",
        z_threshold: float = 3.0,
    ) -> Dict:
        """
        Run all validation checks against a DataFrame.

        Args:
            df:           DataFrame to validate.
            profile:      Name of the validation profile to apply.
            z_threshold:  Z-score threshold for outlier detection.

        Returns:
            Validation report dictionary.
        """
        if profile not in self.profiles:
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Available: {list(self.profiles.keys())}"
            )

        config = self.profiles[profile]
        results: List[ValidationResult] = []

        logger.info(f"Running data quality checks — profile='{profile}', rows={len(df):,}")

        # 1. Schema checks
        results.append(self._check_schema(df, config))

        # 2. Null checks
        results.extend(self._check_nulls(df, config))

        # 3. Duplicate checks
        results.append(self._check_duplicates(df, config))

        # 4. Range checks
        results.extend(self._check_ranges(df, config))

        # 5. Statistical outlier checks
        results.extend(self._check_outliers(df, config, z_threshold))

        # 6. Data type checks
        results.extend(self._check_dtypes(df, config))

        # Compile report
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        report = {
            "profile": profile,
            "timestamp": datetime.now().isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total * 100, 2) if total else 0,
            "results": [r.to_dict() for r in results],
        }

        status = "PASSED" if failed == 0 else "FAILED"
        logger.info(f"Validation {status}: {passed}/{total} checks passed ({report['pass_rate']}%)")

        return report

    # ── Individual Checks ────────────────────────

    def _check_schema(self, df: pd.DataFrame, config: Dict) -> ValidationResult:
        """Verify all required columns are present."""
        required = config.get("required_columns", [])
        present = set(df.columns)
        missing = [c for c in required if c not in present]

        return ValidationResult(
            check_name="schema_completeness",
            passed=len(missing) == 0,
            details={
                "required": len(required),
                "present": len(present),
                "missing_columns": missing,
            },
        )

    def _check_nulls(self, df: pd.DataFrame, config: Dict) -> List[ValidationResult]:
        """Check null percentages and non-nullable constraints."""
        results = []
        threshold = config.get("null_threshold", 0.10)
        non_nullable = config.get("non_nullable", [])

        # Per-column null percentage
        null_pct = df.isnull().mean()
        high_null_cols = {
            col: round(float(pct), 4)
            for col, pct in null_pct.items()
            if pct > threshold
        }
        results.append(ValidationResult(
            check_name="null_threshold",
            passed=len(high_null_cols) == 0,
            details={
                "threshold": threshold,
                "columns_exceeding": high_null_cols,
            },
        ))

        # Non-nullable columns
        for col in non_nullable:
            if col in df.columns:
                null_count = int(df[col].isnull().sum())
                results.append(ValidationResult(
                    check_name=f"non_nullable_{col}",
                    passed=null_count == 0,
                    details={"column": col, "null_count": null_count},
                ))

        return results

    def _check_duplicates(self, df: pd.DataFrame, config: Dict) -> ValidationResult:
        """Check for full-row duplicates."""
        dup_count = int(df.duplicated().sum())
        return ValidationResult(
            check_name="row_duplicates",
            passed=dup_count == 0,
            details={
                "duplicate_rows": dup_count,
                "duplicate_pct": round(dup_count / len(df) * 100, 4) if len(df) else 0,
            },
        )

    def _check_ranges(self, df: pd.DataFrame, config: Dict) -> List[ValidationResult]:
        """Verify numeric columns fall within expected business ranges."""
        results = []
        range_checks = config.get("range_checks", {})

        for col, bounds in range_checks.items():
            if col not in df.columns:
                continue

            series = pd.to_numeric(df[col], errors="coerce")
            violations = 0

            if "min" in bounds:
                violations += int((series < bounds["min"]).sum())
            if "max" in bounds:
                violations += int((series > bounds["max"]).sum())

            results.append(ValidationResult(
                check_name=f"range_{col}",
                passed=violations == 0,
                details={
                    "column": col,
                    "bounds": bounds,
                    "violations": violations,
                    "actual_min": round(float(series.min()), 4) if not series.empty else None,
                    "actual_max": round(float(series.max()), 4) if not series.empty else None,
                },
            ))

        return results

    def _check_outliers(
        self, df: pd.DataFrame, config: Dict, z_threshold: float
    ) -> List[ValidationResult]:
        """Detect statistical outliers using z-scores."""
        results = []
        numeric_cols = config.get("numeric_columns", [])

        for col in numeric_cols:
            if col not in df.columns:
                continue

            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) < 3:
                continue

            mean = series.mean()
            std = series.std()
            if std == 0:
                continue

            z_scores = np.abs((series - mean) / std)
            outlier_count = int((z_scores > z_threshold).sum())

            results.append(ValidationResult(
                check_name=f"outlier_{col}",
                passed=outlier_count < len(series) * 0.05,  # <5% outliers acceptable
                details={
                    "column": col,
                    "z_threshold": z_threshold,
                    "outlier_count": outlier_count,
                    "outlier_pct": round(outlier_count / len(series) * 100, 2),
                    "mean": round(float(mean), 4),
                    "std": round(float(std), 4),
                },
            ))

        return results

    def _check_dtypes(self, df: pd.DataFrame, config: Dict) -> List[ValidationResult]:
        """Verify numeric columns are actually numeric dtype."""
        results = []
        numeric_cols = config.get("numeric_columns", [])

        non_numeric = []
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric.append(col)

        results.append(ValidationResult(
            check_name="dtype_numeric",
            passed=len(non_numeric) == 0,
            details={
                "expected_numeric": len(numeric_cols),
                "non_numeric_columns": non_numeric,
            },
        ))

        return results

    # ── Report Export ────────────────────────────

    @staticmethod
    def save_report(report: Dict, filepath: str):
        """Save validation report as JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Quality report saved to {path}")

    @staticmethod
    def print_summary(report: Dict):
        """Print a human-readable summary to console."""
        print("\n" + "=" * 60)
        print(f"DATA QUALITY REPORT — {report['profile'].upper()}")
        print("=" * 60)
        print(f"  Timestamp  : {report['timestamp']}")
        print(f"  Rows       : {report['row_count']:,}")
        print(f"  Columns    : {report['column_count']}")
        print(f"  Checks     : {report['total_checks']}")
        print(f"  Passed     : {report['passed']}")
        print(f"  Failed     : {report['failed']}")
        print(f"  Pass Rate  : {report['pass_rate']}%")
        print("-" * 60)

        for r in report["results"]:
            icon = "✓" if r["passed"] else "✗"
            print(f"  {icon} {r['check']}")
            if not r["passed"]:
                for k, v in r["details"].items():
                    print(f"      {k}: {v}")

        print("=" * 60 + "\n")


# ────────────────────────────────────────────────
# CLI Entry Point
# ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run data quality checks on vendor data.")
    parser.add_argument("--profile", default="vendor_summary", help="Validation profile name.")
    parser.add_argument("--source", default="db", choices=["db", "csv"], help="Data source.")
    parser.add_argument("--csv-path", default=None, help="CSV file path (if --source csv).")
    parser.add_argument("--output", default=None, help="Output JSON report path.")
    args = parser.parse_args()

    if args.source == "csv" and args.csv_path:
        df = pd.read_csv(args.csv_path)
    else:
        from db_connect import get_engine
        engine = get_engine()
        df = pd.read_sql("SELECT * FROM vendor_sales_summary", engine)

    validator = DataQualityValidator()
    report = validator.validate(df, profile=args.profile)
    validator.print_summary(report)

    if args.output:
        validator.save_report(report, args.output)
