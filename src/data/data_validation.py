"""
============================================================
data_validation.py — Time-Series Data Validation Module
============================================================
Role  : Validate input DataFrame before feeding into pipeline.
Rules :
    1. No missing dates in the time series
    2. No negative values in price / quantity / revenue
    3. No duplicated rows
    4. Revenue consistency  (revenue ≈ price × quantity)

Contract:
    - NEVER modifies the input DataFrame.
    - Returns a structured ValidationReport dataclass.
    - Logs warnings for soft issues, raises DataValidationError
      for hard failures (configurable via `raise_on_error`).
============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s — %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------
class DataValidationError(Exception):
    """Raised when critical validation checks fail."""
    pass


# ---------------------------------------------------------------------------
# Validation Report
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    """
    Structured report returned by validate().

    Attributes
    ----------
    passed : bool
        True only when ALL checks passed (no errors).
    errors : list[str]
        Hard failures — data cannot be trusted.
    warnings : list[str]
        Soft issues — data may still be usable but needs attention.
    details : dict
        Per-check detail dicts for programmatic inspection.
    summary : str
        Human-readable one-liner.
    """
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def summary(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return (
            f"{status} | "
            f"{len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s)"
        )

    def __str__(self) -> str:  # noqa: D105
        lines = [
            "=" * 60,
            f"  VALIDATION REPORT — {self.summary}",
            "=" * 60,
        ]
        if self.errors:
            lines.append("\n🔴 ERRORS (hard failures):")
            for e in self.errors:
                lines.append(f"   • {e}")
        if self.warnings:
            lines.append("\n🟡 WARNINGS (soft issues):")
            for w in self.warnings:
                lines.append(f"   • {w}")
        if not self.errors and not self.warnings:
            lines.append("\n  All checks passed — data looks clean.")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual Checks
# ---------------------------------------------------------------------------

def _check_missing_dates(
    df: pd.DataFrame,
    date_col: str,
    freq: str,
    group_cols: Optional[list[str]],
    report: ValidationReport,
) -> None:
    """
    Check 1 — Missing Dates in Time Series.

    If `group_cols` is provided, the check is done per group
    (e.g., per Brand + SKU combination).
    """
    check_key = "missing_dates"
    report.details[check_key] = {}

    try:
        series = pd.to_datetime(df[date_col])
    except Exception as exc:
        msg = f"Cannot parse '{date_col}' as dates: {exc}"
        report.errors.append(msg)
        logger.error(msg)
        report.passed = False
        return

    if group_cols:
        missing_per_group: dict[str, list] = {}
        for group_vals, grp in df.groupby(group_cols):
            g_dates = pd.to_datetime(grp[date_col]).sort_values()
            full_range = pd.date_range(g_dates.min(), g_dates.max(), freq=freq)
            missing = full_range.difference(g_dates)
            if len(missing) > 0:
                key = str(group_vals)
                missing_per_group[key] = missing.tolist()

        total_missing_groups = len(missing_per_group)
        report.details[check_key] = {
            "grouped": True,
            "groups_with_missing": total_missing_groups,
            "detail": missing_per_group,
        }
        if total_missing_groups > 0:
            msg = (
                f"Missing dates detected in {total_missing_groups} group(s). "
                f"First offending group: {next(iter(missing_per_group))}"
            )
            report.warnings.append(msg)
            logger.warning(msg)
    else:
        dates = series.sort_values()
        full_range = pd.date_range(dates.min(), dates.max(), freq=freq)
        missing = full_range.difference(dates)
        report.details[check_key] = {
            "grouped": False,
            "missing_count": len(missing),
            "missing_dates": missing.tolist()[:20],  # cap for readability
        }
        if len(missing) > 0:
            msg = f"Missing {len(missing)} date(s) in time series (freq='{freq}')."
            report.warnings.append(msg)
            logger.warning(msg)
        else:
            logger.info("✔ No missing dates found.")


def _check_negative_values(
    df: pd.DataFrame,
    columns: list[str],
    report: ValidationReport,
) -> None:
    """
    Check 2 — Negative Values in numeric columns.

    price, quantity, revenue must be ≥ 0.
    """
    check_key = "negative_values"
    report.details[check_key] = {}

    for col in columns:
        if col not in df.columns:
            msg = f"Column '{col}' not found — skipping negative-value check."
            report.warnings.append(msg)
            logger.warning(msg)
            continue

        neg_mask = df[col] < 0
        neg_count = int(neg_mask.sum())
        report.details[check_key][col] = {
            "negative_count": neg_count,
            "negative_row_indices": df.index[neg_mask].tolist()[:20],
        }

        if neg_count > 0:
            msg = f"Column '{col}' has {neg_count} negative value(s)."
            report.errors.append(msg)
            logger.error(msg)
            report.passed = False
        else:
            logger.info(f"✔ No negative values in '{col}'.")


def _check_duplicated_rows(
    df: pd.DataFrame,
    subset: Optional[list[str]],
    report: ValidationReport,
) -> None:
    """
    Check 3 — Duplicated Rows.

    `subset` narrows the uniqueness key (e.g., [date, brand, sku]).
    If None, all columns are compared.
    """
    check_key = "duplicated_rows"

    dup_mask = df.duplicated(subset=subset, keep=False)
    dup_count = int(dup_mask.sum())

    report.details[check_key] = {
        "duplicate_count": dup_count,
        "subset": subset,
        "duplicate_row_indices": df.index[dup_mask].tolist()[:20],
    }

    if dup_count > 0:
        msg = (
            f"Found {dup_count} duplicated row(s)"
            + (f" on subset {subset}." if subset else " (all columns).")
        )
        report.errors.append(msg)
        logger.error(msg)
        report.passed = False
    else:
        logger.info("✔ No duplicated rows found.")


def _check_revenue_consistency(
    df: pd.DataFrame,
    price_col: str,
    quantity_col: str,
    revenue_col: str,
    tolerance: float,
    report: ValidationReport,
) -> None:
    """
    Check 4 — Revenue Consistency.

    Verifies: |revenue - (price × quantity)| / revenue ≤ tolerance
    Rows where revenue == 0 are skipped to avoid division by zero.
    """
    check_key = "revenue_consistency"
    required = [price_col, quantity_col, revenue_col]
    missing_cols = [c for c in required if c not in df.columns]

    if missing_cols:
        msg = f"Revenue consistency check skipped — missing columns: {missing_cols}"
        report.warnings.append(msg)
        logger.warning(msg)
        report.details[check_key] = {"skipped": True, "reason": msg}
        return

    revenue = df[revenue_col]
    computed = df[price_col] * df[quantity_col]

    non_zero_mask = revenue != 0
    relative_error = pd.Series(np.nan, index=df.index)
    relative_error[non_zero_mask] = (
        (revenue[non_zero_mask] - computed[non_zero_mask]).abs()
        / revenue[non_zero_mask].abs()
    )

    inconsistent_mask = relative_error > tolerance
    inconsistent_count = int(inconsistent_mask.sum())

    report.details[check_key] = {
        "tolerance": tolerance,
        "inconsistent_count": inconsistent_count,
        "max_relative_error": float(relative_error.max()),
        "mean_relative_error": float(relative_error.mean()),
        "inconsistent_row_indices": df.index[inconsistent_mask].tolist()[:20],
    }

    if inconsistent_count > 0:
        msg = (
            f"Revenue inconsistency in {inconsistent_count} row(s) "
            f"(tolerance={tolerance:.1%}, "
            f"max_error={relative_error.max():.4%})."
        )
        report.errors.append(msg)
        logger.error(msg)
        report.passed = False
    else:
        logger.info(
            f"✔ Revenue consistent within tolerance ({tolerance:.1%})."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    freq: str = "W",
    group_cols: Optional[list[str]] = None,
    numeric_cols: Optional[list[str]] = None,
    duplicate_subset: Optional[list[str]] = None,
    price_col: str = "price",
    quantity_col: str = "quantity",
    revenue_col: str = "revenue",
    revenue_tolerance: float = 0.01,
    raise_on_error: bool = False,
) -> ValidationReport:
    """
    Run all validation checks on a time-series DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data — will NOT be modified.
    date_col : str
        Name of the date/time column.
    freq : str
        Expected pandas date frequency (e.g. 'D'=daily, 'W'=weekly, 'MS'=monthly).
    group_cols : list[str] | None
        Columns that define a unique time series (e.g. ['brand', 'sku']).
        If provided, missing-date check is run per group.
    numeric_cols : list[str] | None
        Columns to check for negative values.
        Defaults to [price_col, quantity_col, revenue_col].
    duplicate_subset : list[str] | None
        Column subset for duplicate detection.
    price_col : str
        Column name for unit price.
    quantity_col : str
        Column name for quantity sold.
    revenue_col : str
        Column name for revenue / total sales.
    revenue_tolerance : float
        Acceptable relative error for revenue consistency (default 1%).
    raise_on_error : bool
        If True, raise DataValidationError when any hard error is found.

    Returns
    -------
    ValidationReport
        Structured report with passed flag, errors, warnings, and details.

    Examples
    --------
    >>> report = validate(df, date_col="week", freq="W",
    ...                   group_cols=["brand", "sku"],
    ...                   raise_on_error=True)
    >>> print(report)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}.")

    if df.empty:
        raise DataValidationError("Input DataFrame is empty.")

    logger.info(f"Starting validation on DataFrame with shape {df.shape}.")
    report = ValidationReport()

    if numeric_cols is None:
        numeric_cols = [price_col, quantity_col, revenue_col]

    # ── Run checks ──────────────────────────────────────────────────────────
    _check_missing_dates(df, date_col, freq, group_cols, report)
    _check_negative_values(df, numeric_cols, report)
    _check_duplicated_rows(df, duplicate_subset, report)
    _check_revenue_consistency(
        df, price_col, quantity_col, revenue_col, revenue_tolerance, report
    )
    # ────────────────────────────────────────────────────────────────────────

    logger.info(report.summary)

    if raise_on_error and not report.passed:
        raise DataValidationError(
            f"Validation failed with {len(report.errors)} error(s). "
            "See report.errors for details."
        )

    return report
