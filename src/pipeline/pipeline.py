"""
pipeline.py — Full Pipeline Orchestrator
============================================================
Chạy toàn bộ pipeline từ load data đến evaluation.

Usage:
    python -m src.pipeline.pipeline                  # step 1→5
    python -m src.pipeline.pipeline --step 3         # từ step 3
    python -m src.pipeline.pipeline --step 3 --end 4 # chỉ step 3-4
============================================================
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from src.utils.logger import get_logger
from src.utils.config_loader import CONF, ensure_dirs

logger = get_logger(__name__)

_STEP_NAMES = {
    1: "Load & Clean Data",
    2: "Feature Engineering",
    3: "Prophet Stage 1 (per Brand + OOS residuals)",
    4: "LightGBM Stage 2 (Residual Learning)",
    5: "Evaluation & Visualization",
}


def run_pipeline(start_step: int = 1, end_step: int = 5) -> dict[int, str]:
    """
    Chạy pipeline từ start_step đến end_step (inclusive).

    Fix: range(start_step, end_step + 1) thay vì range(1, 6)
    để skip đúng các bước đã chạy.

    Returns
    -------
    dict mapping step_id → kết quả ("PASSED (...s)", "FAILED", "SKIPPED")
    """
    ensure_dirs()

    logger.info("╔" + "═" * 54 + "╗")
    logger.info("║  HYBRID PROPHET-LIGHTGBM DEMAND FORECASTING         ║")
    logger.info(f"║  Steps {start_step} → {end_step}                                       ║")
    logger.info("╚" + "═" * 54 + "╝")

    results: dict[int, str] = {}
    start_total = time.time()

    for idx in range(1, 6):
        if idx < start_step:
            results[idx] = "SKIPPED"
            continue
        if idx > end_step:
            results[idx] = "PENDING"
            continue

        logger.info("\n" + "▓" * 55)
        logger.info(f"  ▶ STEP {idx}: {_STEP_NAMES[idx]}")
        logger.info("▓" * 55)

        t0 = time.time()
        try:
            _run_step(idx)
            elapsed = time.time() - t0
            logger.info(f"  ✓ Step {idx} completed in {elapsed:.1f}s")
            results[idx] = f"PASSED ({elapsed:.1f}s)"
        except Exception as exc:
            logger.error(f"  ❌ Step {idx} FAILED: {exc}", exc_info=True)
            results[idx] = "FAILED"
            break

    # ── Summary ─────────────────────────────────────────────
    total = time.time() - start_total
    logger.info("\n" + "═" * 55)
    logger.info("  PIPELINE SUMMARY")
    logger.info("═" * 55)
    for i in range(1, 6):
        status = results.get(i, "NOT RUN")
        logger.info(f"  Step {i}: {status:<20s} — {_STEP_NAMES[i]}")
    logger.info(f"\n  Total time: {total:.1f}s ({total/60:.1f} min)")
    logger.info("═" * 55)

    return results


def _run_step(idx: int) -> None:
    """Dispatch table cho từng step."""
    if idx == 1:
        _step1_load_and_clean()
    elif idx == 2:
        _step2_feature_engineering()
    elif idx == 3:
        _step3_prophet()
    elif idx == 4:
        _step4_lightgbm()
    elif idx == 5:
        _step5_evaluation()
    else:
        raise ValueError(f"Step {idx} không tồn tại (hợp lệ: 1-5).")


def _step1_load_and_clean() -> None:
    from src.data.data_cleaning import clean
    from src.data.data_validation import validate
    from src.utils.config_loader import DATE_COL, TARGET_COL

    df = clean()

    # Validate sau khi clean (warning mode, không crash pipeline)
    report = validate(
        df,
        date_col=DATE_COL,
        freq="D",
        group_cols=["BRAND", "CATEGORY"],
        numeric_cols=[TARGET_COL],
        duplicate_subset=[DATE_COL, "BRAND", "CATEGORY"],
        raise_on_error=False,
    )
    logger.info(f"Validation: {report.summary}")
    if not report.passed:
        for err in report.errors:
            logger.warning(f"  Validation error: {err}")


def _step2_feature_engineering() -> None:
    from src.features.feature_engineering import engineer
    engineer()


def _step3_prophet() -> None:
    from src.models.prophet_model import run_prophet

    train = pd.read_csv(
        CONF.path_features + "/train_features.csv",
        parse_dates=[CONF.col_date],
    )
    test = pd.read_csv(
        CONF.path_features + "/test_features.csv",
        parse_dates=[CONF.col_date],
    )
    run_prophet(train, test)


def _step4_lightgbm() -> None:
    from src.models.lightgbm_model import run_lightgbm

    train = pd.read_csv(
        CONF.path_features + "/train_with_residuals.csv",
        parse_dates=[CONF.col_date],
    )
    test = pd.read_csv(
        CONF.path_features + "/test_with_residuals.csv",
        parse_dates=[CONF.col_date],
    )
    run_lightgbm(train, test)


def _step5_evaluation() -> None:
    from src.evaluation.metrics import evaluate
    evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run forecasting pipeline")
    parser.add_argument("--step", type=int, default=1,
                        help="Bắt đầu từ step nào (default: 1)")
    parser.add_argument("--end",  type=int, default=5,
                        help="Kết thúc ở step nào (default: 5)")
    args = parser.parse_args()
    run_pipeline(start_step=args.step, end_step=args.end)
