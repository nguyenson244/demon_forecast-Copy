"""
============================================================
train.py — Model Training Entry Point
============================================================
Role: Orchestrate Stage 1 (Prophet) + Stage 2 (LightGBM).
      Có thể chạy độc lập hoặc gọi từ pipeline.py.

Usage:
    python -m src.models.train
    python -m src.models.train --stage 1   # Chỉ Prophet
    python -m src.models.train --stage 2   # Chỉ LightGBM
============================================================
"""

from __future__ import annotations

import argparse
import time

from src.utils.logger import get_logger
from src.utils.config_loader import ensure_dirs

logger = get_logger(__name__)


def train(stage: int = 0) -> dict:
    """
    Huấn luyện mô hình theo stage được chỉ định.

    Parameters
    ----------
    stage : int
        0 = cả 2 stages (mặc định)
        1 = chỉ Prophet Stage 1
        2 = chỉ LightGBM Stage 2

    Returns
    -------
    dict : kết quả mỗi stage {"prophet": ..., "lightgbm": ...}
    """
    ensure_dirs()
    results = {}

    if stage in (0, 1):
        logger.info("▶ Starting Stage 1: Prophet per Brand...")
        t0 = time.time()
        from src.models.prophet_model import run_prophet
        full_df, prophet_models = run_prophet()
        elapsed = time.time() - t0
        results["prophet"] = {
            "full_df": full_df,
            "models":  prophet_models,
            "elapsed": elapsed,
        }
        logger.info(f"⏱ Stage 1 completed in {elapsed:.1f}s")

    if stage in (0, 2):
        logger.info("▶ Starting Stage 2: LightGBM Global...")
        t0 = time.time()
        from src.models.lightgbm_model import run_lightgbm
        model, test_df, train_df = run_lightgbm()
        elapsed = time.time() - t0
        results["lightgbm"] = {
            "model":    model,
            "test_df":  test_df,
            "train_df": train_df,
            "elapsed":  elapsed,
        }
        logger.info(f"⏱ Stage 2 completed in {elapsed:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train Hybrid Prophet-LightGBM Model"
    )
    parser.add_argument(
        "--stage", type=int, default=0,
        help="0=both (default), 1=Prophet only, 2=LightGBM only"
    )
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  MODEL TRAINING — Hybrid Prophet + LightGBM ║")
    logger.info("╚══════════════════════════════════════════════╝")

    total_start = time.time()
    train(stage=args.stage)
    total_elapsed = time.time() - total_start

    logger.info(f"Total training time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
