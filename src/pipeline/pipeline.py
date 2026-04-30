"""
pipeline.py — Full Pipeline Orchestrator
"""

from __future__ import annotations
import argparse
import time
import pandas as pd
from src.utils.logger import get_logger
from src.utils.config_loader import CONF, ensure_dirs

logger = get_logger(__name__)

def run_pipeline(start_step: int = 1, end_step: int = 5):
    ensure_dirs()
    
    steps = {
        1: ("Load & Clean Data", step1_load_and_clean),
        2: ("Feature Engineering", step2_feature_engineering),
        3: ("Prophet Stage 1 (per Brand)", step3_prophet),
        4: ("LightGBM Multi-Cluster Stage 2", step4_lightgbm),
        5: ("Evaluation & Visualization", step5_evaluation),
    }

    logger.info("╔" + "═"*54 + "╗")
    logger.info("║  HYBRID PROPHET-LIGHTGBM DEMAND FORECASTING         ║")
    logger.info("╚" + "═"*54 + "╝")

    start_total = time.time()
    results = {}

    for idx in range(1, 6):
        name, fn = steps[idx]
        if idx < start_step:
            results[idx] = "SKIPPED"
            continue
        if idx > end_step:
            results[idx] = "PENDING"
            continue

        logger.info("\n" + "▓"*55)
        logger.info(f"  ▶ STEP {idx}: {name}")
        logger.info("▓"*55)

        start_s = time.time()
        try:
            fn()
            duration = time.time() - start_s
            logger.info(f"  ✓ Step {idx} completed in {duration:.1f}s")
            results[idx] = f"PASSED ({duration:.1f}s)"
        except Exception as e:
            logger.error(f"  ❌ Step {idx} FAILED: {e}", exc_info=True)
            results[idx] = "FAILED"
            break

    # Summary
    logger.info("\n" + "═"*55)
    logger.info("  PIPELINE SUMMARY")
    logger.info("═"*55)
    for i, res in results.items():
        logger.info(f"  Step {i}: {res} — {steps[i][0]}")
    logger.info(f"\n  Total time: {time.time() - start_total:.1f}s")
    logger.info("═"*55)

def step1_load_and_clean():
    from src.data.data_cleaning import clean
    clean()

def step2_feature_engineering():
    from src.features.feature_engineering import engineer
    engineer()

def step3_prophet():
    from src.models.prophet_model import run_prophet
    # Nạp data từ bước 2
    train = pd.read_csv(CONF.path_features + "/train_features.csv", parse_dates=[CONF.col_date])
    test = pd.read_csv(CONF.path_features + "/test_features.csv", parse_dates=[CONF.col_date])
    run_prophet(train, test)

def step4_lightgbm():
    from src.models.lightgbm_model import run_lightgbm
    # Nạp data từ bước 3 (có chứa residuals)
    train = pd.read_csv(CONF.path_features + "/train_with_residuals.csv", parse_dates=[CONF.col_date])
    test = pd.read_csv(CONF.path_features + "/test_with_residuals.csv", parse_dates=[CONF.col_date])
    run_lightgbm(train, test)

def step5_evaluation():
    from src.evaluation.metrics import evaluate
    # Nạp kết quả từ bước 4
    # (Bước 4 đã lưu file train/test_predictions.csv vào results/metrics)
    evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--end", type=int, default=5)
    args = parser.parse_args()
    run_pipeline(args.step, args.end)
