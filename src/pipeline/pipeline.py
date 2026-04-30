"""
============================================================
pipeline.py — Full Pipeline Orchestrator
============================================================
Thay thế: run_pipeline.py ở root directory.
Role    : Điều phối toàn bộ 5 bước tuần tự.

Usage:
    python -m src.pipeline.pipeline           # Chạy tất cả
    python -m src.pipeline.pipeline --step 3  # Từ bước 3
    python -m src.pipeline.pipeline --step 2 --end 4  # Bước 2→4
============================================================
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

from src.utils.logger import get_logger
from src.utils.config_loader import ensure_dirs

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════
# STEP DEFINITIONS
# ═══════════════════════════════════════════════════════════
@dataclass
class StepResult:
    step_num:  int
    step_name: str
    success:   bool
    elapsed:   float
    output:    object = field(default=None, repr=False)


def _run_step(step_num: int, step_name: str, fn) -> StepResult:
    """Chạy 1 bước, đo thời gian, bắt exception."""
    logger.info(f"\n{'▓' * 55}")
    logger.info(f"  ▶ STEP {step_num}: {step_name}")
    logger.info(f"{'▓' * 55}")

    t0 = time.time()
    try:
        output  = fn()
        elapsed = time.time() - t0
        logger.info(f"  ⏱ Step {step_num} completed in {elapsed:.1f}s")
        return StepResult(step_num, step_name, True, elapsed, output)
    except Exception as e:
        elapsed = time.time() - t0
        import traceback
        logger.error(f"  ⛔ Step {step_num} FAILED after {elapsed:.1f}s: {e}")
        logger.error(traceback.format_exc())
        return StepResult(step_num, step_name, False, elapsed)


# ═══════════════════════════════════════════════════════════
# STEP FUNCTIONS
# ═══════════════════════════════════════════════════════════
def step1_load_and_clean():
    """Bước 1: Load raw data + tiền xử lý."""
    from src.data.data_cleaning import clean
    return clean()


def step2_feature_engineering():
    """Bước 2: Feature Engineering."""
    from src.features.feature_engineering import engineer
    return engineer()


def step3_prophet():
    """Bước 3: Prophet Stage 1 — per Brand."""
    from src.models.prophet_model import run_prophet
    return run_prophet()


def step4_lightgbm():
    """Bước 4: LightGBM Global Stage 2."""
    from src.models.lightgbm_model import run_lightgbm
    return run_lightgbm()


def step5_evaluate():
    """Bước 5: Evaluation & Visualization."""
    from src.evaluation.metrics import evaluate
    return evaluate()


# Ordered steps registry
STEPS: dict[int, tuple[str, callable]] = {
    1: ("Load & Clean Data",         step1_load_and_clean),
    2: ("Feature Engineering",       step2_feature_engineering),
    3: ("Prophet Stage 1 (per Brand)", step3_prophet),
    4: ("LightGBM Global Stage 2",   step4_lightgbm),
    5: ("Evaluation & Visualization", step5_evaluate),
}


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════
def run_pipeline(
    start_step: int = 1,
    end_step:   int = 5,
) -> list[StepResult]:
    """
    Chạy pipeline từ start_step đến end_step (inclusive).

    Parameters
    ----------
    start_step : int  Bước bắt đầu (1–5)
    end_step   : int  Bước kết thúc (1–5)

    Returns
    -------
    list[StepResult]
    """
    ensure_dirs()

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  HYBRID PROPHET-LIGHTGBM DEMAND FORECASTING         ║")
    logger.info("║  Kinh Đô FMCG — src/pipeline/pipeline.py            ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    total_start = time.time()
    results: list[StepResult] = []

    for step_num in range(1, 6):
        step_name, step_fn = STEPS[step_num]

        if step_num < start_step or step_num > end_step:
            logger.info(f"\n  ⏭ Skipping Step {step_num}: {step_name}")
            continue

        result = _run_step(step_num, step_name, step_fn)
        results.append(result)

        if not result.success:
            logger.error(f"\n⛔ Pipeline STOPPED at Step {step_num}")
            break

    # ── Summary ──
    total_elapsed = time.time() - total_start
    logger.info(f"\n{'═' * 55}")
    logger.info("  PIPELINE SUMMARY")
    logger.info(f"{'═' * 55}")
    for r in results:
        status = "✓ PASSED" if r.success else "⛔ FAILED"
        logger.info(f"  Step {r.step_num}: {status} ({r.elapsed:.1f}s) — {r.step_name}")
    logger.info(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    logger.info(f"{'═' * 55}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Demand Forecasting Pipeline — Kinh Đô FMCG"
    )
    parser.add_argument(
        "--step", type=int, default=1,
        help="Bắt đầu từ bước N (1-5). Mặc định: 1"
    )
    parser.add_argument(
        "--end", type=int, default=5,
        help="Kết thúc tại bước N (1-5). Mặc định: 5"
    )
    args = parser.parse_args()
    run_pipeline(start_step=args.step, end_step=args.end)


if __name__ == "__main__":
    main()
