"""
============================================================
backtesting.py — Walk-Forward Validation (Time Series CV)
============================================================
Role: Đánh giá mô hình bằng phương pháp walk-forward
      validation (time-series cross-validation) để mô phỏng
      điều kiện thực tế khi dự báo tương lai.

★ KHÔNG dùng random K-Fold — chỉ walk-forward.
============================================================
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from src.utils.logger import get_logger
from src.utils.config_loader import (
    PATHS, DATE_COL, TARGET_COL, TSCV_N_SPLITS, ensure_dirs,
)
from src.evaluation.metrics import calc_rmse, calc_mape, calc_smape

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════
# BACKTESTING REPORT
# ═══════════════════════════════════════════════════════════
@dataclass
class BacktestReport:
    """Kết quả walk-forward validation."""
    n_splits:      int
    fold_results:  list[dict] = field(default_factory=list)

    @property
    def mean_rmse(self)  -> float:
        return float(np.mean([f["RMSE"]  for f in self.fold_results]))

    @property
    def mean_mape(self)  -> float:
        return float(np.mean([f["MAPE"]  for f in self.fold_results]))

    @property
    def mean_smape(self) -> float:
        return float(np.mean([f["sMAPE"] for f in self.fold_results]))

    @property
    def std_rmse(self) -> float:
        return float(np.std([f["RMSE"] for f in self.fold_results]))

    def __str__(self) -> str:
        lines = [
            "=" * 55,
            f"  WALK-FORWARD BACKTEST REPORT ({self.n_splits} folds)",
            "=" * 55,
            f"  {'Metric':<12} {'Mean':>10} {'Std':>10}",
            f"  {'-'*35}",
            f"  {'RMSE':<12} {self.mean_rmse:>10,.2f} {self.std_rmse:>10,.2f}",
            f"  {'MAPE (%)':<12} {self.mean_mape:>10.2f}",
            f"  {'sMAPE (%)':<12} {self.mean_smape:>10.2f}",
            "=" * 55,
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# WALK-FORWARD SPLITS
# ═══════════════════════════════════════════════════════════
def generate_walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = TSCV_N_SPLITS,
    min_train_ratio: float = 0.5,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Tạo danh sách (train_fold, val_fold) theo walk-forward.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset đã sắp xếp theo thời gian.
    n_splits : int
        Số lượng folds.
    min_train_ratio : float
        Tỷ lệ dữ liệu tối thiểu cho fold train đầu tiên.

    Returns
    -------
    list of (train_fold, val_fold)
    """
    dates_sorted = sorted(df[DATE_COL].unique())
    n_dates      = len(dates_sorted)

    # Minimum train window
    min_train_dates = int(n_dates * min_train_ratio)

    # Chia đều phần còn lại thành n_splits folds
    remaining    = n_dates - min_train_dates
    fold_size    = remaining // (n_splits + 1)

    if fold_size < 1:
        raise ValueError(
            f"Không đủ dữ liệu để tạo {n_splits} folds. "
            f"Thử giảm n_splits hoặc min_train_ratio."
        )

    splits = []
    for i in range(n_splits):
        train_end_idx = min_train_dates + fold_size * i
        val_end_idx   = min(train_end_idx + fold_size, n_dates)

        train_cutoff = dates_sorted[train_end_idx - 1]
        val_start    = dates_sorted[train_end_idx]
        val_end      = dates_sorted[val_end_idx - 1]

        train_fold = df[df[DATE_COL] <= train_cutoff].copy()
        val_fold   = df[(df[DATE_COL] >= val_start) & (df[DATE_COL] <= val_end)].copy()

        splits.append((train_fold, val_fold))
        logger.debug(
            f"Fold {i+1}: Train={len(train_fold):,} rows "
            f"({df[DATE_COL].min().date()} → {train_cutoff.date()}) | "
            f"Val={len(val_fold):,} rows "
            f"({val_start.date()} → {val_end.date()})"
        )

    return splits


# ═══════════════════════════════════════════════════════════
# RUN BACKTEST — Simple (Prophet-only) baseline
# ═══════════════════════════════════════════════════════════
def run_backtest_prophet(
    df: pd.DataFrame,
    n_splits: int = TSCV_N_SPLITS,
) -> BacktestReport:
    """
    Walk-forward backtest sử dụng Prophet standalone.
    Hữu ích để baseline hiệu suất theo từng fold.

    Parameters
    ----------
    df : pd.DataFrame
        Full features DataFrame (đã có DATE_COL và TARGET_COL).
    n_splits : int
        Số lượng folds.

    Returns
    -------
    BacktestReport
    """
    from prophet import Prophet
    from src.models.prophet_model import get_vietnamese_holidays

    splits = generate_walk_forward_splits(df, n_splits=n_splits)
    holidays_df = get_vietnamese_holidays()
    report = BacktestReport(n_splits=len(splits))

    for fold_idx, (train_fold, val_fold) in enumerate(splits):
        logger.info(f"  ── Fold {fold_idx + 1}/{len(splits)} ──")

        # Aggregate daily (total all brands)
        train_ts = (
            train_fold.groupby(DATE_COL)[TARGET_COL].sum()
            .reset_index().rename(columns={DATE_COL: "ds", TARGET_COL: "y"})
            .sort_values("ds")
        )
        val_ts = (
            val_fold.groupby(DATE_COL)[TARGET_COL].sum()
            .reset_index().rename(columns={DATE_COL: "ds", TARGET_COL: "y"})
            .sort_values("ds")
        )

        if len(train_ts) < 30:
            logger.warning(f"Fold {fold_idx+1}: Không đủ dữ liệu train ({len(train_ts)} ngày) → SKIP")
            continue

        try:
            model = Prophet(
                holidays=holidays_df,
                seasonality_mode="multiplicative",
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
            )
            import logging as _logging
            _logging.getLogger("prophet").setLevel(_logging.WARNING)
            model.fit(train_ts)

            forecast = model.predict(val_ts[["ds"]])
            predicted = np.clip(forecast["yhat"].values, 0, None)
            actual    = val_ts["y"].values

            fold_result = {
                "fold":   fold_idx + 1,
                "RMSE":   calc_rmse(actual, predicted),
                "MAPE":   calc_mape(actual, predicted),
                "sMAPE":  calc_smape(actual, predicted),
                "n_train": len(train_ts),
                "n_val":   len(val_ts),
            }
            report.fold_results.append(fold_result)
            logger.info(
                f"    RMSE={fold_result['RMSE']:,.1f} | "
                f"MAPE={fold_result['MAPE']:.2f}% | "
                f"sMAPE={fold_result['sMAPE']:.2f}%"
            )
        except Exception as e:
            logger.error(f"Fold {fold_idx+1} failed: {e}")

    logger.info(str(report))
    return report


# ═══════════════════════════════════════════════════════════
# PLOT BACKTEST RESULTS
# ═══════════════════════════════════════════════════════════
def plot_backtest_results(report: BacktestReport, save_dir: str | None = None) -> None:
    """Vẽ RMSE/MAPE theo từng fold."""
    if not report.fold_results:
        logger.warning("Không có fold results để vẽ.")
        return

    if save_dir is None:
        save_dir = PATHS["figures"]

    folds = [f["fold"]  for f in report.fold_results]
    rmses = [f["RMSE"]  for f in report.fold_results]
    mapes = [f["MAPE"]  for f in report.fold_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(folds, rmses, color="#3498DB", edgecolor="white")
    axes[0].axhline(report.mean_rmse, color="#E74C3C", ls="--", lw=1.5,
                    label=f"Mean={report.mean_rmse:,.0f}")
    axes[0].set_title("RMSE per Fold", fontweight="bold")
    axes[0].set_xlabel("Fold"); axes[0].set_ylabel("RMSE")
    axes[0].legend()

    axes[1].bar(folds, mapes, color="#27AE60", edgecolor="white")
    axes[1].axhline(report.mean_mape, color="#E74C3C", ls="--", lw=1.5,
                    label=f"Mean={report.mean_mape:.1f}%")
    axes[1].set_title("MAPE (%) per Fold", fontweight="bold")
    axes[1].set_xlabel("Fold"); axes[1].set_ylabel("MAPE (%)")
    axes[1].legend()

    plt.suptitle(f"Walk-Forward Backtest — {report.n_splits} Folds",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, "backtest_results.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved backtest plot → {path}")


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════
def backtest(
    df: pd.DataFrame | None = None,
    n_splits: int = TSCV_N_SPLITS,
) -> BacktestReport:
    """
    Chạy walk-forward backtest và lưu kết quả.

    Parameters
    ----------
    df : pd.DataFrame | None
        Full features DataFrame. Nếu None, đọc từ full_features.csv.
    n_splits : int
        Số folds.

    Returns
    -------
    BacktestReport
    """
    ensure_dirs()
    logger.info("=" * 55)
    logger.info(f"WALK-FORWARD BACKTEST — {n_splits} folds")
    logger.info("=" * 55)

    if df is None:
        full_path = os.path.join(PATHS["features"], "full_features.csv")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Chưa có full_features.csv. Chạy feature_engineering trước!")
        df = pd.read_csv(full_path, parse_dates=[DATE_COL])
        logger.info(f"Loaded: {len(df):,} rows")

    report = run_backtest_prophet(df, n_splits=n_splits)
    plot_backtest_results(report)

    # Save summary CSV
    if report.fold_results:
        summary_df  = pd.DataFrame(report.fold_results)
        summary_path = os.path.join(PATHS["metrics"], "backtest_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"✓ Saved backtest summary → {summary_path}")

    return report


if __name__ == "__main__":
    backtest()
