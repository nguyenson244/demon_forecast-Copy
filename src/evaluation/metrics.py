"""
============================================================
metrics.py — Evaluation Metrics & Visualization
============================================================
Migrate từ: src/evaluation.py
Role      : Tính RMSE, MAPE, sMAPE; ARIMA baseline;
            Biểu đồ Forecast vs Actual; Residual Diagnostics.
============================================================
"""

from __future__ import annotations

import os
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

from src.utils.logger import get_logger
from src.utils.config_loader import (
    PATHS, DATE_COL, TARGET_COL, SPLIT_DATE, ensure_dirs,
)

logger = get_logger(__name__)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#FAFAFA",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.labelsize":   12,
})


# ═══════════════════════════════════════════════════════════
# 1. HÀM TÍNH ĐỘ CHÍNH XÁC
# ═══════════════════════════════════════════════════════════
def calc_rmse(actual, predicted) -> float:
    actual, predicted = np.array(actual, float), np.array(predicted, float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))


def calc_mape(actual, predicted, epsilon: float = 1e-8) -> float:
    actual, predicted = np.array(actual, float), np.array(predicted, float)
    mask = (actual > epsilon) & ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100)


def calc_smape(actual, predicted) -> float:
    actual, predicted = np.array(actual, float), np.array(predicted, float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    a, p = actual[mask], predicted[mask]
    denom = (np.abs(a) + np.abs(p)) / 2 + 1e-8
    return float(np.mean(np.abs(a - p) / denom) * 100)


# ═══════════════════════════════════════════════════════════
# 2. ARIMA BASELINE
# ═══════════════════════════════════════════════════════════
def run_arima_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict | None:
    """ARIMA(5,1,2) baseline trên tổng daily."""
    from statsmodels.tsa.arima.model import ARIMA

    logger.info("Running ARIMA(5,1,2) baseline...")
    train_daily = train_df.groupby(DATE_COL)[TARGET_COL].sum().sort_index()
    test_daily  = test_df.groupby(DATE_COL)[TARGET_COL].sum().sort_index()

    try:
        fitted   = ARIMA(train_daily, order=(5, 1, 2)).fit()
        forecast = np.clip(fitted.forecast(steps=len(test_daily)).values, 0, None)

        result = {
            "model":       "ARIMA(5,1,2)",
            "RMSE":        calc_rmse(test_daily.values, forecast),
            "MAPE":        calc_mape(test_daily.values, forecast),
            "sMAPE":       calc_smape(test_daily.values, forecast),
            "predictions": forecast,
            "dates":       test_daily.index,
        }
        logger.info(f"ARIMA RMSE={result['RMSE']:,.1f} | MAPE={result['MAPE']:.2f}% | sMAPE={result['sMAPE']:.2f}%")
        return result
    except Exception as e:
        logger.warning(f"ARIMA failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# 3. BẢNG SO SÁNH
# ═══════════════════════════════════════════════════════════
def create_comparison_table(
    test_df: pd.DataFrame,
    arima_result: dict | None = None,
) -> pd.DataFrame:
    """Prophet vs ARIMA vs Hybrid comparison table."""
    daily = test_df.groupby(DATE_COL).agg({
        TARGET_COL:    "sum",
        "prophet_pred": "sum",
        "final_pred":   "sum",
    }).sort_index()

    actual        = daily[TARGET_COL].values
    prophet_pred  = daily["prophet_pred"].values
    hybrid_pred   = daily["final_pred"].values

    results = [
        {
            "Model":       "Prophet (Standalone)",
            "RMSE":        calc_rmse(actual, prophet_pred),
            "MAPE (%)":    calc_mape(actual, prophet_pred),
            "sMAPE (%)":   calc_smape(actual, prophet_pred),
        }
    ]
    if arima_result:
        results.append({
            "Model":     arima_result["model"],
            "RMSE":      arima_result["RMSE"],
            "MAPE (%)":  arima_result["MAPE"],
            "sMAPE (%)": arima_result["sMAPE"],
        })
    results.append({
        "Model":     "Hybrid (Prophet + LightGBM)",
        "RMSE":      calc_rmse(actual, hybrid_pred),
        "MAPE (%)":  calc_mape(actual, hybrid_pred),
        "sMAPE (%)": calc_smape(actual, hybrid_pred),
    })

    comparison_df = pd.DataFrame(results)
    logger.info(f"\n{comparison_df.to_string(index=False)}")
    return comparison_df


# ═══════════════════════════════════════════════════════════
# 4–6. BIỂU ĐỒ
# ═══════════════════════════════════════════════════════════
def plot_forecast_vs_actual(test_df: pd.DataFrame, save_dir: str | None = None) -> None:
    if save_dir is None:
        save_dir = PATHS["figures"]
    daily = test_df.groupby(DATE_COL).agg({
        TARGET_COL: "sum", "prophet_pred": "sum", "final_pred": "sum"
    }).sort_index()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(daily.index, daily[TARGET_COL],      color="#2C3E50", lw=2, label="Actual")
    ax.plot(daily.index, daily["prophet_pred"],   color="#E74C3C", lw=1.5, ls="--", label="Prophet", alpha=0.7)
    ax.plot(daily.index, daily["final_pred"],     color="#27AE60", lw=1.5, label="Hybrid (Prophet+LightGBM)")
    ax.set_xlabel("Date"); ax.set_ylabel("Total QTY")
    ax.set_title("Forecast vs Actual — Test Set (Daily Aggregate)", fontweight="bold")
    ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator()); plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(save_dir, "forecast_vs_actual.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    logger.info(f"✓ Saved → {path}")


def plot_forecast_per_brand(test_df: pd.DataFrame, top_n: int = 6, save_dir: str | None = None) -> None:
    if save_dir is None:
        save_dir = PATHS["figures"]
    top_brands = test_df.groupby("BRAND")[TARGET_COL].sum().nlargest(top_n).index.tolist()
    fig, axes = plt.subplots((top_n + 1) // 2, 2, figsize=(16, 4 * ((top_n + 1) // 2)))
    axes = axes.flatten()
    for i, brand in enumerate(top_brands):
        ax = axes[i]
        bd = test_df[test_df["BRAND"] == brand].groupby(DATE_COL).agg(
            {TARGET_COL: "sum", "final_pred": "sum"}
        ).sort_index()
        ax.plot(bd.index, bd[TARGET_COL],  color="#2C3E50", lw=1.5, label="Actual")
        ax.plot(bd.index, bd["final_pred"], color="#27AE60", lw=1.5, label="Hybrid", alpha=0.8)
        ax.set_title(brand, fontweight="bold"); ax.legend(fontsize=8); ax.tick_params(axis="x", rotation=45)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Forecast vs Actual — Per Brand (Top Volume)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "forecast_per_brand.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    logger.info(f"✓ Saved → {path}")


def plot_residual_diagnostics(test_df: pd.DataFrame, save_dir: str | None = None) -> None:
    if save_dir is None:
        save_dir = PATHS["figures"]
    daily = test_df.groupby(DATE_COL).agg(
        {TARGET_COL: "sum", "prophet_pred": "sum", "final_pred": "sum"}
    ).sort_index()
    p_res = daily[TARGET_COL] - daily["prophet_pred"]
    h_res = daily[TARGET_COL] - daily["final_pred"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, res, color, title in [
        (axes[0, 0], p_res, "#E74C3C", "Prophet Standalone — Residual Distribution"),
        (axes[0, 1], h_res, "#27AE60", "Hybrid (Prophet+LightGBM) — Residual Distribution"),
    ]:
        ax.hist(res, bins=50, color=color, alpha=0.7, edgecolor="white", density=True)
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_title(title); ax.set_xlabel("Residual"); ax.set_ylabel("Density")
        ax.annotate(f"Std = {res.std():,.0f}", xy=(0.7, 0.9), xycoords="axes fraction",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    axes[1, 0].plot(daily.index, p_res, color="#E74C3C", alpha=0.5, lw=0.8, label="Prophet")
    axes[1, 0].plot(daily.index, h_res, color="#27AE60", alpha=0.8, lw=0.8, label="Hybrid")
    axes[1, 0].axhline(0, color="black", ls="--", lw=0.5)
    axes[1, 0].set_title("Residuals Over Time"); axes[1, 0].legend()

    pd.DataFrame({"Prophet": np.abs(p_res.values), "Hybrid": np.abs(h_res.values)}).boxplot(
        ax=axes[1, 1], column=["Prophet", "Hybrid"]
    )
    axes[1, 1].set_title("|Residual| Comparison (Boxplot)"); axes[1, 1].set_ylabel("Absolute Residual")

    plt.suptitle("RESIDUAL DIAGNOSTICS — Prophet vs Hybrid", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "residual_diagnostics.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    logger.info(f"✓ Saved → {path}")
    logger.info(f"Residual Std — Prophet: {p_res.std():,.1f} | Hybrid: {h_res.std():,.1f} | Reduction: {(p_res.std()-h_res.std())/p_res.std()*100:.1f}%")


def plot_comparison_bars(comparison_df: pd.DataFrame, save_dir: str | None = None) -> None:
    if save_dir is None:
        save_dir = PATHS["figures"]
    metrics   = ["RMSE", "MAPE (%)", "sMAPE (%)"]
    colors    = ["#E74C3C", "#3498DB", "#27AE60"]
    n_models  = len(comparison_df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(metrics):
        if metric not in comparison_df.columns:
            continue
        ax   = axes[i]
        vals = comparison_df[metric].values
        bars = ax.bar(range(n_models), vals, color=colors[:n_models], edgecolor="white", width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:,.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(range(n_models))
        ax.set_xticklabels([m.replace(" (", "\n(") for m in comparison_df["Model"].values], fontsize=9)
        ax.set_title(metric, fontweight="bold", fontsize=13)
        ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1)
    plt.suptitle("MODEL COMPARISON — Error Metrics", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_metrics.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    logger.info(f"✓ Saved → {path}")


def per_brand_metrics(test_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for brand in sorted(test_df["BRAND"].unique()):
        bd = test_df[test_df["BRAND"] == brand]
        results.append({
            "Brand":         brand,
            "N_rows":        len(bd),
            "Actual_Sum":    bd[TARGET_COL].sum(),
            "Hybrid_RMSE":   calc_rmse(bd[TARGET_COL].values,   bd["final_pred"].values),
            "Hybrid_MAPE":   calc_mape(bd[TARGET_COL].values,   bd["final_pred"].values),
            "Hybrid_sMAPE":  calc_smape(bd[TARGET_COL].values,  bd["final_pred"].values),
            "Prophet_RMSE":  calc_rmse(bd[TARGET_COL].values,   bd["prophet_pred"].values),
        })
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════
def evaluate(
    test_df:   pd.DataFrame | None = None,
    train_df:  pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chạy toàn bộ pipeline đánh giá.

    Returns
    -------
    (comparison_df, brand_metrics_df)
    """
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("BƯỚC 5: ĐÁNH GIÁ HIỆU SUẤT & TRỰC QUAN HÓA")
    logger.info("=" * 60)

    if test_df is None:
        pred_path       = os.path.join(PATHS["metrics"], "test_predictions.csv")
        train_pred_path = os.path.join(PATHS["metrics"], "train_predictions.csv")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Chưa có test_predictions.csv. Chạy lightgbm_model trước!")
        test_df  = pd.read_csv(pred_path,       parse_dates=[DATE_COL])
        train_df = pd.read_csv(train_pred_path, parse_dates=[DATE_COL])

    logger.info("[1/5] ARIMA baseline...")
    arima_result = run_arima_baseline(train_df, test_df)

    logger.info("[2/5] Comparison table...")
    comparison_df = create_comparison_table(test_df, arima_result)
    table_path    = os.path.join(PATHS["metrics"], "comparison_metrics.csv")
    comparison_df.to_csv(table_path, index=False)

    logger.info("[3/5] Forecast vs Actual plots...")
    plot_forecast_vs_actual(test_df)
    plot_forecast_per_brand(test_df)

    logger.info("[4/5] Residual diagnostics...")
    plot_residual_diagnostics(test_df)

    logger.info("[5/5] Comparison bars & per-brand metrics...")
    plot_comparison_bars(comparison_df)
    brand_metrics_df = per_brand_metrics(test_df)
    brand_path = os.path.join(PATHS["metrics"], "per_brand_metrics.csv")
    brand_metrics_df.to_csv(brand_path, index=False)

    logger.info("✓ HOÀN TẤT ĐÁNH GIÁ HIỆU SUẤT")
    return comparison_df, brand_metrics_df


def main():
    evaluate()


if __name__ == "__main__":
    main()
