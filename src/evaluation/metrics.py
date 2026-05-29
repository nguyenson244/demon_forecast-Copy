"""
============================================================
metrics.py — Evaluation Metrics & Visualization
============================================================
Role: Tính RMSE, MAPE, WMAPE, sMAPE; ARIMA baseline;
      Biểu đồ Forecast vs Actual; Residual Diagnostics;
      Train vs Test metrics (overfitting check).

WMAPE (Weighted MAPE) là metric chính cho FMCG zero-inflation:
  WMAPE = sum(|actual - pred|) / sum(|actual|)
  Ưu điểm:
    - Không bị undefined khi actual = 0 (khác MAPE)
    - Không bị skew bởi rows có actual nhỏ
    - Tương đương MAE% trên tổng volume — phù hợp KPI business
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
# 1. METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════
def calc_rmse(actual, predicted) -> float:
    a, p = np.array(actual, float), np.array(predicted, float)
    mask = ~(np.isnan(a) | np.isnan(p))
    return float(np.sqrt(np.mean((a[mask] - p[mask]) ** 2)))


def calc_mape(actual, predicted, epsilon: float = 1e-8) -> float:
    """MAPE — không ổn định khi actual ≈ 0. Dùng WMAPE cho FMCG."""
    a, p = np.array(actual, float), np.array(predicted, float)
    mask = (a > epsilon) & ~np.isnan(a) & ~np.isnan(p)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(a[mask] - p[mask]) / a[mask]) * 100)


def calc_wmape(actual, predicted) -> float:
    """
    Weighted MAPE — metric chính cho FMCG zero-inflation.

    WMAPE = sum(|actual - pred|) / sum(|actual|) × 100

    Tại sao tốt hơn MAPE cho FMCG:
      - Không undefined khi actual = 0
      - Volume ngày cao (peak) có trọng số lớn hơn ngày thấp
      - Tương đương MAE percentage trên tổng — dễ giải thích với business
    """
    a, p = np.array(actual, float), np.array(predicted, float)
    mask = ~(np.isnan(a) | np.isnan(p))
    a, p = a[mask], p[mask]
    denom = np.sum(np.abs(a))
    if denom < 1e-8:
        return float("nan")
    return float(np.sum(np.abs(a - p)) / denom * 100)


def calc_smape(actual, predicted) -> float:
    a, p = np.array(actual, float), np.array(predicted, float)
    mask = ~(np.isnan(a) | np.isnan(p))
    a, p = a[mask], p[mask]
    denom = (np.abs(a) + np.abs(p)) / 2 + 1e-8
    return float(np.mean(np.abs(a - p) / denom) * 100)


def calc_mae(actual, predicted) -> float:
    """MAE = mean(|actual - pred|)."""
    a, p = np.array(actual, float), np.array(predicted, float)
    mask = ~(np.isnan(a) | np.isnan(p))
    return float(np.mean(np.abs(a[mask] - p[mask])))


def calc_bias(actual, predicted) -> float:
    """Bias = mean(pred - actual) / mean(actual) × 100. Dương = over-forecast."""
    a, p = np.array(actual, float), np.array(predicted, float)
    mask = ~(np.isnan(a) | np.isnan(p))
    denom = np.mean(np.abs(a[mask])) + 1e-8
    return float(np.mean(p[mask] - a[mask]) / denom * 100)


def _metrics_row(label: str, actual, predicted) -> dict:
    return {
        "Model":       label,
        "RMSE":        calc_rmse(actual, predicted),
        "MAE":         calc_mae(actual, predicted),
        "WMAPE (%)":   calc_wmape(actual, predicted),
        "MAPE (%)":    calc_mape(actual, predicted),
        "sMAPE (%)":   calc_smape(actual, predicted),
        "Bias (%)":    calc_bias(actual, predicted),
    }


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
        result = _metrics_row("ARIMA(5,1,2)", test_daily.values, forecast)
        result["predictions"] = forecast
        result["dates"]       = test_daily.index
        logger.info(
            f"ARIMA RMSE={result['RMSE']:,.1f} | "
            f"WMAPE={result['WMAPE (%)']:.2f}%"
        )
        return result
    except Exception as exc:
        logger.warning(f"ARIMA failed: {exc}")
        return None


# ═══════════════════════════════════════════════════════════
# 3. COMPARISON TABLE (Train + Test)
# ═══════════════════════════════════════════════════════════
def create_comparison_table(
    test_df:  pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    arima_result: dict | None = None,
) -> pd.DataFrame:
    """
    So sánh Prophet vs ARIMA vs Hybrid trên Test.
    Nếu có train_df: thêm Train metrics để check overfitting.
    """
    def _daily_agg(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(DATE_COL).agg({
            TARGET_COL:    "sum",
            "prophet_pred": "sum",
            "final_pred":   "sum",
        }).sort_index()

    test_daily = _daily_agg(test_df)
    act_t  = test_daily[TARGET_COL].values
    p_pred = test_daily["prophet_pred"].values
    h_pred = test_daily["final_pred"].values

    rows = [
        _metrics_row("Prophet (Standalone) — Test", act_t, p_pred),
    ]
    if arima_result:
        rows.append({
            "Model":     arima_result["Model"],
            "RMSE":      arima_result["RMSE"],
            "WMAPE (%)": arima_result["WMAPE (%)"],
            "MAPE (%)":  arima_result["MAPE (%)"],
            "sMAPE (%)": arima_result["sMAPE (%)"],
            "Bias (%)":  arima_result.get("Bias (%)", float("nan")),
        })
    rows.append(_metrics_row("Hybrid (Prophet+LightGBM) — Test", act_t, h_pred))

    # Train metrics (overfitting check)
    if train_df is not None and "final_pred" in train_df.columns:
        train_daily = _daily_agg(train_df)
        act_tr  = train_daily[TARGET_COL].values
        hp_tr   = train_daily["final_pred"].values
        pp_tr   = train_daily["prophet_pred"].values

        rows.append(_metrics_row("Hybrid (Prophet+LightGBM) — Train", act_tr, hp_tr))
        rows.append(_metrics_row("Prophet (Standalone) — Train",       act_tr, pp_tr))

        # Overfitting check
        hybrid_test_wmape  = calc_wmape(act_t,  h_pred)
        hybrid_train_wmape = calc_wmape(act_tr, hp_tr)
        gap = hybrid_test_wmape - hybrid_train_wmape
        if gap > 10:
            logger.warning(
                f"⚠ Overfitting detected: "
                f"Train WMAPE={hybrid_train_wmape:.2f}% "
                f"vs Test WMAPE={hybrid_test_wmape:.2f}% "
                f"(gap={gap:.2f}%)"
            )
        else:
            logger.info(
                f"✔ No overfitting: "
                f"Train WMAPE={hybrid_train_wmape:.2f}% "
                f"Test WMAPE={hybrid_test_wmape:.2f}%"
            )

    df = pd.DataFrame(rows)
    logger.info(f"\n{df.to_string(index=False)}")
    return df


# ═══════════════════════════════════════════════════════════
# 4. BIỂU ĐỒ
# ═══════════════════════════════════════════════════════════
def plot_forecast_vs_actual(test_df: pd.DataFrame, save_dir: str | None = None) -> None:
    if save_dir is None:
        save_dir = PATHS["figures"]
    daily = test_df.groupby(DATE_COL).agg({
        TARGET_COL: "sum", "prophet_pred": "sum", "final_pred": "sum"
    }).sort_index()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(daily.index, daily[TARGET_COL],    color="#2C3E50", lw=2,   label="Actual")
    ax.plot(daily.index, daily["prophet_pred"], color="#E74C3C", lw=1.5, ls="--",
            label="Prophet", alpha=0.7)
    ax.plot(daily.index, daily["final_pred"],   color="#27AE60", lw=1.5,
            label="Hybrid (Prophet+LightGBM)")

    wmape_h = calc_wmape(daily[TARGET_COL], daily["final_pred"])
    wmape_p = calc_wmape(daily[TARGET_COL], daily["prophet_pred"])
    ax.set_title(
        f"Forecast vs Actual — Test Set\n"
        f"Hybrid WMAPE={wmape_h:.2f}%  |  Prophet WMAPE={wmape_p:.2f}%",
        fontweight="bold",
    )
    ax.set_xlabel("Date"); ax.set_ylabel("Total QTY")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(save_dir, "forecast_vs_actual.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    logger.info(f"✓ Saved → {path}")


def plot_forecast_per_brand(
    test_df: pd.DataFrame, top_n: int = 6, save_dir: str | None = None
) -> None:
    if save_dir is None:
        save_dir = PATHS["figures"]
    top_brands = test_df.groupby("BRAND")[TARGET_COL].sum().nlargest(top_n).index
    ncols = 2
    nrows = (top_n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    for i, brand in enumerate(top_brands):
        ax = axes[i]
        bd = test_df[test_df["BRAND"] == brand].groupby(DATE_COL).agg(
            {TARGET_COL: "sum", "final_pred": "sum"}
        ).sort_index()
        wmape = calc_wmape(bd[TARGET_COL], bd["final_pred"])
        ax.plot(bd.index, bd[TARGET_COL],  color="#2C3E50", lw=1.5, label="Actual")
        ax.plot(bd.index, bd["final_pred"], color="#27AE60", lw=1.5,
                label=f"Hybrid (WMAPE={wmape:.1f}%)", alpha=0.8)
        ax.set_title(brand, fontweight="bold")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Forecast vs Actual — Top Brands (Test Set)",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "forecast_per_brand.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    logger.info(f"✓ Saved → {path}")


def plot_residual_diagnostics(
    test_df: pd.DataFrame, save_dir: str | None = None
) -> None:
    if save_dir is None:
        save_dir = PATHS["figures"]
    daily = test_df.groupby(DATE_COL).agg(
        {TARGET_COL: "sum", "prophet_pred": "sum", "final_pred": "sum"}
    ).sort_index()
    p_res = daily[TARGET_COL] - daily["prophet_pred"]
    h_res = daily[TARGET_COL] - daily["final_pred"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, res, color, title in [
        (axes[0, 0], p_res, "#E74C3C", "Prophet — Residual Distribution"),
        (axes[0, 1], h_res, "#27AE60", "Hybrid — Residual Distribution"),
    ]:
        ax.hist(res, bins=50, color=color, alpha=0.7, edgecolor="white", density=True)
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_title(title); ax.set_xlabel("Residual"); ax.set_ylabel("Density")
        ax.annotate(
            f"Std={res.std():,.0f}  Bias={res.mean():+,.0f}",
            xy=(0.05, 0.9), xycoords="axes fraction",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    axes[1, 0].plot(daily.index, p_res, color="#E74C3C", alpha=0.5, lw=0.8, label="Prophet")
    axes[1, 0].plot(daily.index, h_res, color="#27AE60", alpha=0.8, lw=0.8, label="Hybrid")
    axes[1, 0].axhline(0, color="black", ls="--", lw=0.5)
    axes[1, 0].set_title("Residuals Over Time"); axes[1, 0].legend()

    pd.DataFrame({
        "Prophet": np.abs(p_res.values),
        "Hybrid":  np.abs(h_res.values),
    }).boxplot(ax=axes[1, 1])
    axes[1, 1].set_title("|Residual| Comparison"); axes[1, 1].set_ylabel("Abs Residual")

    plt.suptitle(
        "RESIDUAL DIAGNOSTICS — Prophet vs Hybrid",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "residual_diagnostics.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()

    reduction = (p_res.std() - h_res.std()) / p_res.std() * 100
    logger.info(
        f"Residual Std — Prophet: {p_res.std():,.1f} | "
        f"Hybrid: {h_res.std():,.1f} | "
        f"Reduction: {reduction:.1f}%"
    )
    logger.info(f"✓ Saved → {path}")


def plot_comparison_bars(
    comparison_df: pd.DataFrame, save_dir: str | None = None
) -> None:
    if save_dir is None:
        save_dir = PATHS["figures"]

    # Chỉ vẽ Test rows
    test_df = comparison_df[comparison_df["Model"].str.contains("Test")]
    metrics = ["RMSE", "WMAPE (%)", "sMAPE (%)"]
    colors  = ["#E74C3C", "#3498DB", "#27AE60", "#9B59B6"]
    n = len(test_df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(metrics):
        if metric not in test_df.columns:
            continue
        ax   = axes[i]
        vals = test_df[metric].values
        bars = ax.bar(range(n), vals, color=colors[:n], edgecolor="white", width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:,.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
        ax.set_xticks(range(n))
        ax.set_xticklabels(
            [m.replace(" — Test", "").replace(" (", "\n(")
             for m in test_df["Model"].values],
            fontsize=8,
        )
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1)

    plt.suptitle("MODEL COMPARISON — Test Set", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_metrics.png")
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    logger.info(f"✓ Saved → {path}")


# ═══════════════════════════════════════════════════════════
# 5. PER-BRAND METRICS
# ═══════════════════════════════════════════════════════════
def per_brand_metrics(test_df: pd.DataFrame) -> pd.DataFrame:
    """Chi tiết metrics theo từng brand trên test set."""
    rows = []
    for brand in sorted(test_df["BRAND"].unique()):
        bd = test_df[test_df["BRAND"] == brand]
        act  = bd[TARGET_COL].values
        pred = bd["final_pred"].values
        pp   = bd["prophet_pred"].values
        rows.append({
            "Brand":            brand,
            "N_rows":           len(bd),
            "Actual_Sum":       int(act.sum()),
            "Hybrid_RMSE":      calc_rmse(act,  pred),
            "Hybrid_MAE":       calc_mae(act,   pred),
            "Hybrid_WMAPE":     calc_wmape(act, pred),
            "Hybrid_sMAPE":     calc_smape(act, pred),
            "Hybrid_Bias":      calc_bias(act,  pred),
            "Prophet_RMSE":     calc_rmse(act,  pp),
            "Prophet_MAE":      calc_mae(act,   pp),
            "Prophet_WMAPE":    calc_wmape(act, pp),
            "LGBM_Improvement": calc_wmape(act, pp) - calc_wmape(act, pred),
        })
    df = pd.DataFrame(rows).sort_values("Hybrid_WMAPE")
    return df


# ═══════════════════════════════════════════════════════════
# 6. FEATURE IMPORTANCE PLOT
# ═══════════════════════════════════════════════════════════
def plot_feature_importance(save_dir: str | None = None, top_n: int = 20) -> None:
    """Vẽ feature importance từ các model LightGBM đã lưu."""
    import joblib
    if save_dir is None:
        save_dir = PATHS["figures"]

    model_dir = PATHS["models"]
    cluster_names = {0: "Cluster 0 (Stable)", 1: "Cluster 1 (Regular)", 2: "Cluster 2 (Seasonal)"}
    found = False

    for cid in range(3):
        # Two-Part Model (Cluster 2) lưu regressor riêng
        regressor_path = os.path.join(model_dir, f"lgbm_regressor_cluster_{cid}.pkl")
        model_path     = os.path.join(model_dir, f"lgbm_cluster_{cid}.pkl")
        if os.path.exists(regressor_path):
            model = joblib.load(regressor_path)
        elif os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            continue
        fi = pd.Series(model.feature_importances_, index=model.feature_name_).nlargest(top_n)
        found = True

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi)))
        fi.sort_values().plot(kind="barh", ax=ax, color=colors)
        ax.set_title(f"Feature Importance — {cluster_names.get(cid, f'Cluster {cid}')}", fontweight="bold")
        ax.set_xlabel("Importance (Split)")
        plt.tight_layout()
        path = os.path.join(save_dir, f"feature_importance_cluster_{cid}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
        logger.info(f"✓ Saved → {path}")

    if not found:
        logger.warning("Không tìm thấy model LightGBM. Chạy Step 4 trước!")


# ═══════════════════════════════════════════════════════════
# ACF / PACF ANALYSIS
# ═══════════════════════════════════════════════════════════
def plot_acf_pacf(
    train_df: pd.DataFrame,
    save_dir: str | None = None,
    n_lags: int = 60,
) -> None:
    """
    Vẽ ACF và PACF cho 3 brand đại diện (1 per cluster).

    - ACF  (Autocorrelation Function): tương quan giữa chuỗi và phiên bản
      trễ của chính nó → cho thấy lag nào có liên hệ tuyến tính tổng thể.
    - PACF (Partial ACF): tương quan trực tiếp ở lag k sau khi loại bỏ
      ảnh hưởng của các lag trung gian → giúp xác định bậc AR.

    Brand đại diện:
      Cluster 0 (Stable)   → KINH DO BREAD
      Cluster 1 (Regular)  → AFC
      Cluster 2 (Seasonal) → THU
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    if save_dir is None:
        save_dir = PATHS["figures"]

    # Brand đại diện — 1 per cluster
    rep_brands = {
        "KINH DO BREAD": "Cluster 0 — Stable",
        "AFC":           "Cluster 1 — Regular",
        "THU":           "Cluster 2 — Seasonal",
    }

    fig, axes = plt.subplots(
        nrows=len(rep_brands), ncols=2,
        figsize=(16, 5 * len(rep_brands)),
    )
    fig.suptitle(
        "ACF & PACF — Brand Đại Diện Theo Cụm\n"
        "(Dữ liệu Train | Daily Total QTY)",
        fontsize=15, fontweight="bold", y=1.01,
    )

    for row, (brand, label) in enumerate(rep_brands.items()):
        series = (
            train_df[train_df["BRAND"] == brand]
            .groupby(DATE_COL)[TARGET_COL]
            .sum()
            .sort_index()
            .asfreq("D", fill_value=0)
        )

        if len(series) < n_lags + 10:
            logger.warning(f"ACF/PACF: {brand} không đủ dữ liệu ({len(series)} rows).")
            continue

        # ACF
        plot_acf(
            series, lags=n_lags, ax=axes[row, 0],
            alpha=0.05, color="#2196F3", zero=False,
        )
        axes[row, 0].set_title(f"ACF — {brand}\n({label})", fontweight="bold")
        axes[row, 0].set_xlabel("Lag (ngày)")
        axes[row, 0].set_ylabel("Tương quan")
        # Đánh dấu lag đặc biệt
        for lag_mark in [7, 14, 30, 60]:
            if lag_mark <= n_lags:
                axes[row, 0].axvline(lag_mark, color="red", linestyle="--",
                                     alpha=0.4, linewidth=0.8)

        # PACF
        plot_pacf(
            series, lags=n_lags, ax=axes[row, 1],
            alpha=0.05, color="#4CAF50", zero=False, method="ywm",
        )
        axes[row, 1].set_title(f"PACF — {brand}\n({label})", fontweight="bold")
        axes[row, 1].set_xlabel("Lag (ngày)")
        axes[row, 1].set_ylabel("Tương quan riêng phần")
        for lag_mark in [7, 14, 30, 60]:
            if lag_mark <= n_lags:
                axes[row, 1].axvline(lag_mark, color="red", linestyle="--",
                                     alpha=0.4, linewidth=0.8)

    # Chú thích đường đỏ
    fig.text(
        0.5, -0.01,
        "Đường đỏ đứt: lag 7, 14, 30, 60 ngày  |  Vùng xanh: khoảng tin cậy 95%",
        ha="center", fontsize=10, color="gray",
    )

    plt.tight_layout()
    path = os.path.join(save_dir, "acf_pacf_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ ACF/PACF → {path}")


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════
def evaluate(
    test_df:  pd.DataFrame | None = None,
    train_df: pd.DataFrame | None = None,
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
            raise FileNotFoundError(
                "Chưa có test_predictions.csv. Chạy lightgbm_model trước!"
            )
        test_df  = pd.read_csv(pred_path,       parse_dates=[DATE_COL])
        train_df = pd.read_csv(train_pred_path, parse_dates=[DATE_COL]) \
            if os.path.exists(train_pred_path) else None

    # ── Export clean prediction DataFrame ───────────────────
    pred_cols = [DATE_COL, "BRAND", "CATEGORY", TARGET_COL, "prophet_pred", "final_pred"]
    export_cols = [c for c in pred_cols if c in test_df.columns]
    pred_export = test_df[export_cols].copy()
    pred_export.columns = [
        c if c not in ("prophet_pred", "final_pred")
        else ("Prophet_Pred" if c == "prophet_pred" else "Hybrid_Pred")
        for c in pred_export.columns
    ]
    pred_export["Error"] = pred_export["Hybrid_Pred"] - pred_export[TARGET_COL]
    pred_export["AbsError"] = pred_export["Error"].abs()
    pred_export_path = os.path.join(PATHS["metrics"], "predictions_clean.csv")
    pred_export.to_csv(pred_export_path, index=False)
    logger.info(f"✓ Prediction DataFrame → {pred_export_path}  ({len(pred_export):,} rows)")

    logger.info("[1/5] ARIMA baseline...")
    arima_result = run_arima_baseline(train_df, test_df) if train_df is not None else None

    logger.info("[2/5] Comparison table (Train + Test)...")
    comparison_df = create_comparison_table(test_df, train_df, arima_result)
    comparison_df.to_csv(
        os.path.join(PATHS["metrics"], "comparison_metrics.csv"), index=False
    )

    logger.info("[3/5] Forecast vs Actual plots...")
    plot_forecast_vs_actual(test_df)
    plot_forecast_per_brand(test_df)

    logger.info("[4/5] Residual diagnostics...")
    plot_residual_diagnostics(test_df)

    logger.info("[5/6] Comparison bars, per-brand metrics & feature importance...")
    plot_comparison_bars(comparison_df)
    brand_df = per_brand_metrics(test_df)
    brand_df.to_csv(
        os.path.join(PATHS["metrics"], "per_brand_metrics.csv"), index=False
    )
    logger.info(f"\n{brand_df.to_string(index=False)}")
    plot_feature_importance()

    logger.info("[6/6] ACF/PACF analysis (EDA bo sung)...")
    if train_df is not None:
        plot_acf_pacf(train_df)
    else:
        logger.warning("Khong co train_df → bo qua ACF/PACF.")

    logger.info("✓ HOÀN TẤT ĐÁNH GIÁ")
    return comparison_df, brand_df


def main():
    evaluate()


if __name__ == "__main__":
    main()
