"""
============================================================
prophet_model.py — Prophet Stage 1
============================================================
Migrate từ: src/model_prophet.py
Role      : Huấn luyện Prophet ĐỘC LẬP cho từng BRAND,
            trích xuất Residuals R(t) = Y(t) - Ŷ_Prophet(t).

★ CHỐNG RÒ RỈ: Prophet chỉ fit trên Train set.
============================================================
"""

from __future__ import annotations

import os
import warnings
import logging

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

from src.utils.logger import get_logger
from src.utils.config_loader import (
    PATHS, DATE_COL, TARGET_COL, SPLIT_DATE,
    TET_DATES, MID_AUTUMN_DATES, ensure_dirs,
)

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. CẤU HÌNH NGÀY LỄ CHO PROPHET
# ═══════════════════════════════════════════════════════════
def get_vietnamese_holidays() -> pd.DataFrame:
    """Tạo DataFrame holidays cho Prophet với cửa sổ ảnh hưởng."""
    holidays_list = []

    for year, date in TET_DATES.items():
        holidays_list.append({
            "holiday": "tet_nguyen_dan",
            "ds": date,
            "lower_window": -20,
            "upper_window": 2,
        })

    for year, date in MID_AUTUMN_DATES.items():
        holidays_list.append({
            "holiday": "trung_thu",
            "ds": date,
            "lower_window": -60,
            "upper_window": 1,
        })

    for year in range(2023, 2027):
        holidays_list.append({
            "holiday": "quoc_khanh",
            "ds": pd.Timestamp(f"{year}-09-02"),
            "lower_window": -1,
            "upper_window": 1,
        })
        holidays_list.append({
            "holiday": "giai_phong_lao_dong",
            "ds": pd.Timestamp(f"{year}-04-30"),
            "lower_window": -1,
            "upper_window": 2,
        })

    gio_to_dates = {
        2023: "2023-04-29",
        2024: "2024-04-18",
        2025: "2025-04-07",
        2026: "2026-04-26",
    }
    for year, date_str in gio_to_dates.items():
        holidays_list.append({
            "holiday": "gio_to_hung_vuong",
            "ds": pd.Timestamp(date_str),
            "lower_window": -1,
            "upper_window": 0,
        })

    return pd.DataFrame(holidays_list)


# ═══════════════════════════════════════════════════════════
# 2. HUẤN LUYỆN PROPHET CHO 1 BRAND
# ═══════════════════════════════════════════════════════════
def train_prophet_for_brand(
    train_brand: pd.DataFrame,
    test_brand:  pd.DataFrame,
    brand_name:  str,
) -> tuple:
    """
    Huấn luyện Prophet cho 1 brand.

    Returns
    -------
    (model, train_pred, test_pred)  or  (None, None, None) nếu skip.
    """
    from prophet import Prophet

    train_ts = train_brand.groupby(DATE_COL)[TARGET_COL].sum().reset_index()
    train_ts.columns = ["ds", "y"]
    train_ts = train_ts.sort_values("ds").reset_index(drop=True)

    test_ts = test_brand.groupby(DATE_COL)[TARGET_COL].sum().reset_index()
    test_ts.columns = ["ds", "y"]
    test_ts = test_ts.sort_values("ds").reset_index(drop=True)

    if len(train_ts) < 30:
        logger.warning(f"{brand_name}: Chỉ có {len(train_ts)} ngày train → SKIP")
        return None, None, None

    holidays_df = get_vietnamese_holidays()

    model = Prophet(
        growth="linear",
        seasonality_mode="multiplicative",
        holidays=holidays_df,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
    )
    model.fit(train_ts)

    all_dates = (
        pd.concat([train_ts[["ds"]], test_ts[["ds"]]])
        .drop_duplicates()
        .sort_values("ds")
        .reset_index(drop=True)
    )
    forecast = model.predict(all_dates)

    train_pred = forecast[forecast["ds"].isin(train_ts["ds"])][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()
    test_pred = forecast[forecast["ds"].isin(test_ts["ds"])][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()

    train_pred = train_pred.merge(train_ts, on="ds", how="inner")
    train_pred["residual"] = train_pred["y"] - train_pred["yhat"]

    test_pred = test_pred.merge(test_ts, on="ds", how="inner")
    test_pred["residual"] = test_pred["y"] - test_pred["yhat"]

    train_rmse = np.sqrt(np.mean(train_pred["residual"] ** 2))
    logger.info(
        f"  ✓ {brand_name:20s}: Train days={len(train_ts):>4d} | "
        f"Test days={len(test_ts):>4d} | Train RMSE={train_rmse:>12,.1f}"
    )
    return model, train_pred, test_pred


# ═══════════════════════════════════════════════════════════
# 3. PHÂN PHỐI PREDICTIONS VỀ CẤP (DATE, BRAND, CAT)
# ═══════════════════════════════════════════════════════════
def distribute_predictions_to_rows(
    df: pd.DataFrame,
    brand_pred,
    brand_name: str,
    split_date: pd.Timestamp,
) -> pd.DataFrame:
    """Phân phối prophet predictions về từng row theo tỷ lệ category."""
    brand_rows = df[df["BRAND"] == brand_name].copy()

    if brand_pred is None:
        df.loc[brand_rows.index, "prophet_pred"] = np.nan
        df.loc[brand_rows.index, "residual_prophet"] = 0
        return df

    train_rows       = brand_rows[brand_rows[DATE_COL] <= split_date]
    daily_brand_total = train_rows.groupby(DATE_COL)[TARGET_COL].sum()
    categories        = brand_rows["CATEGORY"].unique()

    if len(categories) == 1:
        cat_ratios = {categories[0]: 1.0}
    else:
        cat_ratios = {}
        for cat in categories:
            cat_daily    = train_rows[train_rows["CATEGORY"] == cat].set_index(DATE_COL)[TARGET_COL]
            ratio_series = cat_daily / daily_brand_total.reindex(cat_daily.index)
            ratio_series = ratio_series.replace([np.inf, -np.inf], np.nan).fillna(0)
            cat_ratios[cat] = ratio_series.mean()

        total_ratio = sum(cat_ratios.values())
        if total_ratio > 0:
            cat_ratios = {k: v / total_ratio for k, v in cat_ratios.items()}

    all_pred = (
        pd.concat(brand_pred, ignore_index=True)
        if isinstance(brand_pred, list)
        else brand_pred
    )

    for cat, ratio in cat_ratios.items():
        mask = (df["BRAND"] == brand_name) & (df["CATEGORY"] == cat)
        cat_rows = df[mask].copy()
        merged = cat_rows[[DATE_COL]].merge(
            all_pred[["ds", "yhat"]].rename(columns={"ds": DATE_COL}),
            on=DATE_COL,
            how="left",
        )
        df.loc[mask, "prophet_pred"]     = merged["yhat"].values * ratio
        df.loc[mask, "residual_prophet"] = (
            df.loc[mask, TARGET_COL].values - df.loc[mask, "prophet_pred"].values
        )

    return df


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════
def run_prophet(
    train_df: pd.DataFrame | None = None,
    test_df:  pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Chạy Prophet Stage 1 — vòng lặp cho từng BRAND.

    Returns
    -------
    (full_df_with_residuals, prophet_models_dict)
    """
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("BƯỚC 3 — GIAI ĐOẠN 1: PROPHET PER BRAND")
    logger.info("=" * 60)

    if train_df is None:
        train_path = os.path.join(PATHS["features"], "train_features.csv")
        test_path  = os.path.join(PATHS["features"], "test_features.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Chưa có train_features.csv tại {train_path}.")
        train_df = pd.read_csv(train_path, parse_dates=[DATE_COL])
        test_df  = pd.read_csv(test_path,  parse_dates=[DATE_COL])

    logger.info(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df["prophet_pred"]     = np.nan
    full_df["residual_prophet"] = np.nan

    brands         = sorted(full_df["BRAND"].unique())
    prophet_models = {}

    for brand in brands:
        brand_train = train_df[train_df["BRAND"] == brand]
        brand_test  = test_df[test_df["BRAND"] == brand]

        model, train_pred, test_pred = train_prophet_for_brand(
            brand_train, brand_test, brand
        )
        prophet_models[brand] = model

        if model is not None:
            combined_pred = pd.concat([train_pred, test_pred], ignore_index=True)
            full_df = distribute_predictions_to_rows(full_df, combined_pred, brand, SPLIT_DATE)

    train_res = full_df[full_df[DATE_COL] <= SPLIT_DATE].copy()
    test_res  = full_df[full_df[DATE_COL] >  SPLIT_DATE].copy()

    train_res_path = os.path.join(PATHS["features"], "train_with_residuals.csv")
    test_res_path  = os.path.join(PATHS["features"], "test_with_residuals.csv")
    models_path    = os.path.join(PATHS["models"],   "prophet_models.pkl")

    train_res.to_csv(train_res_path, index=False)
    test_res.to_csv(test_res_path,   index=False)
    joblib.dump(prophet_models, models_path)

    logger.info(f"✓ HOÀN TẤT PROPHET STAGE 1")
    logger.info(f"  Trained: {sum(1 for m in prophet_models.values() if m is not None)}/{len(brands)} brands")
    logger.info(f"  Saved → {train_res_path}")
    logger.info(f"  Saved → {test_res_path}")
    logger.info(f"  Saved → {models_path}")

    return full_df, prophet_models


def main():
    run_prophet()


if __name__ == "__main__":
    main()
