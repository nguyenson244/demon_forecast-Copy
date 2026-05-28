"""
============================================================
prophet_model.py — Prophet Stage 1
============================================================
Role:
  1. Fit Prophet per Brand trên tập Train.
  2. Predict Test (out-of-sample → dùng làm baseline hybrid).
  3. Tạo OOS calibration predictions trong Train:
       - Fit Prophet trên PHẦN ĐẦU của Train (1 - calibration_ratio)
       - Predict PHẦN CUỐI của Train → OOS residual thật
       - LightGBM Stage 2 chỉ train trên phần cuối này

CHỐNG RÒ RỈ:
  - Prophet chỉ fit trên Train set (không thấy Test)
  - OOS residual: Prophet không thấy phần calibration khi fit
  - seasonality_mode = "additive" (phù hợp zero-inflated FMCG)
============================================================
"""

from __future__ import annotations

import logging
import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

from src.utils.logger import get_logger
from src.utils.config_loader import (
    CONF, PATHS, DATE_COL, TARGET_COL, SPLIT_DATE,
    TET_DATES, MID_AUTUMN_DATES, ensure_dirs,
)

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. CẤU HÌNH NGÀY LỄ
# ═══════════════════════════════════════════════════════════
def get_vietnamese_holidays() -> pd.DataFrame:
    """Tạo DataFrame holidays cho Prophet với cửa sổ ảnh hưởng."""
    rows = []

    for year, date in TET_DATES.items():
        rows.append({
            "holiday":      "tet_nguyen_dan",
            "ds":           pd.Timestamp(date),
            "lower_window": -20,
            "upper_window": 2,
        })

    for year, date in MID_AUTUMN_DATES.items():
        rows.append({
            "holiday":      "trung_thu",
            "ds":           pd.Timestamp(date),
            "lower_window": -60,
            "upper_window": 1,
        })

    for year in range(2023, 2027):
        rows.append({
            "holiday":      "quoc_khanh",
            "ds":           pd.Timestamp(f"{year}-09-02"),
            "lower_window": -1,
            "upper_window": 1,
        })
        rows.append({
            "holiday":      "giai_phong_lao_dong",
            "ds":           pd.Timestamp(f"{year}-04-30"),
            "lower_window": -1,
            "upper_window": 2,
        })

    gio_to = {
        2023: "2023-04-29", 2024: "2024-04-18",
        2025: "2025-04-07", 2026: "2026-04-26",
    }
    for _, d in gio_to.items():
        rows.append({
            "holiday":      "gio_to_hung_vuong",
            "ds":           pd.Timestamp(d),
            "lower_window": -1,
            "upper_window": 0,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# 2. XÂY DỰNG PROPHET MODEL
# ═══════════════════════════════════════════════════════════
def _build_prophet(holidays_df: pd.DataFrame):
    """
    Khởi tạo Prophet với cấu hình phù hợp FMCG zero-inflated.

    seasonality_mode = "additive":
      - Đúng cho data có nhiều zero (multiplicative sẽ sinh NaN khi y=0)
      - Seasonal effect cộng thêm vào trend thay vì nhân
    """
    from prophet import Prophet
    return Prophet(
        growth="linear",
        seasonality_mode="additive",       # ← fix: additive cho zero-inflated
        holidays=holidays_df,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
    )


# ═══════════════════════════════════════════════════════════
# 3. FIT + PREDICT CHO 1 BRAND
# ═══════════════════════════════════════════════════════════
def _fit_predict_brand(
    train_ts: pd.DataFrame,
    predict_dates: pd.DataFrame,
    brand_name: str,
    label: str = "",
):
    """
    Fit Prophet trên train_ts, predict predict_dates.

    Parameters
    ----------
    train_ts       : DataFrame với cột 'ds', 'y'
    predict_dates  : DataFrame với cột 'ds'
    brand_name     : tên brand (cho logging)
    label          : nhãn mô tả fit này (e.g. "full", "warmup")

    Returns
    -------
    forecast DataFrame với cột ds, yhat, yhat_lower, yhat_upper
    hoặc None nếu không đủ dữ liệu
    """
    if len(train_ts) < 30:
        logger.warning(f"{brand_name} [{label}]: chỉ {len(train_ts)} ngày → SKIP")
        return None

    holidays_df = get_vietnamese_holidays()
    model = _build_prophet(holidays_df)
    model.fit(train_ts)

    forecast = model.predict(predict_dates)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# ═══════════════════════════════════════════════════════════
# 4. PHÂN PHỐI PREDICTIONS VỀ CẤP ROW
# ═══════════════════════════════════════════════════════════
def _distribute_to_rows(
    df: pd.DataFrame,
    brand_name: str,
    forecast: pd.DataFrame | None,
    pred_col: str,
    residual_col: str | None,
    train_rows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Phân phối prophet yhat về từng (date, brand, category) row
    theo tỷ lệ category được tính trên train data.

    Lý do dùng ratio cố định: tránh dùng future data khi phân phối.
    Category ratio được fit-only-on-train.
    """
    brand_mask = df["BRAND"] == brand_name

    if forecast is None:
        df.loc[brand_mask, pred_col] = np.nan
        if residual_col:
            df.loc[brand_mask, residual_col] = 0.0
        return df

    # Tính ratio từ train data
    daily_brand = train_rows.groupby(DATE_COL)[TARGET_COL].sum()
    categories  = df.loc[brand_mask, "CATEGORY"].unique()

    if len(categories) == 1:
        cat_ratios = {categories[0]: 1.0}
    else:
        raw = {}
        for cat in categories:
            cat_daily = (
                train_rows[train_rows["CATEGORY"] == cat]
                .set_index(DATE_COL)[TARGET_COL]
            )
            ratio = (cat_daily / daily_brand.reindex(cat_daily.index)).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0)
            raw[cat] = float(ratio.mean())

        total = sum(raw.values())
        cat_ratios = {k: v / total for k, v in raw.items()} if total > 0 else {
            cat: 1.0 / len(categories) for cat in categories
        }

    # Merge predictions per category
    fc_map = forecast.set_index("ds")["yhat"]
    for cat, ratio in cat_ratios.items():
        mask = brand_mask & (df["CATEGORY"] == cat)
        dates = df.loc[mask, DATE_COL]
        yhat  = dates.map(fc_map).values * ratio
        df.loc[mask, pred_col] = yhat
        if residual_col:
            df.loc[mask, residual_col] = df.loc[mask, TARGET_COL].values - yhat

    return df


# ═══════════════════════════════════════════════════════════
# 5. TÍNH OOS RESIDUALS CHO LIGHTGBM CALIBRATION
# ═══════════════════════════════════════════════════════════
def _add_oos_residuals(
    full_train_df: pd.DataFrame,
    calibration_ratio: float = 0.25,
) -> pd.DataFrame:
    """
    Tạo cột oos_prophet_pred và use_for_lgbm_train trên tập Train.

    Logic:
      - Với mỗi brand, chia train thành warmup (75%) và calibration (25%)
      - Fit Prophet trên warmup, predict calibration → OOS predictions
      - Rows trong calibration: use_for_lgbm_train = True
      - Rows trong warmup:      use_for_lgbm_train = False (NaN oos_prophet_pred)

    Tại sao đúng:
      - LightGBM học residual từ OOS predictions → không học near-zero in-sample errors
      - Prophet không thấy calibration data khi fit → true OOS
    """
    df = full_train_df.copy()
    df["oos_prophet_pred"]  = np.nan
    df["use_for_lgbm_train"] = False

    holidays_df = get_vietnamese_holidays()
    brands = sorted(df["BRAND"].unique())

    for brand in brands:
        brand_mask = df["BRAND"] == brand
        brand_rows = df[brand_mask].copy()

        # Aggregate theo ngày (Prophet cần daily brand-level)
        daily = (
            brand_rows.groupby(DATE_COL)[TARGET_COL].sum()
            .reset_index()
            .rename(columns={DATE_COL: "ds", TARGET_COL: "y"})
            .sort_values("ds")
        )

        n_total   = len(daily)
        n_warmup  = max(30, int(n_total * (1 - calibration_ratio)))
        warmup_ts = daily.iloc[:n_warmup]
        calib_ts  = daily.iloc[n_warmup:]

        if len(warmup_ts) < 30 or len(calib_ts) < 7:
            logger.warning(
                f"{brand}: warmup={len(warmup_ts)}, calib={len(calib_ts)} "
                f"→ không đủ để tạo OOS residuals."
            )
            continue

        forecast = _fit_predict_brand(
            warmup_ts,
            calib_ts[["ds"]],
            brand,
            label="oos-warmup",
        )
        if forecast is None:
            continue

        # Tính category ratios từ warmup period only
        warmup_cutoff = warmup_ts["ds"].max()
        warmup_rows   = brand_rows[brand_rows[DATE_COL] <= warmup_cutoff]

        calib_dates  = set(calib_ts["ds"].tolist())
        calib_mask   = brand_mask & (df[DATE_COL].isin(calib_dates))

        df = _distribute_to_rows(
            df,
            brand,
            forecast,
            pred_col="oos_prophet_pred",
            residual_col=None,
            train_rows=warmup_rows,
        )

        # Chỉ đánh dấu calibration rows (không đánh warmup rows)
        df.loc[calib_mask, "use_for_lgbm_train"] = True
        # Warmup rows không có oos_prophet_pred → set lại về NaN
        warmup_rows_mask = brand_mask & (~df[DATE_COL].isin(calib_dates))
        df.loc[warmup_rows_mask, "oos_prophet_pred"] = np.nan

        n_calib = calib_mask.sum()
        logger.debug(
            f"  {brand}: warmup={len(warmup_ts)}d, "
            f"calib={len(calib_ts)}d, "
            f"rows_marked={n_calib}"
        )

    n_lgbm = int(df["use_for_lgbm_train"].sum())
    logger.info(
        f"OOS calibration: {n_lgbm:,}/{len(df):,} rows "
        f"({n_lgbm/len(df)*100:.1f}%) sẵn sàng cho LightGBM training."
    )
    return df


# ═══════════════════════════════════════════════════════════
# 6. PUBLIC API
# ═══════════════════════════════════════════════════════════
def run_prophet(
    train_df: pd.DataFrame | None = None,
    test_df:  pd.DataFrame | None = None,
    calibration_ratio: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Chạy Prophet Stage 1.

    Bước thực hiện:
      A. Fit Prophet FULL trên toàn bộ Train per Brand → predict Test
      B. Tạo OOS residuals trong Train cho LightGBM calibration

    Returns
    -------
    (full_df_with_residuals, prophet_models_dict)
    full_df = train + test, có các cột:
      - prophet_pred       : predictions từ full-train model
      - residual_prophet   : actual - prophet_pred
      - oos_prophet_pred   : OOS predictions (chỉ có trong calibration rows)
      - use_for_lgbm_train : True cho calibration rows
    """
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("BƯỚC 3 — PROPHET STAGE 1 (PER BRAND)")
    logger.info("=" * 60)

    if calibration_ratio is None:
        calibration_ratio = getattr(CONF, "lgbm_calibration_ratio", 0.25)

    # ── Load data nếu chưa có ───────────────────────────────
    if train_df is None:
        train_path = os.path.join(PATHS["features"], "train_features.csv")
        test_path  = os.path.join(PATHS["features"], "test_features.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Chưa có train_features.csv tại {train_path}.")
        train_df = pd.read_csv(train_path, parse_dates=[DATE_COL])
        test_df  = pd.read_csv(test_path,  parse_dates=[DATE_COL])

    logger.info(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    # ── Khởi tạo columns ────────────────────────────────────
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df["prophet_pred"]      = np.nan
    full_df["residual_prophet"]  = np.nan
    full_df["oos_prophet_pred"]  = np.nan
    full_df["use_for_lgbm_train"] = False

    brands         = sorted(full_df["BRAND"].unique())
    prophet_models = {}

    # ── A. Fit FULL Prophet per Brand → predict Train + Test ─
    logger.info(f"Fitting full Prophet for {len(brands)} brands...")
    for brand in brands:
        brand_train = train_df[train_df["BRAND"] == brand]
        brand_test  = test_df[test_df["BRAND"] == brand]

        # Aggregate daily cho Prophet
        train_ts = (
            brand_train.groupby(DATE_COL)[TARGET_COL].sum()
            .reset_index()
            .rename(columns={DATE_COL: "ds", TARGET_COL: "y"})
            .sort_values("ds")
        )
        test_ts = (
            brand_test.groupby(DATE_COL)[TARGET_COL].sum()
            .reset_index()
            .rename(columns={DATE_COL: "ds", TARGET_COL: "y"})
            .sort_values("ds")
        )

        all_dates = (
            pd.concat([train_ts[["ds"]], test_ts[["ds"]]])
            .drop_duplicates()
            .sort_values("ds")
        )

        forecast = _fit_predict_brand(train_ts, all_dates, brand, label="full")

        if forecast is not None:
            prophet_models[brand] = True  # model đã fit (lưu pkl riêng nếu cần)
            full_df = _distribute_to_rows(
                full_df, brand, forecast,
                pred_col="prophet_pred",
                residual_col="residual_prophet",
                train_rows=brand_train,
            )

            rmse_train = float(np.sqrt(np.mean(
                (brand_train[TARGET_COL].values
                 - full_df.loc[
                     (full_df["BRAND"] == brand) &
                     (full_df[DATE_COL] <= SPLIT_DATE),
                     "prophet_pred"
                 ].fillna(0).values[:len(brand_train)]) ** 2
            )))
            logger.info(
                f"  ✓ {brand:20s}: "
                f"Train={len(train_ts)}d Test={len(test_ts)}d "
                f"Train-RMSE={rmse_train:,.1f}"
            )
        else:
            prophet_models[brand] = None

    # ── B. Tạo OOS residuals trong Train ────────────────────
    logger.info(f"\nTạo OOS residuals (calibration_ratio={calibration_ratio:.0%})...")
    train_with_oos = _add_oos_residuals(
        full_df[full_df[DATE_COL] <= SPLIT_DATE].copy(),
        calibration_ratio=calibration_ratio,
    )

    # Merge oos columns về full_df
    oos_cols = ["oos_prophet_pred", "use_for_lgbm_train"]
    idx_col  = full_df.index

    train_mask_full = full_df[DATE_COL] <= SPLIT_DATE
    full_df.loc[train_mask_full, oos_cols] = (
        train_with_oos[oos_cols].values
    )

    # ── Lưu kết quả ─────────────────────────────────────────
    train_res = full_df[full_df[DATE_COL] <= SPLIT_DATE].copy()
    test_res  = full_df[full_df[DATE_COL] >  SPLIT_DATE].copy()

    train_res_path = os.path.join(PATHS["features"], "train_with_residuals.csv")
    test_res_path  = os.path.join(PATHS["features"], "test_with_residuals.csv")
    models_path    = os.path.join(PATHS["models"],   "prophet_models.pkl")

    train_res.to_csv(train_res_path, index=False)
    test_res.to_csv(test_res_path,   index=False)
    joblib.dump(prophet_models, models_path)

    n_trained = sum(1 for v in prophet_models.values() if v is not None)
    logger.info(f"\n✓ HOÀN TẤT PROPHET STAGE 1")
    logger.info(f"  Trained: {n_trained}/{len(brands)} brands")
    logger.info(f"  Saved → {train_res_path}")
    logger.info(f"  Saved → {test_res_path}")

    return full_df, prophet_models


def main():
    run_prophet()


if __name__ == "__main__":
    main()
