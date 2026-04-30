"""
============================================================
feature_engineering.py — Feature Engineering Pipeline
============================================================
Migrate từ: src/feature_eng.py
Tạo 80+ features:
  - Lag features (19)           — Trí nhớ ngắn & dài hạn
  - Rolling window stats (20)   — Động lượng & xu hướng
  - Momentum/Volatility (6)     — Tốc độ tăng trưởng
  - Time features (14)          — Lượng giác & chỉ báo thời gian
  - Tết Nguyên Đán (5)          — Đếm ngược & 3 pha
  - Trung Thu (3)               — Đếm ngược mùa bánh
  - Holiday features (3)        — Lễ quốc gia
  - Brand statistics (7)        — Fit on TRAIN only
  - Categorical/Transform (2)   — is_seasonal, cbm_log

★ QUY TẮC CHỐNG RÒ RỈ DỮ LIỆU:
  - Lag & Rolling: dùng .shift() nghiêm ngặt
  - Brand stats: .fit() CHỈ trên Train
  - Split: theo trục thời gian, KHÔNG random
============================================================
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.utils.logger import get_logger
from src.utils.config_loader import (
    PATHS, DATE_COL, TARGET_COL,
    TET_DATES, MID_AUTUMN_DATES, SPLIT_DATE,
    LAG_PERIODS, SAME_WEEKDAY_WEEKS, ROLLING_WINDOWS,
    ensure_dirs,
)

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. LAG FEATURES
# ═══════════════════════════════════════════════════════════
def create_lag_features(group: pd.DataFrame) -> pd.DataFrame:
    """Tạo lag features cho 1 chuỗi thời gian (brand)."""
    target = group[TARGET_COL]
    for lag in LAG_PERIODS:
        group[f"lag_{lag}"] = target.shift(lag)
    for n_weeks in SAME_WEEKDAY_WEEKS:
        group[f"lag_{n_weeks}w_weekday"] = target.shift(7 * n_weeks)
    return group


# ═══════════════════════════════════════════════════════════
# 2. ROLLING WINDOW STATISTICS
# ═══════════════════════════════════════════════════════════
def create_rolling_features(group: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean, std, median, min, max. shift(1) chống leakage."""
    shifted = group[TARGET_COL].shift(1)
    for w in ROLLING_WINDOWS:
        group[f"roll_mean_{w}"]   = shifted.rolling(w, min_periods=1).mean()
        group[f"roll_std_{w}"]    = shifted.rolling(w, min_periods=1).std()
        group[f"roll_median_{w}"] = shifted.rolling(w, min_periods=1).median()
        group[f"roll_min_{w}"]    = shifted.rolling(w, min_periods=1).min()
        group[f"roll_max_{w}"]    = shifted.rolling(w, min_periods=1).max()
    return group


# ═══════════════════════════════════════════════════════════
# 3. MOMENTUM & VOLATILITY
# ═══════════════════════════════════════════════════════════
def create_momentum_features(group: pd.DataFrame) -> pd.DataFrame:
    """Momentum, pct_change, coefficient of variation."""
    target = group[TARGET_COL]
    shifted = target.shift(1)

    group["momentum_7"]   = target.shift(1) - target.shift(8)
    group["momentum_30"]  = target.shift(1) - target.shift(31)
    group["pct_change_1"] = target.pct_change(1).shift(1)
    group["pct_change_7"] = target.pct_change(7).shift(1)

    roll_mean_14 = shifted.rolling(14, min_periods=2).mean()
    roll_std_14  = shifted.rolling(14, min_periods=2).std()
    group["roll_cv_14"] = roll_std_14 / (roll_mean_14 + 1e-8)

    roll_mean_30 = shifted.rolling(30, min_periods=2).mean()
    roll_std_30  = shifted.rolling(30, min_periods=2).std()
    group["roll_cv_30"] = roll_std_30 / (roll_mean_30 + 1e-8)

    return group


# ═══════════════════════════════════════════════════════════
# 4. TIME FEATURES
# ═══════════════════════════════════════════════════════════
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trích xuất đặc trưng thời gian và encoding lượng giác."""
    df = df.copy()
    dt = df[DATE_COL]

    df["year"]         = dt.dt.year
    df["month"]        = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["dayofweek"]    = dt.dt.dayofweek
    df["weekofyear"]   = dt.dt.isocalendar().week.astype(int)
    df["quarter"]      = dt.dt.quarter
    df["is_weekend"]   = (df["dayofweek"] >= 5).astype(int)
    df["is_monday"]    = (df["dayofweek"] == 0).astype(int)
    df["is_first_5days"] = (dt.dt.day <= 5).astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)

    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    return df


# ═══════════════════════════════════════════════════════════
# 5. TẾT NGUYÊN ĐÁN
# ═══════════════════════════════════════════════════════════
def create_tet_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đếm ngược đến Tết và 3 pha hành vi mua sắm."""
    df = df.copy()
    tet_array = np.array(sorted(TET_DATES.values()), dtype="datetime64[ns]")
    dates_np  = df[DATE_COL].values.astype("datetime64[ns]")

    days_to_tet = np.full(len(df), 999, dtype=int)
    for tet in tet_array:
        diff = (tet - dates_np).astype("timedelta64[D]").astype(int)
        mask = (diff >= 0) & (diff < days_to_tet)
        days_to_tet[mask] = diff[mask]

    df["days_to_tet"]    = days_to_tet
    df["is_tet_phase_1"] = ((df["days_to_tet"] >= 20) & (df["days_to_tet"] <= 40)).astype(int)
    df["is_tet_phase_2"] = ((df["days_to_tet"] >= 0)  & (df["days_to_tet"] < 20)).astype(int)

    days_after_tet = np.zeros(len(df), dtype=int)
    for tet in tet_array:
        diff   = (dates_np - tet).astype("timedelta64[D]").astype(int)
        mask   = (diff > 0) & (diff <= 30)
        better = mask & (diff > days_after_tet)
        update = mask & (days_after_tet == 0)
        days_after_tet[better | update] = diff[better | update]

    df["days_after_tet"] = days_after_tet
    df["is_post_tet"]    = (df["days_after_tet"] > 0).astype(int)
    return df


# ═══════════════════════════════════════════════════════════
# 6. TRUNG THU
# ═══════════════════════════════════════════════════════════
def create_mid_autumn_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng mùa bánh Trung Thu."""
    df = df.copy()
    ma_array = np.array(sorted(MID_AUTUMN_DATES.values()), dtype="datetime64[ns]")
    dates_np = df[DATE_COL].values.astype("datetime64[ns]")

    days_to_ma = np.full(len(df), 999, dtype=int)
    for ma in ma_array:
        diff = (ma - dates_np).astype("timedelta64[D]").astype(int)
        mask = (diff >= 0) & (diff < days_to_ma)
        days_to_ma[mask] = diff[mask]

    df["days_to_mid_autumn"]  = days_to_ma
    df["is_mid_autumn_season"] = ((df["days_to_mid_autumn"] >= 0) & (df["days_to_mid_autumn"] <= 75)).astype(int)
    df["is_mid_autumn_peak"]   = ((df["days_to_mid_autumn"] >= 0) & (df["days_to_mid_autumn"] <= 14)).astype(int)
    return df


# ═══════════════════════════════════════════════════════════
# 7. HOLIDAY FEATURES
# ═══════════════════════════════════════════════════════════
def create_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng lễ quốc gia Việt Nam."""
    import holidays as hol

    df = df.copy()
    years = list(range(df[DATE_COL].dt.year.min(), df[DATE_COL].dt.year.max() + 2))
    vn_holidays   = hol.Vietnam(years=years)
    holiday_dates = sorted([pd.Timestamp(d) for d in vn_holidays.keys()])
    holiday_np    = np.array(holiday_dates, dtype="datetime64[ns]")
    dates_np      = df[DATE_COL].values.astype("datetime64[ns]")

    df["is_holiday"] = df[DATE_COL].isin(set(holiday_dates)).astype(int)

    days_to_next   = np.full(len(df), 999, dtype=int)
    days_from_prev = np.full(len(df), 999, dtype=int)

    for h in holiday_np:
        diff      = (h - dates_np).astype("timedelta64[D]").astype(int)
        mask_next = (diff >= 0) & (diff < days_to_next)
        days_to_next[mask_next] = diff[mask_next]
        past_diff = -diff
        mask_prev = (diff <= 0) & (past_diff < days_from_prev)
        days_from_prev[mask_prev] = past_diff[mask_prev]

    df["days_to_next_holiday"]   = days_to_next
    df["days_from_prev_holiday"] = days_from_prev
    return df


# ═══════════════════════════════════════════════════════════
# 8. CATEGORICAL & TRANSFORM
# ═══════════════════════════════════════════════════════════
def create_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """is_seasonal_category và cbm_log."""
    df = df.copy()
    df["is_seasonal_category"] = df["CATEGORY"].isin(["TET", "MOONCAKE"]).astype(int)
    if "Total CBM" in df.columns:
        df["cbm_log"] = np.log1p(df["Total CBM"])
    return df


# ═══════════════════════════════════════════════════════════
# 9. BRAND STATISTICS — FIT CHỈ TRÊN TRAIN
# ═══════════════════════════════════════════════════════════
def create_brand_stats_features(
    df: pd.DataFrame,
    train_mask: pd.Series,
) -> pd.DataFrame:
    """Brand-level stats, fit chỉ trên Train để chống leakage."""
    df         = df.copy()
    train_data = df[train_mask]

    qty_stats = (
        train_data.groupby("BRAND", observed=True)[TARGET_COL]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    qty_stats.columns = ["BRAND", "brand_mean_qty", "brand_std_qty", "brand_median_qty"]

    cbm_stats = (
        train_data.groupby("BRAND", observed=True)["Total CBM"]
        .agg(["mean", "std"])
        .reset_index()
    ) if "Total CBM" in train_data.columns else None

    df = df.merge(qty_stats, on="BRAND", how="left")
    df["qty_zscore_brand"] = (
        (df[TARGET_COL] - df["brand_mean_qty"]) / (df["brand_std_qty"] + 1e-8)
    )

    if cbm_stats is not None:
        cbm_stats.columns = ["BRAND", "brand_mean_cbm", "brand_std_cbm"]
        df = df.merge(cbm_stats, on="BRAND", how="left")
        df["cbm_zscore_brand"] = (
            (df["Total CBM"] - df["brand_mean_cbm"]) / (df["brand_std_cbm"] + 1e-8)
        )

    return df


# ═══════════════════════════════════════════════════════════
# 10. COMPARISON FEATURES
# ═══════════════════════════════════════════════════════════
def create_comparison_features(group: pd.DataFrame) -> pd.DataFrame:
    """is_above_30d_mean."""
    shifted      = group[TARGET_COL].shift(1)
    roll_mean_30 = shifted.rolling(30, min_periods=1).mean()
    group["is_above_30d_mean"] = (shifted > roll_mean_30).astype(int)
    return group


# ═══════════════════════════════════════════════════════════
# 11. CHRONOLOGICAL TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════
def chronological_split(
    df: pd.DataFrame,
    split_date: pd.Timestamp = SPLIT_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tách Train/Test theo trình tự thời gian — KHÔNG random shuffle."""
    train = df[df[DATE_COL] <= split_date].copy()
    test  = df[df[DATE_COL] >  split_date].copy()

    logger.info(f"CHRONOLOGICAL SPLIT — Cutoff: {split_date.date()}")
    logger.info(
        f"  Train: {len(train):>8,} rows | "
        f"{train[DATE_COL].min().date()} → {train[DATE_COL].max().date()}"
    )
    logger.info(
        f"  Test:  {len(test):>8,} rows | "
        f"{test[DATE_COL].min().date()} → {test[DATE_COL].max().date()}"
    )

    max_train = train[DATE_COL].max()
    min_test  = test[DATE_COL].min()
    assert max_train < min_test, (
        f"⛔ DATA LEAKAGE: Train max={max_train} >= Test min={min_test}"
    )
    logger.info(f"✔ Sanity Check PASSED: max(Train)={max_train.date()} < min(Test)={min_test.date()}")
    return train, test


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════
def engineer(cleaned_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chạy toàn bộ Feature Engineering pipeline.

    Parameters
    ----------
    cleaned_df : pd.DataFrame | None
        Cleaned DataFrame từ data_cleaning.clean().
        Nếu None, tự đọc từ data/processed/cleaned_data.csv.

    Returns
    -------
    (full_df, train_df, test_df)
    """
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("BƯỚC 2: FEATURE ENGINEERING")
    logger.info("=" * 60)

    if cleaned_df is None:
        data_path = os.path.join(PATHS["processed"], "cleaned_data.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Chưa có cleaned_data.csv tại {data_path}. Chạy data_cleaning.py trước!"
            )
        cleaned_df = pd.read_csv(data_path, parse_dates=[DATE_COL])
        logger.info(f"Loaded cleaned data: {len(cleaned_df):,} rows")

    df = cleaned_df.sort_values(["BRAND", "CATEGORY", DATE_COL]).reset_index(drop=True)

    logger.info("[1/9] Time features...")
    df = create_time_features(df)

    logger.info("[2/9] Tết Nguyên Đán features...")
    df = create_tet_features(df)

    logger.info("[3/9] Trung Thu features...")
    df = create_mid_autumn_features(df)

    logger.info("[4/9] Holiday features...")
    df = create_holiday_features(df)

    logger.info("[5/9] Categorical & Transform features...")
    df = create_categorical_features(df)

    logger.info("[6-8/9] Lag / Rolling / Momentum features per BRAND...")
    brand_groups = []
    for brand in sorted(df["BRAND"].unique()):
        brand_data = df[df["BRAND"] == brand].copy().sort_values(DATE_COL)
        brand_data = create_lag_features(brand_data)
        brand_data = create_rolling_features(brand_data)
        brand_data = create_momentum_features(brand_data)
        brand_data = create_comparison_features(brand_data)
        brand_groups.append(brand_data)
        logger.debug(f"  ✓ {brand}: {len(brand_data):,} rows")

    df = pd.concat(brand_groups, ignore_index=True)
    df = df.sort_values([DATE_COL, "BRAND", "CATEGORY"]).reset_index(drop=True)

    logger.info("[9/9] Brand statistics (fit only on Train)...")
    train_mask = df[DATE_COL] <= SPLIT_DATE
    df = create_brand_stats_features(df, train_mask)

    train, test = chronological_split(df)

    # Save
    train_path = os.path.join(PATHS["features"], "train_features.csv")
    test_path  = os.path.join(PATHS["features"], "test_features.csv")
    full_path  = os.path.join(PATHS["features"], "full_features.csv")

    train.to_csv(train_path, index=False)
    test.to_csv(test_path,   index=False)
    df.to_csv(full_path,     index=False)

    logger.info(f"✓ HOÀN TẤT — Total features: {len(df.columns)} cột")
    logger.info(f"  Saved train → {train_path}")
    logger.info(f"  Saved test  → {test_path}")
    logger.info(f"  Saved full  → {full_path}")

    return df, train, test


def main():
    engineer()


if __name__ == "__main__":
    main()
