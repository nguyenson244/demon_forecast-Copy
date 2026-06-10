"""
============================================================
feature_engineering.py — Feature Engineering Pipeline
============================================================
Tạo 90+ features:
  - Lag (19)             — trí nhớ ngắn & dài hạn
  - Rolling stats (20)   — động lượng & xu hướng
  - Momentum/CV (6)      — tốc độ tăng trưởng
  - Time (14)            — lượng giác & chỉ báo
  - Tết Nguyên Đán (5)   — 3 pha hành vi mua sắm
  - Trung Thu (3)        — đếm ngược mùa bánh
  - Holiday (3)          — lễ quốc gia (python holidays lib)
  - Extended Holiday (8) — 30/4, 2/9, Giáng Sinh, hè học sinh...
  - Weather (7)          — nhiệt độ, mưa, mùa khô (Open-Meteo)
  - Brand stats (7)      — fit ONLY on Train
  - Categorical (2)      — is_seasonal, cbm_log

CHỐNG RÒ RỈ:
  - Tất cả lag/rolling dùng .shift(1) nghiêm ngặt
  - pct_change: replace inf/nan sau khi tính
  - Brand stats: .fit() chỉ trên Train
  - Split: chronological, KHÔNG random
  - NaN report: log số lượng NaN sau mỗi bước
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
    PATHS, DATE_COL, TARGET_COL, CONF,
    TET_DATES, MID_AUTUMN_DATES, SPLIT_DATE,
    LAG_PERIODS, SAME_WEEKDAY_WEEKS, ROLLING_WINDOWS,
    ensure_dirs,
)
from src.data.data_cleaning import detect_and_handle_outliers

logger = get_logger(__name__)

_INF_CLIP = 1e6  # giá trị clip cho pct_change và các ratio features


# ═══════════════════════════════════════════════════════════
# HELPER: LOG NaN COUNTS
# ═══════════════════════════════════════════════════════════
def _log_nan_summary(df: pd.DataFrame, step: str) -> None:
    """Log tổng hợp NaN counts sau mỗi bước — giúp debug leakage và data quality."""
    nan_cols = df.isnull().sum()
    nan_cols = nan_cols[nan_cols > 0]
    if len(nan_cols) == 0:
        logger.debug(f"[{step}] Không có NaN.")
    else:
        total_nan = nan_cols.sum()
        logger.debug(
            f"[{step}] {len(nan_cols)} cột có NaN, "
            f"tổng {total_nan:,} cells "
            f"({total_nan / df.size * 100:.2f}% của dataset)"
        )


# ═══════════════════════════════════════════════════════════
# 1. LAG FEATURES
# ═══════════════════════════════════════════════════════════
def create_lag_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo lag features cho 1 chuỗi thời gian (brand × category).

    Lưu ý lag_365: sẽ NaN cho toàn bộ năm đầu tiên của mỗi brand.
    Đây là expected behaviour — LightGBM xử lý NaN nội bộ.
    """
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
    """
    Rolling mean, std, median, min, max.
    shift(1) trước khi rolling → chống leakage (không dùng giá trị hiện tại).
    """
    shifted = group[TARGET_COL].shift(1)
    for w in ROLLING_WINDOWS:
        group[f"roll_mean_{w}"]   = shifted.rolling(w, min_periods=1).mean()
        group[f"roll_std_{w}"]    = shifted.rolling(w, min_periods=1).std().fillna(0)
        group[f"roll_median_{w}"] = shifted.rolling(w, min_periods=1).median()
        group[f"roll_min_{w}"]    = shifted.rolling(w, min_periods=1).min()
        group[f"roll_max_{w}"]    = shifted.rolling(w, min_periods=1).max()
    return group


# ═══════════════════════════════════════════════════════════
# 3. MOMENTUM & VOLATILITY
# ═══════════════════════════════════════════════════════════
def create_momentum_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum, pct_change, coefficient of variation.

    Fix pct_change inf:
      - Khi previous = 0 và current > 0 → pct_change = inf
      - Replace inf bằng giá trị lớn nhưng hữu hạn (_INF_CLIP)
      - Replace -inf tương tự
      - NaN giữ nguyên (LightGBM handle internally)
    """
    target  = group[TARGET_COL]
    shifted = target.shift(1)

    group["momentum_7"]  = target.shift(1) - target.shift(8)
    group["momentum_30"] = target.shift(1) - target.shift(31)

    # pct_change: thay thế inf/-inf trước khi shift để tránh leakage
    pct1 = target.pct_change(1)
    pct7 = target.pct_change(7)
    group["pct_change_1"] = (
        pct1.replace([np.inf, -np.inf], np.nan)
            .clip(-_INF_CLIP, _INF_CLIP)
            .shift(1)
    )
    group["pct_change_7"] = (
        pct7.replace([np.inf, -np.inf], np.nan)
            .clip(-_INF_CLIP, _INF_CLIP)
            .shift(1)
    )

    roll_mean_14 = shifted.rolling(14, min_periods=2).mean()
    roll_std_14  = shifted.rolling(14, min_periods=2).std().fillna(0)
    group["roll_cv_14"] = (roll_std_14 / (roll_mean_14 + 1e-8)).clip(0, _INF_CLIP)

    roll_mean_30 = shifted.rolling(30, min_periods=2).mean()
    roll_std_30  = shifted.rolling(30, min_periods=2).std().fillna(0)
    group["roll_cv_30"] = (roll_std_30 / (roll_mean_30 + 1e-8)).clip(0, _INF_CLIP)

    return group


# ═══════════════════════════════════════════════════════════
# 4. TIME FEATURES
# ═══════════════════════════════════════════════════════════
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trích xuất đặc trưng thời gian và encoding lượng giác."""
    df = df.copy()
    dt = df[DATE_COL]

    df["year"]           = dt.dt.year
    df["month"]          = dt.dt.month
    df["day_of_month"]   = dt.dt.day
    df["dayofweek"]      = dt.dt.dayofweek
    df["weekofyear"]     = dt.dt.isocalendar().week.astype(int)
    df["quarter"]        = dt.dt.quarter
    df["is_weekend"]     = (df["dayofweek"] >= 5).astype(int)
    df["is_monday"]      = (df["dayofweek"] == 0).astype(int)
    df["is_first_5days"] = (dt.dt.day <= 5).astype(int)
    df["is_month_end"]   = dt.dt.is_month_end.astype(int)

    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    return df


# ═══════════════════════════════════════════════════════════
# 5. TẾT NGUYÊN ĐÁN
# ═══════════════════════════════════════════════════════════
def create_tet_features(df: pd.DataFrame) -> pd.DataFrame:
    """3 pha hành vi mua sắm quanh Tết."""
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
        diff  = (dates_np - tet).astype("timedelta64[D]").astype(int)
        mask  = (diff > 0) & (diff <= 30)
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

    df["days_to_mid_autumn"]   = days_to_ma
    df["is_mid_autumn_season"] = (
        (df["days_to_mid_autumn"] >= 0) & (df["days_to_mid_autumn"] <= 75)
    ).astype(int)
    df["is_mid_autumn_peak"] = (
        (df["days_to_mid_autumn"] >= 0) & (df["days_to_mid_autumn"] <= 14)
    ).astype(int)

    # days AFTER mid-autumn — model cần biết THU sập nhanh sau peak
    days_after_ma = np.zeros(len(df), dtype=int)
    for ma in ma_array:
        diff = (dates_np - ma).astype("timedelta64[D]").astype(int)
        mask = (diff > 0) & (diff <= 45)
        better = mask & (diff > days_after_ma)
        update = mask & (days_after_ma == 0)
        days_after_ma[better | update] = diff[better | update]
    df["days_after_mid_autumn"] = days_after_ma
    df["is_post_mid_autumn"]    = (days_after_ma > 0).astype(int)

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
# 8. EXTENDED HOLIDAY FEATURES
# ═══════════════════════════════════════════════════════════
def create_extended_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm các ngày lễ quan trọng ảnh hưởng đến tiêu dùng FMCG
    mà python-holidays hoặc Tết/Trung Thu chưa bao gồm.

    Features:
      days_to_liberation_day — đếm ngược đến 30/4 (4-day holiday block)
      is_liberation_window   — trong 4 ngày nghỉ 30/4–1/5
      days_to_national_day   — đếm ngược đến 2/9
      is_national_day_window — trong 2 ngày nghỉ 2–3/9
      is_christmas_newyear   — 24/12–2/1 (tăng mua sắm lễ + quà)
      is_school_holiday      — tháng 6–8 (hè học sinh → tăng snack)
      is_valentine_window    — 12–14/2 (chocolate/gift brands)
      days_since_data_start  — linear trend proxy
    """
    df   = df.copy()
    dt   = df[DATE_COL]
    np_dates = dt.values.astype("datetime64[ns]")
    years = range(dt.dt.year.min(), dt.dt.year.max() + 2)

    # ── 30/4 Giải phóng + 1/5 Lao động (4-day block) ───────
    apr30 = np.array([np.datetime64(f"{y}-04-30", "ns") for y in years])
    d2apr = np.full(len(df), 999, dtype=int)
    for d in apr30:
        diff = ((d - np_dates) / np.timedelta64(1, "D")).astype(int)
        mask = (diff >= 0) & (diff < d2apr)
        d2apr[mask] = diff[mask]
    df["days_to_liberation_day"] = d2apr
    df["is_liberation_window"]   = ((d2apr >= 0) & (d2apr <= 3)).astype(int)

    # ── 2/9 Quốc Khánh ──────────────────────────────────────
    sep2 = np.array([np.datetime64(f"{y}-09-02", "ns") for y in years])
    d2sep = np.full(len(df), 999, dtype=int)
    for d in sep2:
        diff = ((d - np_dates) / np.timedelta64(1, "D")).astype(int)
        mask = (diff >= 0) & (diff < d2sep)
        d2sep[mask] = diff[mask]
    df["days_to_national_day"]   = d2sep
    df["is_national_day_window"] = ((d2sep >= 0) & (d2sep <= 2)).astype(int)

    # ── Giáng Sinh + Tết Dương lịch ────────────────────────
    month, day = dt.dt.month, dt.dt.day
    df["is_christmas_newyear"] = (
        ((month == 12) & (day >= 24)) | ((month == 1) & (day <= 2))
    ).astype(int)

    # ── Hè học sinh (tháng 6–8): tăng snack consumption ────
    df["is_school_holiday"] = ((month >= 6) & (month <= 8)).astype(int)

    # ── Valentine's Day (12–14/2): chocolate/gift brands ───
    df["is_valentine_window"] = ((month == 2) & (day >= 12) & (day <= 14)).astype(int)

    # ── Ngày 20/10 & 20/11 (Women's Day, Teachers' Day) ────
    df["is_vn_gift_day"] = (
        ((month == 10) & (day >= 18) & (day <= 20)) |
        ((month == 11) & (day >= 18) & (day <= 20))
    ).astype(int)

    # ── Linear trend proxy: số ngày kể từ ngày đầu data ────
    t0 = dt.min()
    df["days_since_start"] = (dt - t0).dt.days.astype(int)

    return df


# ═══════════════════════════════════════════════════════════
# 8b. WEATHER FEATURES
# ═══════════════════════════════════════════════════════════
def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weather features từ Open-Meteo API (TP.HCM).
    Fallback tự động nếu API không khả dụng.
    """
    try:
        from src.data.weather_features import build_weather_features
        df = build_weather_features(df)
    except Exception as exc:
        logger.warning(f"Weather features skipped: {exc}")
    return df


# ═══════════════════════════════════════════════════════════
# 9. CATEGORICAL & TRANSFORM
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
    """
    Brand-level statistics, fit chỉ trên Train để chống leakage.

    zscore: dùng shifted target (shift 1) trong khi tính để tránh
    target value hiện tại lọt vào feature.
    """
    df         = df.copy()
    train_data = df[train_mask]

    qty_stats = (
        train_data.groupby("BRAND", observed=True)[TARGET_COL]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    qty_stats.columns = ["BRAND", "brand_mean_qty", "brand_std_qty", "brand_median_qty"]

    df = df.merge(qty_stats, on="BRAND", how="left")

    # Dùng shifted target để tính zscore (chống leakage)
    shifted_target = df.groupby(["BRAND", "CATEGORY"], observed=True)[TARGET_COL].shift(1)
    df["qty_zscore_brand"] = (
        (shifted_target - df["brand_mean_qty"]) / (df["brand_std_qty"].fillna(1) + 1e-8)
    ).clip(-10, 10)

    if "Total CBM" in train_data.columns:
        cbm_stats = (
            train_data.groupby("BRAND", observed=True)["Total CBM"]
            .agg(["mean", "std"])
            .reset_index()
        )
        cbm_stats.columns = ["BRAND", "brand_mean_cbm", "brand_std_cbm"]
        df = df.merge(cbm_stats, on="BRAND", how="left")

        shifted_cbm = df.groupby(["BRAND", "CATEGORY"], observed=True)["Total CBM"].shift(1)
        df["cbm_zscore_brand"] = (
            (shifted_cbm - df["brand_mean_cbm"]) / (df["brand_std_cbm"].fillna(1) + 1e-8)
        ).clip(-10, 10)

    return df


# ═══════════════════════════════════════════════════════════
# 10. COMPARISON FEATURES
# ═══════════════════════════════════════════════════════════
def create_comparison_features(group: pd.DataFrame) -> pd.DataFrame:
    """is_above_30d_mean — tín hiệu momentum so với baseline."""
    shifted      = group[TARGET_COL].shift(1)
    roll_mean_30 = shifted.rolling(30, min_periods=1).mean()
    group["is_above_30d_mean"] = (shifted > roll_mean_30).astype(int)
    return group


# ═══════════════════════════════════════════════════════════
# 11. CHRONOLOGICAL SPLIT
# ═══════════════════════════════════════════════════════════
def chronological_split(
    df: pd.DataFrame,
    split_date: pd.Timestamp = SPLIT_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tách Train/Test theo trình tự thời gian — KHÔNG random shuffle.

    Assert không có data leakage: max(Train) < min(Test).
    """
    train = df[df[DATE_COL] <= split_date].copy()
    test  = df[df[DATE_COL] >  split_date].copy()

    logger.info(f"SPLIT — Cutoff: {split_date.date()}")
    logger.info(
        f"  Train: {len(train):>8,} rows | "
        f"{train[DATE_COL].min().date()} → {train[DATE_COL].max().date()}"
    )
    logger.info(
        f"  Test : {len(test):>8,} rows | "
        f"{test[DATE_COL].min().date()} → {test[DATE_COL].max().date()}"
    )

    if len(train) > 0 and len(test) > 0:
        assert train[DATE_COL].max() < test[DATE_COL].min(), (
            f"⛔ DATA LEAKAGE: Train max={train[DATE_COL].max()} >= Test min={test[DATE_COL].min()}"
        )
        logger.info("✔ Leakage check PASSED")

    return train, test


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════
def engineer(
    cleaned_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chạy toàn bộ Feature Engineering pipeline.

    Parameters
    ----------
    cleaned_df : DataFrame | None
        Cleaned data từ data_cleaning.clean().
        Nếu None, đọc từ data/processed/cleaned_data.csv.

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
                f"Chưa có cleaned_data.csv tại {data_path}. "
                "Chạy data_cleaning.py trước!"
            )
        cleaned_df = pd.read_csv(data_path, parse_dates=[DATE_COL])
        logger.info(f"Loaded cleaned data: {len(cleaned_df):,} rows")

    df = cleaned_df.sort_values(["BRAND", "CATEGORY", DATE_COL]).reset_index(drop=True)

    logger.info("[0/11] Outlier capping (fit on train only — no leakage)...")
    train_only = df[df[DATE_COL] <= SPLIT_DATE]
    df = detect_and_handle_outliers(
        df,
        target_col=TARGET_COL,
        group_cols=CONF.outlier_group_cols,
        iqr_multiplier=CONF.outlier_iqr_multiplier,
        fit_df=train_only,
    )

    logger.info("[1/11] Time features...")
    df = create_time_features(df)

    logger.info("[2/11] Tết Nguyên Đán features...")
    df = create_tet_features(df)

    logger.info("[3/11] Trung Thu features...")
    df = create_mid_autumn_features(df)

    logger.info("[4/11] Holiday features (python-holidays)...")
    df = create_holiday_features(df)

    logger.info("[5/11] Extended holiday features (30/4, 2/9, Giáng Sinh, ...)...")
    df = create_extended_holiday_features(df)

    logger.info("[6/11] Weather features (Open-Meteo TP.HCM)...")
    df = create_weather_features(df)

    logger.info("[7/11] Categorical & Transform features...")
    df = create_categorical_features(df)

    logger.info("[8-10/11] Lag / Rolling / Momentum / Comparison per BRAND × CATEGORY...")
    groups = []
    for (brand, cat), group in df.groupby(["BRAND", "CATEGORY"], observed=True):
        group = group.copy().sort_values(DATE_COL)
        group = create_lag_features(group)
        group = create_rolling_features(group)
        group = create_momentum_features(group)
        group = create_comparison_features(group)
        groups.append(group)

    df = pd.concat(groups, ignore_index=True)
    df = df.sort_values([DATE_COL, "BRAND", "CATEGORY"]).reset_index(drop=True)
    _log_nan_summary(df, "after_lag_rolling")

    logger.info("[11/11] Brand statistics (fit only on Train)...")
    train_mask = df[DATE_COL] <= SPLIT_DATE
    df = create_brand_stats_features(df, train_mask)
    _log_nan_summary(df, "after_brand_stats")

    train, test = chronological_split(df)

    # ── Lưu ─────────────────────────────────────────────────
    train_path = os.path.join(PATHS["features"], "train_features.csv")
    test_path  = os.path.join(PATHS["features"], "test_features.csv")
    full_path  = os.path.join(PATHS["features"], "full_features.csv")

    train.to_csv(train_path, index=False)
    test.to_csv(test_path,   index=False)
    df.to_csv(full_path,     index=False)

    logger.info(f"✓ HOÀN TẤT — {len(df.columns)} cột")
    logger.info(f"  train → {train_path}")
    logger.info(f"  test  → {test_path}")
    logger.info(f"  full  → {full_path}")

    return df, train, test


def main():
    engineer()


if __name__ == "__main__":
    main()
