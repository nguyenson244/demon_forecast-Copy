"""
============================================================
weather_features.py — Open-Meteo Historical Weather
============================================================
Lấy dữ liệu thời tiết TP.HCM từ Open-Meteo API (miễn phí, không cần API key).
Cache kết quả vào data/processed/weather_data.csv để tránh gọi API nhiều lần.
Fallback về số liệu trung bình mùa vụ nếu API không khả dụng.

Tại sao thời tiết quan trọng cho FMCG:
  - Mùa mưa (May–Oct): người mua sắm ít ra ngoài → giảm đơn hàng impulse
  - Ngày nóng >33°C: tăng tiêu thụ nước giải khát, giảm bánh kẹo nặng
  - Mùa khô (Nov–Apr): lễ hội + thời tiết dễ chịu → tăng mua sắm
============================================================
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.config_loader import PATHS, DATE_COL

logger = get_logger(__name__)

# TP.HCM coordinates
_HCM_LAT = 10.8231
_HCM_LON = 106.6297

# Trung bình tháng TP.HCM (fallback khi API không khả dụng)
# Source: Vietnam Meteorological Data
_MONTHLY_TEMP = {
    1: 26.5, 2: 27.3, 3: 28.8, 4: 30.1, 5: 29.4, 6: 28.4,
    7: 28.1, 8: 28.0, 9: 27.7, 10: 27.3, 11: 26.9, 12: 26.3,
}
_MONTHLY_PRECIP_DAILY = {
    1: 0.5, 2: 0.2, 3: 0.4, 4: 1.9, 5: 7.2, 6: 10.5,
    7: 9.8, 8: 9.1, 9: 11.0, 10: 8.9, 11: 4.0, 12: 1.7,
}


# ═══════════════════════════════════════════════════════════
# 1. API FETCH
# ═══════════════════════════════════════════════════════════
def _fetch_from_api(start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Gọi Open-Meteo Archive API.
    Trả về None nếu thất bại (timeout, network error, ...).
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={_HCM_LAT}&longitude={_HCM_LON}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max"
        "&timezone=Asia%2FHo_Chi_Minh"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DemandForecast/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())

        daily = data.get("daily", {})
        df = pd.DataFrame({
            "date":      pd.to_datetime(daily["time"]),
            "temp_mean": daily.get("temperature_2m_mean", [np.nan] * len(daily["time"])),
            "precip":    daily.get("precipitation_sum",   [np.nan] * len(daily["time"])),
            "wind_max":  daily.get("wind_speed_10m_max",  [np.nan] * len(daily["time"])),
        })
        logger.info(
            f"Weather API OK: {len(df)} days "
            f"({df['date'].min().date()} → {df['date'].max().date()})"
        )
        return df

    except Exception as exc:
        logger.warning(f"Weather API failed: {exc}")
        return None


def _seasonal_fallback(start_date: str, end_date: str) -> pd.DataFrame:
    """Trả về trung bình mùa vụ khi API không khả dụng."""
    dates = pd.date_range(start_date, end_date, freq="D")
    df = pd.DataFrame({"date": dates})
    df["temp_mean"] = df["date"].dt.month.map(_MONTHLY_TEMP)
    df["precip"]    = df["date"].dt.month.map(_MONTHLY_PRECIP_DAILY)
    df["wind_max"]  = 14.0
    logger.warning(
        f"Using seasonal weather fallback for "
        f"{df['date'].min().date()} → {df['date'].max().date()}"
    )
    return df


# ═══════════════════════════════════════════════════════════
# 2. CACHE + FETCH
# ═══════════════════════════════════════════════════════════
def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Lấy dữ liệu thời tiết TP.HCM, cache vào CSV.

    Parameters
    ----------
    start_date : str  — "YYYY-MM-DD"
    end_date   : str  — "YYYY-MM-DD"

    Returns
    -------
    DataFrame với cột: date, temp_mean, precip, wind_max
    """
    cache_path = Path(PATHS["processed"]) / "weather_data.csv"

    # Check cache coverage
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        if (
            cached["date"].min() <= pd.Timestamp(start_date)
            and cached["date"].max() >= pd.Timestamp(end_date)
        ):
            logger.info(f"Weather loaded from cache: {cache_path}")
            return cached

    # Fetch from API
    df = _fetch_from_api(start_date, end_date)
    if df is None:
        df = _seasonal_fallback(start_date, end_date)
    else:
        # Fill remaining NaN với fallback
        df["temp_mean"] = df["temp_mean"].fillna(
            df["date"].dt.month.map(_MONTHLY_TEMP)
        )
        df["precip"] = df["precip"].fillna(
            df["date"].dt.month.map(_MONTHLY_PRECIP_DAILY)
        )
        df["wind_max"] = df["wind_max"].fillna(14.0)

    df.to_csv(cache_path, index=False)
    return df


# ═══════════════════════════════════════════════════════════
# 3. BUILD WEATHER FEATURES
# ═══════════════════════════════════════════════════════════
def build_weather_features(df_main: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weather features vào DataFrame chính.

    Features tạo ra:
      temp_mean         — nhiệt độ trung bình ngày (°C)
      precip            — lượng mưa (mm)
      temp_roll7        — rolling 7-day mean temperature (shift 1 để chống leakage)
      precip_roll7      — rolling 7-day sum precip (shift 1)
      is_hot_day        — nhiệt độ > 33°C (FMCG giảm khi quá nóng)
      is_rainy_day      — mưa > 10mm (giảm offline shopping)
      is_dry_cool_season — tháng 11–2: mùa khô mát → tăng tiêu dùng
    """
    start = df_main[DATE_COL].min().strftime("%Y-%m-%d")
    end   = df_main[DATE_COL].max().strftime("%Y-%m-%d")

    weather = fetch_weather(start, end).sort_values("date").reset_index(drop=True)

    # Rolling features: shift(1) tránh dùng same-day weather làm feature
    shifted_temp   = weather["temp_mean"].shift(1)
    shifted_precip = weather["precip"].shift(1)
    weather["temp_roll7"]   = shifted_temp.rolling(7, min_periods=1).mean()
    weather["precip_roll7"] = shifted_precip.rolling(7, min_periods=1).sum()

    weather["is_hot_day"]        = (weather["temp_mean"] > 33.0).astype(int)
    weather["is_rainy_day"]      = (weather["precip"] > 10.0).astype(int)
    weather["is_dry_cool_season"] = weather["date"].dt.month.isin([11, 12, 1, 2]).astype(int)

    keep_cols = [
        "date", "temp_mean", "precip",
        "temp_roll7", "precip_roll7",
        "is_hot_day", "is_rainy_day", "is_dry_cool_season",
    ]
    weather_feat = weather[keep_cols].rename(columns={"date": DATE_COL})

    # Merge — left join để không mất rows chính
    df_merged = df_main.merge(weather_feat, on=DATE_COL, how="left")

    # Fill NaN từ fallback seasonal nếu có dates nằm ngoài range
    month = df_merged[DATE_COL].dt.month
    df_merged["temp_mean"]         = df_merged["temp_mean"].fillna(month.map(_MONTHLY_TEMP))
    df_merged["precip"]            = df_merged["precip"].fillna(month.map(_MONTHLY_PRECIP_DAILY))
    df_merged["temp_roll7"]        = df_merged["temp_roll7"].fillna(month.map(_MONTHLY_TEMP))
    df_merged["precip_roll7"]      = df_merged["precip_roll7"].fillna(0.0)
    df_merged["is_hot_day"]        = df_merged["is_hot_day"].fillna(0).astype(int)
    df_merged["is_rainy_day"]      = df_merged["is_rainy_day"].fillna(0).astype(int)
    df_merged["is_dry_cool_season"] = df_merged["is_dry_cool_season"].fillna(0).astype(int)

    n_weather = len([c for c in keep_cols if c != DATE_COL])
    logger.info(f"Weather features merged: {n_weather} columns")
    return df_merged
