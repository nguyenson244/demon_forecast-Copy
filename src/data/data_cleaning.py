"""
============================================================
data_cleaning.py — Data Preparation & Cleaning
============================================================
Migrate từ: src/data_prep.py
Role      : Tiền xử lý dữ liệu thô:
              1. Chuẩn hóa kiểu dữ liệu (dtypes)
              2. Báo cáo chất lượng dữ liệu
              3. Tạo lưới đầy đủ (full date grid) — xử lý zero-inflation
              4. Xuất → data/processed/cleaned_data.csv
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
    PATHS, DATE_COL, TARGET_COL, ensure_dirs
)
from src.data.data_loader import load_raw_data

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. CHUẨN HÓA KIỂU DỮ LIỆU
# ═══════════════════════════════════════════════════════════
def format_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuyển đổi dtype theo đặc tả Data Dictionary.
      - ACTUALSHIPDATE → datetime64
      - CATEGORY, WHSEID, BRAND, Week → category
      - Day → int
      - Total QTY, Total CBM → float
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    for col in ["CATEGORY", "WHSEID", "BRAND", "Week"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    if "Day" in df.columns:
        df["Day"] = df["Day"].astype(int)
    df[TARGET_COL] = df[TARGET_COL].astype(float)
    if "Total CBM" in df.columns:
        df["Total CBM"] = df["Total CBM"].astype(float)
    logger.debug("Dtypes formatted.")
    return df


# ═══════════════════════════════════════════════════════════
# 2. BÁO CÁO CHẤT LƯỢNG DỮ LIỆU
# ═══════════════════════════════════════════════════════════
def report_data_quality(df: pd.DataFrame) -> None:
    """Log báo cáo missing values, zero-inflation, thống kê cơ bản."""
    logger.info("─" * 50)
    logger.info("BÁO CÁO CHẤT LƯỢNG DỮ LIỆU")
    logger.info("─" * 50)

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"Missing values:\n{missing[missing > 0].to_string()}")
    else:
        logger.info("✔ Không có missing values.")

    # Zero-inflation per category
    if "CATEGORY" in df.columns:
        logger.info("── Zero-Inflation Analysis ──")
        for cat in df["CATEGORY"].unique():
            subset = df[df["CATEGORY"] == cat]
            zero_count = (subset[TARGET_COL] == 0).sum()
            total = len(subset)
            pct = zero_count / total * 100 if total > 0 else 0
            flag = " ⚠ ZERO-INFLATED" if pct > 50 else ""
            logger.info(
                f"  {cat:12s}: {zero_count:>6,}/{total:>6,} zero rows "
                f"({pct:5.1f}%){flag}"
            )

    # Basic stats
    logger.info(
        f"Date range : {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}"
    )
    logger.info(f"Total rows : {len(df):,}")


# ═══════════════════════════════════════════════════════════
# 3. TẠO LƯỚI ĐẦY ĐỦ (FULL DATE GRID)
# ═══════════════════════════════════════════════════════════
def create_full_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo lưới đầy đủ (date × brand × category) để xử lý zero-inflation.
    Các ngày không có giao dịch được điền Total QTY = 0, Total CBM = 0.

    Điều này đảm bảo các hàm lag/shift hoạt động đúng theo
    trục thời gian lịch (calendar) thay vì theo vị trí hàng.
    """
    df = df.copy()

    existing_pairs = df[["BRAND", "CATEGORY"]].drop_duplicates()

    full_dates = pd.date_range(
        df[DATE_COL].min(),
        df[DATE_COL].max(),
        freq="D",
    )

    grid = existing_pairs.merge(
        pd.DataFrame({DATE_COL: full_dates}),
        how="cross",
    )

    logger.info(
        f"Grid size: {len(grid):,} rows "
        f"({len(existing_pairs)} pairs × {len(full_dates)} days)"
    )

    df_full = grid.merge(df, on=[DATE_COL, "BRAND", "CATEGORY"], how="left")

    # Điền zero cho ngày không có giao dịch
    df_full[TARGET_COL] = df_full[TARGET_COL].fillna(0.0)
    if "Total CBM" in df_full.columns:
        df_full["Total CBM"] = df_full["Total CBM"].fillna(0.0)
    if "WHSEID" in df_full.columns:
        df_full["WHSEID"] = df_full["WHSEID"].fillna("BKD1")

    # Tính lại Week và Day từ date
    df_full["Day"] = df_full[DATE_COL].dt.dayofweek + 1
    day_names = {
        1: "Monday", 2: "Tuesday", 3: "Wednesday",
        4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday",
    }
    week_num = df_full[DATE_COL].dt.isocalendar().week.astype(int)
    day_name = df_full["Day"].map(day_names)
    df_full["Week"] = week_num.astype(str) + "." + day_name

    df_full = df_full.sort_values(
        [DATE_COL, "BRAND", "CATEGORY"]
    ).reset_index(drop=True)

    df_full[TARGET_COL] = df_full[TARGET_COL].clip(lower=0)
    if "Total CBM" in df_full.columns:
        df_full["Total CBM"] = df_full["Total CBM"].clip(lower=0)

    filled = len(df_full) - len(df)
    logger.info(f"Filled {filled:,} zero-sale rows. Final: {len(df_full):,} rows.")
    return df_full


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════
def clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Chạy toàn bộ pipeline làm sạch dữ liệu.

    Parameters
    ----------
    df : pd.DataFrame | None
        DataFrame thô đã load. Nếu None, tự động gọi load_raw_data().

    Returns
    -------
    pd.DataFrame
        DataFrame đã làm sạch, sẵn sàng cho Feature Engineering.
    """
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU THÔ")
    logger.info("=" * 60)

    if df is None:
        logger.info("[1/4] Loading raw data...")
        df = load_raw_data()

    logger.info("[2/4] Formatting dtypes...")
    df = format_dtypes(df)

    logger.info("[3/4] Data quality check...")
    report_data_quality(df)

    logger.info("[4/4] Creating full date grid (xử lý zero-inflation)...")
    df = create_full_grid(df)

    out_path = os.path.join(PATHS["processed"], "cleaned_data.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"✓ HOÀN TẤT — Saved → {out_path}")
    return df


# ── Standalone entry point ───────────────────────────────
def main():
    clean()


if __name__ == "__main__":
    main()
