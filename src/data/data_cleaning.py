"""
============================================================
data_cleaning.py — Data Preparation & Cleaning
============================================================
Role:
  1. Chuẩn hóa kiểu dữ liệu (dtypes)
  2. Báo cáo chất lượng dữ liệu
  3. Tạo lưới đầy đủ (full date grid) — xử lý zero-inflation
  4. Phát hiện và cap outliers (IQR-based, per group)
  5. Xuất → data/processed/cleaned_data.csv
============================================================
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.utils.logger import get_logger
from src.utils.config_loader import CONF, PATHS, DATE_COL, TARGET_COL, ensure_dirs
from src.data.data_loader import load_raw_data

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. CHUẨN HÓA DTYPES
# ═══════════════════════════════════════════════════════════
def format_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Chuyển đổi dtype theo Data Dictionary."""
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
# 2. BÁO CÁO CHẤT LƯỢNG
# ═══════════════════════════════════════════════════════════
def report_data_quality(df: pd.DataFrame) -> None:
    """Log báo cáo missing values, zero-inflation, thống kê cơ bản."""
    logger.info("─" * 50)
    logger.info("BÁO CÁO CHẤT LƯỢNG DỮ LIỆU")
    logger.info("─" * 50)

    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"Missing values:\n{missing[missing > 0].to_string()}")
    else:
        logger.info("✔ Không có missing values.")

    if "CATEGORY" in df.columns:
        logger.info("── Zero-Inflation per Category ──")
        for cat in df["CATEGORY"].unique():
            subset = df[df["CATEGORY"] == cat]
            zero_count = (subset[TARGET_COL] == 0).sum()
            total = len(subset)
            pct = zero_count / total * 100 if total > 0 else 0
            flag = " ⚠ ZERO-INFLATED" if pct > 50 else ""
            logger.info(
                f"  {str(cat):15s}: {zero_count:>6,}/{total:>6,} "
                f"({pct:5.1f}%){flag}"
            )

    logger.info(
        f"Date range : {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}"
    )
    logger.info(f"Total rows : {len(df):,}")


# ═══════════════════════════════════════════════════════════
# 3. FULL DATE GRID
# ═══════════════════════════════════════════════════════════
def create_full_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo lưới đầy đủ (date × brand × category) để xử lý zero-inflation.
    Ngày không có giao dịch được điền Total QTY = 0.

    Đảm bảo lag/shift hoạt động đúng theo calendar time.
    """
    df = df.copy()
    existing_pairs = df[["BRAND", "CATEGORY"]].drop_duplicates()
    full_dates = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")

    grid = existing_pairs.merge(
        pd.DataFrame({DATE_COL: full_dates}),
        how="cross",
    )

    logger.info(
        f"Grid: {len(grid):,} rows "
        f"({len(existing_pairs)} pairs × {len(full_dates)} days)"
    )

    df_full = grid.merge(df, on=[DATE_COL, "BRAND", "CATEGORY"], how="left")

    df_full[TARGET_COL] = df_full[TARGET_COL].fillna(0.0)
    if "Total CBM" in df_full.columns:
        df_full["Total CBM"] = df_full["Total CBM"].fillna(0.0)
    if "WHSEID" in df_full.columns:
        # Lấy từ config thay vì hardcode
        df_full["WHSEID"] = df_full["WHSEID"].fillna(CONF.default_whseid)

    # Tính lại Week và Day từ date
    df_full["Day"] = df_full[DATE_COL].dt.dayofweek + 1
    day_names = {
        1: "Monday", 2: "Tuesday", 3: "Wednesday",
        4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday",
    }
    week_num = df_full[DATE_COL].dt.isocalendar().week.astype(int)
    df_full["Week"] = week_num.astype(str) + "." + df_full["Day"].map(day_names)

    df_full = df_full.sort_values([DATE_COL, "BRAND", "CATEGORY"]).reset_index(drop=True)

    df_full[TARGET_COL] = df_full[TARGET_COL].clip(lower=0)
    if "Total CBM" in df_full.columns:
        df_full["Total CBM"] = df_full["Total CBM"].clip(lower=0)

    filled = len(df_full) - len(df)
    logger.info(f"Filled {filled:,} zero-sale rows. Total: {len(df_full):,}")
    return df_full


# ═══════════════════════════════════════════════════════════
# 4. OUTLIER DETECTION & CAPPING (IQR-based)
# ═══════════════════════════════════════════════════════════
def detect_and_handle_outliers(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    method: str | None = None,
    group_cols: list[str] | None = None,
    iqr_multiplier: float | None = None,
) -> pd.DataFrame:
    """
    Phát hiện và CAP outliers bằng IQR per group.

    Logic:
      upper_bound = Q3 + k * IQR
      lower_bound = max(0, Q1 - k * IQR)  ← không clip âm cho QTY

    Tại sao k=3.0 thay vì 1.5:
      - FMCG có peak mùa vụ Trung Thu/Tết — peak thật sự cao gấp 10x thường ngày
      - k=1.5 sẽ cap cả peak thật → mất signal quan trọng
      - k=3.0 chỉ loại bỏ outlier thực sự bất thường

    Tại sao cap thay vì remove:
      - Time series cần liên tục — xóa row sẽ phá vỡ cấu trúc ngày
      - Cap giữ nguyên pattern nhưng giảm ảnh hưởng của extreme values

    Parameters
    ----------
    method         : "iqr" (default) — hiện chỉ hỗ trợ IQR
    group_cols     : tính IQR per group (e.g., per BRAND + CATEGORY)
    iqr_multiplier : nhân tử k (default từ config: 3.0)
    """
    if target_col not in df.columns:
        logger.warning(f"Cột '{target_col}' không tồn tại → skip.")
        return df

    if method is None:
        method = CONF.outlier_method
    if group_cols is None:
        group_cols = CONF.outlier_group_cols
    if iqr_multiplier is None:
        iqr_multiplier = CONF.outlier_iqr_multiplier

    df = df.copy()
    total_capped = 0

    valid_group_cols = [c for c in group_cols if c in df.columns]

    if valid_group_cols:
        for keys, group in df.groupby(valid_group_cols, observed=True):
            idx = group.index
            vals = group[target_col]
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            lower = max(0.0, q1 - iqr_multiplier * iqr)
            upper = q3 + iqr_multiplier * iqr

            before = vals.copy()
            df.loc[idx, target_col] = vals.clip(lower=lower, upper=upper)
            n_capped = int((df.loc[idx, target_col] != before).sum())
            total_capped += n_capped
    else:
        vals = df[target_col]
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lower = max(0.0, q1 - iqr_multiplier * iqr)
        upper = q3 + iqr_multiplier * iqr
        before = vals.copy()
        df[target_col] = vals.clip(lower=lower, upper=upper)
        total_capped = int((df[target_col] != before).sum())

    pct_capped = total_capped / len(df) * 100
    logger.info(
        f"Outlier capping (IQR k={iqr_multiplier}): "
        f"{total_capped:,} values capped ({pct_capped:.2f}% of rows)"
    )
    return df


# ═══════════════════════════════════════════════════════════
# 5. PHÂN TÍCH TƯƠNG QUAN
# ═══════════════════════════════════════════════════════════
def analyze_correlation(df: pd.DataFrame) -> dict:
    """Phân tích tương quan giữa các numeric columns."""
    logger.info("─ Analyzing Correlations ─")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        logger.warning("Không đủ numeric columns.")
        return {}

    corr_matrix = df[numeric_cols].corr()
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.8:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], val))

    if high_corr:
        logger.info("High correlations (|r| > 0.8):")
        for c1, c2, v in high_corr:
            logger.info(f"  {c1:20s} ↔ {c2:20s}: {v:+.4f}")
    else:
        logger.info("  Không có tương quan cao (|r| > 0.8)")

    return {
        "correlation_matrix":    corr_matrix,
        "high_correlation_pairs": high_corr,
        "numeric_columns":       numeric_cols,
    }


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════
def clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Chạy toàn bộ pipeline làm sạch dữ liệu.

    Parameters
    ----------
    df : DataFrame | None
        Raw data đã load. Nếu None, tự gọi load_raw_data().

    Returns
    -------
    DataFrame đã làm sạch.
    """
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU THÔ")
    logger.info("=" * 60)

    if df is None:
        logger.info("[1/5] Loading raw data...")
        df = load_raw_data()

    logger.info("[2/5] Formatting dtypes...")
    df = format_dtypes(df)

    logger.info("[3/5] Data quality check...")
    report_data_quality(df)

    logger.info("[4/5] Creating full date grid (zero-inflation)...")
    df = create_full_grid(df)

    logger.info("[5/5] Outlier detection & capping (IQR per group)...")
    df = detect_and_handle_outliers(
        df,
        target_col=TARGET_COL,
        group_cols=CONF.outlier_group_cols,
        iqr_multiplier=CONF.outlier_iqr_multiplier,
    )

    out_path = os.path.join(PATHS["processed"], "cleaned_data.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"✓ HOÀN TẤT — Saved → {out_path}")
    logger.info(
        f"  Shape: {df.shape} | "
        f"Date: {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}"
    )
    return df


def main():
    clean()


if __name__ == "__main__":
    main()
