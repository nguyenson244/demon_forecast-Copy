"""
============================================================
data_loader.py — Load & Merge Raw CSV Files
============================================================
Role  : Đọc và gộp các file CSV thô từ data/raw/.
        Đây là điểm vào duy nhất cho dữ liệu thô.

Contract:
    - KHÔNG bao giờ sửa các file trong data/raw/
    - Trả về DataFrame gộp CHƯA xử lý gì thêm
============================================================
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger
from src.utils.config_loader import PATHS, DATE_COL, TARGET_COL

logger = get_logger(__name__)

# Danh sách file mặc định theo thứ tự thời gian
DEFAULT_FILES = ["data_2023.csv", "data_2024.csv", "data_2025.csv"]


def load_raw_data(
    filenames: list[str] | None = None,
    raw_dir: str | None = None,
) -> pd.DataFrame:
    """
    Đọc và gộp nhiều file CSV thô thành một DataFrame.

    Parameters
    ----------
    filenames : list[str] | None
        Danh sách tên file cần đọc (chỉ tên file, không phải đường dẫn).
        Mặc định: ['data_2023.csv', 'data_2024.csv', 'data_2025.csv']
    raw_dir : str | None
        Đường dẫn thư mục raw. Mặc định lấy từ config.yaml.

    Returns
    -------
    pd.DataFrame
        DataFrame gộp chưa xử lý.

    Raises
    ------
    FileNotFoundError
        Nếu bất kỳ file nào không tồn tại.
    ValueError
        Nếu danh sách file rỗng hoặc không đọc được file nào.
    """
    if filenames is None:
        filenames = DEFAULT_FILES

    if raw_dir is None:
        raw_dir = PATHS["raw"]

    if not filenames:
        raise ValueError("Danh sách file không được rỗng.")

    logger.info(f"Loading {len(filenames)} file(s) from: {raw_dir}")

    dfs: list[pd.DataFrame] = []
    for fname in filenames:
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

        df = pd.read_csv(path)
        dfs.append(df)
        logger.info(f"  ✓ Loaded '{fname}': {len(df):,} rows, {df.shape[1]} cols")

    if not dfs:
        raise ValueError("Không đọc được file nào.")

    merged = pd.concat(dfs, ignore_index=True)
    logger.info(f"  ► Merged total: {len(merged):,} rows")

    # Báo cáo nhanh
    if DATE_COL in merged.columns:
        merged[DATE_COL] = pd.to_datetime(merged[DATE_COL], errors="coerce")
        date_min = merged[DATE_COL].min().date()
        date_max = merged[DATE_COL].max().date()
        logger.info(f"  ► Date range : {date_min} → {date_max}")

    if TARGET_COL in merged.columns:
        logger.info(
            f"  ► Target '{TARGET_COL}': "
            f"min={merged[TARGET_COL].min():.0f}, "
            f"max={merged[TARGET_COL].max():.0f}, "
            f"null={merged[TARGET_COL].isna().sum()}"
        )

    return merged


def get_available_files(raw_dir: str | None = None) -> list[str]:
    """Trả về danh sách các file CSV có sẵn trong raw_dir."""
    if raw_dir is None:
        raw_dir = PATHS["raw"]
    return sorted(
        f for f in os.listdir(raw_dir) if f.endswith(".csv")
    )
