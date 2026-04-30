"""
config_loader.py — YAML Config Loader
============================================================
Đọc config/config.yaml và expose các hằng số đã được type-annotate
để các module khác import giống như config.py cũ — nhưng từ YAML.

Usage
-----
    from src.utils.config_loader import cfg, PATHS, COLUMNS, HOLIDAYS

    raw_dir    = PATHS["raw"]
    date_col   = COLUMNS["date"]
    tet_dates  = HOLIDAYS["tet"]
============================================================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # src/utils/ → project/
_CONFIG_PATH  = _PROJECT_ROOT / "config" / "config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Load once at import time
# ---------------------------------------------------------------------------
cfg: dict[str, Any] = _load_yaml(_CONFIG_PATH)

# ── Paths (absolute) ────────────────────────────────────────────────────────
def _abs(rel: str) -> str:
    return str(_PROJECT_ROOT / rel)

PATHS: dict[str, str] = {k: _abs(v) for k, v in cfg["paths"].items()}

# ── Column names ────────────────────────────────────────────────────────────
COLUMNS: dict[str, str] = cfg["columns"]
DATE_COL:   str = COLUMNS["date"]
TARGET_COL: str = COLUMNS["target"]

# ── Split date ───────────────────────────────────────────────────────────────
SPLIT_DATE: pd.Timestamp = pd.Timestamp(cfg["split_date"])

# ── Holidays ─────────────────────────────────────────────────────────────────
_raw_holidays = cfg.get("holidays", {})

TET_DATES: dict[int, pd.Timestamp] = {
    int(yr): pd.Timestamp(dt)
    for yr, dt in _raw_holidays.get("tet", {}).items()
}

MID_AUTUMN_DATES: dict[int, pd.Timestamp] = {
    int(yr): pd.Timestamp(dt)
    for yr, dt in _raw_holidays.get("mid_autumn", {}).items()
}

HOLIDAYS: dict[str, dict] = {
    "tet":        TET_DATES,
    "mid_autumn": MID_AUTUMN_DATES,
}

# ── Feature engineering params ───────────────────────────────────────────────
_fe = cfg.get("features", {})
LAG_PERIODS:        list[int] = _fe.get("lag_periods", [])
SAME_WEEKDAY_WEEKS: list[int] = _fe.get("same_weekday_weeks", [])
ROLLING_WINDOWS:    list[int] = _fe.get("rolling_windows", [])
ROLLING_STATS:      list[str] = _fe.get("rolling_stats", [])

# ── LightGBM params ──────────────────────────────────────────────────────────
_lgb             = cfg.get("lightgbm", {})
LIGHTGBM_DEVICE: str = _lgb.get("device", "cpu")
OPTUNA_N_TRIALS: int = _lgb.get("optuna_n_trials", 50)
TSCV_N_SPLITS:   int = _lgb.get("tscv_n_splits", 5)

# ── API params ───────────────────────────────────────────────────────────────
_api      = cfg.get("api", {})
API_HOST: str = _api.get("host", "0.0.0.0")
API_PORT: int = _api.get("port", 8000)


# ---------------------------------------------------------------------------
# Helper: ensure all output directories exist
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create all output directories if they don't exist."""
    for key in ["processed", "features", "models", "figures", "metrics", "logs"]:
        os.makedirs(PATHS[key], exist_ok=True)
