"""
config_loader.py — Centralized Config Object with Full Backward Compatibility
"""

import os
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH  = _PROJECT_ROOT / "config" / "config.yaml"

def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class Config:
    def __init__(self):
        self._cfg = _load_yaml(_CONFIG_PATH)
        
        # Paths
        self.path_root      = str(_PROJECT_ROOT)
        self.path_raw       = self._abs(self._cfg["paths"]["raw"])
        self.path_processed = self._abs(self._cfg["paths"]["processed"])
        self.path_features  = self._abs(self._cfg["paths"]["features"])
        self.path_models    = self._abs(self._cfg["paths"]["models"])
        self.path_figures   = self._abs(self._cfg["paths"]["figures"])
        self.path_metrics   = self._abs(self._cfg["paths"]["metrics"])
        self.path_logs      = self._abs(self._cfg["paths"]["logs"])
        
        # Backward compat for PATHS dict
        self.PATHS = {
            "raw": self.path_raw,
            "processed": self.path_processed,
            "features": self.path_features,
            "models": self.path_models,
            "figures": self.path_figures,
            "metrics": self.path_metrics,
            "logs": self.path_logs
        }
        
        # Columns
        self.col_date   = self._cfg["columns"]["date"]
        self.col_brand  = self._cfg["columns"]["brand"]
        self.col_cat    = self._cfg["columns"]["category"]
        self.col_target = self._cfg["columns"]["target"]
        
        # Split
        self.split_date = pd.Timestamp(self._cfg["split_date"])
        
        # Clustering
        self.cluster_mapping = self._cfg.get("cluster_mapping", {})
        
        # Holidays
        _hol = self._cfg.get("holidays", {})
        self.tet_dates = {int(y): pd.Timestamp(d) for y, d in _hol.get("tet", {}).items()}
        self.mid_autumn_dates = {int(y): pd.Timestamp(d) for y, d in _hol.get("mid_autumn", {}).items()}
        self.holidays_dict = {"tet": self.tet_dates, "mid_autumn": self.mid_autumn_dates}

        # Features
        _fe = self._cfg.get("features", {})
        self.lag_periods = _fe.get("lag_periods", [])
        self.rolling_windows = _fe.get("rolling_windows", [])
        self.rolling_stats = _fe.get("rolling_stats", [])
        self.same_weekday_weeks = _fe.get("same_weekday_weeks", [])

        # LightGBM
        self.lgbm_params = self._cfg.get("lightgbm", {}).get("params", {})
        self.lgbm_optuna_trials = self._cfg.get("lightgbm", {}).get("optuna_n_trials", 50)

    def _abs(self, rel_path: str) -> str:
        return str(_PROJECT_ROOT / rel_path)

# Instance duy nhất
CONF = Config()

# --- EXPORTS CHO COMPATIBILITY ---
PATHS = CONF.PATHS
DATE_COL = CONF.col_date
TARGET_COL = CONF.col_target
SPLIT_DATE = CONF.split_date

TET_DATES = CONF.tet_dates
MID_AUTUMN_DATES = CONF.mid_autumn_dates
HOLIDAYS = CONF.holidays_dict

LAG_PERIODS = CONF.lag_periods
ROLLING_WINDOWS = CONF.rolling_windows
ROLLING_STATS = CONF.rolling_stats
SAME_WEEKDAY_WEEKS = CONF.same_weekday_weeks

def ensure_dirs():
    for p in [CONF.path_processed, CONF.path_features, CONF.path_models, 
              CONF.path_figures, CONF.path_metrics, CONF.path_logs]:
        os.makedirs(p, exist_ok=True)
