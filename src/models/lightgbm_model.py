"""
============================================================
lightgbm_model.py — LightGBM Global Stage 2
============================================================
Migrate từ: src/model_lightgbm.py
Role      : Huấn luyện MỘT mô hình LightGBM Global.
            Target = R(t) = Residuals từ Prophet Stage 1.
            Ŷ_Final(t) = Ŷ_Prophet(t) + R̂_LightGBM(t)

★ CHỐNG RÒ RỈ: Time Series CV (KHÔNG random K-Fold).
============================================================
"""

from __future__ import annotations

import os
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.utils.logger import get_logger
from src.utils.config_loader import (
    PATHS, DATE_COL, TARGET_COL,
    LIGHTGBM_DEVICE, OPTUNA_N_TRIALS, TSCV_N_SPLITS,
    ensure_dirs,
)

logger = get_logger(__name__)

# Cột không phải feature
NON_FEATURE_COLS = [
    DATE_COL, TARGET_COL, "Total CBM", "WHSEID", "Week",
    "prophet_pred", "residual_prophet", "Day",
]
CATEGORICAL_COLS = ["BRAND", "CATEGORY"]


# ═══════════════════════════════════════════════════════════
# 1. CHUẨN BỊ FEATURES
# ═══════════════════════════════════════════════════════════
def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Label-encode categoricals, build X matrix, extract residual target."""
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col + "_encoded"] = df[col].astype("category").cat.codes

    target_col = "residual_prophet"
    y = df[target_col].copy()

    drop_cols   = NON_FEATURE_COLS + CATEGORICAL_COLS
    feature_cols = [c for c in df.columns if c not in drop_cols and c != target_col]
    X = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)

    logger.info(f"  Features: {X.shape[1]} cols | Samples: {X.shape[0]:,} | NaN target: {y.isna().sum()}")
    return X, y, feature_cols


# ═══════════════════════════════════════════════════════════
# 2. TIME SERIES K-FOLD CV
# ═══════════════════════════════════════════════════════════
def time_series_kfold(dates: np.ndarray, n_splits: int = TSCV_N_SPLITS) -> list:
    """Walk-forward time series splits — KHÔNG random."""
    sorted_idx = dates.argsort()
    n          = len(sorted_idx)
    fold_size  = n // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        val_end   = min(fold_size * (i + 2), n)
        train_idx = sorted_idx[:train_end]
        val_idx   = sorted_idx[train_end:val_end]
        if len(val_idx) > 0:
            splits.append((train_idx, val_idx))
    return splits


# ═══════════════════════════════════════════════════════════
# 3. OPTUNA OBJECTIVE
# ═══════════════════════════════════════════════════════════
def create_objective(X_train, y_train, train_dates, device):
    def objective(trial):
        params = {
            "objective":       "regression",
            "metric":          "rmse",
            "boosting_type":   "gbdt",
            "device":          device,
            "verbosity":       -1,
            "n_jobs":          -1,
            "learning_rate":   trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "n_estimators":    trial.suggest_int("n_estimators", 200, 2000, step=100),
            "max_depth":       trial.suggest_int("max_depth", 4, 12),
            "num_leaves":      trial.suggest_int("num_leaves", 15, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq":    trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1":       trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2":       trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }

        splits    = time_series_kfold(train_dates)
        cv_scores = []

        for train_idx, val_idx in splits:
            X_tr, y_tr   = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.iloc[val_idx],   y_train.iloc[val_idx]

            valid_tr  = ~y_tr.isna()
            valid_val = ~y_val.isna()
            X_tr,  y_tr  = X_tr[valid_tr],   y_tr[valid_tr]
            X_val, y_val = X_val[valid_val], y_val[valid_val]

            if len(X_tr) == 0 or len(X_val) == 0:
                continue

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            pred = model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val.values - pred) ** 2))
            cv_scores.append(rmse)

        return np.mean(cv_scores) if cv_scores else float("inf")

    return objective


# ═══════════════════════════════════════════════════════════
# 4. TRAIN GLOBAL LIGHTGBM
# ═══════════════════════════════════════════════════════════
def train_global_lightgbm(
    X_train,
    y_train,
    train_dates,
    n_trials: int = OPTUNA_N_TRIALS,
    device:   str = LIGHTGBM_DEVICE,
):
    """Optuna HPO + final model training."""
    # GPU availability check
    actual_device = device
    try:
        test_model = lgb.LGBMRegressor(device=device, n_estimators=5, verbosity=-1)
        test_model.fit(X_train.iloc[:100].fillna(0), y_train.iloc[:100].fillna(0))
        logger.info(f"✓ GPU available — device='{device}'")
    except Exception as e:
        actual_device = "cpu"
        logger.warning(f"GPU unavailable ({str(e)[:60]}) → fallback CPU")

    study = optuna.create_study(
        direction="minimize",
        study_name="lightgbm_hpo",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    objective = create_objective(X_train, y_train, train_dates, actual_device)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best CV RMSE: {study.best_value:,.2f}")

    best_params = {**study.best_params, **{
        "objective":     "regression",
        "metric":        "rmse",
        "boosting_type": "gbdt",
        "device":        actual_device,
        "verbosity":     -1,
        "n_jobs":        -1,
    }}

    valid_mask = ~y_train.isna()
    X_final, y_final = X_train[valid_mask].copy(), y_train[valid_mask].copy()

    logger.info(f"Training final model on {len(X_final):,} samples...")
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_final, y_final)
    logger.info("✓ Final model trained.")

    return final_model, study.best_params, study


# ═══════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE PLOT
# ═══════════════════════════════════════════════════════════
def plot_feature_importance(model, feature_names: list[str], top_n: int = 30) -> None:
    import matplotlib.pyplot as plt

    importance = model.feature_importances_
    indices    = np.argsort(importance)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), importance[indices][::-1], color="#4A90D9", edgecolor="#2D5F8A")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (split)")
    ax.set_title(f"LightGBM Global — Top {top_n} Features", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(PATHS["figures"], "feature_importance.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved feature importance → {save_path}")


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════
def run_lightgbm(
    train_df: pd.DataFrame | None = None,
    test_df:  pd.DataFrame | None = None,
) -> tuple:
    """
    Chạy LightGBM Global Stage 2.

    Returns
    -------
    (model, test_df_with_preds, train_df_with_preds)
    """
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("BƯỚC 4 — GIAI ĐOẠN 2: LIGHTGBM GLOBAL")
    logger.info("=" * 60)

    if train_df is None:
        train_path = os.path.join(PATHS["features"], "train_with_residuals.csv")
        test_path  = os.path.join(PATHS["features"], "test_with_residuals.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Chưa có train_with_residuals.csv. Chạy prophet_model.py trước!")
        train_df = pd.read_csv(train_path, parse_dates=[DATE_COL])
        test_df  = pd.read_csv(test_path,  parse_dates=[DATE_COL])

    logger.info(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    logger.info("[1/4] Preparing features...")
    X_train, y_train, feature_cols = prepare_features(train_df)
    X_test,  _,       _            = prepare_features(test_df)
    train_dates = train_df[DATE_COL].values.astype("datetime64[ns]")

    logger.info("[2/4] Training LightGBM Global with Optuna HPO...")
    model, best_params, study = train_global_lightgbm(X_train, y_train, train_dates)

    logger.info("[3/4] Generating final predictions...")
    residual_pred_test  = model.predict(X_test.replace([np.inf, -np.inf], np.nan))
    final_pred_test     = np.clip(test_df["prophet_pred"].values + residual_pred_test, 0, None)

    test_df["lgbm_residual_pred"] = residual_pred_test
    test_df["final_pred"]         = final_pred_test
    test_df["actual"]             = test_df[TARGET_COL]

    residual_pred_train = model.predict(X_train.replace([np.inf, -np.inf], np.nan))
    train_df["lgbm_residual_pred"] = residual_pred_train
    train_df["final_pred"]         = np.clip(train_df["prophet_pred"].values + residual_pred_train, 0, None)
    train_df["actual"]             = train_df[TARGET_COL]

    logger.info("[4/4] Saving results...")

    pred_cols = [DATE_COL, "BRAND", "CATEGORY", TARGET_COL,
                 "prophet_pred", "lgbm_residual_pred", "final_pred"]

    pred_path       = os.path.join(PATHS["metrics"], "test_predictions.csv")
    train_pred_path = os.path.join(PATHS["metrics"], "train_predictions.csv")
    model_path      = os.path.join(PATHS["models"],  "lightgbm_global.pkl")
    params_path     = os.path.join(PATHS["models"],  "lightgbm_best_params.pkl")

    test_df[pred_cols].to_csv(pred_path,       index=False)
    train_df[pred_cols].to_csv(train_pred_path, index=False)
    joblib.dump(model,       model_path)
    joblib.dump(best_params, params_path)
    plot_feature_importance(model, feature_cols)

    logger.info(f"✓ HOÀN TẤT LIGHTGBM GLOBAL STAGE 2")
    logger.info(f"  Saved predictions → {pred_path}")
    logger.info(f"  Saved model       → {model_path}")

    return model, test_df, train_df


def main():
    run_lightgbm()


if __name__ == "__main__":
    main()
