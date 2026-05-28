"""
============================================================
lightgbm_model.py — LightGBM Stage 2 (Residual Learning)
============================================================
Kiến trúc Hybrid:
  Stage 1 (Prophet): học trend + seasonality → prophet_pred
  Stage 2 (LightGBM): học residual OOS → lgbm_residual_pred
  Final: prophet_pred + lgbm_residual_pred (clipped ≥ 0)

CẢI TIẾN v2:
  1. Per-brand Prophet fallback:
     Nếu LightGBM làm tệ hơn Prophet trên calib holdout
     → revert về prophet_pred cho brand đó.
  2. Two-Part Model cho Cluster 2 (Seasonal brands):
     Stage 1: LGBMClassifier → P(sale > 0)
     Stage 2: LGBMRegressor → residuals (positive days only)
     Final:   (prophet_pred + residual) × P_sale
     Giúp suppress over-prediction trong off-season.

CHỐNG LEAKAGE:
  - LightGBM train ONLY trên phần calibration (OOS residual)
  - eval_set dùng validation split thật sự (không dùng train data)
  - Features loại bỏ target và CBM hiện tại
============================================================
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# XGBoost optional — ensemble nếu có, solo LightGBM nếu không
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from src.utils.logger import get_logger
from src.utils.config_loader import CONF

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# Cột bị loại khỏi features (target, volume hiện tại, leaky)
# ─────────────────────────────────────────────────────────────
_EXCLUDE_COLS = {
    CONF.col_target,
    "cluster",
    "lgbm_residual_pred",
    "final_pred",
    "prophet_pred",
    "oos_prophet_pred",
    "Total CBM",
    "cbm_log",
    "residual_prophet",
    "use_for_lgbm_train",
}


# ═══════════════════════════════════════════════════════════
# 1. CHỌN FEATURES
# ═══════════════════════════════════════════════════════════
def select_features(df: pd.DataFrame) -> list[str]:
    """Chọn tất cả cột số, loại trừ cột target và leaky columns."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in _EXCLUDE_COLS]


# ═══════════════════════════════════════════════════════════
# 2. OPTUNA HPO
# ═══════════════════════════════════════════════════════════
def optimize_hpo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
) -> dict:
    """
    Tối ưu hóa hyperparameters bằng Optuna.

    Dùng X_val/y_val thật sự (không phải train) để đánh giá.
    Early stopping trên validation set → chống overfit.
    """
    def objective(trial: optuna.Trial) -> float:
        # n_estimators KHÔNG đưa vào search space — cố định 500,
        # để early stopping tự tìm best_iter ổn định hơn trên small datasets.
        params = {
            "n_estimators":     500,
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 120),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":     trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples":trial.suggest_int("min_child_samples", 10, 80),
            "lambda_l1":        trial.suggest_float("lambda_l1", 1e-8, 5.0, log=True),
            "lambda_l2":        trial.suggest_float("lambda_l2", 1e-8, 5.0, log=True),
            "verbosity":        -1,
            "random_state":     42,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

        best_iter = getattr(model, "best_iteration_", None)
        # Enforce minimum 150 trees — tránh underfitting trên small val sets
        safe_iter = max(150, int(best_iter)) if (best_iter and best_iter > 0) else 500
        trial.set_user_attr("best_iter", safe_iter)

        preds = model.predict(X_val)
        return float(np.sqrt(np.mean((y_val.values - preds) ** 2)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params.copy()
    best["n_estimators"] = study.best_trial.user_attrs.get("best_iter", 500)
    logger.info(f"  Optuna best n_estimators: {best['n_estimators']}")
    return best


# ═══════════════════════════════════════════════════════════
# 3. TRAIN + PREDICT CHO 1 CLUSTER
# ═══════════════════════════════════════════════════════════
def _train_cluster(
    c_train: pd.DataFrame,
    c_test: pd.DataFrame,
    features: list[str],
    cluster_id: int,
    base_params: dict,
    optuna_trials: int,
) -> tuple[lgb.LGBMRegressor, pd.DataFrame, pd.DataFrame]:
    """
    Huấn luyện LightGBM cho 1 cluster.

    Target = actual - oos_prophet_pred  (OOS residual thật sự)
    eval_set = 20% cuối của calibration data (time-based split)

    Returns: (model, c_train_with_pred, c_test_with_pred)
    """
    # ── Lấy phần có OOS residual (calibration set) ─────────
    calib_mask = c_train["use_for_lgbm_train"].fillna(False).astype(bool)
    c_calib = c_train[calib_mask].copy()

    if len(c_calib) < 50:
        logger.warning(
            f"Cluster {cluster_id}: calibration set quá nhỏ "
            f"({len(c_calib)} rows) → dùng toàn bộ train với in-sample residual."
        )
        # Fallback: dùng toàn bộ train với prophet_pred in-sample
        # (vẫn sai nhưng tránh crash; cần thêm dữ liệu)
        c_calib = c_train.copy()
        y_calib = c_calib[CONF.col_target] - c_calib["prophet_pred"].fillna(0)
    else:
        y_calib = c_calib[CONF.col_target] - c_calib["oos_prophet_pred"]

    X_calib = c_calib[features]

    # Loại bỏ NaN target (brand bị skip trong Prophet)
    valid_mask = y_calib.notna() & X_calib.notna().all(axis=1)
    X_calib = X_calib[valid_mask]
    y_calib = y_calib[valid_mask]

    if len(X_calib) < 20:
        logger.warning(f"Cluster {cluster_id}: Không đủ dữ liệu sau khi lọc NaN → skip.")
        c_train["lgbm_residual_pred"] = 0.0
        c_test["lgbm_residual_pred"] = 0.0
        return None, c_train, c_test

    # ── Time-based train/val split trong calibration set ───
    n_val = max(1, int(len(X_calib) * 0.2))
    X_tr, X_val = X_calib.iloc[:-n_val], X_calib.iloc[-n_val:]
    y_tr, y_val = y_calib.iloc[:-n_val], y_calib.iloc[-n_val:]

    logger.info(
        f"  Cluster {cluster_id} | "
        f"calib={len(X_calib):,} rows | "
        f"lgbm_train={len(X_tr):,} | val={len(X_val):,}"
    )

    # ── HPO (Optuna) hoặc dùng params mặc định ─────────────
    if optuna_trials > 0:
        logger.info(f"  Chạy Optuna ({optuna_trials} trials)...")
        best_params = optimize_hpo(X_tr, y_tr, X_val, y_val, optuna_trials)
        joblib.dump(
            best_params,
            Path(CONF.path_models) / f"lgbm_params_cluster_{cluster_id}.pkl",
        )
    else:
        best_params = base_params.copy()

    # ── Huấn luyện với early stopping trên val set thật ────
    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],          # ← val set thật, không phải train
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # Log feature importance top-10
    fi = pd.Series(model.feature_importances_, index=features).nlargest(10)
    logger.info(f"  Top-10 features (cluster {cluster_id}):\n{fi.to_string()}")

    joblib.dump(model, Path(CONF.path_models) / f"lgbm_cluster_{cluster_id}.pkl")

    # ── Predict ─────────────────────────────────────────────
    X_full_train = c_train[features].fillna(0)
    X_full_test  = c_test[features].fillna(0)

    c_train = c_train.copy()
    c_test  = c_test.copy()

    lgbm_pred_train = model.predict(X_full_train)
    lgbm_pred_test  = model.predict(X_full_test)

    # ── XGBoost Ensemble (nếu có) ───────────────────────────
    if _HAS_XGB:
        xgb_params = {
            "n_estimators":     500,
            "learning_rate":    best_params.get("learning_rate", 0.05),
            "max_depth":        6,
            "subsample":        best_params.get("bagging_fraction", 0.8),
            "colsample_bytree": best_params.get("feature_fraction", 0.8),
            "min_child_weight": best_params.get("min_child_samples", 20),
            "reg_alpha":        best_params.get("lambda_l1", 0.1),
            "reg_lambda":       best_params.get("lambda_l2", 0.1),
            "random_state":     42,
            "verbosity":        0,
            "tree_method":      "hist",
        }
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        joblib.dump(xgb_model, Path(CONF.path_models) / f"xgb_cluster_{cluster_id}.pkl")

        xgb_pred_train = xgb_model.predict(X_full_train)
        xgb_pred_test  = xgb_model.predict(X_full_test)

        # Blend 50/50
        c_train["lgbm_residual_pred"] = 0.5 * lgbm_pred_train + 0.5 * xgb_pred_train
        c_test["lgbm_residual_pred"]  = 0.5 * lgbm_pred_test  + 0.5 * xgb_pred_test
        logger.info(f"  Ensemble LightGBM+XGBoost (50/50) applied for cluster {cluster_id}")
    else:
        logger.warning(
            f"  XGBoost not installed — solo LightGBM for cluster {cluster_id}. "
            "Run: pip install xgboost"
        )
        c_train["lgbm_residual_pred"] = lgbm_pred_train
        c_test["lgbm_residual_pred"]  = lgbm_pred_test

    return model, c_train, c_test


# ═══════════════════════════════════════════════════════════
# 4. TWO-PART MODEL (Cluster 2 — Extreme Seasonal)
# ═══════════════════════════════════════════════════════════
def _train_two_part_cluster(
    c_train: pd.DataFrame,
    c_test: pd.DataFrame,
    features: list[str],
    cluster_id: int,
    base_params: dict,
) -> tuple[lgb.LGBMRegressor, pd.DataFrame, pd.DataFrame]:
    """
    Two-Part Model cho Cluster 2 (THU, TRANG VANG, HAMPER, CADBURY, KOKO).

    Vấn đề: các brand này bán cực kỳ ít ngoài mùa (zero-inflated 70–80%).
    LightGBM thông thường over-predict trong off-season vì học từ peak patterns.

    Giải pháp Two-Part:
      Stage 1 — Classifier (toàn bộ train):
        Input: tất cả features
        Target: binary → (actual > 0)
        Output: P_sale = xác suất ngày có doanh số

      Stage 2 — Regressor (calibration + positive days only):
        Input: features của ngày actual > 0
        Target: residual = actual - oos_prophet_pred
        Output: lgbm_residual_pred

      Final:
        lgbm_residual_pred × P_sale  (gate residual bởi xác suất bán)
        → prophet_pred × P_sale + lgbm_residual_pred × P_sale

    Kết quả: off-season predictions bị suppress về ~0 khi P_sale ≈ 0.
    """
    # ── Stage 1: Classifier trên toàn bộ train ──────────────
    X_all    = c_train[features].fillna(0)
    y_binary = (c_train[CONF.col_target] > 0).astype(int)

    n_pos = int(y_binary.sum())
    n_neg = len(y_binary) - n_pos
    logger.info(
        f"  Cluster {cluster_id} (Two-Part) | "
        f"pos={n_pos} ({n_pos/len(y_binary)*100:.1f}%) "
        f"neg={n_neg}"
    )

    if n_pos < 20:
        logger.warning(
            f"  Cluster {cluster_id}: quá ít ngày bán ({n_pos}) "
            "→ fallback regular training"
        )
        return _train_cluster(c_train, c_test, features, cluster_id, base_params, 0)

    cls_params = {
        "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
        "feature_fraction": 0.8, "min_child_samples": 10,
        "class_weight": "balanced", "verbosity": -1, "random_state": 42,
    }
    classifier = lgb.LGBMClassifier(**cls_params)
    classifier.fit(X_all, y_binary)

    # ── Stage 2: Regressor chỉ trên calib positive days ─────
    calib_mask = c_train["use_for_lgbm_train"].fillna(False).astype(bool)
    c_calib_pos = c_train[calib_mask & (c_train[CONF.col_target] > 0)].copy()

    if len(c_calib_pos) < 15:
        logger.warning(
            f"  Cluster {cluster_id}: calib positive rows quá ít "
            f"({len(c_calib_pos)}) → dùng toàn bộ calib"
        )
        c_calib_pos = c_train[calib_mask].copy()

    y_residual = c_calib_pos[CONF.col_target] - c_calib_pos["oos_prophet_pred"].fillna(0)
    X_calib    = c_calib_pos[features].fillna(0)

    valid_mask = y_residual.notna()
    X_calib, y_residual = X_calib[valid_mask], y_residual[valid_mask]

    n_val = max(1, int(len(X_calib) * 0.2))
    X_tr, X_val = X_calib.iloc[:-n_val], X_calib.iloc[-n_val:]
    y_tr, y_val = y_residual.iloc[:-n_val], y_residual.iloc[-n_val:]

    logger.info(
        f"  Cluster {cluster_id} regressor | "
        f"train={len(X_tr)} val={len(X_val)}"
    )

    regressor = lgb.LGBMRegressor(**base_params)
    regressor.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )

    fi = pd.Series(regressor.feature_importances_, index=features).nlargest(10)
    logger.info(f"  Top-10 features (cluster {cluster_id} regressor):\n{fi.to_string()}")

    # ── Predict ──────────────────────────────────────────────
    X_train_full = c_train[features].fillna(0)
    X_test_full  = c_test[features].fillna(0)

    p_sale_train = classifier.predict_proba(X_train_full)[:, 1]
    p_sale_test  = classifier.predict_proba(X_test_full)[:, 1]

    resid_train = regressor.predict(X_train_full)
    resid_test  = regressor.predict(X_test_full)

    c_train = c_train.copy()
    c_test  = c_test.copy()

    # Gate residual bởi P_sale → suppress off-season over-prediction
    c_train["lgbm_residual_pred"] = resid_train * p_sale_train
    c_test["lgbm_residual_pred"]  = resid_test  * p_sale_test
    c_train["p_sale"] = p_sale_train
    c_test["p_sale"]  = p_sale_test

    joblib.dump(
        classifier,
        Path(CONF.path_models) / f"lgbm_classifier_cluster_{cluster_id}.pkl",
    )
    joblib.dump(
        regressor,
        Path(CONF.path_models) / f"lgbm_regressor_cluster_{cluster_id}.pkl",
    )

    return regressor, c_train, c_test


# ═══════════════════════════════════════════════════════════
# 5. PER-BRAND PROPHET FALLBACK
# ═══════════════════════════════════════════════════════════
def _apply_per_brand_fallback(
    train_final: pd.DataFrame,
    test_final: pd.DataFrame,
    tolerance: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Đánh giá WMAPE per-brand trên calibration holdout.
    Nếu Hybrid WMAPE > Prophet WMAPE + tolerance → revert về Prophet.

    Lý do:
      - KINH DO CAKE: LightGBM thêm noise không có ích (tệ hơn 21pp)
      - Fallback nhẹ nhàng: chỉ set lgbm_residual_pred = 0 cho brand đó
      - Không ảnh hưởng đến các brand khác

    Parameters
    ----------
    tolerance : float
        Ngưỡng tối thiểu (pp) Hybrid phải cải thiện được so với Prophet.
        Default 2.0 — tránh revert khi cải thiện rất nhỏ.
    """
    brands = sorted(train_final["BRAND"].unique())
    fallback_brands = []

    for brand in brands:
        calib = train_final[
            (train_final["BRAND"] == brand) &
            (train_final.get("use_for_lgbm_train", pd.Series(False, index=train_final.index)).fillna(False))
        ]
        if len(calib) < 10:
            continue

        act = calib[CONF.col_target].values
        pp  = calib["prophet_pred"].fillna(0).values
        hp  = calib["final_pred"].fillna(0).values

        denom = float(np.sum(np.abs(act))) + 1e-8
        prophet_wmape = float(np.sum(np.abs(act - pp)) / denom * 100)
        hybrid_wmape  = float(np.sum(np.abs(act - hp)) / denom * 100)

        if hybrid_wmape > prophet_wmape + tolerance:
            fallback_brands.append(brand)
            logger.info(
                f"  ⟳ FALLBACK {brand}: "
                f"calib Hybrid={hybrid_wmape:.1f}% > Prophet={prophet_wmape:.1f}% "
                f"(+{hybrid_wmape - prophet_wmape:.1f}pp) → use Prophet-only"
            )
            # Revert: zero out LightGBM residual
            for df in (train_final, test_final):
                mask = df["BRAND"] == brand
                df.loc[mask, "lgbm_residual_pred"] = 0.0
                df.loc[mask, "final_pred"] = (
                    df.loc[mask, "prophet_pred"].fillna(0).clip(lower=0)
                )
                if "p_sale" in df.columns:
                    df.loc[mask, "p_sale"] = 1.0
        else:
            logger.info(
                f"  ✓ KEEP    {brand}: "
                f"calib Hybrid={hybrid_wmape:.1f}% <= Prophet={prophet_wmape:.1f}%"
            )

    if fallback_brands:
        logger.info(
            f"\nFallback brands ({len(fallback_brands)}): "
            + ", ".join(fallback_brands)
        )
    else:
        logger.info("No brands reverted to Prophet — LightGBM helped all.")

    return train_final, test_final, set(fallback_brands)


# ═══════════════════════════════════════════════════════════
# 6. BLEND WEIGHT + BIAS CORRECTION
# ═══════════════════════════════════════════════════════════
def _apply_blend_and_bias(
    train_final: pd.DataFrame,
    test_final: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hai bước post-processing per brand, tính trên calibration holdout:

    Bước 1 — Blend weight:
      Tìm alpha tối ưu: final = alpha × prophet_pred + (1-alpha) × hybrid_pred
      Grid search [0.0, 0.1, ..., 1.0] theo WMAPE trên calib set.
      alpha=0 → thuần Hybrid, alpha=1 → thuần Prophet.
      Giúp các brand mà LightGBM cải thiện ít có thể dựa nhiều hơn vào Prophet.

    Bước 2 — Bias correction:
      factor = sum(actual_calib) / sum(blended_calib)
      Clip [0.5, 2.0] để tránh correction quá cực đoan.
      Giảm over/under-forecast hệ thống per brand.

    Chống leakage:
      - Cả alpha và factor đều tính từ calib set (out-of-sample đối với Prophet,
        in-sample đối với LightGBM nhưng chỉ là scalar correction → ít overfit).
      - Áp dụng cho cả train (diagnostic) và test (prediction thực).
    """
    brands = sorted(train_final["BRAND"].unique())
    alpha_grid = np.arange(0.0, 1.05, 0.1)
    blend_meta: dict[str, dict] = {}

    for brand in brands:
        calib = train_final[
            (train_final["BRAND"] == brand) &
            (train_final.get(
                "use_for_lgbm_train",
                pd.Series(False, index=train_final.index)
            ).fillna(False))
        ]
        if len(calib) < 10:
            blend_meta[brand] = {"alpha": 0.0, "factor": 1.0}
            continue

        act = calib[CONF.col_target].values
        pp  = calib["prophet_pred"].fillna(0).values
        hp  = calib["final_pred"].fillna(0).values
        denom = float(np.sum(np.abs(act))) + 1e-8

        # ── Bước 1: tìm alpha tối ưu ────────────────────────
        best_alpha, best_wmape = 0.0, float("inf")
        for alpha in alpha_grid:
            blended = alpha * pp + (1.0 - alpha) * hp
            wmape = float(np.sum(np.abs(act - blended)) / denom * 100)
            if wmape < best_wmape:
                best_wmape, best_alpha = wmape, alpha

        blended_calib = best_alpha * pp + (1.0 - best_alpha) * hp

        # ── Bước 2: bias correction ──────────────────────────
        sum_act     = float(np.sum(act))
        sum_blended = float(np.sum(blended_calib))

        if sum_act < 1e-8 or sum_blended < 1e-8:
            factor = 1.0  # brand 0-sale: không correction
        else:
            factor = float(np.clip(sum_act / sum_blended, 0.5, 2.0))

        blend_meta[brand] = {"alpha": float(best_alpha), "factor": float(factor)}

        logger.info(
            f"  {brand:20s}: alpha={best_alpha:.1f}  "
            f"bias={factor:.3f}  calib_WMAPE={best_wmape:.1f}%"
        )

        # ── Áp dụng cho train và test ────────────────────────
        for df in (train_final, test_final):
            mask = df["BRAND"] == brand
            pp_b = df.loc[mask, "prophet_pred"].fillna(0)
            hp_b = df.loc[mask, "final_pred"].fillna(0)
            df.loc[mask, "final_pred"] = (
                (best_alpha * pp_b + (1.0 - best_alpha) * hp_b) * factor
            ).clip(lower=0)

    return train_final, test_final, blend_meta


# ═══════════════════════════════════════════════════════════
# 7. PUBLIC API
# ═══════════════════════════════════════════════════════════
def run_lightgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Huấn luyện đa mô hình LightGBM theo cụm (cluster).

    Parameters
    ----------
    train_df : DataFrame
        Train features + prophet_pred + oos_prophet_pred + use_for_lgbm_train
    test_df : DataFrame
        Test features + prophet_pred

    Returns
    -------
    (train_final, test_final) — có thêm cột lgbm_residual_pred, final_pred
    """
    logger.info("=" * 60)
    logger.info("BƯỚC 4: LIGHTGBM MULTI-CLUSTER (RESIDUAL LEARNING)")
    logger.info("=" * 60)

    # ── Gán cluster ─────────────────────────────────────────
    cluster_map = CONF.cluster_mapping
    train_df = train_df.copy()
    test_df  = test_df.copy()

    train_df["cluster"] = train_df[CONF.col_brand].map(cluster_map).fillna(1).astype(int)
    test_df["cluster"]  = test_df[CONF.col_brand].map(cluster_map).fillna(1).astype(int)

    # ── Chọn features ───────────────────────────────────────
    features = select_features(train_df)
    unique_clusters = sorted(train_df["cluster"].unique())
    base_params = CONF.lgbm_params or {
        "n_estimators":     500,
        "learning_rate":    0.05,
        "num_leaves":       63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_child_samples":20,
        "lambda_l1":        0.1,
        "lambda_l2":        0.1,
        "verbosity":        -1,
        "random_state":     42,
    }
    optuna_trials = CONF.lgbm_optuna_trials

    logger.info(f"Clusters: {unique_clusters} | Features: {len(features)}")
    logger.info(
        f"OOS calibration: "
        f"{train_df['use_for_lgbm_train'].sum():,}/{len(train_df):,} rows"
        if "use_for_lgbm_train" in train_df.columns
        else "WARNING: use_for_lgbm_train column missing — fallback to in-sample"
    )

    all_train, all_test = [], []

    for cid in unique_clusters:
        logger.info(f"\n--- Cluster {cid} ---")
        c_train = train_df[train_df["cluster"] == cid].copy()
        c_test  = test_df[test_df["cluster"] == cid].copy()

        if len(c_train) == 0:
            logger.warning(f"Cluster {cid}: không có dữ liệu train → skip.")
            continue

        # Cluster 2 (Seasonal) dùng Two-Part Model
        if cid == 2:
            logger.info(f"  [Two-Part Model] Cluster {cid} (Seasonal brands)")
            _, c_train, c_test = _train_two_part_cluster(
                c_train, c_test, features, cid, base_params
            )
        else:
            _, c_train, c_test = _train_cluster(
                c_train, c_test, features, cid, base_params, optuna_trials
            )
        all_train.append(c_train)
        all_test.append(c_test)

    # ── Gộp kết quả ─────────────────────────────────────────
    sort_cols = [CONF.col_date, CONF.col_brand]
    train_final = pd.concat(all_train).sort_values(sort_cols).reset_index(drop=True)
    test_final  = pd.concat(all_test).sort_values(sort_cols).reset_index(drop=True)

    # p_sale column: 1.0 cho clusters 0/1, từ classifier cho cluster 2
    for df in (train_final, test_final):
        if "p_sale" not in df.columns:
            df["p_sale"] = 1.0
        else:
            df["p_sale"] = df["p_sale"].fillna(1.0)

    for df in (train_final, test_final):
        prophet_p = df["prophet_pred"].fillna(0)
        lgbm_res  = df["lgbm_residual_pred"].fillna(0)
        p_sale    = df["p_sale"]
        # Two-Part: gate toàn bộ prediction của Cluster 2 bởi P_sale
        # Cluster 0/1: p_sale = 1.0 → không thay đổi
        df["final_pred"] = ((prophet_p + lgbm_res) * p_sale).clip(lower=0)

    # ── Per-brand Prophet fallback ───────────────────────────
    logger.info("\n─ Per-Brand Fallback Check ─")
    train_final, test_final, fallback_brands = _apply_per_brand_fallback(train_final, test_final)

    # ── Blend weight + Bias correction ───────────────────────
    logger.info("\n─ Blend Weight + Bias Correction ─")
    train_final, test_final, blend_meta = _apply_blend_and_bias(train_final, test_final)

    # ── Lưu brand forecast metadata cho API ─────────────────
    brand_meta: dict[str, dict] = {}
    for brand in sorted(CONF.cluster_mapping.keys()):
        m = blend_meta.get(brand, {})
        brand_meta[brand] = {
            "alpha":       m.get("alpha", 0.0),
            "factor":      m.get("factor", 1.0),
            "is_fallback": brand in fallback_brands,
            "cluster":     int(CONF.cluster_mapping.get(brand, 1)),
        }
    meta_path = Path(CONF.path_models) / "brand_forecast_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as _f:
        json.dump(brand_meta, _f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Brand metadata → {meta_path}")

    # ── Lưu kết quả ─────────────────────────────────────────
    out = Path(CONF.path_metrics)
    train_final.to_csv(out / "train_predictions.csv", index=False)
    test_final.to_csv(out / "test_predictions.csv",  index=False)
    logger.info(f"✓ Hoàn tất LightGBM. Kết quả → {out}")

    # ── Log metrics nhanh ───────────────────────────────────
    act  = test_final[CONF.col_target].values
    pred = test_final["final_pred"].values
    mask = ~np.isnan(act) & ~np.isnan(pred)
    rmse = float(np.sqrt(np.mean((act[mask] - pred[mask]) ** 2)))
    wmape = float(
        np.sum(np.abs(act[mask] - pred[mask])) / (np.sum(np.abs(act[mask])) + 1e-8) * 100
    )
    logger.info(f"  Test RMSE={rmse:,.1f} | WMAPE={wmape:.2f}%")

    return train_final, test_final
