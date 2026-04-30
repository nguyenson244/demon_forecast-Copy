import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import optuna
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config_loader import CONF

logger = get_logger(__name__)

def optimize_hpo(X, y, n_trials=50):
    """Tối ưu hóa Hyperparameters bằng Optuna."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'verbosity': -1,
            'random_state': 42
        }
        split = int(len(X) * 0.8)
        X_t, X_v = X.iloc[:split], X.iloc[split:]
        y_t, y_v = y.iloc[:split], y.iloc[split:]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], eval_metric='rmse', callbacks=[lgb.early_stopping(20, verbose=False)])
        preds = model.predict(X_v)
        return np.sqrt(np.mean((y_v - preds)**2))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def run_lightgbm(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Huấn luyện đa mô hình LightGBM dựa trên phân cụm (Cluster).
    """
    logger.info("============================================================")
    logger.info("BƯỚC 4: HUẤN LUYỆN ĐA MÔ HÌNH LIGHTGBM (MULTI-CLUSTER)")
    logger.info("============================================================")

    # 1. Xác định Cụm
    cluster_map = CONF.cluster_mapping
    train_df['cluster'] = train_df[CONF.col_brand].map(cluster_map).fillna(1).astype(int)
    test_df['cluster'] = test_df[CONF.col_brand].map(cluster_map).fillna(1).astype(int)
    
    # 2. Chọn đặc trưng (Chỉ lấy cột số)
    # Loại bỏ các cột không phải số (BRAND, CATEGORY, v.v.)
    features = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Loại bỏ các cột mục tiêu và ID
    exclude = [CONF.col_target, 'cluster', 'lgbm_residual_pred', 'final_pred', 'prophet_pred']
    features = [f for f in features if f not in exclude]
    
    unique_clusters = sorted(train_df['cluster'].unique())
    logger.info(f"Phát hiện {len(unique_clusters)} cụm: {unique_clusters}")
    logger.info(f"Sử dụng {len(features)} đặc trưng dạng số.")

    all_test_preds = []
    all_train_preds = []
    
    for cluster_id in unique_clusters:
        logger.info(f"\n--- Đang xử lý Cụm {cluster_id} ---")
        c_train = train_df[train_df['cluster'] == cluster_id].copy()
        c_test = test_df[test_df['cluster'] == cluster_id].copy()
        
        if len(c_train) == 0: continue
            
        y_train = c_train[CONF.col_target] - c_train['prophet_pred']
        X_train = c_train[features]
        X_test = c_test[features]

        best_params = CONF.lgbm_params
        if CONF.lgbm_optuna_trials > 0:
            logger.info(f"Chạy Optuna ({CONF.lgbm_optuna_trials} trials)...")
            best_params = optimize_hpo(X_train, y_train, CONF.lgbm_optuna_trials)
            joblib.dump(best_params, Path(CONF.path_models) / f"lightgbm_params_cluster_{cluster_id}.pkl")

        # Huấn luyện
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='rmse', callbacks=[lgb.log_evaluation(period=0)])

        joblib.dump(model, Path(CONF.path_models) / f"lightgbm_cluster_{cluster_id}.pkl")

        # Dự báo
        c_train['lgbm_residual_pred'] = model.predict(X_train)
        c_test['lgbm_residual_pred'] = model.predict(X_test)
        all_train_preds.append(c_train)
        all_test_preds.append(c_test)

    # Gộp và lưu
    train_final = pd.concat(all_train_preds).sort_values([CONF.col_date, CONF.col_brand])
    test_final = pd.concat(all_test_preds).sort_values([CONF.col_date, CONF.col_brand])

    for df in [train_final, test_final]:
        df['final_pred'] = (df['prophet_pred'] + df['lgbm_residual_pred']).clip(lower=0)

    train_final.to_csv(Path(CONF.path_metrics) / "train_predictions.csv", index=False)
    test_final.to_csv(Path(CONF.path_metrics) / "test_predictions.csv", index=False)
    
    logger.info(f"✓ Hoàn tất. Kết quả lưu tại {CONF.path_metrics}")
    return train_final, test_final
