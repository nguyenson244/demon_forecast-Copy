"""
app.py — FastAPI Production Server (Multi-Cluster Support)
"""

import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from src.utils.config_loader import CONF
from src.utils.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="Kinh Do Demand Forecast API", version="2.0")

# --- NẠP MÔ HÌNH ---
try:
    # 1. Nạp các mô hình Prophet (Dictionary)
    prophet_models = joblib.load(Path(CONF.path_models) / "prophet_models.pkl")
    
    # 2. Nạp các mô hình LightGBM theo từng Cụm
    lgbm_models = {}
    cluster_map = CONF.cluster_mapping
    unique_clusters = set(cluster_map.values())
    
    for cid in unique_clusters:
        model_path = Path(CONF.path_models) / f"lightgbm_cluster_{cid}.pkl"
        if model_path.exists():
            lgbm_models[cid] = joblib.load(model_path)
            logger.info(f"✓ Loaded LightGBM Cluster {cid}")
        else:
            logger.warning(f"⚠ Missing model for Cluster {cid} at {model_path}")

except Exception as e:
    logger.error(f"Failed to load models: {e}")
    prophet_models = {}
    lgbm_models = {}

class ForecastRequest(BaseModel):
    brand: str
    date: str  # YYYY-MM-DD

@app.get("/")
def health_check():
    return {"status": "online", "models_loaded": list(lgbm_models.keys())}

@app.post("/predict")
def predict(req: ForecastRequest):
    brand = req.brand.upper()
    if brand not in prophet_models:
        raise HTTPException(status_code=404, detail=f"Brand '{brand}' not found.")
    
    # 1. Dự báo Prophet (Stage 1)
    try:
        m_prophet = prophet_models[brand]
        future = pd.DataFrame({'ds': [pd.to_datetime(req.date)]})
        forecast_p = m_prophet.predict(future)
        prophet_val = forecast_p['yhat'].values[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prophet error: {e}")

    # 2. Dự báo LightGBM (Stage 2)
    try:
        # Xác định cụm của Brand
        cluster_id = CONF.cluster_mapping.get(brand, 1) # Mặc định cụm 1 nếu không thấy
        model_lgbm = lgbm_models.get(cluster_id)
        
        if model_lgbm:
            # LƯU Ý: Trong thực tế, bạn cần tạo lại các đặc trưng (Lags, Holidays) cho ngày req.date
            # Ở đây mình giả định một cơ chế đơn giản hoặc dùng giá trị trung bình residuals
            # Để demo nhanh, mình sẽ dùng giá trị dự báo từ Prophet làm base
            residual_correction = 0 # Placeholder
            
            # Nếu bạn có file features mới nhất, bạn có thể lookup tại đây
            final_val = prophet_val + residual_correction
        else:
            final_val = prophet_val

    except Exception as e:
        logger.error(f"LGBM Error: {e}")
        final_val = prophet_val

    return {
        "brand": brand,
        "date": req.date,
        "cluster": CONF.cluster_mapping.get(brand, "unknown"),
        "prophet_base": round(float(prophet_val), 2),
        "final_prediction": round(float(max(0, final_val)), 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=CONF._cfg["api"]["host"], port=CONF._cfg["api"]["port"])
