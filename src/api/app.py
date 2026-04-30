"""
============================================================
app.py — FastAPI Forecast API
============================================================
Role   : Expose endpoint dự báo nhu cầu qua REST API.
         Sử dụng model đã train (prophet_models.pkl +
         lightgbm_global.pkl) từ thư mục models/.

Endpoints:
    GET  /                    Health check
    GET  /info                Thông tin model
    POST /forecast            Dự báo theo brand/category
    POST /forecast/batch      Dự báo hàng loạt

Usage:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
============================================================
"""

from __future__ import annotations

import os
from datetime import datetime, date
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.logger import get_logger
from src.utils.config_loader import PATHS, DATE_COL, TARGET_COL

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Demand Forecasting API",
    description="Hybrid Prophet + LightGBM Demand Forecasting — Kinh Đô FMCG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model Cache — load once at startup
# ---------------------------------------------------------------------------
_models: dict = {}


def _load_models() -> dict:
    """Load prophet_models.pkl và lightgbm_global.pkl (lazy, cached)."""
    global _models
    if _models:
        return _models

    prophet_path = os.path.join(PATHS["models"], "prophet_models.pkl")
    lgbm_path    = os.path.join(PATHS["models"], "lightgbm_global.pkl")

    if not os.path.exists(prophet_path):
        raise RuntimeError(
            f"Prophet model không tồn tại: {prophet_path}. "
            "Chạy pipeline trước: python -m src.pipeline.pipeline"
        )
    if not os.path.exists(lgbm_path):
        raise RuntimeError(
            f"LightGBM model không tồn tại: {lgbm_path}. "
            "Chạy pipeline trước: python -m src.pipeline.pipeline"
        )

    _models["prophet"] = joblib.load(prophet_path)
    _models["lgbm"]    = joblib.load(lgbm_path)
    logger.info("✓ Models loaded successfully.")
    return _models


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    brand:      str               = Field(..., example="KinhDo",   description="Tên Brand")
    category:   str               = Field(..., example="TET",       description="Category sản phẩm")
    start_date: date              = Field(..., example="2025-07-01", description="Ngày bắt đầu dự báo")
    end_date:   date              = Field(..., example="2025-12-31", description="Ngày kết thúc dự báo")
    horizon:    Optional[int]     = Field(None, ge=1, le=365,
                                          description="Số ngày dự báo (ưu tiên hơn end_date nếu truyền)")


class ForecastPoint(BaseModel):
    date:          str
    prophet_pred:  float
    final_pred:    float
    lower_bound:   float
    upper_bound:   float


class ForecastResponse(BaseModel):
    brand:        str
    category:     str
    start_date:   str
    end_date:     str
    n_days:       int
    total_pred:   float
    forecast:     list[ForecastPoint]


class BatchForecastRequest(BaseModel):
    requests: list[ForecastRequest]


class ModelInfoResponse(BaseModel):
    brands_available:     list[str]
    model_files:          dict[str, str]
    api_version:          str


# ---------------------------------------------------------------------------
# Forecast Logic
# ---------------------------------------------------------------------------
def _generate_prophet_forecast(
    brand: str,
    start_date: pd.Timestamp,
    end_date:   pd.Timestamp,
    models:     dict,
) -> pd.DataFrame:
    """Dùng Prophet model đã train để dự báo khoảng thời gian mới."""
    prophet_models = models["prophet"]

    if brand not in prophet_models or prophet_models[brand] is None:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy Prophet model cho brand '{brand}'. "
                   f"Brands có sẵn: {list(prophet_models.keys())}"
        )

    model      = prophet_models[brand]
    future_df  = pd.DataFrame({
        "ds": pd.date_range(start_date, end_date, freq="D")
    })
    forecast   = model.predict(future_df)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def _apply_lightgbm_correction(
    prophet_forecast: pd.DataFrame,
    models:           dict,
) -> np.ndarray:
    """
    Placeholder: áp dụng LightGBM correction lên Prophet forecast.
    Trong production sẽ cần build feature vector đầy đủ.
    Hiện tại trả về 0 (prophet-only prediction).
    """
    # NOTE: Để build đầy đủ cần feature_engineering pipeline
    # cho các ngày tương lai. Hiện trả về 0 residual.
    return np.zeros(len(prophet_forecast))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status":    "ok",
        "service":   "Demand Forecasting API",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/info", response_model=ModelInfoResponse, tags=["Info"])
async def model_info():
    """Thông tin về model đã load và brands có sẵn."""
    try:
        models = _load_models()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    prophet_path = os.path.join(PATHS["models"], "prophet_models.pkl")
    lgbm_path    = os.path.join(PATHS["models"], "lightgbm_global.pkl")

    brands = [
        b for b, m in models["prophet"].items()
        if m is not None
    ]

    return ModelInfoResponse(
        brands_available=sorted(brands),
        model_files={
            "prophet":  prophet_path,
            "lightgbm": lgbm_path,
        },
        api_version="1.0.0",
    )


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
async def forecast(req: ForecastRequest):
    """
    Dự báo nhu cầu theo brand/category cho khoảng thời gian chỉ định.

    - **brand**: Tên brand (phải khớp với brand đã train)
    - **start_date / end_date**: Khoảng thời gian dự báo
    - **horizon**: Số ngày dự báo từ start_date (override end_date nếu truyền)
    """
    try:
        models = _load_models()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    start_ts = pd.Timestamp(req.start_date)
    if req.horizon:
        end_ts = start_ts + pd.Timedelta(days=req.horizon - 1)
    else:
        end_ts = pd.Timestamp(req.end_date)

    if end_ts < start_ts:
        raise HTTPException(status_code=400, detail="end_date phải sau start_date.")

    try:
        prophet_fc = _generate_prophet_forecast(req.brand, start_ts, end_ts, models)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prophet forecast failed: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {e}")

    lgbm_correction = _apply_lightgbm_correction(prophet_fc, models)

    prophet_pred = np.clip(prophet_fc["yhat"].values, 0, None)
    final_pred   = np.clip(prophet_pred + lgbm_correction, 0, None)
    lower_bound  = np.clip(prophet_fc["yhat_lower"].values, 0, None)
    upper_bound  = np.clip(prophet_fc["yhat_upper"].values, 0, None)

    forecast_points = [
        ForecastPoint(
            date          = row["ds"].strftime("%Y-%m-%d"),
            prophet_pred  = round(float(prophet_pred[i]), 2),
            final_pred    = round(float(final_pred[i]),   2),
            lower_bound   = round(float(lower_bound[i]),  2),
            upper_bound   = round(float(upper_bound[i]),  2),
        )
        for i, (_, row) in enumerate(prophet_fc.iterrows())
    ]

    return ForecastResponse(
        brand       = req.brand,
        category    = req.category,
        start_date  = start_ts.strftime("%Y-%m-%d"),
        end_date    = end_ts.strftime("%Y-%m-%d"),
        n_days      = len(forecast_points),
        total_pred  = round(float(final_pred.sum()), 2),
        forecast    = forecast_points,
    )


@app.post("/forecast/batch", tags=["Forecast"])
async def forecast_batch(batch_req: BatchForecastRequest):
    """Dự báo hàng loạt cho nhiều brand/category cùng lúc."""
    results = []
    for req in batch_req.requests:
        try:
            result = await forecast(req)
            results.append({"status": "ok", "data": result})
        except HTTPException as e:
            results.append({
                "status":  "error",
                "brand":   req.brand,
                "detail":  e.detail,
            })
    return {"results": results, "total": len(results)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    from src.utils.config_loader import API_HOST, API_PORT

    uvicorn.run(
        "src.api.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )
