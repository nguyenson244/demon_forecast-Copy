# Dự Báo Nhu Cầu FMCG — Kinh Đô
> Hybrid Prophet + LightGBM · 15 nhãn hàng · Daily forecasting · WMAPE 20.02%

---

## Mục Lục
1. [Bài Toán](#1-bài-toán)
2. [Dữ Liệu](#2-dữ-liệu)
3. [Kiến Trúc Pipeline](#3-kiến-trúc-pipeline)
4. [Các Kỹ Thuật Đã Dùng](#4-các-kỹ-thuật-đã-dùng)
5. [Kết Quả](#5-kết-quả)
6. [Cấu Trúc Thư Mục](#6-cấu-trúc-thư-mục)
7. [Cài Đặt & Chạy](#7-cài-đặt--chạy)
8. [Công Nghệ](#8-công-nghệ)

---

## 1. Bài Toán

**Đối tượng:** Tập đoàn Kinh Đô — nhà sản xuất FMCG bánh kẹo hàng đầu Việt Nam.

**Mục tiêu:** Dự báo sản lượng xuất kho hàng ngày (`Total QTY`) theo từng nhãn hàng (brand) để hỗ trợ lập kế hoạch sản xuất và phân phối.

**Khó khăn đặc thù của dữ liệu FMCG:**

| Thách thức | Mô tả |
|---|---|
| **Zero-inflation** | Nhiều ngày không có đơn hàng → MAPE truyền thống bị undefined |
| **Mùa vụ cực đoan** | Tết, Trung Thu tạo đỉnh gấp 10–50× ngày thường |
| **Không đồng nhất** | Kinh Đô Bread bán đều; Thu/Trang Vang chỉ bán mùa Trung Thu |
| **Dữ liệu thưa** | Cadbury, Koko, Trang Vang bán = 0 trong toàn tập test |

---

## 2. Dữ Liệu

| Thuộc tính | Chi tiết |
|---|---|
| Nguồn | Dữ liệu xuất kho nội bộ Kinh Đô |
| Cột chính | `ACTUALSHIPDATE`, `BRAND`, `CATEGORY`, `Total QTY` |
| Cột phụ | `WHSEID` (kho — 1 kho BKD1), `Total CBM` (thể tích) |
| Thời gian | 2023 – 2025 |
| Tần suất | Daily (hàng ngày) |
| Số brand | 15 brand, 4 category (FRESH, DRY, MOONCAKE, TET) |
| Train/Test split | Trước 30/06/2025 → train · Sau 30/06/2025 → test |

---

## 3. Kiến Trúc Pipeline

```
Bước 1: Load & Clean Data
         → Aggregate theo kho, full date grid, outlier capping (IQR k=3)
         ↓
Bước 2: Feature Engineering  (90+ features)
         → Lag, Rolling, Holiday (Tết/Trung Thu/Extended), Weather, Brand stats
         ↓
Bước 3: Prophet Stage 1  (per brand)
         → OOS calibration: fit 75% train, predict 25% cuối → residual thật
         ↓
Bước 4: LightGBM Stage 2  (per cluster)
         → Cluster 0/1: LightGBM + XGBoost ensemble (50/50) + Optuna HPO
         → Cluster 2:   Two-Part Model (Classifier × Regressor)
         → Per-brand fallback · Blend weight · Bias correction
         ↓
Bước 5: Evaluation & Visualization
         → WMAPE, RMSE, MAE · ACF/PACF · Feature importance · Comparison
```

---

## 4. Các Kỹ Thuật Đã Dùng

### 4.1 Hybrid Prophet + LightGBM

Prophet học **trend + seasonality**, LightGBM học **residual** (sai số còn lại của Prophet):

```
final_pred = prophet_pred + lgbm_residual_pred   (clip ≥ 0)
```

Lý do dùng Hybrid: Prophet tốt cho mùa vụ dài hạn, LightGBM tốt cho tín hiệu ngắn hạn (lag, rolling) — hai mô hình bổ sung cho nhau.

### 4.2 OOS Calibration — Chống Data Leakage

```
Train data (100%)
├── 75% đầu  → fit Prophet
└── 25% cuối → Prophet predict (OOS) → residual thật → LightGBM train
```

LightGBM chỉ học từ residual **out-of-sample** — đảm bảo không có leakage.

### 4.3 Phân Cụm Brand

| Cluster | Nhãn | Brand | Mô hình Tầng 2 |
|---|---|---|---|
| **0** | Stable | Kinh Đô Bread, Kinh Đô Cake, Solite, Cosy | LightGBM + XGBoost |
| **1** | Regular | AFC, Oreo, LU, Ritz, Slide, Kinh Đô Biscuit | LightGBM + XGBoost |
| **2** | Seasonal | Thu, Trang Vang, Hamper, Cadbury, Koko | Two-Part Model |

### 4.4 Two-Part Model (Cluster 2 — Seasonal)

Giải quyết zero-inflation cực cao của brand mùa vụ:

```
Thành phần A: LGBMClassifier → P(sale > 0)   — "ngày này có bán không?"
Thành phần B: LGBMRegressor  → residual       — "nếu bán thì bao nhiêu?"

final_pred = (prophet_pred + residual_B) × P(sale > 0)_A
```

P(sale > 0) hoạt động như "gate" — tự động suppress dự báo trong off-season.

### 4.5 Per-Brand Prophet Fallback

```python
if hybrid_WMAPE > prophet_WMAPE + 2.0:
    final_pred = prophet_pred   # LightGBM thêm noise → revert về Prophet
```

Brands fallback: Cadbury, Kinh Đô Cake, Koko, Trang Vang.

### 4.6 XGBoost Ensemble (Cluster 0 & 1)

```
residual_pred = 0.5 × LightGBM_pred + 0.5 × XGBoost_pred
```

LightGBM (leaf-wise) + XGBoost (depth-wise) → mắc lỗi ở các điểm khác nhau → ensemble giảm variance.

### 4.7 Blend Weight + Bias Correction Per Brand

```
# Bước 1: tìm α tối ưu per brand
blended = α × prophet_pred + (1 - α) × hybrid_pred
α* = argmin WMAPE(calib),  α ∈ {0.0, 0.1, ..., 1.0}

# Bước 2: bias correction
factor = clip(sum_actual_calib / sum_blended_calib, 0.5, 2.0)
final_pred = blended × factor
```

### 4.8 Feature Engineering — 90+ Features

| Nhóm | Số lượng | Mô tả |
|---|---|---|
| Lag | 17 | lag_1, 2, 3, 5, 7, 14, 21, 28, 30, 60, 90, 120, 150, 180, 364, 365, 366 |
| Same weekday | 4 | Cùng thứ 1/2/3/4 tuần trước |
| Rolling stats | 20 | mean/std/median/min/max — cửa sổ 7, 14, 30, 90 ngày |
| Thời gian | 14 | Sin/cos day_of_week, month, day_of_year |
| Tết Nguyên Đán | 5 | days_to_tet, is_pre/tet/post_tet, tet_intensity |
| Trung Thu | 5 | days_to_mid_autumn, is_mooncake_season, days_after, is_post |
| Ngày lễ | 3 | 30/4, 2/9, 1/1 (thư viện `holidays`) |
| Extended holiday | 8 | Liberation Day, National Day, Christmas, hè học sinh... |
| Thời tiết | 7 | temp_mean, precip, wind_max, rolling 7 ngày, is_hot/rainy/dry |
| Brand stats | 7 | mean, median, std, percentile — fit ONLY trên Train |
| Categorical | 2 | is_seasonal, cbm_log |

> Chống leakage: tất cả lag/rolling dùng `.shift(1)`. Brand stats chỉ `.fit()` trên Train.

### 4.9 Optuna HPO

Bayesian optimization tự động tìm siêu tham số LightGBM per cluster:
- 20 trials · Walk-forward CV 5 folds · Minimize WMAPE
- Không gian tìm kiếm: `learning_rate`, `num_leaves`, `feature_fraction`, `bagging_fraction`, `min_child_samples`, `lambda_l1`, `lambda_l2`

### 4.10 WMAPE — Metric Chính

```
WMAPE = Σ|actual - pred| / Σ|actual| × 100%
```

Được chọn thay MAPE vì: không bị undefined khi actual = 0, không bị skew bởi ngày bán thấp, tương đương % sai số trên tổng volume.

---

## 5. Kết Quả

### So Sánh Tổng Thể (Tập Test)

| Mô hình | WMAPE (%) | RMSE | MAE |
|---|---|---|---|
| ARIMA(5,1,2) — Baseline | 39.04 | 645,513 | — |
| Prophet (Standalone) | 21.26 | 398,714 | 278,896 |
| **Hybrid Prophet + LightGBM** | **20.02** | **374,295** | **262,622** |

Hybrid cải thiện **1.24pp WMAPE** và **6.1% RMSE** so với Prophet thuần.

### Kết Quả Per Brand

| Brand | Cluster | WMAPE Hybrid | WMAPE Prophet | Cải thiện |
|---|---|---|---|---|
| Kinh Đô Bread | Stable | **13.8%** | 22.5% | +8.7pp |
| Hamper | Seasonal | **21.1%** | 41.7% | +20.6pp |
| Kinh Đô Cake | Stable | 30.3% | 30.1% | fallback |
| Solite | Stable | **36.2%** | 41.0% | +4.8pp |
| Oreo | Regular | **37.2%** | 41.9% | +4.7pp |
| Cosy | Stable | **37.5%** | 45.3% | +7.8pp |
| Slide | Regular | 42.1% | 40.2% | fallback |
| AFC | Regular | **46.9%** | 53.9% | +7.0pp |
| Ritz | Regular | 47.7% | 46.8% | fallback |
| Kinh Đô Biscuit | Regular | 48.4% | 42.9% | fallback |
| LU | Regular | **64.8%** | 84.3% | +19.5pp |
| THU | Seasonal | **78.8%** | 107.7% | +28.9pp |

> Train WMAPE = 20.70% · Test WMAPE = 20.02% → không có dấu hiệu overfitting.

---

## 6. Cấu Trúc Thư Mục

```
deman_forecast/
├── run_pipeline.py              # Entry point chính
├── config/config.yaml           # Cấu hình tập trung
├── requirements.txt
├── data/
│   ├── raw/                     # Dữ liệu gốc (CSV theo năm)
│   ├── processed/               # Sau bước clean
│   └── features/                # Sau bước feature engineering
├── models/                      # Model artifacts (.pkl, .json)
├── results/
│   ├── metrics/                 # CSV đánh giá
│   └── figures/                 # Biểu đồ PNG
├── src/
│   ├── data/                    # data_cleaning, loader, validation, weather
│   ├── features/                # feature_engineering
│   ├── models/                  # prophet_model, lightgbm_model
│   ├── pipeline/                # pipeline.py
│   ├── evaluation/              # metrics, backtesting
│   └── utils/                   # config_loader, logger
└── notebooks/EDA.ipynb
```

---

## 7. Cài Đặt & Chạy

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Chạy toàn bộ pipeline (5 bước)
python run_pipeline.py

# 3. Chạy từ bước cụ thể
python -m src.pipeline.pipeline --step 4 --end 5
```

**Kết quả xuất ra:**

| File | Nội dung |
|---|---|
| `results/metrics/per_brand_metrics.csv` | WMAPE / RMSE / MAE từng brand |
| `results/metrics/comparison_metrics.csv` | So sánh Prophet vs Hybrid vs ARIMA |
| `results/metrics/predictions_clean.csv` | Actual vs Predicted từng ngày |
| `results/figures/forecast_vs_actual.png` | Biểu đồ dự báo tổng hợp |
| `results/figures/forecast_per_brand.png` | Biểu đồ từng brand |
| `results/figures/acf_pacf_analysis.png` | ACF/PACF — phân tích tương quan lag |
| `results/figures/feature_importance_cluster_*.png` | Top features LightGBM |

---

## 8. Công Nghệ

| Thư viện | Mục đích |
|---|---|
| `prophet` ≥ 1.1 | Stage 1 — trend + seasonality |
| `lightgbm` ≥ 4.0 | Stage 2 — residual learning |
| `xgboost` ≥ 2.0 | Ensemble với LightGBM |
| `optuna` ≥ 3.0 | Hyperparameter optimization |
| `scikit-learn` ≥ 1.3 | Walk-forward CV, metrics |
| `statsmodels` ≥ 0.14 | ARIMA baseline |
| `pandas` / `numpy` | Xử lý dữ liệu |
| `holidays` ≥ 0.40 | Ngày lễ Việt Nam |
| `lunardate` / `vnlunar` | Âm lịch (Tết, Trung Thu) |
