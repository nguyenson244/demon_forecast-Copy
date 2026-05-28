# Dự Báo Nhu Cầu FMCG — Kinh Đô
## Tổng Quan Dự Án End-to-End

---

## 1. Bài Toán

**Đối tượng:** Tập đoàn Kinh Đô — nhà sản xuất FMCG bánh kẹo hàng đầu Việt Nam (các nhãn hàng: Kinh Đô Bread/Cake/Biscuit, Solite, Cosy, AFC, Oreo, LU, Ritz, Slide, Thu, Trang Vang, Hamper, Cadbury, Koko).

**Bài toán:** Dự báo sản lượng xuất kho hàng ngày (`Total QTY`) theo từng nhãn hàng (brand) để hỗ trợ lập kế hoạch sản xuất và phân phối.

**Khó khăn đặc thù:**
- **Zero-inflation:** Nhiều ngày không có đơn hàng → MAPE truyền thống bị undefined hoặc sai lệch nghiêm trọng.
- **Tính mùa vụ mạnh:** Tết Nguyên Đán, Trung Thu tạo ra các đỉnh cực lớn (gấp 10–20× ngày thường).
- **Không đồng nhất giữa các brand:** Kinh Đô Bread bán đều quanh năm; Thu/Trang Vang chỉ bán trong mùa Trung Thu.
- **Dữ liệu thưa:** Một số brand (Cadbury, Koko, Trang Vang) có lượng bán bằng 0 trong toàn bộ tập test.

---

## 2. Dữ Liệu

| Thuộc tính | Chi tiết |
|---|---|
| Nguồn | Dữ liệu xuất kho nội bộ Kinh Đô |
| Cột chính | `ACTUALSHIPDATE`, `BRAND`, `CATEGORY`, `Total QTY` |
| Cột phụ | `WHSEID` (kho), `Total CBM` (thể tích) |
| Thời gian | 2023 – 2025 |
| Tần suất | Daily (hàng ngày) |
| Train/Test split | Trước 30/06/2025 (train) / Sau 30/06/2025 (test) |

---

## 3. Kiến Trúc Pipeline

Pipeline được tổ chức thành **5 bước tuần tự**:

```
Bước 1: Load & Clean Data
         ↓
Bước 2: Feature Engineering  (90+ features)
         ↓
Bước 3: Prophet Stage 1      (trend + seasonality)
         ↓
Bước 4: LightGBM Stage 2     (residual learning)
         ↓
Bước 5: Evaluation & Visualization
```

---

## 4. Các Ý Tưởng Kỹ Thuật Đã Sử Dụng

### 4.1 Mô Hình Hybrid: Prophet + LightGBM

**Ý tưởng cốt lõi:** Kết hợp điểm mạnh của hai mô hình khác nhau:

| Mô hình | Vai trò | Điểm mạnh |
|---|---|---|
| **Prophet (Meta)** | Stage 1 — học trend + seasonality | Không cần feature engineering, xử lý tốt missing dates, tích hợp holidays |
| **LightGBM** | Stage 2 — học phần dư (residual) OOS | Học pattern phức tạp, xử lý được 90+ features, tốc độ cao |

**Cơ chế:**
```
prophet_pred  → dự báo xu hướng và mùa vụ
residual      = actual - prophet_pred  (chỉ trên tập OOS)
lgbm          → học residual từ 90+ features
final_pred    = prophet_pred + lgbm_residual_pred  (clip ≥ 0)
```

**Lý do không dùng một mô hình đơn lẻ:**
- Prophet tốt cho trend/seasonality nhưng bỏ qua các tín hiệu ngắn hạn (lag, rolling).
- LightGBM tốt cho tín hiệu ngắn hạn nhưng khó capture seasonality dài (Trung Thu, Tết).
- Hybrid bổ sung cho nhau → WMAPE tổng thể thấp hơn cả hai mô hình đơn lẻ.

---

### 4.2 OOS Calibration — Chống Rò Rỉ Dữ Liệu (Data Leakage)

**Vấn đề:** Nếu LightGBM train trên residual in-sample (tức là dùng cùng dữ liệu đã dùng để fit Prophet), residual sẽ gần bằng 0 → LightGBM không học được gì, nhưng lại overfit nghiêm trọng trên tập test.

**Giải pháp — OOS Calibration:**
```
Train data (100%)
├── 75% đầu → fit Prophet
└── 25% cuối → Prophet predict (OOS) → tính residual thật
                                        → LightGBM train trên residual này
```

Bằng cách này, LightGBM chỉ được học từ residual **out-of-sample** — tức là sai số thật của Prophet trên dữ liệu chưa thấy — đảm bảo không có leakage.

---

### 4.3 Phân Cụm Brand (Cluster-based Training)

Thay vì train một mô hình toàn cục, các brand được phân thành 3 cụm dựa trên hành vi bán hàng:

| Cluster | Nhãn | Brand | Đặc điểm |
|---|---|---|---|
| **0** | Stable | Kinh Đô Bread, Kinh Đô Cake, Solite, Cosy | Bán đều quanh năm, ít biến động |
| **1** | Regular | AFC, Oreo, LU, Ritz, Slide, Kinh Đô Biscuit | Biến động vừa, đỉnh nhẹ các dịp lễ |
| **2** | Seasonal | Thu, Trang Vang, Hamper, Cadbury, Koko | Đỉnh cực cao dịp Trung Thu / Tết |

**Lý do:** Mỗi cụm có pattern nhu cầu khác nhau → cần siêu tham số và cấu trúc mô hình khác nhau.

---

### 4.4 Two-Part Model cho Cluster 2 (Seasonal)

**Vấn đề với Cluster 2:** Brand như Thu, Trang Vang có hàng trăm ngày bán = 0 (off-season), xen kẽ đỉnh cực cao dịp Trung Thu. Mô hình hồi quy thông thường có xu hướng over-predict trong off-season.

**Giải pháp — Mô hình Hai Thành Phần:**
```
Thành phần 1: LGBMClassifier → P(sale > 0)  — xác suất có đơn hàng
Thành phần 2: LGBMRegressor  → residual      — chỉ train trên ngày có bán

final_pred = (prophet_pred + residual) × P(sale > 0)
```

P(sale > 0) hoạt động như một "gate" — tự động suppress dự báo trong off-season về gần 0.

---

### 4.5 Per-Brand Prophet Fallback

**Ý tưởng:** Không phải brand nào LightGBM cũng cải thiện được kết quả. Với các brand mà mô hình hybrid làm tệ hơn Prophet trên tập calibration, tự động revert về Prophet-only.

**Cơ chế:**
```python
# Với mỗi brand, so sánh trên tập calibration holdout:
if hybrid_WMAPE > prophet_WMAPE + 2.0:
    # LightGBM thêm noise → revert về prophet
    lgbm_residual_pred = 0
    final_pred = prophet_pred
```

**Brands fallback trong kết quả cuối:** Cadbury, Kinh Đô Cake, Koko, Trang Vang.

---

### 4.6 XGBoost Ensemble (Cluster 0 và 1)

**Ý tưởng:** Kết hợp hai thuật toán gradient boosting khác nhau (LightGBM và XGBoost) để giảm variance:

```
residual_pred = 0.5 × LightGBM_pred + 0.5 × XGBoost_pred
```

LightGBM và XGBoost sử dụng cách xây dựng cây khác nhau (leaf-wise vs depth-wise) → kết hợp 50/50 thường cho kết quả ổn định hơn từng mô hình đơn lẻ.

---

### 4.7 Blend Weight + Bias Correction Per Brand

**Bước 1 — Blend weight optimization:**
Tìm tỷ lệ pha trộn tối ưu giữa Prophet và Hybrid cho từng brand:
```
blended = α × prophet_pred + (1 - α) × hybrid_pred
α* = argmin WMAPE(calib)  với α ∈ {0.0, 0.1, ..., 1.0}
```
Brand nào LightGBM cải thiện ít → α cao hơn (dựa nhiều vào Prophet).

**Bước 2 — Bias correction:**
Hiệu chỉnh systematic over/under-forecast per brand:
```
factor = sum(actual_calib) / sum(blended_calib)
factor = clip(factor, 0.5, 2.0)   # tránh correction cực đoan
final_pred = blended × factor
```

---

### 4.8 Feature Engineering — 90+ Features

| Nhóm feature | Số lượng | Mô tả |
|---|---|---|
| **Lag** | 17 | lag_1, 2, 3, 5, 7, 14, 21, 28, 30, 60, 90, 120, 150, 180, 364, 365, 366 |
| **Same weekday** | 4 | Lag 1/2/3/4 tuần trước cùng thứ |
| **Rolling stats** | 20 | mean/std/median/min/max trên cửa sổ 7, 14, 30, 90 ngày |
| **Momentum/CV** | 6 | Tốc độ tăng trưởng, hệ số biến thiên |
| **Time** | 14 | Sin/cos của day_of_week, month, day_of_year (lượng giác hóa) |
| **Tết Nguyên Đán** | 5 | days_to_tet, is_pre_tet, is_tet, is_post_tet, tet_intensity |
| **Trung Thu** | 5 | days_to_mid_autumn, is_mooncake_season, days_after_mid_autumn, is_post_mid_autumn |
| **Ngày lễ** | 3 | Python `holidays` library (30/4, 2/9, 1/1...) |
| **Extended holiday** | 8 | Liberation Day, National Day, Christmas, hè học sinh, Valentine... |
| **Thời tiết** | 7 | temp_mean, precip, wind_max, rolling 7 ngày, is_hot_day, is_rainy_day, is_dry_season |
| **Brand stats** | 7 | Trung bình, median, std, percentile — fit ONLY trên Train |
| **Categorical** | 2 | is_seasonal (cluster 2), cbm_log |

**Nguyên tắc chống leakage:** Tất cả lag/rolling dùng `.shift(1)` nghiêm ngặt. Brand stats chỉ `.fit()` trên tập train.

---

### 4.9 Optuna Hyperparameter Optimization (HPO)

Thay vì tuning tay, sử dụng **Optuna** (Bayesian optimization) để tìm siêu tham số tốt nhất cho LightGBM per cluster:

- **20 trials** mỗi cluster
- **Không gian tìm kiếm:** learning_rate, num_leaves, feature_fraction, bagging_fraction, min_child_samples, lambda_l1, lambda_l2
- **Objective:** Minimize WMAPE trên walk-forward cross-validation (5 folds)
- `n_estimators = 500` cố định, early stopping quyết định số cây thực tế (tối thiểu 150 để tránh underfit)

---

### 4.10 Outlier Detection — IQR-based Capping

Phát hiện và giới hạn outlier theo nhóm `[BRAND, CATEGORY]` bằng phương pháp IQR:
```
upper_bound = Q3 + 3.0 × IQR
lower_bound = Q1 - 3.0 × IQR
```
Hệ số 3.0 (thay vì 1.5 tiêu chuẩn) để không loại bỏ các đỉnh hợp lệ (Tết, Trung Thu).

---

### 4.11 Metric Chính — WMAPE

**WMAPE** (Weighted Mean Absolute Percentage Error) được chọn thay vì MAPE vì:

```
WMAPE = Σ|actual - pred| / Σ|actual|  × 100%
```

| Tính chất | MAPE | WMAPE |
|---|---|---|
| Khi actual = 0 | Undefined (÷0) | Vẫn tính được |
| Bị skew bởi ngày bán ít | Có | Không |
| Ý nghĩa business | Khó diễn giải | = % sai số trên tổng volume |

---

## 5. Kết Quả

### 5.1 So Sánh Tổng Thể (Tập Test)

| Mô hình | WMAPE (%) | RMSE | MAE |
|---|---|---|---|
| ARIMA(5,1,2) — Baseline | 39.04 | 645,513 | — |
| **Prophet (Standalone)** | **21.26** | 398,714 | 278,896 |
| **Hybrid Prophet+LightGBM** | **20.40** | 376,916 | 267,560 |

→ Hybrid cải thiện **0.86 điểm phần trăm WMAPE** so với Prophet thuần, giảm **5.5% RMSE**.

### 5.2 Kết Quả Per Brand

| Brand | WMAPE Hybrid (%) | WMAPE Prophet (%) | Cải thiện |
|---|---|---|---|
| Kinh Đô Bread | **14.6%** | 22.5% | +7.9pp |
| Hamper | **21.1%** | 41.7% | +20.6pp |
| Kinh Đô Cake | 30.3% | 30.1% | -0.2pp (fallback) |
| Cosy | **37.7%** | 45.3% | +7.6pp |
| Oreo | **38.3%** | 41.9% | +3.7pp |
| Solite | **39.1%** | 41.0% | +1.9pp |
| Slide | 42.2% | 40.2% | -2.0pp (fallback) |
| AFC | **46.4%** | 53.9% | +7.6pp |
| Ritz | 47.4% | 46.8% | -0.6pp (fallback) |
| Kinh Đô Biscuit | 48.3% | 42.9% | -5.4pp (fallback) |
| LU | **63.7%** | 84.3% | +20.6pp |
| THU | **78.8%** | 107.7% | +28.9pp |

### 5.3 Nhận Xét

- **Hiệu quả nhất:** Hamper (+20.6pp), THU (+28.9pp), LU (+20.6pp) — các brand có pattern phức tạp.
- **Fallback về Prophet:** 4 brand (Cadbury, Koko, Trang Vang có tập test = 0 sale; Slide/Ritz/Kinh Đô Biscuit/Kinh Đô Cake LightGBM thêm noise).
- **Kết quả Train vs Test:** WMAPE Train=21.04%, Test=20.40% → không có dấu hiệu overfitting đáng kể.

---

## 6. Cấu Trúc Thư Mục

```
deman_forecast/
├── run_pipeline.py              # Entry point — chạy toàn bộ pipeline
├── config/config.yaml           # Cấu hình tập trung (paths, params, holidays)
├── requirements.txt
├── data/
│   ├── raw/                     # Dữ liệu gốc
│   ├── processed/               # Sau khi clean
│   └── features/                # Sau khi feature engineering
├── models/                      # File .pkl các mô hình đã train
│   ├── prophet_models.pkl
│   ├── lgbm_cluster_{0,1,2}.pkl
│   ├── xgb_cluster_{0,1}.pkl
│   ├── lgbm_classifier_cluster_2.pkl
│   ├── lgbm_regressor_cluster_2.pkl
│   └── brand_forecast_metadata.json
├── results/
│   ├── metrics/                 # CSV kết quả đánh giá
│   └── figures/                 # Biểu đồ PNG
├── src/
│   ├── data/                    # data_cleaning, data_loader, data_validation, weather_features
│   ├── features/                # feature_engineering
│   ├── models/                  # prophet_model, lightgbm_model
│   ├── pipeline/                # pipeline.py (orchestrator)
│   ├── evaluation/              # metrics, backtesting
│   └── utils/                   # config_loader, logger
└── notebooks/EDA.ipynb          # Phân tích dữ liệu khám phá
```

---

## 7. Công Nghệ Sử Dụng

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| `prophet` | ≥ 1.1 | Mô hình Stage 1 — trend + seasonality |
| `lightgbm` | ≥ 4.0 | Mô hình Stage 2 — residual learning |
| `xgboost` | ≥ 2.0 | Ensemble với LightGBM |
| `optuna` | ≥ 3.0 | Hyperparameter optimization (Bayesian) |
| `scikit-learn` | ≥ 1.3 | Walk-forward cross-validation, metrics |
| `pandas` / `numpy` | latest | Xử lý dữ liệu |
| `holidays` | ≥ 0.40 | Ngày lễ Việt Nam tự động |
| `lunardate` / `vnlunar` | latest | Chuyển đổi âm lịch (Tết, Trung Thu) |
| `statsmodels` | ≥ 0.14 | ARIMA baseline |

---

## 8. Cách Chạy

```bash
# Cài đặt môi trường
pip install -r requirements.txt

# Chạy toàn bộ pipeline (tất cả 5 bước)
python run_pipeline.py

# Chạy từ bước cụ thể (ví dụ: chỉ step 4-5)
python -m src.pipeline.pipeline --step 4 --end 5
```

**Kết quả xuất ra:**
- `results/metrics/per_brand_metrics.csv` — WMAPE/RMSE/MAE từng brand
- `results/metrics/comparison_metrics.csv` — So sánh Prophet vs Hybrid vs ARIMA
- `results/metrics/predictions_clean.csv` — Actual vs Predicted từng ngày
- `results/figures/*.png` — Biểu đồ dự báo, feature importance, residuals
