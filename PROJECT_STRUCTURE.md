# 📁 Project Structure — Hybrid Prophet-LightGBM Demand Forecasting

> **Dự án:** Dự báo nhu cầu sản phẩm FMCG (Kinh Đô) sử dụng mô hình lai Prophet + LightGBM  
> **Mục tiêu:** Pipeline tự động 5 bước từ dữ liệu thô → dự báo → đánh giá kết quả

---

## 🗂️ Cây Thư Mục

```
deman_forecast/
│
├── 📄 run_pipeline.py              # Điểm vào chính — chạy toàn bộ 5 bước tuần tự
├── 📄 requirements.txt             # Danh sách thư viện phụ thuộc
├── 📄 structure.md                 # Ghi chú cấu trúc thư mục (bản gốc)
├── 📄 PROJECT_STRUCTURE.md         # Tài liệu cấu trúc dự án (file này)
│
├── 📁 data/                        # Toàn bộ dữ liệu theo từng giai đoạn xử lý
│   ├── 📁 raw/                     # ⚠️ Dữ liệu thô gốc — KHÔNG BAO GIỜ SỬA
│   │   ├── data_2023.csv           # Dữ liệu bán hàng năm 2023
│   │   ├── data_2024.csv           # Dữ liệu bán hàng năm 2024
│   │   └── data_2025.csv           # Dữ liệu bán hàng năm 2025
│   │
│   ├── 📁 processed/               # Dữ liệu đã làm sạch và xử lý outliers
│   │   └── cleaned_data.csv        # Output của Bước 1 (data_prep.py)
│   │
│   └── 📁 features/                # Dữ liệu đã feature engineering, sẵn sàng train
│       ├── full_features.csv       # Toàn bộ dataset sau khi tạo features
│       ├── train_features.csv      # Tập huấn luyện (Stage 1 - Prophet)
│       ├── test_features.csv       # Tập kiểm tra (Stage 1 - Prophet)
│       ├── train_with_residuals.csv # Tập train kèm Residuals từ Prophet
│       └── test_with_residuals.csv  # Tập test kèm Residuals từ Prophet
│
├── 📁 src/                         # Mã nguồn Python cho từng bước pipeline
│   ├── __init__.py                 # Khai báo package Python
│   ├── config.py                   # Cấu hình trung tâm (đường dẫn, tham số, hyperparameters)
│   ├── data_prep.py                # [Bước 1] Tiền xử lý, làm sạch, xử lý Outliers
│   ├── feature_eng.py              # [Bước 2] Tạo đặc trưng: lag, rolling window, ngày lễ
│   ├── model_prophet.py            # [Bước 3] Huấn luyện Prophet theo từng Brand + trích Residuals
│   ├── model_lightgbm.py           # [Bước 4] Huấn luyện LightGBM Global trên Residuals
│   └── evaluation.py               # [Bước 5] Tính RMSE/MAPE/sMAPE + vẽ biểu đồ
│
├── 📁 models_saved/                # Lưu trữ mô hình đã huấn luyện (dùng lại không cần train lại)
│   ├── prophet_models.pkl          # Các mô hình Prophet (per Brand) đã lưu (~1.5 MB)
│   ├── lightgbm_global.pkl         # Mô hình LightGBM Global đã lưu (~2 MB)
│   └── lightgbm_best_params.pkl    # Hyperparameters tốt nhất của LightGBM
│
├── 📁 notebooks/                   # Jupyter Notebooks để EDA và thử nghiệm
│   └── EDA.ipynb                   # Phân tích khám phá dữ liệu (Exploratory Data Analysis)
│
└── 📁 results/                     # Kết quả đầu ra sau khi chạy pipeline
    ├── 📁 figures/                  # Biểu đồ hình ảnh (Forecast vs Actual, Residual plots)
    └── 📁 metrics/                  # Bảng số liệu độ chính xác (.csv: RMSE, MAPE, sMAPE)
```

---

## ⚙️ Luồng Pipeline 5 Bước

```
data/raw/          data/processed/    data/features/     models_saved/      results/
[data_2023~25]  →  [cleaned_data]  →  [train/test]    →  [prophet.pkl    →  [figures/]
                                      [residuals]         lightgbm.pkl]     [metrics/]
     │                  │                  │                   │                 │
  [Bước 1]           [Bước 2]          [Bước 3]           [Bước 4]          [Bước 5]
  data_prep       feature_eng       model_prophet      model_lightgbm      evaluation
```

| Bước | File            | Mô tả                                              | Input                  | Output                       |
|------|-----------------|-----------------------------------------------------|------------------------|------------------------------|
| 1    | `data_prep.py`  | Làm sạch, merge 3 năm, xử lý outliers/missing      | `data/raw/*.csv`       | `data/processed/cleaned_data.csv` |
| 2    | `feature_eng.py`| Tạo lag, rolling mean/std, đặc trưng thời gian, lễ | `cleaned_data.csv`     | `data/features/full_features.csv` |
| 3    | `model_prophet.py` | Train Prophet/Brand, trích Residuals             | `train/test_features`  | `*_with_residuals.csv`, `prophet_models.pkl` |
| 4    | `model_lightgbm.py` | Train LightGBM Global trên Residuals           | `*_with_residuals.csv` | `lightgbm_global.pkl`        |
| 5    | `evaluation.py` | Tính metric, vẽ biểu đồ so sánh                   | Model + test data      | `results/figures/`, `results/metrics/` |

---

## 🚀 Cách Chạy Pipeline

```bash
# Chạy toàn bộ 5 bước từ đầu
python run_pipeline.py

# Chạy từ bước cụ thể (ví dụ: bỏ qua bước 1-2, chạy từ bước 3)
python run_pipeline.py --step 3
```

---

## 📦 Thư Viện Chính (`requirements.txt`)

| Thư viện       | Mục đích                                          |
|----------------|---------------------------------------------------|
| `pandas`       | Xử lý và thao tác dữ liệu dạng bảng              |
| `numpy`        | Tính toán số học, mảng đa chiều                   |
| `prophet`      | Mô hình dự báo chuỗi thời gian (Stage 1)          |
| `lightgbm`     | Mô hình Gradient Boosting tốc độ cao (Stage 2)    |
| `scikit-learn` | Preprocessing, cross-validation, metrics          |
| `matplotlib`   | Vẽ biểu đồ trực quan hóa kết quả                  |
| `seaborn`      | Biểu đồ thống kê nâng cao                         |
| `joblib`       | Lưu và tải lại các mô hình (.pkl)                 |

---

## 🏗️ Kiến Trúc Mô Hình Lai (Hybrid Architecture)

```
            ┌──────────────────────────────────────┐
            │         Dữ liệu Đầu Vào              │
            │   (Sales per Brand per Date)          │
            └──────────────────┬───────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Feature Engineering │
                    │ (Lag, Rolling, Holiday)│
                    └──────────┬───────────┘
                               │
               ┌───────────────▼────────────────┐
               │       Stage 1: PROPHET          │
               │    (Huấn luyện theo từng Brand) │
               │   → Dự báo xu hướng & mùa vụ   │
               └───────────────┬────────────────┘
                               │
               ┌───────────────▼────────────────┐
               │         RESIDUALS               │
               │  (= Thực tế − Dự báo Prophet)  │
               └───────────────┬────────────────┘
                               │
               ┌───────────────▼────────────────┐
               │    Stage 2: LIGHTGBM GLOBAL     │
               │   (Học từ Residuals + Features) │
               │   → Bù trừ phần chưa dự báo    │
               └───────────────┬────────────────┘
                               │
            ┌──────────────────▼───────────────────┐
            │        Kết Quả Dự Báo Cuối Cùng      │
            │  Final = Prophet_pred + LightGBM_pred │
            └──────────────────────────────────────┘
```

---

*Tài liệu được tạo tự động — cập nhật lần cuối: 2026-04-30*
