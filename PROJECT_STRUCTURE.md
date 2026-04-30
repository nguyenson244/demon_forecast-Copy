# Cấu Trúc Dự Án (Project Structure) — Phiên Bản Modular

> **Dự án:** Dự báo nhu cầu sản phẩm FMCG (Kinh Đô) sử dụng mô hình lai Prophet + LightGBM
> **Kiến trúc:** Feature-based Package Layout (Sạch - Chuyên nghiệp - Dễ mở rộng)

---

## 🗂️ Cây Thư Mục Hoàn Chỉnh

```text
deman_forecast/
│
├── 📄 run_pipeline.py              # Điểm vào chính — Chạy toàn bộ Pipeline 5 bước
├── 📄 README.md                    # Hướng dẫn cài đặt và vận hành nhanh
├── 📄 TECHNICAL_DOCUMENTATION.md   # Giải thích ý tưởng thiết kế và nội dung chi tiết
├── 📄 PROJECT_STRUCTURE.md         # Tài liệu cấu trúc thư mục (file này)
├── 📄 requirements.txt             # Danh sách thư viện (Pandas, Prophet, LightGBM, FastAPI...)
├── 📄 .gitignore                   # Ngăn chặn upload file rác/nặng lên GitHub
│
├── 📁 config/                      # Quản lý cấu hình tập trung
│   └── config.yaml                 # Chứa tham số ngày lễ, mốc lag, config model
│
├── 📁 data/                        # Quản lý dữ liệu đa tầng
│   ├── 📁 raw/                     # Dữ liệu gốc (2023, 2024, 2025)
│   ├── 📁 processed/               # Dữ liệu sau khi làm sạch (cleaned_data.csv)
│   └── 📁 features/                # Dữ liệu sau khi tạo 80+ đặc trưng (train/test_features.csv)
│
├── 📁 src/                         # Mã nguồn cốt lõi (Source Code)
│   ├── 📁 api/                     # Triển khai Web Service
│   │   └── app.py                  # API FastAPI (Dự báo thời gian thực)
│   │
│   ├── 📁 data/                    # Xử lý dữ liệu
│   │   ├── data_loader.py          # Đọc và gộp file CSV thô
│   │   ├── data_cleaning.py        # Tiền xử lý, xử lý zero-inflation
│   │   └── data_validation.py      # Kiểm tra chất lượng và logic dữ liệu
│   │
│   ├── 📁 features/                # Đặc trưng hóa
│   │   └── feature_engineering.py  # Tạo Lags, Rolling windows, đặc trưng Ngày lễ
│   │
│   ├── 📁 models/                  # Huấn luyện mô hình
│   │   ├── prophet_model.py        # Stage 1: Prophet per Brand
│   │   ├── lightgbm_model.py       # Stage 2: LightGBM Global (Residuals)
│   │   └── train.py                # Entry point cho việc huấn luyện
│   │
│   ├── 📁 evaluation/              # Đánh giá & Kiểm định
│   │   ├── metrics.py              # Tính RMSE, MAPE và vẽ biểu đồ kết quả
│   │   └── backtesting.py          # Đánh giá chéo chuỗi thời gian (Walk-forward)
│   │
│   ├── 📁 pipeline/                # Điều phối (Orchestration)
│   │   └── pipeline.py             # Script điều khiển luồng công việc 5 bước
│   │
│   └── 📁 utils/                   # Công cụ hỗ trợ (Helpers)
│       ├── logger.py               # Ghi nhật ký hệ thống tập trung
│       └── config_loader.py        # Đọc file YAML và expose hằng số
│
├── 📁 models/                      # Lưu trữ các Model đã huấn luyện (.pkl)
│   ├── prophet_models.pkl          # File model Prophet cho các Brand
│   └── lightgbm_global.pkl         # File model LightGBM toàn cục
│
├── 📁 results/                     # Đầu ra của hệ thống
│   ├── 📁 figures/                 # Biểu đồ Forecast vs Actual, Feature Importance
│   └── 📁 metrics/                 # Bảng so sánh độ lỗi, Backtest summary
│
├── 📁 logs/                        # Nhật ký thực thi
│   └── pipeline.log                # Lưu vết lịch sử chạy của toàn bộ hệ thống
│
└── 📁 notebooks/                   # Môi trường nghiên cứu (R&D)
    └── EDA.ipynb                   # Phân tích dữ liệu khám phá (Dùng cho báo cáo)
```

---

## 🏗️ Quy Trình Dữ Liệu (Data Flow)

1.  **Bước 1 (Loading & Cleaning)**: Dữ liệu thô từ `data/raw/` được gộp lại, xử lý giá trị âm, điền khuyết các ngày không bán được hàng thành 0. Lưu vào `data/processed/`.
2.  **Bước 2 (Feature Engineering)**: Tạo ra 80+ cột dữ liệu mới dựa trên lịch sử và ngày lễ. Chia thành tập Train/Test theo thời gian. Lưu vào `data/features/`.
3.  **Bước 3 (Stage 1 - Prophet)**: Huấn luyện Prophet cho từng Brand để học xu hướng dài hạn. Trích xuất sai số (Residuals) để làm mục tiêu cho bước sau.
4.  **Bước 4 (Stage 2 - LightGBM)**: Huấn luyện 1 model LightGBM duy nhất trên toàn bộ dữ liệu để học cách sửa lỗi cho Prophet. Tìm tham số tốt nhất qua Optuna.
5.  **Bước 5 (Evaluation)**: Tính toán sai số cuối cùng, vẽ biểu đồ so sánh thực tế và dự báo. Lưu vào `results/`.

---

## 📌 Ghi Chú Cho Lập Trình Viên

- **Cấu hình**: Tuyệt đối không sửa các tham số (ngày lễ, tham số model) trong file `.py`. Hãy sửa trong `config/config.yaml`.
- **Thêm tính năng**: Nếu muốn thêm đặc trưng mới, hãy sửa file `src/features/feature_engineering.py`.
- **API**: Để kiểm tra API, hãy truy cập Swagger UI tại `/docs` sau khi bật server.
- **Log**: Luôn kiểm tra `logs/pipeline.log` nếu thấy Pipeline dừng đột ngột.
