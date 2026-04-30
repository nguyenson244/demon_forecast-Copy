# Dự Án Dự Báo Nhu Cầu Kinh Đô FMCG (Hybrid Prophet-LightGBM)

Hệ thống dự báo nhu cầu hàng ngày dựa trên kiến trúc Hybrid: Prophet xử lý xu hướng/mùa vụ + LightGBM xử lý các đặc trưng phi tuyến và Residuals.

## 📁 Cấu Trúc Dự Án
- `src/data/`: Tải và làm sạch dữ liệu.
- `src/features/`: Tạo lag, rolling window, đặc trưng ngày lễ (Tết, Trung Thu).
- `src/models/`: Huấn luyện Prophet (Stage 1) và LightGBM (Stage 2).
- `src/api/`: FastAPI phục vụ dự báo thời gian thực.
- `config/`: Quản lý hằng số và tham số qua file `config.yaml`.

## 🚀 Hướng Dẫn Chạy

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2. Huấn luyện Model (Toàn bộ Pipeline)
Chạy lệnh này để xử lý dữ liệu và tạo model mới:
```bash
python run_pipeline.py
```

### 3. Khởi chạy API Server
Sau khi đã có model, chạy server để phục vụ yêu cầu dự báo:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
- Truy cập Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

## 🛠 Tùy Chỉnh Cấu Hình
Bạn có thể thay đổi các tham số như:
- Danh sách ngày lễ (Tết, Trung Thu)
- Các mốc Lag (7, 14, 30 ngày...)
- Tham số huấn luyện LightGBM (Optuna trials)

Tất cả nằm trong file: `config/config.yaml`.

---
*Ghi chú: Đảm bảo bạn đang đứng ở thư mục gốc của dự án khi chạy các lệnh trên.*
