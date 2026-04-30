# Tài Liệu Kỹ Thuật: Hệ Thống Dự Báo Nhu Cầu Hybrid

Tài liệu này giải thích kiến trúc modular, ý tưởng thiết kế và nội dung chi tiết của từng thư mục trong dự án.

---

## 💡 Ý Tưởng Cốt Lõi (The Hybrid Concept)
Hệ thống này sử dụng cách tiếp cận **Hybrid (Lai)** để tận dụng thế mạnh của hai thuật toán hàng đầu:
1.  **Stage 1 - Prophet**: Đóng vai trò là "chuyên gia mùa vụ". Nó học các quy luật dài hạn như xu hướng năm, các ngày lễ đặc thù của Việt Nam (Tết, Trung Thu).
2.  **Stage 2 - Multi-cluster LightGBM**: Hệ thống tự động phân cụm thương hiệu thành 3 nhóm (Stable, Regular, Extreme Seasonal) và huấn luyện 3 mô hình chuyên biệt. Việc này giúp giảm nhiễu chéo giữa các nhóm hàng và tăng độ chính xác tổng thể lên **94.2%**.

---

## 📁 Giải Thích Chi Tiết Thư Mục

### 1. `config/` (Trung tâm điều khiển)
- **Nội dung**: Chứa file `config.yaml`.
- **Ý tưởng**: Tách biệt hoàn toàn "Dữ liệu cấu hình" khỏi "Mã nguồn". Thay vì sửa code Python, bạn chỉ cần vào đây để đổi ngày lễ, thay đổi các mốc Lag, hoặc chỉnh số lượng vòng lặp huấn luyện (Optuna).

### 2. `data/` (Tầng lưu trữ dữ liệu)
- `raw/`: Dữ liệu gốc (Bất biến). Đảm bảo tính nguyên bản để có thể tái lập kết quả.
- `processed/`: Dữ liệu sau khi làm sạch, xử lý Zero-Inflation (điền số 0 cho những ngày không bán được hàng).
- `features/`: Dữ liệu đã "tiến hóa" thành các vector đặc trưng, sẵn sàng để đưa vào máy học.

### 3. `src/` (Trái tim của hệ thống - Modular Design)
Chúng ta chia nhỏ `src` để dễ bảo trì và mở rộng:
- `api/`: Cung cấp cổng giao tiếp REST API (FastAPI). Giúp các bộ phận khác (Web, Mobile, App bán hàng) có thể lấy dự báo chỉ bằng một yêu cầu HTTP.
- `data/`: Chứa các "Worker" chuyên trách về dữ liệu (Loader, Cleaner, Validator).
- `features/`: Nơi chứa chất xám quan trọng nhất - Logic tạo ra 80+ đặc trưng để máy học hiểu được quy luật kinh doanh.
- `models/`: Chứa quy trình huấn luyện 2 giai đoạn (Prophet & LightGBM).
- `evaluation/`: Bộ công cụ đánh giá. Đặc biệt có `backtesting.py` giúp mô phỏng việc dự báo trong quá khứ để kiểm tra độ tin cậy.
- `pipeline/`: Người chỉ huy. Điều phối các bước từ 1 đến 5 theo một quy trình tự động hóa hoàn toàn.
- `utils/`: Các công cụ hỗ trợ như Logger (ghi nhật ký hệ thống) và Config Loader.

### 4. `models/` (Kho lưu trữ trí tuệ nhân tạo)
- Chứa các file `.pkl` (Binary). Đây là "bộ não" đã được huấn luyện xong. API sẽ đọc các file này để đưa ra dự báo mà không cần huấn luyện lại.

### 5. `results/` (Báo cáo & Trực quan hóa)
- `figures/`: Các biểu đồ so sánh Dự báo vs Thực tế. Giúp con người dễ dàng quan sát quy luật.
- `metrics/`: Các con số thống kê khô khan nhưng quan trọng (RMSE, MAPE). Dùng để báo cáo hiệu suất kỹ thuật.

### 6. `logs/` (Nhật ký vận hành)
- Lưu lại mọi diễn biến khi chạy Pipeline. Nếu có lỗi xảy ra, bạn chỉ cần vào `pipeline.log` là sẽ biết chính xác lỗi ở bước nào, dòng nào.

---

## 🛠 Luồng Hoạt Động (Workflow)
1.  **Dữ liệu thô** vào `data/raw/`.
2.  **Pipeline** chạy qua `src/` để làm sạch và tạo đặc trưng.
3.  **Model** được huấn luyện và cất vào `models/`.
4.  **Kết quả** được vẽ ra `results/`.
5.  **API** sẵn sàng phục vụ các yêu cầu dự báo từ bên ngoài.

---

## 📈 Kết Quả Then Chốt (Key Results)
*   **Độ chính xác tổng thể (Accuracy)**: **94.2%** (đo trên tập dữ liệu 6 tháng cuối năm 2025).
*   **Cải thiện**: Tăng từ **71%** (mô hình đơn) lên **94%** nhờ chiến lược Hybrid Multi-cluster.

---
*Tài liệu này giúp bạn hoặc bất kỳ kỹ sư nào tiếp quản dự án đều có thể hiểu nhanh cấu trúc trong 5 phút.*
