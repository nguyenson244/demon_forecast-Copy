# Mô Hình Sử Dụng Theo Từng Cụm — Giải Thích Chi Tiết

---

## Tổng Quan Kiến Trúc

Mọi cụm đều dùng **kiến trúc Hybrid 2 tầng**:

```
Tầng 1 (chung cho tất cả): Prophet  → học trend + seasonality
Tầng 2 (khác nhau theo cụm): LightGBM / XGBoost / Two-Part Model → học phần dư (residual)
```

Lý do tách tầng 2 theo cụm: mỗi nhóm brand có **pattern nhu cầu khác nhau căn bản** → cần mô hình phù hợp với từng đặc điểm đó.

---

## Cụm 0 — Stable (Ổn Định)

**Brand:** Kinh Đô Bread, Kinh Đô Cake, Solite, Cosy

### Đặc điểm dữ liệu
- Bán đều quanh năm, ít ngày bằng 0
- Biến động thấp, không có đỉnh mùa vụ cực đoan
- Volume lớn nhất trong 3 cụm (Solite: ~94 triệu, Kinh Đô Bread: ~47 triệu)

### Mô hình sử dụng

#### Tầng 1: Prophet
```
Đầu vào: chuỗi thời gian daily theo brand
Đầu ra:  prophet_pred  (trend + weekly + yearly seasonality)
```
**Tại sao Prophet?**
Các brand Stable có trend rõ ràng và seasonality tuần/năm ổn định → Prophet capture tốt mà không cần feature thủ công.

#### Tầng 2: LightGBM + XGBoost Ensemble (50/50)
```
Đầu vào: 90+ features (lag, rolling, holiday, weather...)
Mục tiêu học: residual = actual - prophet_pred  (chỉ trên OOS calib)
Kết quả: residual_pred = 0.5 × LightGBM_pred + 0.5 × XGBoost_pred
```
**Tại sao LightGBM?**
- Với dữ liệu ổn định, residual của Prophet là các dao động ngắn hạn (promotional spike, thời tiết, lag effect) → LightGBM rất mạnh trong việc học từ nhiều features tương tác.
- Xử lý tốt NaN từ lag dài (lag_365 NaN năm đầu) mà không cần imputation.

**Tại sao thêm XGBoost (Ensemble)?**
- LightGBM (leaf-wise) và XGBoost (depth-wise) xây dựng cây theo cách khác nhau → hai mô hình mắc sai số ở các điểm khác nhau.
- Trung bình 50/50 giảm variance mà không tăng bias → kết quả ổn định hơn, đặc biệt với brand volume lớn như Solite, Kinh Đô Bread nơi sai số tuyệt đối rất nhạy cảm.

#### Kết quả (Tập Test)
| Brand | WMAPE Hybrid | WMAPE Prophet | Cải thiện |
|---|---|---|---|
| Kinh Đô Bread | **13.8%** | 22.5% | +8.7pp |
| Kinh Đô Cake | 30.3% | 30.1% | ≈0 (fallback) |
| Solite | **36.2%** | 41.0% | +4.8pp |
| Cosy | **37.5%** | 45.3% | +7.8pp |

---

## Cụm 1 — Regular (Thông Thường)

**Brand:** AFC, Oreo, LU, Ritz, Slide, Kinh Đô Biscuit

### Đặc điểm dữ liệu
- Biến động vừa, có đỉnh nhẹ dịp lễ nhưng không cực đoan
- Tỷ lệ ngày bán = 0 vừa phải (~20–40%)
- Volume trung bình, phân bổ tương đối đều

### Mô hình sử dụng

#### Tầng 1: Prophet
```
Đầu vào: chuỗi thời gian daily theo brand
Đầu ra:  prophet_pred
```
**Tại sao Prophet?**
Các brand Regular vẫn có seasonality tuần/tháng rõ ràng. Prophet phù hợp vì xử lý được khoảng thiếu dữ liệu (ngày không có đơn) mà không bị crash như ARIMA.

#### Tầng 2: LightGBM + XGBoost Ensemble (50/50)
```
Cấu trúc: giống Cụm 0
Điểm khác: Optuna HPO tìm siêu tham số riêng cho Cụm 1
           (learning_rate, num_leaves, lambda... khác Cụm 0)
```
**Tại sao giống Cụm 0 nhưng HPO riêng?**
- Cấu trúc ensemble phù hợp cho cả 2 cụm, nhưng Cụm 1 có zero-inflation cao hơn và distribution khác → cần siêu tham số riêng để tránh overfit vào các ngày bán thấp.
- Optuna tự động tìm `min_child_samples` cao hơn để ignore noise từ ngày bán lẻ.

#### Per-Brand Fallback (áp dụng một số brand)
Một số brand trong Cụm 1 (Slide, Ritz, Kinh Đô Biscuit) có LightGBM làm tệ hơn Prophet trên calibration holdout → tự động revert về Prophet-only:
```
Nếu: hybrid_WMAPE > prophet_WMAPE + 2.0 điểm %
Thì: final_pred = prophet_pred  (bỏ qua phần dư LightGBM)
```

#### Kết quả (Tập Test)
| Brand | WMAPE Hybrid | WMAPE Prophet | Cải thiện |
|---|---|---|---|
| Oreo | **37.2%** | 41.9% | +4.7pp |
| AFC | **46.9%** | 53.9% | +7.0pp |
| LU | **64.8%** | 84.3% | +19.5pp |
| Ritz | 47.7% | 46.8% | -0.9pp (fallback) |
| Slide | 42.1% | 40.2% | -1.9pp (fallback) |
| Kinh Đô Biscuit | 48.4% | 42.9% | -5.5pp (fallback) |

---

## Cụm 2 — Seasonal (Mùa Vụ)

**Brand:** THU, Trang Vang, Hamper, Cadbury, Koko

### Đặc điểm dữ liệu
- Đỉnh cực cao dịp Trung Thu / Tết (gấp 10–50× ngày thường)
- **Zero-inflation rất cao:** Trang Vang, Cadbury, Koko có hàng trăm ngày bán = 0 liên tiếp
- Off-season gần như không có doanh thu

### Vấn đề với mô hình thông thường
Nếu dùng LightGBM hồi quy đơn giản như Cụm 0/1:
- Mô hình sẽ **over-predict** trong off-season (dự báo ra số dương trong khi thực tế = 0)
- Lý do: mô hình học rằng "trung bình có bán" và không phân biệt được ngày on/off season rõ ràng

### Mô hình sử dụng

#### Tầng 1: Prophet
```
Đầu vào: chuỗi thời gian daily theo brand
Đầu ra:  prophet_pred
```
**Tại sao Prophet vẫn phù hợp?**
Prophet có thể học được đỉnh Trung Thu hàng năm thông qua Fourier seasonality và custom holidays. Tuy nhiên vẫn có xu hướng over-predict off-season vì fit trên tổng thể → cần Tầng 2 hỗ trợ.

#### Tầng 2: Two-Part Model (Mô Hình Hai Thành Phần)

Đây là điểm **khác biệt hoàn toàn** so với Cụm 0 và 1:

```
Thành phần A — LGBMClassifier:
  Học: P(sale > 0) — xác suất ngày đó có bán hàng
  Train: toàn bộ tập calib (cả ngày bán = 0 và > 0)
  class_weight = "balanced"  (vì số ngày = 0 >> số ngày > 0)

Thành phần B — LGBMRegressor:
  Học: residual = actual - prophet_pred
  Train: CHỈ các ngày có actual > 0 (loại ngày không bán)
  Tránh học noise từ ngày off-season

Kết hợp:
  final_pred = (prophet_pred + residual_B) × P(sale > 0)_A
               └─────────────────────────┘   └──────────┘
                    dự báo volume nếu có bán   "gate" on/off
```

**Tại sao cần hai thành phần riêng biệt?**

| Câu hỏi | Mô hình xử lý |
|---|---|
| "Ngày này có bán không?" | Classifier (A) |
| "Nếu có bán thì bán bao nhiêu?" | Regressor (B) |

Nếu chỉ dùng Regressor đơn: không phân biệt được "ngày không bán" → luôn dự báo số dương → WMAPE tệ trong off-season.

**P(sale > 0) hoạt động như một "gate":**
- Off-season: P ≈ 0.05 → `final_pred ≈ 0` ✓
- Trước Trung Thu: P ≈ 0.95 → `final_pred ≈ prophet + residual` ✓

#### Kết quả (Tập Test)
| Brand | WMAPE Hybrid | WMAPE Prophet | Cải thiện |
|---|---|---|---|
| Hamper | **21.1%** | 41.7% | +20.6pp |
| THU | **78.8%** | 107.7% | +28.9pp |
| Trang Vang | 0% (sale=0 toàn tập test) | — | fallback |
| Cadbury | 0% (sale=0 toàn tập test) | — | fallback |
| Koko | 0% (sale=0 toàn tập test) | — | fallback |

---

## Tổng Hợp So Sánh 3 Cụm

| | **Cụm 0 — Stable** | **Cụm 1 — Regular** | **Cụm 2 — Seasonal** |
|---|---|---|---|
| **Tầng 1** | Prophet | Prophet | Prophet |
| **Tầng 2** | LightGBM + XGBoost | LightGBM + XGBoost | Two-Part Model |
| **Lý do khác biệt** | Volume lớn, cần giảm variance | Zero-inflation vừa, cần HPO riêng | Zero-inflation cực cao, cần gate on/off |
| **HPO (Optuna)** | 20 trials | 20 trials | Không (Two-Part có cấu trúc riêng) |
| **Fallback về Prophet** | Kinh Đô Cake | Slide, Ritz, Kinh Đô Biscuit | Trang Vang, Cadbury, Koko |
| **WMAPE trung bình cụm** | ~29.5% | ~47.5% | ~78.8% (chỉ tính THU, Hamper) |

---

## Các Hướng Cải Thiện Tiếp Theo

Phần này giải thích **3 hướng cải thiện cụ thể**, cách chúng hoạt động, và ảnh hưởng đến pipeline hiện tại như thế nào.

---

### Hướng 1 — Croston's Method cho Brand Intermittent (Slide, Ritz, Kinh Đô Biscuit)

#### Vấn đề hiện tại
Ba brand này đang bị **fallback về Prophet-only** vì LightGBM thêm noise. Nhưng Prophet cũng không tốt lắm vì nó được thiết kế cho chuỗi liên tục, không phải chuỗi gián đoạn (có nhiều ngày bán = 0 xen kẽ không theo quy luật).

**Ví dụ pattern Slide:**
```
Ngày 1:  0
Ngày 2:  0
Ngày 3:  8,500  ← đột nhiên có đơn
Ngày 4:  0
Ngày 5:  0
Ngày 6:  0
Ngày 7:  12,300 ← lại có đơn
```
Prophet và LightGBM đều cố gắng dự báo *con số* cho mỗi ngày → thất bại vì pattern quá ngẫu nhiên.

#### Croston's Method hoạt động như thế nào?
Croston tách bài toán thành **2 câu hỏi riêng biệt**:

```
Câu hỏi 1: Khoảng cách trung bình giữa 2 lần có đơn là bao nhiêu ngày?
           → Dùng Exponential Smoothing trên "inter-arrival time"

Câu hỏi 2: Khi có đơn thì giá trị trung bình là bao nhiêu?
           → Dùng Exponential Smoothing trên "demand size"

Kết hợp:
  forecast = demand_size / inter_arrival_time
           = "nếu có 1 đơn mỗi 4 ngày, và đơn trung bình 8,000"
           → dự báo mỗi ngày = 8,000 / 4 = 2,000
```

#### Ảnh hưởng đến pipeline hiện tại

```
TRƯỚC (hiện tại):
  Fallback brands → Prophet-only → final_pred

SAU KHI THÊM CROSTON:
  Fallback brands → Croston's Method → final_pred
                    (thay thế Prophet cho các brand này)
```

**Mức độ thay đổi code:** Thấp — chỉ cần thêm hàm Croston vào bước fallback trong `lightgbm_model.py`, không đụng đến cấu trúc 5 bước pipeline.

**Thư viện cần thêm:** `statsforecast` (Nixtla) — có sẵn hàm `CrostonOptimized`.

**Kết quả kỳ vọng:** WMAPE Slide/Ritz/Kinh Đô Biscuit giảm ~5–15pp.

---

### Hướng 2 — Quantile Regression cho Brand Biến Động Cao (LU, AFC)

#### Vấn đề hiện tại
LU (64.8%) và AFC (46.9%) có biến động cực lớn do yếu tố khuyến mãi và cạnh tranh — những yếu tố **không có trong dữ liệu hiện tại**. Dù cải thiện mô hình đến đâu, dự báo một điểm duy nhất sẽ luôn có sai số lớn.

#### Quantile Regression hoạt động như thế nào?
Thay vì dự báo **một con số** (ví dụ: 5,000 thùng), dự báo **một khoảng**:

```
Dự báo thông thường:   final_pred = 5,000
Quantile Regression:   Q10 = 2,000  (bi quan — chỉ có 10% ngày thấp hơn này)
                       Q50 = 5,000  (trung bình)
                       Q90 = 9,500  (lạc quan — chỉ có 10% ngày cao hơn này)
```

**Tại sao phù hợp cho LU, AFC?**
- Khi không có dữ liệu khuyến mãi, mô hình không thể biết ngày nào sẽ bán đột biến
- Khoảng [Q10, Q90] truyền đạt sự **không chắc chắn** này đến người dùng
- Người lập kế hoạch sản xuất có thể dùng Q90 để dự phòng an toàn

#### LightGBM hỗ trợ Quantile Loss sẵn:
```python
# Thay đổi duy nhất trong lightgbm_model.py:
model = lgb.LGBMRegressor(objective="quantile", alpha=0.5)   # Q50
model_q10 = lgb.LGBMRegressor(objective="quantile", alpha=0.1)  # Q10
model_q90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9)  # Q90
```

#### Ảnh hưởng đến pipeline hiện tại

```
TRƯỚC:
  LightGBM → một giá trị residual → final_pred (1 cột)

SAU:
  LightGBM × 3 → residual_q10, residual_q50, residual_q90
               → final_pred_q10, final_pred_q50, final_pred_q90 (3 cột)
```

**Mức độ thay đổi code:** Trung bình — cần sửa `_train_cluster()` để train 3 model thay vì 1, và sửa `metrics.py` để tính Pinball Loss thay vì WMAPE cho các cột quantile.

**Không thay đổi:** Bước 1 (Prophet), Bước 2 (Feature Engineering), Bước 3 (OOS calib), cấu trúc 5 bước.

**Kết quả kỳ vọng:** WMAPE Q50 tương đương hiện tại, nhưng thêm được thông tin về độ không chắc chắn — giá trị thực tiễn cao hơn cho báo cáo.

---

### Hướng 3 — Tách Model Theo Mùa cho THU (Peak / Off-season)

#### Vấn đề hiện tại
THU có **2 chế độ hoàn toàn khác nhau**:

```
Chế độ 1 — Off-season (300 ngày/năm):  bán = 0  hoặc rất thấp
Chế độ 2 — Peak season (60 ngày/năm):  bán đột biến, tăng 50× so với off-season
```

Two-Part Model hiện tại cố gắng học cả 2 chế độ trong cùng một mô hình → mô hình bị "kéo" theo off-season (vì chiếm 83% dữ liệu) → peak season bị under-predict.

#### Tách model theo mùa hoạt động như thế nào?

```
Định nghĩa ngưỡng mùa:
  Peak season:    từ 60 ngày trước Trung Thu đến 15 ngày sau
  Off-season:     toàn bộ thời gian còn lại

Train 2 model riêng biệt:
  Model A (Peak):     chỉ train trên ~120 rows peak season
                      → học pattern tăng trưởng trước Trung Thu
                      → học pattern giảm sau Trung Thu

  Model B (Off):      chỉ train trên ~600 rows off-season
                      → học khi nào bán lẻ (nếu có)
                      → thường dự báo gần 0

Khi dự báo:
  Nếu ngày thuộc peak window → dùng Model A
  Nếu không                  → dùng Model B
```

#### So sánh với Two-Part Model hiện tại

| | **Two-Part Model (hiện tại)** | **Split Season Model (đề xuất)** |
|---|---|---|
| Số model | 2 (Classifier + Regressor) | 2 (Peak + Off-season) |
| Dữ liệu train mỗi model | Tất cả (trộn lẫn peak và off) | Tách rõ ràng theo mùa |
| Điểm mạnh | Tự động học ranh giới on/off | Mỗi model tập trung sâu vào 1 chế độ |
| Điểm yếu | Bị kéo lệch bởi off-season | Phải định nghĩa ngưỡng mùa thủ công |

#### Ảnh hưởng đến pipeline hiện tại

```
TRƯỚC:
  Cụm 2 → _train_two_part_cluster() → 1 bộ dự báo

SAU:
  Cụm 2 → _train_peak_model()     (cho THU, Hamper)
         → _train_offseason_model()
         → chọn model theo ngày khi dự báo
```

**Mức độ thay đổi code:** Trung bình — thêm hàm mới trong `lightgbm_model.py` cho Cụm 2 THU/Hamper, không ảnh hưởng Cụm 0/1 và không đổi 5 bước pipeline.

**Kết quả kỳ vọng:** THU giảm từ 78.8% xuống ~55–65%.

---

### Tổng Hợp Ảnh Hưởng Đến Pipeline

| Hướng cải thiện | Thay đổi ở đâu | Mức độ phức tạp | Kỳ vọng cải thiện |
|---|---|---|---|
| Croston's Method (Slide/Ritz/KB) | `lightgbm_model.py` — bước fallback | Thấp | -5–15pp WMAPE 3 brands |
| Quantile Regression (LU/AFC) | `lightgbm_model.py` — hàm `_train_cluster` | Trung bình | Không giảm WMAPE trực tiếp, thêm khoảng tin cậy |
| Split Season Model (THU) | `lightgbm_model.py` — hàm riêng Cụm 2 | Trung bình | THU: -15–25pp |

**Điều không thay đổi dù implement hướng nào:**
- Bước 1: Load & Clean Data
- Bước 2: Feature Engineering (90+ features)
- Bước 3: Prophet Stage 1
- Bước 5: Evaluation & Visualization
- Toàn bộ logic Cụm 0 và Cụm 1 (trừ Quantile nếu áp dụng)

---

## Sơ Đồ Tổng Thể — Kiến Trúc Hiện Tại

```
15 Brands
    │
    ├─── Cụm 0 (Stable) ────► Prophet → LightGBM + XGBoost → Blend/Bias
    │    KINH DO BREAD                    (Ensemble 50/50)
    │    KINH DO CAKE
    │    SOLITE
    │    COSY
    │
    ├─── Cụm 1 (Regular) ───► Prophet → LightGBM + XGBoost → Blend/Bias
    │    AFC                              (Ensemble 50/50)
    │    OREO                             (HPO riêng)
    │    LU
    │    RITZ
    │    SLIDE
    │    KINH DO BISCUIT
    │
    └─── Cụm 2 (Seasonal) ──► Prophet → Two-Part Model     → Blend/Bias
         THU                             (Classifier × Regressor)
         TRANG VANG                       áp dụng cho TẤT CẢ
         HAMPER                           5 brand trong cụm
         CADBURY
         KOKO
              │
              ▼
    Per-Brand Fallback Check
    (Trang Vang, Cadbury, Koko sale=0 → revert Prophet-only)
              │
              ▼
    Blend Weight Optimization (α per brand)
    Bias Correction (factor per brand)
              │
              ▼
         final_pred ≥ 0
```

---

## Sơ Đồ Tổng Thể Sau Khi Áp Dụng Tất Cả Cải Thiện (Đề Xuất)

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         15 BRANDS — DỮ LIỆU ĐẦU VÀO                    ║
╚══════════════════════════════════════════════════════════════════════════╝
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
   │   CỤM 0         │   │   CỤM 1         │   │   CỤM 2         │
   │   STABLE        │   │   REGULAR       │   │   SEASONAL      │
   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
            │                     │                     │
            ▼                     │                     │
     [TẦNG 1 — PROPHET]           │                     │
     (tất cả brand)               │                     │
            │                     │                     │
            ▼                     │                     │
   ┌─────────────────┐            │                     │
   │ LightGBM+XGBoost│            │                     │
   │  Ensemble 50/50 │            │                     │
   │ (Optuna HPO cụm0│            │                     │
   └────────┬────────┘            │                     │
            │                     ▼                     │
            │          ┌──────────────────────┐         │
            │          │  [TẦNG 1 — PROPHET]  │         │
            │          │  (tất cả brand)      │         │
            │          └──────────┬───────────┘         │
            │                     │                     │
            │          ┌──────────┴──────────┐          │
            │          │                     │          │
            │          ▼                     ▼          │
            │  ┌──────────────┐   ┌─────────────────┐  │
            │  │ AFC, LU      │   │ SLIDE, RITZ,    │  │
            │  │ (WMAPE cao)  │   │ KINH DO BISCUIT │  │
            │  ├──────────────┤   │ (intermittent)  │  │
            │  │LightGBM+XGB  │   ├─────────────────┤  │
            │  │+ Quantile    │   │ CROSTON'S       │  │
            │  │Regression    │   │ METHOD          │  │
            │  │→ Q10/Q50/Q90 │   │(thay Prophet-   │  │
            │  └──────┬───────┘   │ only fallback)  │  │
            │         │           └────────┬────────┘  │
            │         │                    │           │
            │  ┌──────────────┐            │           │
            │  │ OREO         │            │           │
            │  │ (ổn định)    │            │           │
            │  │LightGBM+XGB  │            │           │
            │  │Ensemble 50/50│            │           │
            │  └──────┬───────┘            │           │
            │         │                    │           │
            └────┬────┘                    │           │
                 │                         │           ▼
                 │                         │  ┌─────────────────┐
                 │                         │  │  [TẦNG 1—PROPHET│
                 │                         │  │  tất cả brand]  │
                 │                         │  └────────┬────────┘
                 │                         │           │
                 │                         │  ┌────────┴────────┐
                 │                         │  │                 │
                 │                         │  ▼                 ▼
                 │                         │ ┌──────────┐  ┌──────────────┐
                 │                         │ │THU,HAMPER│  │TRANG VANG,   │
                 │                         │ ├──────────┤  │CADBURY, KOKO │
                 │                         │ │  SPLIT   │  ├──────────────┤
                 │                         │ │  SEASON  │  │ PROPHET-ONLY │
                 │                         │ │  MODEL   │  │ (sale=0 test)│
                 │                         │ │Peak+Off  │  └──────┬───────┘
                 │                         │ └────┬─────┘         │
                 │                         │      │               │
                 └─────────────────────────┴──────┴───────────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │   PER-BRAND FALLBACK   │
                              │ (Hybrid tệ → revert)   │
                              └────────────┬───────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │  BLEND WEIGHT (α)      │
                              │  BIAS CORRECTION       │
                              │  (per brand)           │
                              └────────────┬───────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │    final_pred ≥ 0      │
                              └────────────────────────┘
```

**Tóm tắt thay đổi so với kiến trúc hiện tại:**

| Brand | Hiện tại | Sau cải thiện |
|---|---|---|
| AFC, LU | LightGBM+XGBoost → 1 giá trị | LightGBM+XGBoost+Quantile → Q10/Q50/Q90 |
| Slide, Ritz, Kinh Đô Biscuit | Prophet-only (fallback) | Croston's Method |
| THU, Hamper | Two-Part Model (Classifier×Regressor) | Split Season (Peak + Off-season) |
| Trang Vang, Cadbury, Koko | Prophet-only (sale=0) | Giữ nguyên |
| Còn lại | Không đổi | Không đổi |

---

## Sơ Đồ Nếu Áp Dụng Cải Thiện Hướng 3 — Split Season (Chưa Implement)

```
Cụm 2 (Seasonal)
    │
    ├── TRANG VANG ──► Prophet-only (sale=0 toàn test, không cải thiện được)
    ├── CADBURY    ──► Prophet-only (sale=0 toàn test, không cải thiện được)
    ├── KOKO       ──► Prophet-only (sale=0 toàn test, không cải thiện được)
    │
    ├── THU    ─────► Prophet → Split Season Model   → Blend/Bias
    │                           ├── Model A (Peak)
    │                           │   train: ~120 rows mùa Trung Thu
    │                           │   60 ngày trước → 15 ngày sau Trung Thu
    │                           └── Model B (Off-season)
    │                               train: ~600 rows ngoài mùa
    │                               dự báo gần 0
    │
    └── HAMPER ─────► Prophet → Split Season Model   → Blend/Bias
                                (cùng cấu trúc như THU)

Lý do chỉ áp dụng cho THU và Hamper:
  - Trang Vang / Cadbury / Koko: sale = 0 toàn tập test
    → không có gì để cải thiện, giữ Prophet-only
  - THU (78.8%) và Hamper (21.1%): còn dư địa cải thiện rõ ràng
    → Split Season có thể giảm thêm 15–25pp cho THU
```
