# KẾ HOẠCH THÍ NGHIỆM TUẦN 8 — TỐI ƯU HÓA, CẢI TIẾN VÀ SO SÁNH MÔ HÌNH
**Dự án:** `FaceDetection_Week4`  
**Trạng thái:** Đề xuất thiết kế & ước tính chi phí. **Chưa thực thi các thí nghiệm này.**

---

## 1. Động lực từ Bằng chứng Tiền kiểm toán

Dựa trên [limitations_evidence.csv](limitations_evidence.csv) và [preflight_report.md](preflight_report.md), hệ thống cửa sổ $24 \times 24$ hiện tại bộc lộ hai nhóm hạn chế có bằng chứng định lượng rõ ràng:
1. **Quá khớp (*Overfitting*) & Độ phức tạp chưa được tiết chế:** `Decision Tree` ($\Delta F_1 = +0{,}0606$, CV std $= 0{,}0185$, Test $F_1 = 0{,}8359$), `Random Forest` ($\Delta F_1 = +0{,}0381$, `max_depth=None` fit đạt $0{,}9997$ trên train), và `AdaBoost` ($\Delta F_1 = +0{,}0280$, fit lâu nhất $8{,}8947\text{ s}/$fold).
2. **Tỷ lệ báo động giả vượt ngưỡng $\text{FPR} \le 5\%$ và nhiễu chiều đặc trưng:** `k-NN` ($\text{CV FPR} = 7{,}38\%$, $\text{Test FPR} = 7{,}80\%$), `Logistic Regression` ($\text{CV FPR} = 6{,}56\%$, $\text{Test FPR} = 7{,}12\%$), cùng tốc độ suy luận chậm của `SVM RBF` ($0{,}0750\text{ s}/$fold) trong không gian 258 đặc trưng Haar chồng lấn.

---

## 2. Hai Nhóm Giải pháp Đề xuất (Gắn với Hạn chế Có Bằng chứng)

### Giải pháp 1: Lựa chọn Đặc trưng Haar-like bên trong `Pipeline`
- **Hạn chế mục tiêu:** Không gian $258$ đặc trưng Haar có nhiều ô chữ nhật liền kề/chồng lấn nhau theo bước dịch 4 px và 3 kích thước block ($4, 6, 8$), dễ gây hiệu ứng "lời nguyền số chiều" cho `k-NN`, đa cộng tuyến cho `Logistic Regression`, chia nhánh nhiễu cho `Decision Tree` và tăng chi phí huấn luyện/suy luận cho `AdaBoost`, `Random Forest`, `SVM RBF`.
- **Cấu trúc cài đặt bắt buộc (chống rò rỉ):** Đưa bộ chọn đặc trưng vào thành một bước nằm **bên trong `Pipeline`** (trước `clf`, sau `StandardScaler` đối với mô hình cần chuẩn hóa), chỉ `.fit()` trên phần train của fold đang xét:
  - Phương án A (Lọc thống kê đơn biến — chi phí rất thấp): `SelectKBest(score_func=f_classif, k=...)` với các mức khảo sát dự kiến $k \in \{64, 128, 192, 258\}$ (hoặc tỷ lệ phần trăm).
  - Phương án B (Lọc theo mô hình / L1 — cho mô hình tuyến tính/cây): `SelectFromModel` dùng `LogisticRegression(l1_ratio=1.0, solver='saga'|'liblinear')` hoặc tầm quan trọng cây nông, giữ ở dạng nhánh thí nghiệm riêng có kiểm soát.
- **Giả thuyết kiểm chứng:** Giảm số chiều xuống tập con đặc trưng mang tính phân biệt cao có thể: (1) cải thiện $\text{FPR}$ và $F_1$ của `k-NN` và `Logistic Regression`; (2) rút ngắn đáng kể thời gian `fit` của `AdaBoost`/`Random Forest` và thời gian `predict` của `SVM RBF`.

### Giải pháp 2: Điều chỉnh Độ phức tạp, Chính quy hóa và Tối ưu Siêu tham số bằng CV Kiểm soát Nhóm
- **Hạn chế mục tiêu:** Quá khớp ở mô hình cây (`Decision Tree`, `Random Forest`), chi phí/biên quyết định chưa tối ưu ở `AdaBoost`, `SVM RBF`, `Logistic Regression` và `k-NN`.
- **Không gian tìm kiếm đề xuất (giữ gọn để khả thi trong ngân sách chạy):**
  1. **AdaBoost (Trọng tâm):**
     - `estimator__max_depth`: `[1, 2]`
     - `n_estimators`: `[100, 200, 300]`
     - `learning_rate`: `[0.1, 0.5, 1.0]`
  2. **SVM RBF:**
     - `C`: `[0.5, 1.0, 2.0, 5.0, 10.0]`
     - `gamma`: `['scale', 0.002, 0.005, 0.01]`
  3. **Random Forest:**
     - `max_depth`: `[10, 15, 25, None]`
     - `min_samples_leaf`: `[2, 4, 8]`
     - `max_features`: `['sqrt', 0.2]`
     - `n_estimators`: `[150, 200]`
  4. **Logistic Regression:**
     - `C`: `[0.05, 0.1, 0.5, 1.0, 2.0, 5.0]`
     - `l1_ratio` (trên `solver='saga'` hoặc `liblinear` tùy phiên bản sklearn 1.9.1; cần kiểm tra API không deprecated): `[0.0, 0.5, 1.0]`
  5. **k-NN:**
     - `n_neighbors`: `[3, 5, 7, 11, 15]`
     - `weights`: `['uniform', 'distance']`
     - `metric`: `['euclidean', 'manhattan', 'cosine']`
  6. **Decision Tree:**
     - `max_depth`: `[4, 5, 7, 10]`
     - `min_samples_leaf`: `[2, 5, 10, 20]`
     - `ccp_alpha`: `[0.0, 0.001, 0.005]`

---

## 3. Thiết kế So sánh Từng Bước (Ablation Study)

Mỗi thuật toán được đánh giá qua 4 cấu hình nối tiếp trên cùng bộ chia fold để tách bạch tác động thực sự của từng thay đổi:

| Cấu hình Ablation | Thành phần Pipeline | Mục tiêu trả lời |
| :--- | :--- | :--- |
| **`S0` — Baseline Tuần 6** | Cấu hình gốc Tuần 6 (258 đặc trưng, siêu tham số mặc định) | Mốc đối chiếu chuẩn trên đúng 5 outer fold. |
| **`S1` — Chỉ Feature Selection** | `Pipeline([..., ('select', SelectKBest), ('clf', S0_clf)])` | Tác động riêng của chọn đặc trưng khi giữ nguyên siêu tham số gốc. |
| **`S2` — Chỉ Tuning Siêu tham số** | `Pipeline([..., ('clf', Tuned_clf)])` trên đủ 258 đặc trưng | Tác động riêng của điều chỉnh độ phức tạp/chính quy hóa. |
| **`S3` — Kết hợp (`FS + Tuning`)** | `Pipeline([..., ('select', Tuned_SelectKBest), ('clf', Tuned_clf)])` | Có cộng hưởng hay giảm hiệu năng khi phối hợp cả hai? |

*Nguyên tắc báo cáo:* Mọi trường hợp cải tiến không làm tăng $F_1$/$\text{FPR}$ hoặc làm giảm hiệu năng đều phải được ghi nhận và phân tích đầy đủ, không chọn lọc chỉ giữ kết quả đẹp.

---

## 4. Giao thức Đánh giá Chống Rò rỉ và Nested Group CV

1. **Bảo toàn tuyệt đối phân chia tập và cột nhóm:**
   - Dùng lại nguyên vẹn `split_indices.npz` (`2.511` Train / `540` Val / `538` Test) và cột `group` trong `metadata.csv`.
   - `DummyClassifier(strategy='most_frequent')` giữ nguyên làm mốc tham chiếu cố định.
2. **Nested Group Cross-Validation trên $2.511$ mẫu Train:**
   - **Vòng ngoài (*Outer CV — 5 folds*):** Dùng chính xác phân công 5 fold từ `reports/tables/week6_cv_assignments.csv` (`StratifiedGroupKFold`, `~2.008` train / `~502` val mỗi fold) để so sánh trực tiếp 1-1 với kết quả Tuần 6 (`S0`).
   - **Vòng trong (*Inner CV — 3 folds*):** Trên mỗi phần outer-train (`~2.008` mẫu), sử dụng `StratifiedGroupKFold(n_splits=3)` dựa trên cột `group` để tìm tập đặc trưng/siêu tham số tốt nhất (theo tiêu chí tối ưu $F_1$ hoặc điểm ràng buộc $\text{Recall} \ge 0{,}90, \text{FPR} \le 0{,}05$ — quy ước rõ trước khi chạy).
   - **Phân biệt hai loại điểm CV:**
     - *Inner Selection CV Score:* Chỉ dùng để chọn ứng viên cấu hình trong vòng trong (có thiên lệch lạc quan nhẹ do dùng để chọn).
     - *Outer Evaluation CV Score:* Điểm trung bình và độ lệch chuẩn trên 5 outer folds — thước đo khách quan duy nhất dùng để so sánh `S0`, `S1`, `S2`, `S3`.
3. **Cách ly Validation giữ riêng và Test:**
   - Không dùng $540$ mẫu Validation hay $538$ mẫu Test để chọn đặc trưng hay tìm kiếm siêu tham số trong Nested CV.
   - Sau khi chốt ứng viên thắng cuộc từ Outer CV, mới fit trên toàn bộ $2.511$ mẫu Train và kiểm tra thêm ngưỡng trên $540$ mẫu Validation (nếu cần), trước khi đối chiếu một lần cuối trên $538$ mẫu Test (ghi nhận rõ tập Test đã từng được đánh giá ở Tuần 6 nên đây là bước kiểm chứng cuối cùng có đối chiếu, không phải tập ẩn tuyệt đối).

---

## 5. Ước tính Số lượt Fit và Chi phí Chạy (Trước khi Triển khai)

Cơ sở ước tính lấy trực tiếp từ `fit_time_cv_mean_s` thực đo ở Tuần 6 trên $~2.008$ mẫu (`n_jobs=1`), với phần cứng hiện có là **8 nhân / 16 luồng logic (AMD Ryzen 7 7840H, 15.19 GiB RAM)**. Lưu ý: ở vòng trong `Inner 3-fold CV`, mỗi lượt fit dùng `~1.339` mẫu (nhanh hơn khoảng $1{,}3\times\text{–}1{,}5\times$ so với outer fit, nhưng cộng thêm chi phí predict trên outer/inner).

| Mô hình | Fit Time Outer thực đo (`s`) | Thiết kế tìm kiếm đề xuất (Nested 5×3) | Số ứng viên (`S2`/`S3`) | Tổng số lượt fit (`S1` + `S2` + `S3` + refit outer) | Ước tính thời gian tuần tự (`n_jobs=1`) | Ước tính khi song song (`n_jobs=4..8`) |
| :--- | :---: | :--- | :---: | :--- | :---: | :---: |
| **AdaBoost** | $8{,}895$ | `RandomizedSearchCV(n_iter=10)` cho `S2` & `S3`; `SelectKBest(k=[64,128,192])` cho `S1` | 10 | $S1(15+5) + S2(5\times 30+5) + S3(5\times 30+5) = \mathbf{330}$ lượt | $\approx 35\text{ – }45$ phút *(nặng nhất)* | $\approx 6\text{ – }10$ phút |
| **Random Forest** | $5{,}114$ | `RandomizedSearchCV(n_iter=12)` cho `S2` & `S3`; `SelectKBest(k=3 mức)` cho `S1` | 12 | $S1(20) + S2(5\times 36+5) + S3(5\times 36+5) = \mathbf{390}$ lượt | $\approx 25\text{ – }33$ phút | $\approx 4\text{ – }7$ phút |
| **SVM RBF** | $0{,}155$ | `GridSearchCV` 16 tổ hợp cho `S2`, `RandomizedSearchCV(n_iter=16)` cho `S3` | 16 | $S1(20) + S2(5\times 48+5) + S3(5\times 48+5) = \mathbf{510}$ lượt | $\approx 1{,}5\text{ – }2{,}5$ phút | $< 1$ phút |
| **Decision Tree** | $0{,}375$ | `GridSearchCV` 24 tổ hợp cho `S2`, `n_iter=20` cho `S3` | 24 / 20 | $S1(20) + S2(365) + S3(305) = \mathbf{690}$ lượt | $\approx 3\text{ – }4$ phút | $< 1$ phút |
| **Logistic Reg.** | $0{,}096$ | `GridSearchCV` 12 tổ hợp cho `S2` & `S3` | 12 | $S1(20) + S2(185) + S3(185) = \mathbf{390}$ lượt | $\approx 0{,}8\text{ – }1{,}5$ phút | $< 0{,}5$ phút |
| **k-NN** | $0{,}010$ *(pred $0{,}018$)* | `GridSearchCV` 20 tổ hợp cho `S2` & `S3` | 20 | $S1(20) + S2(305) + S3(305) = \mathbf{630}$ lượt | $\approx 0{,}5\text{ – }1$ phút | $< 0{,}5$ phút |
| **TỔNG CỘNG** | — | Toàn bộ 6 mô hình (Ablation `S1`, `S2`, `S3` với Nested 5×3 CV) | — | **~2.940 lượt fit** | **~66 – 87 phút** | **~13 – 20 phút** |

### Biện pháp kiểm soát chi phí khi triển khai
1. Giới hạn không gian tìm kiếm của `AdaBoost` và `Random Forest` bằng `RandomizedSearchCV` (`n_iter=10..12`), đồng thời đặt `n_jobs=4..8` (trên tối đa 16 luồng) để tránh bão hòa RAM (`3,47 GiB` trống tại thời điểm đo `environment.json`; đặt `pre_dispatch='2*n_jobs'`).
2. Khảo sát chi phí `SelectKBest` trước (`S1`, chỉ 20 lượt fit/mô hình $\approx 3\text{ phút}$ toàn bộ) để xem các mức $k$ khả dĩ trước khi nhân lưới với `S3`.
3. Lưu kết quả trung gian sau từng mô hình (checkpoint CSV) để không mất công tính toán nếu cần dừng giữa chừng.
