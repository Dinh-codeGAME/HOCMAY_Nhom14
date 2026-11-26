# BẢN TÓM TẮT DỰ ÁN HỌC MÁY
## Đề tài: Phát hiện khuôn mặt người trong ảnh (*Face Detection in Images*)

- **Thư mục dự án:** `FaceDetection_Week4`
- **Môi trường thực nghiệm:** Python 3.12.4, `scikit-learn==1.9.1`, `numpy==2.4.6`, `pandas==3.0.5`, `joblib==1.5.3`, `streamlit`

---

## 1. Tổng quan Đề tài và Mục tiêu Nghiên cứu

- **Bài toán cốt lõi:** Phân loại nhị phân (*Binary Classification*) trên các ảnh con mức xám kích thước **$24 \times 24$ điểm ảnh** nhằm phân biệt hai lớp:
  - Lớp `1` (*Face*): Khuôn mặt người.
  - Lớp `0` (*Non-face*): Vùng nền không chứa khuôn mặt.
- **Bài toán mở rộng:** Tích hợp bộ phát hiện và định vị khuôn mặt tự động trên ảnh đầy đủ kích thước tùy ý (*Full Image Face Detector*) bằng cách kết hợp kim tự tháp ảnh đa tỷ lệ (*Image Pyramid*), cửa sổ trượt (*Sliding Window*), trích xuất 258 đặc trưng Haar-like, phân loại bằng Pipeline đã huấn luyện của dự án và lọc khung trùng lặp bằng *Non-Maximum Suppression (NMS)*.
- **Mục tiêu kỹ thuật thiết kế (trên bài toán phân loại ảnh con):** Đạt đồng thời độ nhạy cao ($\text{Recall} \ge 0{,}90$) để hạn chế bỏ sót khuôn mặt và tỷ lệ báo động giả thấp ($\text{FPR} \le 0{,}05$) để hạn chế nhận nhầm vùng nền.
- **Thuật toán trọng tâm:** **AdaBoost** (*Decision Stump*), đối sánh với 5 thuật toán học máy khác (SVM RBF, Random Forest, Logistic Regression, k-NN, Decision Tree) và 1 mô hình cơ sở tham chiếu (`DummyClassifier`).

---

## 2. Bộ Dữ liệu Thực nghiệm và Kiểm soát Rò rỉ Dữ liệu

- **Tổng quy mô:** **$3.589$ mẫu** ảnh xám $24 \times 24$ (`images_24x24.npy`, kiểu `uint8`). Toàn bộ $3.589$ ảnh khớp $100\%$ mã băm SHA-256 với `metadata.csv`.
  - **Nguồn Face (`1`):** LFW (*Labeled Faces in the Wild*).
  - **Nguồn Non-face (`0`):** Caltech 101 (`BACKGROUND_Google`).
- **Quy ước nhóm chống rò rỉ (*Group Splitting*):**
  - Lớp Face được gom nhóm theo cá nhân (`person_id`).
  - Lớp Non-face được gom nhóm theo ảnh nền nguồn gốc (`source_image_id`).
  - Bảo đảm không có sự giao thoa nhóm giữa các tập dữ liệu và giữa các fold kiểm định chéo.
- **Phân chia tập dữ liệu cố định (`split_indices.npz`):**
  - **Train:** $2.511$ mẫu ($1.277$ face, $1.234$ non-face).
  - **Held-out Validation:** $540$ mẫu ($280$ face, $260$ non-face) — dùng để khảo sát ngưỡng và khóa giao thức.
  - **Held-out Test:** $538$ mẫu ($243$ face, $295$ non-face) — giữ kín để đánh giá cuối cùng theo giao thức đã xác định từ train/validation.

---

## 3. Trích xuất Đặc trưng Haar-like

- **Không gian đặc trưng:** **$258$ đặc trưng Haar-like** (`haar_features.npy`, kiểu `float32`), được định nghĩa trong `data_dictionary_haar.csv`.
- **Năm mẫu hình học:** `two_x`, `two_y`, `three_x`, `three_y`, `four` với các kích thước ô (*block*) $4$, $6$, $8$ pixel và bước dịch $4$ pixel trên cửa sổ $24 \times 24$.
- **Kỹ thuật tính toán:** Chuẩn hóa điểm ảnh bằng cách chia cho $255{,}0$ (`float32`), tính ảnh tích phân (*Integral Image*, đệm 1 hàng và 1 cột `0`, `float64`) để tính tổng vùng chữ nhật trong thời gian hằng số, sau đó chuẩn hóa theo diện tích ô ($\text{factor} \times \text{block}^2$, với $\text{factor} = 1$ cho `two_x`/`two_y` và $\text{factor} = 2$ cho các mẫu còn lại).
- **Kiểm chứng độ khớp:** Module dùng chung [haar_extractor.py](haar_extractor.py) tái tạo lại toàn bộ $2.511$ mẫu train với sai số tuyệt đối lớn nhất là $0{,}00\times 10^0$ (vượt xa dung sai $10^{-5}$).

---

## 4. Cấu hình Mô hình và Giao thức Thực nghiệm

Các mô hình được đóng gói thành `Pipeline` độc lập (với `StandardScaler()` đặt bên trong Pipeline cho các thuật toán nhạy cảm thang đo để tránh rò rỉ dữ liệu) và cố định `random_state = 42` ở các bước ngẫu nhiên:

| Mô hình | Tiền xử lý | Cấu hình chính | Hàm điểm số |
| :--- | :--- | :--- | :--- |
| **AdaBoost** *(Trọng tâm)* | `passthrough` | `DecisionTreeClassifier(max_depth=1)`, `n_estimators=100`, `learning_rate=1.0` | `decision_function` |
| **SVM RBF** | `StandardScaler()` | `C=1.0`, `kernel='rbf'`, `gamma='scale'`, `probability=False` | `decision_function` |
| **Random Forest** | `passthrough` | `n_estimators=200`, `min_samples_leaf=2`, `n_jobs=1` | `predict_proba[:, 1]` |
| **Logistic Regression** | `StandardScaler()` | `C=1.0`, `solver='lbfgs'`, `max_iter=3000` | `decision_function` |
| **k-NN** | `StandardScaler()` | `n_neighbors=5`, `weights='uniform'`, `metric='minkowski'`, `p=2` | `predict_proba[:, 1]` |
| **Decision Tree** | `passthrough` | `max_depth=5`, `min_samples_leaf=2` | `predict_proba[:, 1]` |
| **DummyClassifier** *(Baseline)* | `passthrough` | `strategy='most_frequent'` (luôn dự đoán lớp đa số `1`) | `predict_proba[:, 1]` |

### Quy trình thực nghiệm
1. **Kiểm định chéo 5-fold (`StratifiedGroupKFold`) trên 2.511 mẫu train:** Lưu phân công fold tại `reports/tables/week6_cv_assignments.csv`, ghi nhận $35$ lượt chạy (7 mô hình $\times$ 5 fold) và $17.577$ dự đoán ngoài fold (OOF) cùng phân tích lỗi theo $821$ mẫu sai và $1.128$ nhóm.
2. **Huấn luyện trên toàn bộ tập Train & Khảo sát ngưỡng trên Validation ($540$ mẫu):**
   - Tìm ngưỡng `score >= threshold` thỏa mãn $\text{Recall} \ge 0{,}90$ và $\text{FPR} \le 0{,}05$, ưu tiên tối đa hóa $F_1$ (phân xử hòa: ưu tiên $\text{FPR}$ thấp hơn, rồi $\text{Recall}$ cao hơn).
   - Nếu không có ngưỡng nào đạt tiêu chí kép, giữ nguyên `Pipeline.predict()`.
   - Khóa toàn bộ ngưỡng, siêu tham số và thông tin môi trường trong [week6_final_protocol.json](reports/tables/week6_final_protocol.json).
3. **Bảo toàn độ chính xác số thực:** Điểm số trong `week6_validation_predictions.csv` và `week6_test_predictions.csv` được lưu với định dạng `float_format="%.17g"` (đối soát nạp lại bằng `float_precision="round_trip"` đạt **$0$ sai khác** trên toàn bộ $3.780$ dòng validation và $3.766$ dòng test). Quy tắc mặc định được ghi chuẩn là `Pipeline.predict()`.

---

## 5. Kết quả Thực nghiệm Chính

### 5.1. Kiểm định chéo 5-fold trên tập Train (`validation_cv`, Mean ± Std)

| Mô hình | Recall | FPR | F1-score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **SVM RBF** | $0{,}9632 \pm 0{,}0190$ | $0{,}0227 \pm 0{,}0095$ | **$0{,}9703 \pm 0{,}0093$** | $0{,}9934 \pm 0{,}0033$ |
| **AdaBoost** | $0{,}9702 \pm 0{,}0193$ | $0{,}0405 \pm 0{,}0108$ | $0{,}9656 \pm 0{,}0116$ | $0{,}9930 \pm 0{,}0037$ |
| **Random Forest** | $0{,}9514 \pm 0{,}0196$ | $0{,}0284 \pm 0{,}0158$ | $0{,}9616 \pm 0{,}0110$ | $0{,}9932 \pm 0{,}0035$ |
| **Logistic Regression** | $0{,}9530 \pm 0{,}0132$ | $0{,}0656 \pm 0{,}0218$ | $0{,}9453 \pm 0{,}0100$ | $0{,}9831 \pm 0{,}0070$ |
| **k-NN** | $0{,}9405 \pm 0{,}0169$ | $0{,}0738 \pm 0{,}0172$ | $0{,}9350 \pm 0{,}0099$ | $0{,}9711 \pm 0{,}0072$ |
| **Decision Tree** | $0{,}9146 \pm 0{,}0177$ | $0{,}1167 \pm 0{,}0275$ | $0{,}9024 \pm 0{,}0185$ | $0{,}9320 \pm 0{,}0246$ |
| **DummyClassifier** | $1{,}0000 \pm 0{,}0000$ | $1{,}0000 \pm 0{,}0000$ | $0{,}6742 \pm 0{,}0012$ | $0{,}5000 \pm 0{,}0000$ |

### 5.2. Đánh giá cuối cùng trên tập Test giữ riêng ($538$ mẫu)

| Mô hình | Chế độ quyết định | TN / FP / FN / TP | Accuracy | Recall | FPR | F1-score | ROC-AUC | Đạt mục tiêu |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM RBF** | `Pipeline.predict()` | 285 / 10 / 7 / 236 | $0{,}9684$ | $0{,}9712$ | $0{,}0339$ | **$0{,}9652$** | **$0{,}9942$** | Đạt |
| | `score >= -0.018045` | 285 / 10 / 7 / 236 | $0{,}9684$ | $0{,}9712$ | $0{,}0339$ | **$0{,}9652$** | **$0{,}9942$** | **Đạt** |
| **AdaBoost** | `Pipeline.predict()` | 282 / 13 / 9 / 234 | $0{,}9591$ | $0{,}9630$ | $0{,}0441$ | $0{,}9551$ | $0{,}9935$ | Đạt |
| | `score >= 0.026787` | 284 / 11 / 11 / 232 | $0{,}9591$ | $0{,}9547$ | $0{,}0373$ | $0{,}9547$ | $0{,}9935$ | **Đạt** |
| **Random Forest** | `Pipeline.predict()` | 287 / 8 / 13 / 230 | $0{,}9610$ | $0{,}9465$ | $0{,}0271$ | $0{,}9563$ | $0{,}9929$ | Đạt |
| | `score >= 0.506167` | 288 / 7 / 15 / 228 | $0{,}9591$ | $0{,}9383$ | $0{,}0237$ | $0{,}9540$ | $0{,}9929$ | **Đạt** |
| **Logistic Reg.** | `Pipeline.predict()` | 269 / 26 / 8 / 235 | $0{,}9368$ | $0{,}9671$ | $0{,}0881$ | $0{,}9325$ | $0{,}9815$ | Không đạt FPR |
| | `score >= 0.538914` | 274 / 21 / 12 / 231 | $0{,}9387$ | $0{,}9506$ | $0{,}0712$ | $0{,}9333$ | $0{,}9815$ | Không đạt FPR |
| **k-NN** | `Pipeline.predict()` | 272 / 23 / 13 / 230 | $0{,}9331$ | $0{,}9465$ | $0{,}0780$ | $0{,}9274$ | $0{,}9816$ | Không đạt FPR |
| **Decision Tree** | `Pipeline.predict()` | 240 / 55 / 29 / 214 | $0{,}8439$ | $0{,}8807$ | $0{,}1864$ | $0{,}8359$ | $0{,}8925$ | Không đạt |
| **DummyClassifier**| `Pipeline.predict()` | 0 / 295 / 0 / 243 | $0{,}4517$ | $1{,}0000$ | $1{,}0000$ | $0{,}6223$ | $0{,}5000$ | Mốc cơ sở |

### 5.3. Phân tích đánh đổi (*Trade-off*) của AdaBoost trên tập Test
- Khi áp dụng ngưỡng $\text{score} \ge 0{,}026787$, số báo động giả $\text{FP}$ giảm từ $13$ xuống $11$ ($\text{FPR}$ giảm từ $4{,}41\%$ xuống $3{,}73\%$).
- Đổi lại, số mẫu bỏ sót $\text{FN}$ tăng từ $9$ lên $11$ ($\text{Recall}$ giảm từ $96{,}30\%$ xuống $95{,}47\%$), và $F_1$ giảm nhẹ từ $0{,}9551$ xuống $0{,}9547$.
- Việc hạ $\text{FPR}$ là sự đánh đổi có chủ đích theo mục tiêu thiết kế, không phải sự cải thiện đồng thời mọi chỉ số.

---

## 6. Bộ Phát hiện Tự động và Ứng dụng Demo Streamlit

### 6.1. Bộ phát hiện trên ảnh đầy đủ ([face_detector.py](face_detector.py))
- **Tiền xử lý ảnh:** Hỗ trợ JPG/PNG, tự động hiệu chỉnh xoay EXIF (`ImageOps.exif_transpose`), chuyển sang ảnh mức xám và giới hạn cạnh lớn nhất ở `800` px (`max_image_dim=800`) khi xử lý để kiểm soát bộ nhớ.
- **Quét đa tỷ lệ:** Sử dụng *Image Pyramid* (mặc định `scale_factor=1.25` trong class, tùy chỉnh từ `1.1–1.4` trên giao diện) với cửa sổ trượt $24 \times 24$ (bước trượt `stride=6` px), trích xuất đặc trưng và suy luận theo từng batch $512$ cửa sổ.
- **Lọc khung và quy đổi tọa độ:** Triệt tiêu khung trùng lặp bằng hàm NMS thuần NumPy (`iou_threshold=0.3`), hỗ trợ điểm số âm/dương và quy đổi tọa độ $(x, y, w, h)$ chính xác về kích thước ảnh gốc.
- **Kiểm chứng thực tế (`test_demo_end_to_end.py`):** Trên ảnh mặt LFW (`Aaron_Eckhart_0001.jpg`, $250 \times 250$), bộ phát hiện AdaBoost quét $1.158$ cửa sổ trong $0{,}205\text{ s}$ và định vị khung chính tại $(x=77, y=77, w=103, h=103)$ với $\text{score}=+0{,}5808$; trên ảnh nền Caltech `image_0003.jpg` ($144 \times 144$), quét $239$ cửa sổ và trả về $0$ khung giả.

### 6.2. Giao diện Streamlit ([app_demo.py](app_demo.py))
Cung cấp 2 chế độ làm việc độc lập:
1. **Phát hiện khuôn mặt trong ảnh (mặc định):** Tải ảnh hoặc chọn ảnh mẫu dự án, tùy chỉnh tham số quét/NMS/ngưỡng thử nghiệm demo, vẽ khung bao và cho phép tải ảnh kết quả, kèm công cụ xem lại đúng ảnh con $24 \times 24$ và 258 đặc trưng Haar của từng khung phát hiện.
2. **Minh họa phân loại ảnh con (24×24):** Chọn mẫu từ $2.511$ ảnh train, thực hiện trích xuất đặc trưng trực tiếp, đối chiếu nhãn dự đoán và so sánh đồng thời 6 mô hình cùng `DummyClassifier`.

---

## 7. Danh mục Sản phẩm và Lệnh Chạy Nhanh

### Cấu trúc tệp chính
- **Nghiên cứu & Báo cáo Tuần 6:** [02_Week6_Training_FaceDetection.ipynb](02_Week6_Training_FaceDetection.ipynb), [README_WEEK6.md](README_WEEK6.md), [requirements-week6.txt](requirements-week6.txt), `reports/tables/*`, `reports/figures/*`.
- **Ứng dụng Demo & Suy luận:** [app_demo.py](app_demo.py), [face_detector.py](face_detector.py), [haar_extractor.py](haar_extractor.py), [train_export_demo.py](train_export_demo.py), [test_demo_end_to_end.py](test_demo_end_to_end.py), [README_DEMO.md](README_DEMO.md), [requirements-demo.txt](requirements-demo.txt), `models/demo/*.joblib`, [demo_metadata.json](models/demo/demo_metadata.json).
- **Gói bàn giao độc lập:**
  - `week6_review_bundle.zip` ($36$ tệp, $11{,}60\text{ MB}$): Gói toàn bộ kết quả đánh giá và mã nguồn Tuần 6.
  - `week6_demo_bundle.zip` ($20$ tệp, $8{,}41\text{ MB}$): Gói chạy ứng dụng Demo độc lập.

### Lệnh khởi chạy ứng dụng Demo
```bash
# Kiểm tra toàn diện đầu cuối
python test_demo_end_to_end.py

# Mở giao diện Demo Streamlit
python -m streamlit run app_demo.py
```

---

## 8. Giới hạn và Hướng Phát triển

- **Giới hạn hiện tại:** Bộ phát hiện trên ảnh đầy đủ quét qua một bộ phân loại đơn lẻ trên kim tự tháp ảnh chứ chưa huấn luyện cấu trúc phân tầng liên tiếp (*Attentional Cascade*) đầy đủ kiểu Viola–Jones; đồng thời chất lượng định vị trên ảnh đầy đủ cần được đánh giá thêm trên tập ảnh có nhãn hộp chuẩn (*Ground-truth Bounding Boxes*) do các chỉ số tuần 6 chỉ đánh giá ở mức phân loại cửa sổ $24 \times 24$.
- **Hướng mở rộng:** Bổ sung dữ liệu khuôn mặt đa góc nghiêng/che khuất, áp dụng *Hard Negative Mining* trên tập ảnh nền lớn để giảm thêm báo động giả khi quét toàn ảnh, và xây dựng tầng lọc *Cascade* để tăng tốc độ quét trên ảnh độ phân giải cao.
