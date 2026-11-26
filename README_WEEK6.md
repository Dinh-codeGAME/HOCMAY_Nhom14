# Hướng dẫn Chạy thực nghiệm Tuần 6 — Đồ án Học máy: Phát hiện khuôn mặt người trong ảnh

Tài liệu này cung cấp hướng dẫn cài đặt môi trường, tổ chức thư mục và thực thi notebook `02_Week6_Training_FaceDetection.ipynb` đảm bảo tính độc lập và tái lập kết quả thực nghiệm từ Tiêu chí 1 đến Đánh giá cuối cùng Tuần 6.

---

## 1. Môi trường thực thi và Cài đặt

- **Phiên bản Python sử dụng:** `Python 3.12.4` (hệ điều hành Windows 64-bit).
- **Cài đặt thư viện:**
  Để tránh việc cài đặt nhầm vào môi trường Python khác khi máy tính có nhiều phiên bản Python, khuyến nghị sử dụng trực tiếp module `pip` của chính trình thông dịch Python đang dùng:
  ```bash
  python -m pip install -r requirements-week6.txt
  ```
- **Đăng ký và chọn Jupyter Kernel:**
  Đảm bảo môi trường Python đã được cài đặt kernel `ipykernel`:
  ```bash
  python -m ipykernel install --user --name python3 --display-name "Python 3 (ipykernel)"
  ```
  Khi mở notebook trong VS Code, JupyterLab hoặc Google Colab, chọn đúng kernel `Python 3 (ipykernel)`.

---

## 2. Cấu trúc thư mục dự án

Toàn bộ mã nguồn sử dụng đường dẫn tương đối xuất phát từ thư mục gốc dự án (`FaceDetection_Week4`). Cấu trúc thư mục chuẩn như sau:

```
FaceDetection_Week4/
├── 02_Week6_Training_FaceDetection.ipynb   # Notebook chính tuần 6 (đầy đủ đầu ra thực thi từ TC1 đến Đánh giá cuối)
├── README_WEEK6.md                         # Hướng dẫn chạy và mô tả kỹ thuật
├── requirements-week6.txt                  # Danh mục thư viện cố định phiên bản
├── data/
│   └── processed/                          # Dữ liệu đặc trưng và ảnh đã xử lý
│       ├── haar_features.npy               # Ma trận 258 đặc trưng Haar (3589, 258)
│       ├── labels.npy                      # Vector nhãn nhị phân (3589,)
│       ├── images_24x24.npy                # Mảng 3.589 ảnh xám gốc 24x24 (uint8)
│       ├── metadata.csv                    # Siêu dữ liệu quản lý nhóm, mã băm sha256 và nguồn ảnh
│       ├── split_indices.npz               # Chỉ số phân chia train/val/test
│       ├── haar_dataset_splits.npz         # Dữ liệu đặc trưng chia sẵn
│       └── data_dictionary_haar.csv        # Từ điển mô tả 258 đặc trưng Haar
├── reports/
│   ├── figures/                            # Biểu đồ đánh giá trực quan (độ phân giải ≥ 300 dpi)
│   │   ├── week6_oof_confusion_matrices.png# Ma trận nhầm lẫn OOF gộp 7 mô hình (4x2 dọc)
│   │   ├── week6_cv_mean_roc.png           # Đường cong ROC trung bình 5-fold (2 khung dọc)
│   │   ├── week6_errors_adaboost.png       # Lưới 4x4 mẫu lỗi tiêu biểu (FP & FN) của AdaBoost
│   │   ├── week6_errors_svm_rbf.png        # Lưới 4x4 mẫu lỗi tiêu biểu (FP & FN) của SVM RBF
│   │   ├── week6_test_confusion_matrices.png# Ma trận nhầm lẫn trên tập Test (4x2 dọc)
│   │   └── week6_test_roc.png              # Đường cong ROC trên tập Test (2 khung dọc)
│   └── tables/                             # Thư mục lưu bảng kết quả và minh chứng
│       ├── week6_cv_assignments.csv        # Phân công fold kiểm định chéo đã lưu
│       ├── week6_cv_audit.csv              # Thẩm định phân chia 5 fold
│       ├── week6_baseline_folds.csv        # Kết quả baseline chi tiết từng fold
│       ├── week6_baseline_summary.csv      # Bảng tổng hợp hiệu năng baseline
│       ├── week6_pipeline_config.csv       # Cấu hình khởi tạo 6 Pipeline
│       ├── week6_pipeline_checks.csv       # Kết quả kiểm tra 6 Pipeline trên Fold 1
│       ├── week6_reproducibility_check.csv # Bảng đối soát chạy độc lập
│       ├── week6_models_cv_folds.csv       # Kết quả kiểm định chéo 35 fold (7 mô hình x 5 fold)
│       ├── week6_models_cv_summary.csv     # Bảng tổng hợp hiệu năng dạng dài (mean ± std)
│       ├── week6_models_oof_predictions.csv# 17.577 dự đoán ngoài fold (OOF) của 7 mô hình
│       ├── week6_oof_confusion_matrices.csv# Bảng số liệu ma trận nhầm lẫn OOF gộp
│       ├── week6_cv_roc_points.csv         # Bảng 3.500 điểm nội suy đường cong ROC trung bình
│       ├── week6_error_samples.csv         # 821 mẫu dự đoán sai của 6 mô hình học máy
│       ├── week6_error_groups.csv          # Bảng tổng hợp lỗi theo 1.128 nhóm độc lập
│       ├── week6_error_examples.csv        # 32 mẫu ảnh lỗi minh họa kèm thông số chi tiết
│       ├── week6_validation_predictions.csv# 3.780 dự đoán trên tập Validation (đầy đủ độ chính xác)
│       ├── week6_threshold_selection.csv   # Kết quả khảo sát và lựa chọn ngưỡng Validation (default_rule: Pipeline.predict())
│       ├── week6_final_protocol.json       # Giao thức khóa ngưỡng, siêu tham số và phiên bản thư viện
│       ├── week6_test_predictions.csv      # 3.766 dự đoán trên tập Test (đầy đủ độ chính xác)
│       └── week6_test_results.csv          # Bảng kết quả đánh giá cuối cùng trên tập Test
└── week6_review_bundle.zip                 # Gói lưu trữ bàn giao độc lập đầy đủ
```

---

## 3. Quy ước Dữ liệu, Nhãn và Không gian đặc trưng

1. **Đơn vị mẫu và không gian đặc trưng:**
   - Dữ liệu gồm **$3.589$ mẫu** ảnh con kích thước $24 	imes 24$ điểm ảnh (`images_24x24.npy`, kiểu `uint8`).
   - Đối chiếu tính toàn vẹn từng ảnh với cột `sha256` trong `metadata.csv` bằng hàm băm SHA-256.
   - Các mô hình học máy sử dụng **$258$ đặc trưng Haar-like** (kiểu `float32`) làm đầu vào. Con số $24 	imes 24 = 576$ là số lượng điểm ảnh (pixels) của cửa sổ ảnh thô, **không phải** số đặc trưng đầu vào của các mô hình hiện tại.
2. **Quy ước nhãn:**
   - Nhãn `1`: Khuôn mặt (*face*).
   - Nhãn `0`: Phi khuôn mặt / Vùng nền (*non-face*).
3. **Quy ước nhóm định danh (chống rò rỉ dữ liệu):**
   - Lớp Face: Chia nhóm theo định danh từng cá nhân (`person_id`).
   - Lớp Non-face: Chia nhóm theo ảnh nền nguồn gốc (`source_image_id`).
4. **Phân chia tập dữ liệu:**
   - **Tập Train:** $2.511$ mẫu ($1.277$ face, $1.234$ non-face).
   - **Tập Validation giữ riêng:** $540$ mẫu ($280$ face, $260$ non-face) — dùng để khảo sát ngưỡng và kiểm định điều kiện thiết kế.
   - **Tập Test giữ riêng:** $538$ mẫu ($243$ face, $295$ non-face) — dùng để đánh giá độc lập sau cùng theo giao thức đã khóa.

---

## 4. Cách Thực thi Notebook

1. Mở IDE (VS Code / Jupyter Notebook) với thư mục làm việc (*working directory*) là thư mục gốc dự án (`FaceDetection_Week4`).
2. Mở tệp `02_Week6_Training_FaceDetection.ipynb`.
3. Nhấp chọn **Restart Kernel and Run All Cells** (Khởi động lại Kernel và chạy toàn bộ 51 cells).
4. **Cơ chế tái sử dụng fold:**
   - Notebook đọc và kiểm tra tệp `reports/tables/week6_cv_assignments.csv` đã lưu, tuyệt đối **không ghi đè**.
   - Cả mô hình cơ sở và sáu thuật toán học máy đều sử dụng chung danh sách `cv_splits` trích xuất từ tệp phân công này.
5. **Cảnh báo `FutureWarning` của `SVC`:**
   - Khi chạy mô hình `SVM RBF`, Scikit-learn phát ra cảnh báo `FutureWarning: The 'probability' parameter was deprecated in 1.9...`. Đây là thông báo về thay đổi API tương lai của thư viện, hoàn toàn không phải lỗi huấn luyện. Đề tài giữ nguyên cấu hình `probability=False` và sử dụng `decision_function` để lấy điểm khoảng cách tới siêu phẳng lề, không bật hiệu chỉnh xác suất và không tắt cảnh báo để duy trì tính minh bạch khoa học.

---

## 5. Phạm vi thực hiện của Tuần 6

- **Các nội dung đã hoàn thành:**
  1. **Tiêu chí 1:** Lựa chọn và phân tích cơ sở lý thuyết của 6 thuật toán học máy và mô hình cơ sở tham chiếu.
  2. **Tiêu chí 2:** Xây dựng mô hình cơ sở (`DummyClassifier` - Most Frequent) và đánh giá trên 5 fold phân tầng theo nhóm.
  3. **Tiêu chí 3:** Đóng gói tiền xử lý (`StandardScaler` hoặc `passthrough`) và bộ phân loại vào `Pipeline` độc lập, cố định tính ngẫu nhiên, kiểm tra kỹ thuật trên Fold 1.
  4. **Tiêu chí 4:** Đảm bảo mã nguồn Python có cấu trúc chuẩn, chạy độc lập không phụ thuộc đường dẫn tuyệt đối máy cá nhân, đóng gói `week6_review_bundle.zip`.
  5. **Tiêu chí 5:** Huấn luyện và đánh giá kiểm định chéo 5-fold toàn diện trên **$2.511$ mẫu train** cho toàn bộ 7 mô hình, tính toán trung bình số học và độ lệch chuẩn ($	ext{ddof}=0$), xuất $17.577$ dự đoán ngoài fold (OOF).
  6. **Tiêu chí 6:** Bổ sung công cụ đánh giá trực quan từ kết quả kiểm định chéo đã lưu:
     - Thẩm định 4 điều kiện toàn vẹn và nhất quán của dữ liệu OOF đối soát với siêu dữ liệu và kết quả CV.
     - Xuất ma trận nhầm lẫn gộp ($TN, FP, FN, TP$, độ nhạy, độ đặc hiệu) dạng hình ảnh `week6_oof_confusion_matrices.png` ($\ge 300$ dpi, bố cục 4x2 dọc) và bảng CSV `week6_oof_confusion_matrices.csv`.
     - Vẽ đường cong ROC trung bình 5-fold kèm dải độ lệch chuẩn $\pm 1\,	ext{std}$ gồm 2 khung nhìn dọc (`week6_cv_mean_roc.png`, $\ge 300$ dpi) và bảng 3.500 điểm nội suy tại `week6_cv_roc_points.csv`.
  7. **Tiêu chí 7:** Phân tích mẫu/nhóm dự đoán sai và đối chiếu kết quả huấn luyện với kiểm định:
     - Thẩm định 100% mã băm SHA-256 của $3.589$ ảnh gốc `images_24x24.npy`.
     - Trích xuất $821$ mẫu dự đoán sai của 6 mô hình học máy vào `week6_error_samples.csv`, khớp hoàn toàn với ma trận nhầm lẫn.
     - Tổng hợp lỗi theo $1.128$ nhóm độc lập vào `week6_error_groups.csv` (mẫu số lấy từ tổng số mẫu OOF của nhóm).
     - Trực quan hóa $32$ mẫu ảnh lỗi tiêu biểu dạng lưới $4 	imes 4$ cho AdaBoost và SVM RBF (`week6_errors_adaboost.png`, `week6_errors_svm_rbf.png`, `week6_error_examples.csv`), nhận xét đặc trưng thị giác khách quan.
     - Đối chiếu F1 huấn luyện và kiểm định CV ($\Delta F_1$), đánh giá đa chiều không gán ngưỡng tùy tiện.
  8. **Đánh giá cuối cùng Tuần 6:**
     - Huấn luyện lại trên toàn bộ $2.511$ mẫu train. Khảo sát ngưỡng quyết định trên tập Validation giữ riêng ($540$ mẫu) với điều kiện tiên quyết $	ext{Recall} \ge 0{,}90$ và $	ext{FPR} \le 0{,}05$.
     - Khóa giao thức thực nghiệm trước khi đánh giá Test vào `week6_final_protocol.json`, ghi nhận đầy đủ siêu tham số, random_state và phiên bản thư viện.
     - Tập test dùng để đánh giá cuối theo cấu hình và ngưỡng đã xác định từ train/validation, xuất `week6_test_predictions.csv` (lưu đầy đủ độ chính xác `float_format="%.17g"`), `week6_test_results.csv`, `week6_test_confusion_matrices.png` và `week6_test_roc.png`.
     - **Phân tích đánh đổi (*trade-off*) kỹ thuật của AdaBoost trên tập Test:**
       + Khi áp dụng ngưỡng tối ưu $	ext{score} \ge 0{,}026787$, số lượng báo động giả $	ext{FP}$ giảm từ $13$ xuống $11$ mẫu (tỷ lệ $	ext{FPR}$ giảm từ $4{,}41\%$ xuống $3{,}73\%$), thỏa mãn mục tiêu khống chế báo động giả $	ext{FPR} \le 5\%$.
       + Đổi lại, số lượng khuôn mặt bị bỏ sót $	ext{FN}$ tăng từ $9$ lên $11$ mẫu (độ nhạy $	ext{Recall}$ giảm từ $96{,}30\%$ xuống $95{,}47\%$).
       + Điểm tổng hòa $F_1$ giảm nhẹ từ $0{,}9551$ xuống $0{,}9547$.
       + Phân tích này khẳng định việc hạ $	ext{FPR}$ là sự đánh đổi có chủ đích theo tiêu chí thiết kế, không phải là sự cải thiện đồng thời mọi chỉ số đo lường.
     - **Quy tắc quyết định mặc định:** Được xác định chính xác là `Pipeline.predict()`, không mô tả chung là `predict_proba >= 0.5` do hiện tượng hòa xác suất tại ngưỡng $0{,}5$ (Random Forest có 1 mẫu validation và Decision Tree có 3 mẫu validation / 4 mẫu test có điểm xác suất bằng đúng $0{,}5$ được `predict()` gán về lớp 0 theo cơ chế argmax).
     - **Bảo toàn độ chính xác số thực:** Các bảng dự đoán validation và test được lưu với định dạng `float_format="%.17g"`. Khi nạp lại bằng `pd.read_csv(..., float_precision="round_trip")`, việc tái tạo nhãn dự đoán từ điểm số và ngưỡng đã khóa đạt tỷ lệ khớp $100\%$ ($0$ sai khác).
     - Làm rõ giới hạn của bài toán: Phân loại ô ảnh con $24 	imes 24$ điểm ảnh chưa phải là hệ thống phát hiện khuôn mặt toàn cảnh (Full Face Detector).
