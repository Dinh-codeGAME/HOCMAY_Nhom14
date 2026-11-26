# BÁO CÁO TIỀN KIỂM TOÁN TUẦN 8 — PHÁT HIỆN KHUÔN MẶT NGƯỜI TRONG ẢNH
**Dự án:** `FaceDetection_Week4`  
**Giai đoạn:** Bài kiểm tra 3 – Tuần 8 (Tối ưu hóa, Cải tiến và So sánh Mô hình — Giai đoạn Tiền kiểm toán)  
**Trạng thái:** Chỉ đọc, kiểm chứng bằng chứng và lập kế hoạch. Chưa chạy tối ưu siêu tham số, chưa chỉnh sửa kết quả Tuần 6 và chưa xuất LaTeX.

---

## 1. Kiểm kê Trạng thái Thực tế Dự án

### 1.1. Danh sách tệp tìm thấy
- **Notebook & Tài liệu:**
  - `02_Week6_Training_FaceDetection.ipynb` (51 cells, $1.562.668$ bytes)
  - `../01_Week4_EDA_FaceDetection.ipynb` (tại thư mục cha `C:\Users\mrddd\Documents`)
  - `README_WEEK6.md`, `README_DEMO.md`, `PROJECT_SUMMARY.md`
  - `requirements-week6.txt`, `requirements-demo.txt`
- **Mã nguồn trích xuất, phát hiện & demo:**
  - `haar_extractor.py`, `train_export_demo.py`, `face_detector.py`, `app_demo.py`, `test_demo_end_to_end.py`, `create_demo_bundle.py`
- **Dữ liệu đã xử lý (`data/processed/`, 8 tệp):**
  - `images_24x24.npy`, `haar_features.npy`, `labels.npy`, `metadata.csv`, `split_indices.npz`, `haar_dataset_splits.npz`, `data_dictionary_haar.csv`, `patch_metadata_before_split.csv`
  - Kèm 11 tệp kiểm toán nguồn & EDA trong `data/metadata/` và ảnh thô LFW/Caltech 101 trong `data/raw/`.
- **Bảng kết quả & Giao thức (`reports/tables/`, 31 tệp):**
  - Nhóm Tuần 4/EDA & kiểm toán nguồn (11 tệp): `feature_matrix_summary.csv`, `quality_source_*`, `split_leakage_audit.csv`, `split_summary.csv`.
  - Nhóm Tuần 6 (20 tệp): `week6_cv_assignments.csv`, `week6_cv_audit.csv`, `week6_baseline_folds.csv`, `week6_baseline_summary.csv`, `week6_pipeline_config.csv`, `week6_pipeline_checks.csv`, `week6_reproducibility_check.csv`, `week6_models_cv_folds.csv`, `week6_models_cv_summary.csv`, `week6_models_oof_predictions.csv`, `week6_oof_confusion_matrices.csv`, `week6_cv_roc_points.csv`, `week6_error_samples.csv`, `week6_error_groups.csv`, `week6_error_examples.csv`, `week6_validation_predictions.csv`, `week6_threshold_selection.csv`, `week6_final_protocol.json`, `week6_test_predictions.csv`, `week6_test_results.csv`.
- **Biểu đồ (`reports/figures/`, 12 tệp PNG):** 6 hình Tuần 4 EDA và 6 hình Tuần 6 (`week6_oof_confusion_matrices.png`, `week6_cv_mean_roc.png`, `week6_errors_adaboost.png`, `week6_errors_svm_rbf.png`, `week6_test_confusion_matrices.png`, `week6_test_roc.png`).
- **Mô hình Demo đã xuất (`models/demo/`, 8 tệp):** 7 tệp `.joblib` (`adaboost.joblib`, `svm_rbf.joblib`, `random_forest.joblib`, `logistic_regression.joblib`, `knn.joblib`, `decision_tree.joblib`, `dummy_classifier.joblib`) và `demo_metadata.json`.
- **Gói bàn giao hiện có:** `week6_review_bundle.zip` ($12.168.490$ bytes), `week6_demo_bundle.zip` ($8.822.257$ bytes).

### 1.2. Tệp và công cụ còn thiếu
1. **Không có mã nguồn LaTeX (`.tex`):** Tìm kiếm đệ quy trong `FaceDetection_Week4` và thư mục cha `C:\Users\mrddd\Documents` trả về `0` tệp `.tex`.
2. **Không có trình biên dịch LaTeX cục bộ:** Cả `pdflatex`, `xelatex`, `lualatex`, `latexmk`, `tectonic`, `bibtex`, `biber`, `pandoc` đều không có trên máy.
3. **Không có nhãn hộp bao chuẩn (*Ground-truth Bounding Boxes*) cho ảnh đầy đủ:** Dữ liệu dự án chỉ gán nhãn nhị phân ở mức ô ảnh $24 \times 24$ (`labels.npy`), chưa có tọa độ khuôn mặt chuẩn trên ảnh LFW/ảnh ngoài để đo `mAP` hoặc `Recall` định vị.
4. **Một số thư viện tối ưu/GBDT chưa cài:** `optuna`, `xgboost`, `lightgbm`, `skimage` hiện có giá trị `null` (chỉ có `scikit-learn==1.9.1`, `scipy==1.18.1`).

### 1.3. Mâu thuẫn và bất nhất phát hiện khi đối chiếu tệp thực tế
1. **Mâu thuẫn `default_rule` trong `models/demo/demo_metadata.json`:**
   - Mã nguồn `train_export_demo.py` đã được sửa `default_rule` thành `"Pipeline.predict()"`, khớp với `week6_final_protocol.json` và `week6_threshold_selection.csv`.
   - Tuy nhiên, tệp `models/demo/demo_metadata.json` đang lưu trên đĩa (thời điểm xuất `2026-11-13T04:48:48+00:00`) chưa được chạy xuất lại sau lần sửa đó, nên vẫn ghi `"decision_function >= 0"`, `"predict_proba >= 0.5"`, `"predict() [most_frequent]"`.
2. **Tham số `"algorithm": "SAMME.R"` không tồn tại trong `AdaBoostClassifier` thực tế (`scikit-learn 1.9.1`):**
   - `reports/tables/week6_final_protocol.json` ghi `"algorithm": "SAMME.R"` trong phần `hyperparameters` của AdaBoost.
   - Kiểm tra trực tiếp `clf.get_params()` từ `models/demo/adaboost.joblib` cho thấy `AdaBoostClassifier` của `scikit-learn 1.9.1` chỉ có 4 tham số (`estimator`, `learning_rate`, `n_estimators`, `random_state`) và không còn tham số `algorithm` (`SAMME.R` đã bị loại bỏ từ scikit-learn 1.6+).
3. **Khác biệt nhẹ giá trị mặc định `scale_factor` giữa mã và giao diện:**
   - `FullImageFaceDetector.__init__` trong `face_detector.py` đặt mặc định `scale_factor=1.25`, còn slider trên `app_demo.py` đặt mặc định `1.2`.
4. **Khác biệt khai báo phụ thuộc:**
   - `requirements-week6.txt` khóa phiên bản `==` cho 9 gói nhưng không liệt kê `joblib` (dù được dùng trong demo/xuất mô hình), còn `requirements-demo.txt` dùng ràng buộc `>=`.

---

## 2. Xác minh Bảo toàn Kết quả Tuần 6 (Đọc từ tệp thực tế)

Toàn bộ mã băm SHA-256 của dữ liệu, phân chia tập, bảng kết quả và mã nguồn đã được ghi nhận trong [week6_manifest_sha256.json](week6_manifest_sha256.json).

- **Dữ liệu gốc đã xử lý:**
  - `images_24x24.npy`: `(3589, 24, 24)` `uint8`.
  - `haar_features.npy`: `(3589, 258)` `float32`.
  - `labels.npy`: `(3589,)` `int64` (`1800` mẫu `1`, `1789` mẫu `0`).
  - Cột nhóm: `group` (`person_id` cho Face, `source_image_id` cho Non-face).
- **Phân chia Train / Validation / Test (`split_indices.npz`):**
  - `train`: **$2.511$ mẫu** ($1.277$ face, $1.234$ non-face), $1.128$ nhóm duy nhất.
  - `val`: **$540$ mẫu** ($280$ face, $260$ non-face), $243$ nhóm duy nhất.
  - `test`: **$538$ mẫu** ($243$ face, $295$ non-face), $243$ nhóm duy nhất.
  - Giao thoa chỉ số mẫu giữa 3 tập = `[0, 0, 0]`; giao thoa nhóm = `[0, 0, 0]`.
- **Phân công 5 fold (`week6_cv_assignments.csv`):**
  - SHA-256: `b87eaaf62171abca86d80378965be12268d0449d2ac038c467f47b3e42992ab3`.
  - Mỗi fold kiểm định có $501\text{–}503$ mẫu, $0$ giao thoa mẫu, $0$ giao thoa nhóm và đủ 2 lớp.
- **Cấu hình 6 mô hình và baseline:** Đã xác minh trực tiếp qua `clf.get_params()` từ 7 tệp `.joblib`, khớp với `week6_pipeline_config.csv` (trừ ghi chú dư `"algorithm": "SAMME.R"` trong protocol JSON).
- **Trạng thái sử dụng Validation và Test trong lịch sử thực nghiệm:**
  - `540` mẫu Validation đã được dùng ở Tuần 6 để khảo sát ngưỡng quyết định (`week6_threshold_selection.csv`, `week6_validation_predictions.csv`).
  - `538` mẫu Test **đã được thực thi và đã có kết quả đánh giá** (`week6_test_predictions.csv`, `week6_test_results.csv`, `week6_test_confusion_matrices.png`). Do đó, trong báo cáo Tuần 8 **không được mô tả tập Test là chưa từng được xem**; mọi bước chọn đặc trưng, điều chỉnh độ phức tạp và tìm kiếm siêu tham số Tuần 8 phải thực hiện hoàn toàn bên trong $2.511$ mẫu Train (qua Group CV / Nested Group CV), giữ nguyên tuyệt đối ranh giới không rò rỉ sang Validation/Test.

---

## 3. Bằng chứng Hạn chế cho Tiêu chí 1 Tuần 8

### 3.1. Phân tách phạm vi đánh giá
- **(a) Hiệu năng phân loại cửa sổ ảnh $24 \times 24$:** Có đầy đủ số liệu định lượng qua kiểm định chéo 5-fold ($2.511$ mẫu Train), phân tích OOF/nhóm lỗi ($1.128$ nhóm), Validation ($540$ mẫu) và Test ($538$ mẫu) — chi tiết trong [limitations_evidence.csv](limitations_evidence.csv).
- **(b) Hiệu năng định vị trên ảnh đầy đủ (`face_detector.py`):** Hiện chỉ có bằng chứng định tính từ thử nghiệm chạy [test_demo_end_to_end.py](file:///c:/Users/mrddd/Documents/FaceDetection_Week4/test_demo_end_to_end.py) trên 1 ảnh LFW (`Aaron_Eckhart_0001.jpg`, $250 \times 250$) và 1 ảnh nền Caltech (`image_0003.jpg`, $144 \times 144$). **Chưa có nhãn hộp bao chuẩn nên không suy ra Accuracy, Recall hay mAP định vị.**

### 3.2. Bảng bằng chứng định lượng phân loại cửa sổ $24 \times 24$

| Mô hình | F1 Train CV | F1 Val CV (Mean ± Std) | Chênh lệch ($\Delta F_1$) | Recall CV | FPR CV | OOF FP / FN | Fit Time CV (s) | Pred Time CV (s) | Test F1 / FPR (Chế độ chọn) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM RBF** | $0{,}9884$ | $0{,}9703 \pm 0{,}0093$ | $+0{,}0181$ | $0{,}9632$ | $0{,}0227$ | 28 / 47 | $0{,}1554$ | $0{,}0750$ | $0{,}9652$ / $0{,}0339$ |
| **AdaBoost** | $0{,}9936$ | $0{,}9656 \pm 0{,}0116$ | $+0{,}0280$ | $0{,}9702$ | $0{,}0405$ | 50 / 38 | $8{,}8947$ | $0{,}0341$ | $0{,}9547$ / $0{,}0373$ |
| **Random Forest** | $0{,}9997$ | $0{,}9616 \pm 0{,}0110$ | $+0{,}0381$ | $0{,}9514$ | $0{,}0284$ | 35 / 62 | $5{,}1141$ | $0{,}0317$ | $0{,}9540$ / $0{,}0237$ |
| **Logistic Reg.** | $0{,}9598$ | $0{,}9453 \pm 0{,}0100$ | $+0{,}0145$ | $0{,}9530$ | $0{,}0656$ | 81 / 60 | $0{,}0957$ | $0{,}0014$ | $0{,}9333$ / $0{,}0712$ |
| **k-NN** | $0{,}9561$ | $0{,}9350 \pm 0{,}0099$ | $+0{,}0211$ | $0{,}9405$ | $0{,}0738$ | 91 / 76 | $0{,}0101$ | $0{,}0182$ | $0{,}9274$ / $0{,}0780$ |
| **Decision Tree** | $0{,}9629$ | $0{,}9024 \pm 0{,}0185$ | $+0{,}0606$ | $0{,}9146$ | $0{,}1167$ | 144 / 109 | $0{,}3749$ | $0{,}0008$ | $0{,}8359$ / $0{,}1864$ |
| **DummyClassifier**| $0{,}6742$ | $0{,}6742 \pm 0{,}0012$ | $-0{,}0000$ | $1{,}0000$ | $1{,}0000$ | 1234 / 0 | $0{,}0009$ | $0{,}0002$ | $0{,}6223$ / $1{,}0000$ |

### 3.3. Phân tách Hạn chế có bằng chứng, Giả thuyết và Thông tin chưa đo

1. **Hạn chế đã có bằng chứng trực tiếp:**
   - **Quá khớp (*Overfitting*) rõ rệt ở mô hình cây:** `Decision Tree` có $\Delta F_1 = +0{,}0606$ (`0.9629` train vs `0.9024` CV, độ lệch chuẩn cao nhất `0.0185`), tụt mạnh xuống $F_1 = 0{,}8359$, $\text{Recall} = 0{,}8807 < 0{,}90$ và $\text{FPR} = 0{,}1864$ trên Test. `Random Forest` đạt gần bão hòa tuyệt đối trên train (`0.9997`) nhưng giảm $0{,}0381$ xuống `0.9616` trên CV (`62 FN`). `AdaBoost` cũng có khoảng cách $\Delta F_1 = +0{,}0280$ (`0.9936` vs `0.9656`).
   - **Không kiểm soát được $\text{FPR} \le 5\%$ ở 3 thuật toán:** `Logistic Regression` ($\text{CV FPR}=6{,}56\%$, Test mặc định $8{,}81\%$, ngay cả sau ngưỡng Val vẫn đạt $\text{Test FPR}=7{,}12\%$ với $21\text{ FP}$), `k-NN` ($\text{CV FPR}=7{,}38\%$, $0$ ngưỡng đạt trên Val, $\text{Test FPR}=7{,}80\%$ với $23\text{ FP}$) và `Decision Tree` ($\text{CV FPR}=11{,}67\%$, $\text{Test FPR}=18{,}64\%$ với $55\text{ FP}$).
   - **Chi phí thời gian huấn luyện và suy luận:** `AdaBoost` có thời gian huấn luyện CV lâu nhất ($8{,}8947\text{ s}/$fold), kế đến là `Random Forest` ($5{,}1141\text{ s}/$fold). Ở chiều ngược lại, `SVM RBF` fit rất nhanh ($0{,}1554\text{ s}$) nhưng có thời gian suy luận trên `~502` mẫu kiểm định chậm nhất ($0{,}0750\text{ s}/$fold, gấp $>2\times$ AdaBoost và $>54\times$ Logistic Regression).
   - **Lỗi tập trung theo nhóm khó (bằng chứng từ `week6_error_groups.csv`):** Một số nhóm cá nhân LFW bị toàn bộ hoặc hầu hết các mô hình bỏ sót (ví dụ `lfw_person:2750` sai $2/2$ FN trên cả 6 mô hình; `lfw_person:2058` sai $3/3$ FN trên Random Forest & k-NN và $2/3$ FN trên AdaBoost & SVM RBF) cùng một số ảnh nền Caltech cấu trúc phức tạp gây cụm báo động giả (như `image_0248.jpg` sai $4/4$ FP trên Decision Tree, `image_0310.jpg` sai $3/4$ FP trên k-NN, `image_0403.jpg` sai $2/4$ FP trên AdaBoost).
   - **Đánh đổi ngưỡng của AdaBoost trên Test:** Áp dụng ngưỡng $\ge 0{,}026787$ giảm FP từ $13 \to 11$ nhưng tăng FN từ $9 \to 11$ và giảm nhẹ $F_1$ từ $0{,}9551 \to 0{,}9547$.
   - **Quan sát định tính khi quét ảnh đầy đủ:** Khi chạy `FullImageFaceDetector` (AdaBoost, `score_threshold=0.25`, `stride=6`, `scale_factor=1.2`, NMS `IoU=0.3`) trên `Aaron_Eckhart_0001.jpg` ($250 \times 250$), mô hình quét $1.158$ cửa sổ trong $0{,}205\text{ s}$ và giữ lại 5 khung sau NMS: khung #1 khớp vùng mặt (`x=77, y=77, w=103, h=103, score=+0.5808`), nhưng 4 khung còn lại (`score=+0.2630` đến `+0.3254`) khoanh nhầm vào vùng cổ áo/nền phía dưới. Trên ảnh nền `image_0003.jpg`, quét $239$ cửa sổ cho $0$ khung.

2. **Giả thuyết cần thí nghiệm Tuần 8 để kiểm chứng:**
   - Trong 258 đặc trưng Haar (cửa sổ trượt $4$ px với các block $4, 6, 8$ chồng lấn nhau) tồn tại đa cộng tuyến và các đặc trưng ít thông tin, làm suy giảm mô hình dựa trên khoảng cách (`k-NN`), mô hình tuyến tính (`Logistic Regression`), gây nhiễu cho cây quyết định và làm tăng chi phí fit của `AdaBoost`/`Random Forest`.
   - Việc chọn lọc đặc trưng bên trong Pipeline và điều chỉnh tham số chính quy hóa/độ phức tạp (`max_depth`, `min_samples_leaf`, `ccp_alpha`, `C`, `gamma`, `learning_rate`, `n_neighbors`, `weights`) có thể giảm khoảng cách train–CV hoặc hạ FPR trên nhóm mô hình hiện có.

3. **Thông tin chưa được đo:**
   - Định lượng mức độ dư thừa/tương quan giữa 258 đặc trưng Haar.
   - Độ trễ quét ảnh đầy đủ có kiểm soát theo từng mô hình (nhất là chi phí suy luận của `SVM RBF` và `k-NN` khi quét $>1.000$ cửa sổ).
   - Số lượng Support Vectors của `SVM RBF` trên từng fold.
   - Mọi chỉ số định vị định lượng (`mAP`, `IoU Recall`, `FPPI`) trên ảnh đầy đủ do chưa có bộ nhãn hộp bao chuẩn.
