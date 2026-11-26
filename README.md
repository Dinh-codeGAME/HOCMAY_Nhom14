# Đồ án Học máy — Nhóm 14 (CNTT K22H, ICTU)
## Đề tài: Ứng dụng Học máy Phát hiện Khuôn mặt Người trong Ảnh

Repository lưu trữ mã nguồn, kết quả thực nghiệm (Tuần 4 – Tuần 8), ứng dụng Demo Streamlit và mã nguồn báo cáo LaTeX (`bao_cao/main.tex`) cho đề tài phát hiện khuôn mặt người trong ảnh sử dụng đặc trưng **Haar-like** (258 đặc trưng) và **AdaBoost** cùng 5 thuật toán đối sánh (**SVM RBF**, **Random Forest**, **Logistic Regression**, **k-NN**, **Decision Tree**) và mốc cơ sở (`DummyClassifier`).

---

## 1. Cấu trúc Repository

```text
FaceDetection_Week4/
├── 01_Week4_EDA_FaceDetection.ipynb     # Notebook Tuần 4: EDA, xử lý ảnh & trích xuất 258 đặc trưng Haar
├── 02_Week6_Training_FaceDetection.ipynb# Notebook Tuần 6: Baseline, Pipeline 5-fold CV, phân tích lỗi & Test
├── app_demo.py                          # Giao diện Demo Streamlit (Phát hiện trên ảnh đầy đủ + Ảnh con 24x24)
├── face_detector.py                     # Bộ phát hiện đa tỷ lệ (Image Pyramid + Sliding Window + NMS NumPy)
├── haar_extractor.py                    # Module trích xuất 258 đặc trưng Haar-like và suy luận chuẩn hóa
├── train_export_demo.py                 # Kịch bản huấn luyện trên 2.511 mẫu train và xuất mô hình .joblib
├── test_demo_end_to_end.py              # Kiểm thử đầu cuối cho module trích xuất và bộ phát hiện
├── requirements-week6.txt               # Thư viện cho thực nghiệm Tuần 6 & Tuần 8
├── requirements-demo.txt                # Thư viện cho ứng dụng Demo Streamlit
├── README_WEEK6.md                      # Hướng dẫn chi tiết thực nghiệm Tuần 6
├── README_DEMO.md                       # Hướng dẫn cài đặt và vận hành Demo
├── PROJECT_SUMMARY.md                   # Tóm tắt tổng quan dự án
├── bao_cao/
│   ├── main.tex                         # Báo cáo LaTeX (biên dịch bằng XeLaTeX)
│   └── images/                          # Hình minh họa báo cáo (gồm các biểu đồ Tuần 4, 6, 8)
├── models/demo/
│   └── demo_metadata.json               # Metadata các mô hình demo (tệp .joblib tạo lại khi chạy script)
├── reports/
│   ├── tables/                          # Bảng kết quả, kiểm toán rò rỉ và week6_cv_assignments.csv
│   └── figures/                         # Biểu đồ kết quả 300 dpi
└── week8/
    ├── run_tc3_adaboost_selectkbest.py  # Tiêu chí 3: AdaBoost 258 vs SelectKBest(f_classif, k=128)
    ├── run_tc4_decision_tree_depth.py   # Tiêu chí 4: Decision Tree max_depth=5 vs max_depth=3
    ├── run_tc5_dt_nested_gridsearch.py  # Tiêu chí 5: Nested Group CV (5x3) GridSearchCV cho Decision Tree
    ├── run_tc7_final_adaboost.py        # Tiêu chí 7: Đánh giá lại AdaBoost 258 trên 538 mẫu Test
    └── *.csv, *.json, *.md              # Minh chứng tiền kiểm toán và kết quả thực nghiệm Tuần 8
```

---

## 2. Chuẩn bị Dữ liệu (Bắt buộc trước khi chạy lại)

Thư mục ảnh thô (`data/raw/`) và dữ liệu đã xử lý (`data/processed/`) được loại trừ khỏi Git theo `.gitignore` và được chia sẻ riêng trong nhóm. Để chạy lại các notebook, script Tuần 8 hoặc ứng dụng Demo, hãy giải nén/đặt các tệp dữ liệu đúng vị trí sau:

### 2.1. Dữ liệu đã xử lý đặt vào `data/processed/`
Các tệp bắt buộc:
- `data/processed/images_24x24.npy` — Mảng $3.589$ ảnh xám $24 \times 24$ (`uint8`).
- `data/processed/haar_features.npy` — Ma trận đặc trưng Haar-like $(3589, 258)$ (`float32`).
- `data/processed/labels.npy` — Nhãn nhị phân $(3589,)$ (`1`: face, `0`: non-face).
- `data/processed/metadata.csv` — Siêu dữ liệu, mã định danh nhóm (`group`) và mã băm SHA-256.
- `data/processed/split_indices.npz` — Chỉ số chia tập cố định (`train=2511`, `val=540`, `test=538`).
- `data/processed/haar_dataset_splits.npz` — Dữ liệu đặc trưng đã tách sẵn theo 3 tập.
- `data/processed/data_dictionary_haar.csv` — Từ điển mô tả 258 đặc trưng Haar-like.
- `data/processed/patch_metadata_before_split.csv` — Siêu dữ liệu trước bước chia tập (tham khảo Tuần 4).

### 2.2. Tái tạo dữ liệu từ ảnh thô (tùy chọn)
Nếu muốn chạy lại từ đầu notebook `01_Week4_EDA_FaceDetection.ipynb`, đặt dữ liệu nguồn vào `data/raw/` (ảnh mặt LFW trong `data/raw/faces/` và ảnh nền Caltech 101 trong `data/raw/backgrounds/caltech101/BACKGROUND_Google/`).

---

## 3. Hướng dẫn Cài đặt, Chạy Thực nghiệm và Khởi chạy Demo

### 3.1. Cài đặt môi trường Python (Python 3.12)
```bash
# Chạy thực nghiệm Tuần 6 & Tuần 8
python -m pip install -r requirements-week6.txt

# Hoặc chạy ứng dụng Demo
python -m pip install -r requirements-demo.txt
```

### 3.2. Tái tạo mô hình `.joblib` và chạy Demo Streamlit
Các tệp trọng số `models/demo/*.joblib` không lưu trên GitHub. Sinh lại toàn bộ 7 Pipeline đã huấn luyện trên 2.511 mẫu train bằng lệnh:

```bash
python train_export_demo.py
```

Sau khi tạo xong mô hình trong `models/demo/`, kiểm thử đầu cuối và mở giao diện Demo:

```bash
python test_demo_end_to_end.py
python -m streamlit run app_demo.py
```
Chi tiết vận hành xem thêm tại `README_DEMO.md`.

### 3.3. Chạy lại thực nghiệm Tuần 6 & Tuần 8
- **Tuần 6:** Mở `02_Week6_Training_FaceDetection.ipynb` và chạy *Restart Kernel and Run All Cells* (notebook tự động đọc `reports/tables/week6_cv_assignments.csv` để giữ nguyên 5 fold theo nhóm). Xem thêm `README_WEEK6.md`.
- **Tuần 8:** Chạy trực tiếp từng kịch bản từ thư mục gốc dự án:
  ```bash
  python week8/run_tc3_adaboost_selectkbest.py
  python week8/run_tc4_decision_tree_depth.py
  python week8/run_tc5_dt_nested_gridsearch.py
  python week8/run_tc7_final_adaboost.py
  ```

---

## 4. Biên dịch Báo cáo LaTeX (`bao_cao/main.tex`)

Tải thư mục `bao_cao/` lên Overleaf và chọn trình biên dịch **XeLaTeX** (chạy 2 lần để cập nhật đầy đủ mục lục và tham chiếu chéo), hoặc biên dịch cục bộ khi đã cài TeX Live / MiKTeX:

```bash
cd bao_cao
latexmk -xelatex main.tex
```
