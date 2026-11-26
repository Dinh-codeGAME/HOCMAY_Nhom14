# HƯỚNG DẪN CÀI ĐẶT, VẬN HÀNH VÀ BÁO CÁO ỨNG DỤNG DEMO
## Đề tài: Phát hiện khuôn mặt người trong ảnh — Tuần 6 (Tích hợp Định vị Khuôn mặt Tự động)

---

## 1. Giới thiệu và Phạm vi Kỹ thuật của Ứng dụng Demo

Ứng dụng Demo được phát triển nhằm phục vụ việc **báo cáo và trình bày trực tiếp cho giảng viên hướng dẫn** thông qua giao diện đồ họa tương tác (Streamlit). Chương trình tích hợp hai chế độ hoạt động chuyên biệt:

1. **Chế độ chính: Phát hiện khuôn mặt tự động trên ảnh đầy đủ (*Full Image Face Detection*)**
   - Người dùng tải lên một bức ảnh bất kỳ (JPG/PNG) hoặc chọn ảnh mẫu có sẵn của dự án.
   - Nhấp chọn **“Bắt đầu phát hiện khuôn mặt”**, chương trình tự động quét đa tỷ lệ, trích xuất đặc trưng, suy luận và vẽ các khung bao (*bounding boxes*) định vị khuôn mặt trên ảnh gốc. Người dùng **không phải tự chọn vùng mặt**.
   - Cung cấp tính năng phân tích giải thích: người dùng có thể nhấp chọn từng khung bao để xem ảnh con $24\times 24$ điểm ảnh và biên độ các đặc trưng Haar-like tương ứng.
2. **Chế độ bổ trợ: Minh họa phân loại ảnh con $24\times 24$ (*Sub-window Classification*)**
   - Giữ nguyên toàn bộ chức năng kiểm thử trên $2.511$ mẫu thuộc tập huấn luyện (*train split*).
   - Minh họa cơ chế tính ảnh tích phân, trích xuất 258 đặc trưng Haar và so sánh đồng thời 6 mô hình học máy trên một cửa sổ ảnh con.

---

## 2. Kiến trúc và Nguyên lý của Bộ Phát hiện Tự động (`face_detector.py`)

Bộ phát hiện tự động được xây dựng hoàn toàn từ mã nguồn nội bộ của dự án, sử dụng trực tiếp các Pipeline máy học đã huấn luyện trong `models/demo/`:

1. **Chuẩn hóa ảnh đầu vào:**
   - Đọc ảnh đa định dạng (JPG, JPEG, PNG).
   - Tự động hiệu chỉnh góc xoay theo metadata EXIF bằng `PIL.ImageOps.exif_transpose`.
   - Giới hạn kích thước tối đa (`max_image_dim = 800` px) nhằm kiểm soát bộ nhớ và thời gian quét, ghi nhận tỷ lệ co `downscale_ratio` để quy đổi tọa độ ngược lại ảnh gốc ban đầu.
   - Chuyển đổi sang ảnh mức xám (*Grayscale*).
2. **Quét cửa sổ trượt đa tỷ lệ (Kim tự tháp ảnh — *Image Pyramid*):**
   - Co nhỏ ảnh theo hệ số tỷ lệ `scale_factor` (mặc định $1{,}2$ hoặc $1{,}25$) qua từng bậc.
   - Trượt cửa sổ kích thước chuẩn $24 \times 24$ pixel với bước trượt `stride` (mặc định 6 hoặc 8 px).
   - Mỗi cửa sổ trên ảnh co nhỏ tương ứng với một vùng có kích thước $w = \text{round}(24 / \text{scale})$ trên ảnh gốc.
   - Bỏ qua các tỷ lệ nằm ngoài dải kích thước thiết lập `[min_face_size, max_face_size]`.
3. **Trích xuất đặc trưng và suy luận theo lô (Batch Processing):**
   - Thu thập toàn bộ các mảng $24\times 24$ của từng bậc tỷ lệ.
   - Xử lý theo từng batch (mặc định 512 cửa sổ) đưa vào `haar_extractor.extract_haar_features()`, tính ảnh tích phân và trích xuất $258$ đặc trưng Haar-like.
   - Đưa vector đặc trưng vào Pipeline đã nạp (`AdaBoost`, `SVM RBF`, `Random Forest`, ...).
   - Chế độ mặc định sử dụng `Pipeline.predict() == 1`. Hỗ trợ chế độ thử nghiệm đặt ngưỡng điểm số (`score >= threshold`).
4. **Triệt tiêu cực đại cục bộ (Non-Maximum Suppression — NMS):**
   - Cài đặt NMS thuần túy bằng thư viện NumPy, hỗ trợ trực tiếp các điểm quyết định có giá trị âm hoặc dương của `decision_function` hoặc xác suất `predict_proba`.
   - Sắp xếp các khung bao ứng viên theo điểm số giảm dần, tính chỉ số giao trên hợp (IoU) và loại bỏ các khung bao trùng lặp vượt ngưỡng `iou_threshold` (mặc định $0{,}3$).
5. **Quy đổi tọa độ:**
   - Nhân tọa độ khung bao với `downscale_ratio` để đưa về đúng hệ tọa độ điểm ảnh của ảnh gốc ban đầu.
   - Khống chế tọa độ không vượt quá biên ảnh gốc.

---

## 3. Cấu trúc Thư mục Dự án

```text
FaceDetection_Week4/
├── app_demo.py                      # Ứng dụng Streamlit tích hợp 2 chế độ (chính & bổ trợ)
├── face_detector.py                 # Mô-đun tự động phát hiện khuôn mặt đa tỷ lệ + NMS
├── haar_extractor.py                # Mô-đun trích xuất 258 đặc trưng Haar-like và suy luận
├── train_export_demo.py             # Kịch bản huấn luyện và xuất 7 Pipeline mô hình ra joblib
├── requirements-demo.txt            # Danh mục thư viện phụ thuộc bắt buộc
├── README_DEMO.md                   # Tài liệu hướng dẫn sử dụng và kịch bản trình bày
├── week6_demo_bundle.zip            # Gói nén chuyển giao độc lập (mã nguồn, mô hình, dữ liệu)
│
├── models/
│   └── demo/                        # Thư mục lưu trữ 7 mô hình đã fit trên tập Train
│       ├── adaboost.joblib          # Pipeline AdaBoost (100 Decision Stumps) - Mặc định
│       ├── svm_rbf.joblib           # Pipeline SVM kernel RBF (StandardScaler + SVC)
│       ├── random_forest.joblib     # Pipeline Random Forest (200 cây quyết định)
│       ├── logistic_regression.joblib# Pipeline Logistic Regression (StandardScaler + lbfgs)
│       ├── knn.joblib               # Pipeline k-NN (StandardScaler + k=5)
│       ├── decision_tree.joblib     # Pipeline Decision Tree đơn (max_depth=5)
│       ├── dummy_classifier.joblib  # Pipeline DummyClassifier (mốc cơ sở tham chiếu)
│       └── demo_metadata.json       # Tệp metadata: siêu tham số, ngưỡng và phiên bản
│
└── data/
    ├── processed/                   # 6 tệp dữ liệu thực nghiệm tuần 6
    │   ├── images_24x24.npy         # 3.589 ảnh con 24x24 uint8
    │   ├── haar_features.npy        # Ma trận 3.589 x 258 đặc trưng Haar float32
    │   ├── labels.npy               # 3.589 nhãn nhị phân
    │   ├── metadata.csv             # Siêu dữ liệu nguồn ảnh và mã băm sha256
    │   ├── split_indices.npz        # Phân chia train (2.511), val (540), test (538)
    │   └── data_dictionary_haar.csv # Từ điển định nghĩa 258 đặc trưng Haar
    └── raw/                         # Dữ liệu ảnh thô dùng làm mẫu thử nghiệm
        ├── faces/lfw_cache/...      # Ảnh khuôn mặt đầy đủ thật (LFW 250x250)
        └── backgrounds/caltech101/..# Ảnh nền đầy đủ thật không có mặt (Caltech 101)
```

---

## 4. Cài đặt Môi trường và Khởi chạy

### 4.1. Cài đặt phụ thuộc
Chạy lệnh sau trong Terminal (PowerShell / Command Prompt):

```bash
pip install -r requirements-demo.txt
```

### 4.2. Huấn luyện và xuất mô hình (nếu chạy lần đầu)
```bash
python train_export_demo.py
```

### 4.3. Kiểm thử tự động toàn diện
```bash
python test_demo_end_to_end.py
```

### 4.4. Khởi chạy ứng dụng giao diện Demo
```bash
python -m streamlit run app_demo.py
```
*(Trình duyệt sẽ tự động mở giao diện tại `http://localhost:8501`)*

---

## 5. Kịch bản Trình bày Từng Bước cho Giảng viên

### Bước 1: Mở chế độ chính "Phát hiện khuôn mặt trong ảnh"
- Giới thiệu giao diện: Đây là chế độ mở mặc định khi khởi động ứng dụng.
- Nêu rõ nguyên lý: Ứng dụng thực hiện quét ảnh đầy đủ tự động, không yêu cầu người dùng phải tự cắt chọn vùng mặt.

### Bước 2: Thử nghiệm trên ảnh khuôn mặt có sẵn của dự án
- Tại thanh bên trái mục **"📁 Nguồn ảnh đầu vào"**, chọn **"Sử dụng ảnh mẫu có sẵn của dự án"**.
- Chọn ảnh `Ảnh mặt 1 (LFW: Aaron Eckhart)`.
- Giữ mô hình mặc định: **AdaBoost** (thuật toán trọng tâm đề tài).
- Bấm nút **"🔍 Bắt đầu phát hiện khuôn mặt"**:
  - Quan sát thanh tiến trình quét qua các bậc kim tự tháp ảnh.
  - Quan sát kết quả: Bức ảnh được khoanh vùng khuôn mặt bằng khung màu xanh lá kèm nhãn và điểm số quyết định.
  - Quan sát thẻ số liệu thống kê: Số lượng khuôn mặt phát hiện, tổng số cửa sổ đã quét (ví dụ: $1.158$ cửa sổ), thời gian xử lý thực tế (ví dụ: $0{,}18\text{ s}$).

### Bước 3: Chứng minh quy trình suy luận bằng tính năng giải thích ảnh con
- Cuộn xuống mục **"🔬 Phân tích giải thích: Kiểm tra ảnh con 24×24 tương ứng"**.
- Trình bày cho giảng viên: Khung bao được phát hiện thực chất được trích xuất về ma trận điểm ảnh $24\times 24$ (phóng đại 8x hiển thị sắc nét lưới pixel).
- Chỉ cho giảng viên thấy vector $258$ đặc trưng Haar-like và các đặc trưng có biên độ dao động mạnh nhất tại vùng mắt, mũi, miệng.

### Bước 4: Thử nghiệm trên ảnh nền không có khuôn mặt (Kiểm tra báo động giả)
- Tại thanh bên trái, chọn `Ảnh nền 1 (Caltech: Không có mặt #0003)`.
- Bấm **"🔍 Bắt đầu phát hiện khuôn mặt"**.
- Quan sát kết quả: Hệ thống quét qua hàng trăm cửa sổ và thông báo rõ ràng:
  > *"Không phát hiện được khuôn mặt với cấu hình hiện tại."*
- Phân tích cho giảng viên: Mô hình kiểm soát tốt báo động sai trên vùng nền đồng nhất.

### Bước 5: Thử nghiệm tải ảnh bất kỳ từ máy tính
- Chọn mục **"Tải ảnh từ máy tính (JPG/PNG)"**.
- Tải một ảnh chân dung thực tế.
- Bấm **"Bắt đầu phát hiện khuôn mặt"** và tải ảnh kết quả bằng nút **"💾 Tải ảnh kết quả về máy"**.
- Trình bày tính năng tùy chỉnh tham số quét: điều chỉnh `stride`, `min_face_size`, `scale_factor` và `iou_threshold` trong mục Cấu hình nâng cao.

### Bước 6: Chuyển sang chế độ "Minh họa phân loại ảnh con"
- Chọn radio tại thanh bên: **"🔬 Minh họa phân loại ảnh con (24×24)"**.
- Bấm **"🎲 Chọn ngẫu nhiên"** và bấm **"📊 So sánh sáu mô hình"**.
- Trình bày bảng đối sánh giữa 6 thuật toán học máy và mô hình cơ sở DummyClassifier để giảng viên thấy rõ quá trình đánh giá ở mức ảnh con đã thực hiện trong tuần 6.

---

## 6. Tính Toàn vẹn Khoa học và Giới hạn Đồ án

1. **Tuyệt đối không dùng mô hình ngoài:** Toàn bộ khung phát hiện đều do chính các mô hình đã fit trong dự án suy luận từ 258 đặc trưng Haar. Không dùng bất kỳ mô hình pretrained hay thư viện OpenCV Haar cascade có sẵn.
2. **Đo lường thời gian trung thực:** Báo cáo thời gian xử lý thực tế đo bằng `time.perf_counter()`, không tự nhận là hệ thống chạy "thời gian thực" (*real-time*).
3. **Phân biệt ranh giới bài toán:**
   - Các bảng chỉ số trong báo cáo tuần 6 ($F_1$, Recall, FPR, ROC-AUC) đánh giá bài toán **phân loại ảnh con $24 \times 24$**.
   - Bài toán định vị trên ảnh đầy đủ phụ thuộc thêm vào bước trượt cửa sổ, tỷ lệ kim tự tháp và ngưỡng NMS. Do ảnh tải lên không có nhãn hộp chuẩn (Ground-truth Bounding Boxes), hệ thống **không tự tính điểm Accuracy** hoặc khẳng định kết quả đúng/sai tuyệt đối.
4. **Điểm số hiển thị nguyên bản:** Điểm quyết định từ `decision_function` được giữ nguyên giá trị đại số lề, không tự đổi thành phần trăm xác suất giả định.
