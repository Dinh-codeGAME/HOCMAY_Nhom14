# ĐÁNH GIÁ ĐIỀU KIỆN TỰ ĐỘNG HÓA — TUẦN 8
**Dự án:** `FaceDetection_Week4`  
**Mục đích:** Xác minh năng lực phần cứng, phần mềm, CLI và đề xuất giao thức điều phối bằng tệp có kiểm soát chất lượng.

---

## 1. Kết quả Kiểm tra Môi trường Thực tế trên Máy

Thông số chi tiết được ghi nhận tự động trong [environment.json](environment.json):

### 1.1. Phần cứng & Hệ điều hành
- **Hệ điều hành:** Microsoft Windows 11 Home (`Windows-11-10.0.26300-SP0`).
- **CPU:** `AMD Ryzen 7 7840H w/ Radeon 780M Graphics` — **8 nhân vật lý, 16 luồng logic**, xung nhịp tối đa `3801 MHz`.
- **RAM:** Tổng dung lượng `15.19 GiB` (`15.927.668 KB`), dung lượng trống tại thời điểm ghi nhận `environment.json` là `3.47 GiB` (`3.642.348 KB`). *Lưu ý:* Với 3.47 GiB RAM trống, khi chạy song song (`n_jobs`), cần giới hạn `n_jobs` (4–8 worker) và `pre_dispatch` để tránh tràn bộ nhớ.

### 1.2. Python & Thư viện Học máy
- **Python:** `3.12.4` (64-bit).
- **Đã cài đặt:** `numpy==2.4.6`, `pandas==3.0.5`, `scikit-learn==1.9.1`, `scipy==1.18.1`, `joblib==1.5.3`, `matplotlib==3.11.1`, `seaborn==0.13.2`, `Pillow==12.2.0`, `streamlit==1.67.0`, `nbformat==5.11.1`, `nbclient==0.10.4`, `ipykernel==7.4.0`, `cv2==4.13.0`.
- **Chưa cài đặt:** `optuna`, `lightgbm`, `xgboost`, `skimage` (giá trị `null`). Hoàn toàn đủ điều kiện chạy `GridSearchCV` / `RandomizedSearchCV` / `HalvingRandomSearchCV` và `Pipeline` của `scikit-learn 1.9.1` mà không cần cài thêm thư viện ngoài.

### 1.3. Antigravity CLI (`agy`) & Khả năng Chạy Không Tương Tác
- **Trạng thái:** Đã cài đặt tại `C:\Users\mrddd\.gemini\bin\agy.exe`, phiên bản **`1.5.18`**.
- **Khả năng xác minh từ `agy --help`:**
  - **Chạy không tương tác:** Hỗ trợ `--print` / `-p` (và `--prompt`) để thực thi một câu lệnh rồi trả kết quả và thoát; hỗ trợ `--print-timeout` để đặt giới hạn thời gian chạy.
  - **Xuất JSON có cấu trúc:** Hỗ trợ `--output-format json` và `--output-format stream-json` (kết hợp `--input-format stream-json`), kèm **`--json-schema`** để ép kết quả đầu ra tuân thủ lược đồ JSON.
  - **Kiểm soát quyền và môi trường:** Hỗ trợ `--mode` (`accept-edits`, `auto`, `plan`), `--sandbox`, `--add-dir`, `--effort` (`low|medium|high|xhigh|max`), `--log-file`.

### 1.4. Skills hiện có liên quan đến ML và LaTeX
- **Cấp người dùng (`C:\Users\mrddd\.gemini\config\skills`):** Có `ictu-thesis-writer` (hỗ trợ cấu trúc báo cáo học thuật/LaTeX/DOCX và kiểm chứng bằng chứng codebase).
- **Tích hợp sẵn (`builtin\skills`):** Có `run-workflow` (điều phối đa tác nhân bằng Python), `automation`, `native_swarm`, v.v.
- **Cấp dự án (`FaceDetection_Week4/.gemini`):** Chưa có thư mục kỹ năng/cấu hình cục bộ riêng cho dự án.

### 1.5. Công cụ Biên dịch LaTeX
- **Trạng thái:** **CHƯA CÓ** bất kỳ công cụ nào (`pdflatex`, `xelatex`, `lualatex`, `latexmk`, `tectonic`, `bibtex`, `biber`, `pandoc` đều không tìm thấy trên `PATH`).
- **Hệ quả:** Quy trình tự động hóa có thể sinh mã nguồn LaTeX (`.tex`) sau khi kiểm tra nghiệm thu đạt chuẩn, nhưng **không thể tự biên dịch ra PDF để kiểm tra lỗi trình bày/tràn lề cục bộ** trên máy này (bước biên dịch PDF phải gắn nhãn `BLOCKED` cục bộ hoặc giao người dùng dán lên Overleaf).

---

## 2. Đề xuất Quy trình Điều phối Tự động Bằng Tệp (File-based Orchestration)

Để tự động hóa các thí nghiệm Tuần 8 một cách có kiểm soát, minh bạch và chống gian lận số liệu, đề xuất tổ chức thư mục `week8/orchestration/` với 4 loại tệp giao tiếp chuẩn:

```text
week8/orchestration/
├── task_spec.json          # Đầu vào bất biến: mục tiêu, phạm vi, ràng buộc & ngưỡng dừng
├── run_log.jsonl           # Nhật ký từng lần chạy (thời gian, cấu hình, trạng thái)
├── experiment_results.csv  # Kết quả thực nghiệm thô đầy đủ (Outer/Inner CV, thời gian)
└── review_verdict.json     # Kết luận kiểm toán độc lập (PASS / REVISE / BLOCKED) + SHA-256
```

### 2.1. Vai trò các tệp điều phối
1. **`task_spec.json` (Đặc tả nhiệm vụ & Điều kiện nghiệm thu):**
   - Khai báo cứng đường dẫn dữ liệu đã khóa (`week6_manifest_sha256.json`), danh sách outer folds (`week6_cv_assignments.csv`), ngân sách tối đa (`max_revisions = 2`, `max_wall_clock_minutes = 30`), và điều kiện nghiệm thu kỹ thuật (không rò rỉ nhóm, đủ kết quả cả 4 bước ablation `S0–S3`, không chạm tập Test trong khi tìm kiếm).
2. **`experiment_results.csv` & `run_log.jsonl` (Bằng chứng thực thi):**
   - Lưu từng lượt chạy kèm seed, tham số, Inner Selection Score, Outer Evaluation Score, `fit_time`, `predict_time`, và mã thoát (*exit code*).
3. **`review_verdict.json` (Kết luận kiểm tra gắn với SHA-256):**
   - Do tác nhân kiểm toán độc lập (hoặc script kiểm định + `agy -p --output-format json --json-schema ...`) xuất ra với 3 trạng thái duy nhất:
     - `PASS`: Đạt đủ kiểm định chống rò rỉ, khớp số liệu, không có lỗi logic.
     - `REVISE`: Phát hiện lỗi kỹ thuật/thiếu kiểm tra — kèm danh sách mục cần sửa cụ thể và đếm `revision_count`.
     - `BLOCKED`: Thiếu công cụ/dữ liệu bắt buộc (ví dụ thiếu trình biên dịch LaTeX cho bước xuất PDF, hoặc lỗi môi trường không tự khắc phục được).
   - **Ràng buộc toàn vẹn bằng SHA-256:** `review_verdict.json` bắt buộc phải lưu cặp `{file_path: sha256}` của **mã nguồn thí nghiệm**, **dữ liệu đầu vào** và **`experiment_results.csv`**. Trước khi chuyển sang bước tiếp theo, hệ thống đối chiếu lại mã băm; nếu bất kỳ tệp nào thay đổi sau khi duyệt, trạng thái `PASS` tự động hết hiệu lực và bắt buộc chạy kiểm tra lại.

### 2.2. Cổng kiểm soát LaTeX (*LaTeX Gate*) & Giới hạn vòng sửa
- **Cổng LaTeX:** Tệp hoặc đoạn mã LaTeX **chỉ được phép sinh hoặc chèn số liệu** khi thỏa mãn đồng thời:
  1. `review_verdict.json` tồn tại và có `verdict == "PASS"`.
  2. Toàn bộ mã băm SHA-256 trong `review_verdict.json` khớp $100\%$ với các tệp đang có trên đĩa.
  3. Mọi con số đưa vào LaTeX phải được trích xuất tự động từ `experiment_results.csv` đã duyệt (không nhập tay).
- **Giới hạn vòng lặp chống tối ưu ép điểm:**
  - Giới hạn cứng số vòng sửa tối đa: **`max_revisions = 2`** (chỉ sửa lỗi kỹ thuật/cấu trúc, **không** dùng vòng lặp để chạy thêm hàng loạt lưới tham số nhằm ép điểm Validation/Test).
  - Giới hạn cứng thời gian từng lần gọi `agy -p`: **`--print-timeout 600s`**, và tổng ngân sách thời gian toàn đợt thí nghiệm: **30 phút**.
  - Nếu hết `max_revisions` mà chưa `PASS`, chuyển ngay sang `BLOCKED` và dừng để báo cáo người dùng thay vì lặp vô hạn.
