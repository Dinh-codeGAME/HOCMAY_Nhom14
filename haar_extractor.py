"""
haar_extractor.py - Module trích xuất đặc trưng Haar-like và suy luận cho mô hình phát hiện khuôn mặt.
Dự án: Phát hiện khuôn mặt người trong ảnh (Tuần 6)
Thư mục: FaceDetection_Week4

Quy chuẩn kỹ thuật:
- Cửa sổ ảnh con: 24x24 pixel, ảnh mức xám uint8 [0, 255].
- Chuẩn hóa điểm ảnh: chia cho 255.0 (float32).
- Ảnh tích phân (Integral Image): np.pad((1, 0), (1, 0)) và cumsum hai chiều (float64).
- 258 đặc trưng Haar-like thuộc 5 loại mẫu (two_x, two_y, three_x, three_y, four)
  với kích thước ô (block) 4, 6, 8 pixel và bước dịch 4 pixel.
- Chuẩn hóa diện tích: chia cho (factor * block * block) với factor=1 (two_x, two_y) và factor=2 (khác).
- Đầu ra đặc trưng: vector ma trận (N, 258) kiểu np.float32.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

SIZE = 24

# Định nghĩa các mẫu Haar-like theo tọa độ ô và trọng số
PATTERNS = {
    "two_x": [(0, 0, -1), (1, 0, 1)],
    "two_y": [(0, 0, -1), (0, 1, 1)],
    "three_x": [(0, 0, 1), (1, 0, -2), (2, 0, 1)],
    "three_y": [(0, 0, 1), (0, 1, -2), (0, 2, 1)],
    "four": [
        (0, 0, 1), (1, 0, -1),
        (0, 1, -1), (1, 1, 1)
    ]
}


def generate_specs() -> List[Tuple[str, int, int, int]]:
    """
    Sinh danh sách thông số 258 đặc trưng Haar-like theo thứ tự cố định.
    Trả về danh sách các bộ (kind, x, y, block).
    """
    specs = []
    for block in (4, 6, 8):
        for kind, pattern in PATTERNS.items():
            width = (max(dx for dx, dy, weight in pattern) + 1) * block
            height = (max(dy for dx, dy, weight in pattern) + 1) * block
            for y in range(0, SIZE - height + 1, 4):
                for x in range(0, SIZE - width + 1, 4):
                    specs.append((kind, x, y, block))
    return specs


SPECS = generate_specs()


def get_feature_names() -> List[str]:
    """Trả về danh sách 258 tên đặc trưng tương ứng với SPECS."""
    return [f"haar_{kind}_x{x}_y{y}_b{block}" for kind, x, y, block in SPECS]


def extract_haar_features(image_input: Union[np.ndarray, list]) -> np.ndarray:
    """
    Trích xuất 258 đặc trưng Haar-like từ một ảnh 24x24 hoặc một mảng N ảnh 24x24.

    Tham số:
        image_input: Mảng numpy 2D (24, 24) hoặc 3D (N, 24, 24) mức xám.
    
    Trả về:
        feats: Mảng float32 shape (1, 258) nếu đầu vào là 1 ảnh, hoặc (N, 258) nếu đầu vào là batch.
    """
    img_arr = np.asarray(image_input)
    is_single = False
    if img_arr.ndim == 2:
        if img_arr.shape != (SIZE, SIZE):
            raise ValueError(f"Ảnh đầu vào phải có kích thước ({SIZE}, {SIZE}), nhận được {img_arr.shape}")
        img_arr = img_arr[np.newaxis, :, :]
        is_single = True
    elif img_arr.ndim == 3:
        if img_arr.shape[1:] != (SIZE, SIZE):
            raise ValueError(f"Mỗi ảnh trong batch phải có kích thước ({SIZE}, {SIZE}), nhận được {img_arr.shape[1:]}")
    else:
        raise ValueError(f"Đầu vào phải là mảng 2D hoặc 3D, nhận được ndim={img_arr.ndim}")

    # Chuẩn hóa giá trị điểm ảnh sang khoảng [0, 1] kiểu float32
    images_f = img_arr.astype(np.float32) / 255.0

    # Tính ảnh tích phân (Integral Image) có đệm 1 hàng và 1 cột 0 ở góc trên-trái
    ii = np.pad(
        images_f,
        ((0, 0), (1, 0), (1, 0))
    ).cumsum(axis=1, dtype=np.float64).cumsum(axis=2, dtype=np.float64)

    N = len(images_f)
    num_features = len(SPECS)
    feats = np.empty((N, num_features), dtype=np.float32)

    for col, (kind, x, y, block) in enumerate(SPECS):
        values = np.zeros(N, dtype=np.float64)
        for dx, dy, weight in PATTERNS[kind]:
            rx = x + dx * block
            ry = y + dy * block
            # Rectangle sum từ ảnh tích phân: II[y+h, x+w] - II[y, x+w] - II[y+h, x] + II[y, x]
            values += weight * (
                ii[:, ry + block, rx + block]
                - ii[:, ry, rx + block]
                - ii[:, ry + block, rx]
                + ii[:, ry, rx]
            )
        factor = 1 if kind in ("two_x", "two_y") else 2
        feats[:, col] = values / (factor * block * block)

    return feats if not is_single else feats.reshape(1, num_features)


def predict_single_image(
    pipeline,
    image_24x24: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, Union[int, float, str, np.ndarray, None]]:
    """
    Thực hiện quy trình trích xuất đặc trưng và suy luận thực tế trên một ảnh 24x24:
    Ảnh 24x24 -> Trích xuất 258 đặc trưng -> Pipeline -> Nhãn dự đoán và Điểm số.

    Tham số:
        pipeline: Pipeline đã fit của sklearn.
        image_24x24: Mảng 2D (24, 24) uint8.
        threshold: Ngưỡng quyết định tùy chọn (nếu None, dùng quy tắc mặc định của mô hình).

    Trả về:
        dict chứa:
            - y_pred_default: Nhãn phân loại mặc định (0 hoặc 1)
            - y_pred_selected: Nhãn phân loại sau khi áp dụng ngưỡng (0 hoặc 1)
            - score: Giá trị điểm số thô (decision_function hoặc xác suất lớp 1)
            - score_type: "decision_function" hoặc "predict_proba"
            - score_label: "Điểm quyết định" hoặc "Xác suất ước lượng lớp face"
            - threshold_applied: Ngưỡng đã áp dụng (nếu có)
            - features: Mảng (1, 258) đặc trưng đã trích xuất
    """
    feats = extract_haar_features(image_24x24)  # shape (1, 258)
    
    # Dự đoán mặc định từ Pipeline
    pred_def = int(pipeline.predict(feats)[0])
    
    clf = pipeline.named_steps["clf"]
    score = None
    score_type = "none"
    score_label = "Không áp dụng"

    if hasattr(clf, "decision_function"):
        score_val = pipeline.decision_function(feats)[0]
        score = float(score_val)
        score_type = "decision_function"
        score_label = "Điểm quyết định"
    elif hasattr(clf, "predict_proba"):
        classes = list(clf.classes_)
        if 1 in classes:
            c1 = classes.index(1)
            score_val = pipeline.predict_proba(feats)[0, c1]
            score = float(score_val)
            score_type = "predict_proba"
            score_label = "Xác suất ước lượng lớp face"

    # Áp dụng ngưỡng nếu được cấu hình
    if threshold is not None and score is not None and np.isfinite(threshold):
        pred_sel = int(score >= threshold)
    else:
        pred_sel = pred_def

    return {
        "y_pred_default": pred_def,
        "y_pred_selected": pred_sel,
        "score": score,
        "score_type": score_type,
        "score_label": score_label,
        "threshold_applied": threshold,
        "features": feats
    }

