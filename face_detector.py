"""
face_detector.py - Mô-đun tự động định vị khuôn mặt trên ảnh đầy đủ (Full Face Detector).
Dự án: Phát hiện khuôn mặt người trong ảnh (Tuần 6)
Thư mục: FaceDetection_Week4

Quy chuẩn kỹ thuật:
- Đầu vào: Ảnh đầy đủ kích thước tùy ý (JPG/PNG).
- Xử lý hướng xoay EXIF và chuyển sang ảnh mức xám.
- Quét đa tỷ lệ (Multi-scale Image Pyramid) kết hợp cửa sổ trượt (Sliding Window).
- Mỗi vùng quét được trích xuất trực tiếp qua 258 đặc trưng Haar-like (haar_extractor.py).
- Phân loại bằng chính Pipeline đã huấn luyện của dự án (models/demo/*.joblib).
- Lọc khung bao trùng lặp bằng thuật toán Non-Maximum Suppression (NMS) dựa trên chỉ số IoU.
- Quy đổi tọa độ chính xác về hệ quy chiếu ảnh gốc ban đầu.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from haar_extractor import extract_haar_features


def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.3
) -> List[int]:
    """
    Thuật toán Non-Maximum Suppression (NMS) bằng NumPy.

    Tham số:
        boxes: Mảng float shape (N, 4) chứa tọa độ [x, y, w, h].
        scores: Mảng float shape (N,) chứa điểm số phân loại (hỗ trợ cả âm và dương).
        iou_threshold: Ngưỡng IoU tối đa để triệt tiêu khung trùng lặp.

    Trả về:
        Danh sách các chỉ số được giữ lại (theo thứ tự điểm số giảm dần).
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


class FullImageFaceDetector:
    """
    Bộ phát hiện khuôn mặt tự động trên ảnh đầy đủ sử dụng:
    - Image Pyramid + Sliding Window 24x24
    - 258 Haar-like features
    - Pipeline phân loại đã fit của dự án
    - Non-Maximum Suppression (NMS)
    """

    def __init__(
        self,
        pipeline,
        scale_factor: float = 1.25,
        stride: int = 6,
        min_face_size: int = 36,
        max_face_size: Optional[int] = None,
        iou_threshold: float = 0.3,
        score_threshold: Optional[float] = None,
        use_threshold: bool = False,
        max_image_dim: int = 800,
        batch_size: int = 512
    ):
        """
        Khởi tạo bộ phát hiện khuôn mặt.

        Tham số:
            pipeline: Pipeline scikit-learn đã fit trên 2.511 mẫu train.
            scale_factor: Hệ số thu nhỏ ảnh qua từng bậc của kim tự tháp (e.g. 1.25).
            stride: Bước trượt pixel trên ảnh con 24x24 của từng bậc tỷ lệ (e.g. 4, 6, 8).
            min_face_size: Kích thước cạnh nhỏ nhất của khuôn mặt cần quét (theo pixel ảnh gốc).
            max_face_size: Kích thước cạnh lớn nhất của khuôn mặt cần quét (mặc định bằng kích thước ảnh).
            iou_threshold: Ngưỡng IoU của NMS để gộp các khung bao trùng nhau.
            score_threshold: Ngưỡng điểm số tùy chọn (nếu use_threshold=True).
            use_threshold: Có áp dụng ngưỡng điểm số hay dùng Pipeline.predict() mặc định.
            max_image_dim: Kích thước cạnh tối đa để xử lý (tránh tràn bộ nhớ trên ảnh quá lớn).
            batch_size: Kích thước batch trích xuất đặc trưng và suy luận.
        """
        self.pipeline = pipeline
        self.scale_factor = max(1.1, float(scale_factor))
        self.stride = max(2, int(stride))
        self.min_face_size = max(24, int(min_face_size))
        self.max_face_size = max_face_size
        self.iou_threshold = float(iou_threshold)
        self.score_threshold = float(score_threshold) if score_threshold is not None else None
        self.use_threshold = bool(use_threshold)
        self.max_image_dim = int(max_image_dim)
        self.batch_size = int(batch_size)

        # Xác định phương thức trích xuất điểm số của Pipeline
        clf = pipeline.named_steps["clf"]
        if hasattr(clf, "decision_function"):
            self.score_type = "decision_function"
            self.score_label = "Điểm quyết định"
        elif hasattr(clf, "predict_proba"):
            self.score_type = "predict_proba"
            self.score_label = "Xác suất ước lượng lớp face"
        else:
            self.score_type = "predict"
            self.score_label = "Nhãn phân loại"

    def detect(
        self,
        image_input: Union[Image.Image, np.ndarray, str, Path],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Union[List[dict], int, float, str]]:
        """
        Thực hiện quy trình phát hiện khuôn mặt tự động trên ảnh đầu vào.

        Trả về từ điển gồm:
            - detections: Danh sách các khung phát hiện [{'x', 'y', 'w', 'h', 'score', 'patch_24x24'}, ...]
            - total_windows_scanned: Tổng số cửa sổ đã quét qua tất cả các tỷ lệ
            - positive_windows_before_nms: Số lượng cửa sổ được dự đoán dương tính trước NMS
            - faces_detected: Số lượng khuôn mặt sau khi áp dụng NMS
            - time_scan_seconds: Thời gian tạo cửa sổ trượt
            - time_infer_seconds: Thời gian trích xuất Haar và suy luận
            - time_total_seconds: Tổng thời gian thực thi
            - downscale_ratio: Tỷ lệ co ảnh tiền xử lý so với ảnh gốc
            - score_type: Loại điểm số ("decision_function" hoặc "predict_proba")
            - score_label: Tên hiển thị loại điểm số
        """
        t_start = time.perf_counter()

        # 1. Nạp ảnh và chuẩn hóa hướng EXIF
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            raise TypeError(f"Đầu vào không hợp lệ: {type(image_input)}")

        # Xoay đúng chiều theo EXIF metadata
        pil_img = ImageOps.exif_transpose(pil_img)
        orig_w, orig_h = pil_img.size

        # Giới hạn kích thước ảnh xử lý để kiểm soát bộ nhớ và thời gian tính toán
        proc_img = pil_img
        downscale_ratio = 1.0
        max_dim = max(orig_w, orig_h)
        if max_dim > self.max_image_dim:
            downscale_ratio = max_dim / float(self.max_image_dim)
            new_w = int(round(orig_w / downscale_ratio))
            new_h = int(round(orig_h / downscale_ratio))
            proc_img = pil_img.resize((new_w, new_h), Image.BILINEAR)

        # Chuyển sang ảnh mức xám
        gray_img = proc_img.convert("L")
        proc_w, proc_h = gray_img.size

        effective_max_face = self.max_face_size or max(proc_w, proc_h)
        # Quy đổi min_face và max_face sang hệ quy chiếu proc_img
        min_win_proc = int(round(self.min_face_size / downscale_ratio))
        max_win_proc = int(round(effective_max_face / downscale_ratio))

        if progress_callback:
            progress_callback(0.1, "Đang khởi tạo kim tự tháp ảnh và các cửa sổ trượt...")

        # 2. Sinh các cửa sổ trượt theo Kim tự tháp ảnh (Image Pyramid)
        t_scan_start = time.perf_counter()
        candidate_patches = []
        candidate_boxes_proc = []

        scale = 1.0
        while True:
            scaled_w = int(round(proc_w * scale))
            scaled_h = int(round(proc_h * scale))

            # Dừng nếu kích thước ảnh nhỏ hơn kích thước cửa sổ chuẩn 24x24
            if scaled_w < 24 or scaled_h < 24:
                break

            # Kích thước khung bao tương ứng trên ảnh xử lý
            win_proc = int(round(24.0 / scale))
            
            # Chỉ xét các tỷ lệ nằm trong dải kích thước mong muốn
            if win_proc >= min_win_proc and win_proc <= max_win_proc:
                scaled_pil = gray_img.resize((scaled_w, scaled_h), Image.BILINEAR)
                scaled_arr = np.array(scaled_pil, dtype=np.uint8)

                for y in range(0, scaled_h - 24 + 1, self.stride):
                    for x in range(0, scaled_w - 24 + 1, self.stride):
                        patch_24 = scaled_arr[y : y + 24, x : x + 24]
                        x_proc = int(round(x / scale))
                        y_proc = int(round(y / scale))

                        candidate_patches.append(patch_24)
                        candidate_boxes_proc.append((x_proc, y_proc, win_proc, win_proc))

            scale /= self.scale_factor

        t_scan = time.perf_counter() - t_scan_start
        total_scanned = len(candidate_patches)

        if total_scanned == 0:
            return {
                "detections": [],
                "total_windows_scanned": 0,
                "positive_windows_before_nms": 0,
                "faces_detected": 0,
                "time_scan_seconds": t_scan,
                "time_infer_seconds": 0.0,
                "time_total_seconds": time.perf_counter() - t_start,
                "downscale_ratio": downscale_ratio,
                "score_type": self.score_type,
                "score_label": self.score_label
            }

        # 3. Trích xuất đặc trưng Haar và suy luận theo từng batch
        t_infer_start = time.perf_counter()
        pos_boxes = []
        pos_scores = []
        pos_patches = []

        total_batches = int(np.ceil(total_scanned / self.batch_size))
        clf = self.pipeline.named_steps["clf"]

        for b_idx in range(total_batches):
            b_start = b_idx * self.batch_size
            b_end = min(b_start + self.batch_size, total_scanned)

            batch_arr = np.array(candidate_patches[b_start:b_end], dtype=np.uint8)
            batch_feats = extract_haar_features(batch_arr)

            # Trích xuất điểm số
            if self.score_type == "decision_function":
                scores = self.pipeline.decision_function(batch_feats)
            elif self.score_type == "predict_proba":
                classes = list(clf.classes_)
                c1 = classes.index(1) if 1 in classes else 0
                scores = self.pipeline.predict_proba(batch_feats)[:, c1]
            else:
                scores = self.pipeline.predict(batch_feats).astype(float)

            # Xác định điều kiện dự đoán dương tính
            if self.use_threshold and self.score_threshold is not None:
                is_pos = (scores >= self.score_threshold)
            else:
                is_pos = (self.pipeline.predict(batch_feats) == 1)

            pos_indices = np.where(is_pos)[0]
            for p_idx in pos_indices:
                global_idx = b_start + p_idx
                pos_boxes.append(candidate_boxes_proc[global_idx])
                pos_scores.append(float(scores[p_idx]))
                pos_patches.append(candidate_patches[global_idx])

            if progress_callback:
                frac = 0.1 + 0.8 * ((b_idx + 1) / total_batches)
                progress_callback(frac, f"Đang suy luận: batch {b_idx + 1}/{total_batches} ({b_end}/{total_scanned} cửa sổ)...")

        t_infer = time.perf_counter() - t_infer_start
        pos_before_nms = len(pos_boxes)

        # 4. Áp dụng Non-Maximum Suppression (NMS)
        if pos_before_nms > 0:
            if progress_callback:
                progress_callback(0.95, "Đang áp dụng Non-Maximum Suppression (NMS)...")

            boxes_arr = np.array(pos_boxes, dtype=np.float32)
            scores_arr = np.array(pos_scores, dtype=np.float32)

            keep_indices = nms_boxes(boxes_arr, scores_arr, iou_threshold=self.iou_threshold)

            # 5. Quy đổi tọa độ về kích thước ảnh gốc
            detections = []
            for k in keep_indices:
                x_p, y_p, w_p, h_p = pos_boxes[k]
                score_val = pos_scores[k]
                patch_arr = pos_patches[k]

                # Quy đổi tọa độ
                x_o = int(round(x_p * downscale_ratio))
                y_o = int(round(y_p * downscale_ratio))
                w_o = int(round(w_p * downscale_ratio))
                h_o = int(round(h_p * downscale_ratio))

                # Đảm bảo khung bao không tràn biên ảnh gốc
                x_o = max(0, min(x_o, orig_w - 1))
                y_o = max(0, min(y_o, orig_h - 1))
                w_o = max(1, min(w_o, orig_w - x_o))
                h_o = max(1, min(h_o, orig_h - y_o))

                detections.append({
                    "x": x_o,
                    "y": y_o,
                    "w": w_o,
                    "h": h_o,
                    "score": score_val,
                    "patch_24x24": patch_arr
                })
        else:
            detections = []

        if progress_callback:
            progress_callback(1.0, "Hoàn tất phát hiện khuôn mặt.")

        t_total = time.perf_counter() - t_start

        return {
            "detections": detections,
            "total_windows_scanned": total_scanned,
            "positive_windows_before_nms": pos_before_nms,
            "faces_detected": len(detections),
            "time_scan_seconds": t_scan,
            "time_infer_seconds": t_infer,
            "time_total_seconds": t_total,
            "downscale_ratio": downscale_ratio,
            "score_type": self.score_type,
            "score_label": self.score_label
        }


def draw_bounding_boxes(
    pil_image: Image.Image,
    detections: List[dict],
    line_width: int = 3,
    color: str = "#00FF00"
) -> Image.Image:
    """
    Vẽ các khung bao khuôn mặt lên ảnh gốc kèm nhãn và điểm số.

    Tham số:
        pil_image: Ảnh gốc đối tượng PIL Image.
        detections: Danh sách các khung bao từ FullImageFaceDetector.
        line_width: Độ dày nét vẽ khung.
        color: Màu sắc nét vẽ (mặc định xanh lá #00FF00).

    Trả về:
        Ảnh PIL Image mới đã được vẽ các khung bao.
    """
    img_draw = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img_draw)

    # Thử nạp font chữ rõ ràng
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for idx, det in enumerate(detections, start=1):
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        score = det["score"]

        # Vẽ hình chữ nhật
        draw.rectangle([x, y, x + w, y + h], outline=color, width=line_width)

        # Nhãn text
        if isinstance(score, float):
            label_text = f"Face #{idx}: {score:+.3f}"
        else:
            label_text = f"Face #{idx}"

        # Tính kích thước hộp nhãn text
        bbox = draw.textbbox((x, y), label_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Vẽ nền nhãn text
        label_y = max(0, y - text_h - 4)
        draw.rectangle([x, label_y, x + text_w + 6, label_y + text_h + 4], fill=color)
        draw.text((x + 3, label_y + 2), label_text, fill="black", font=font)

    return img_draw
