import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from PIL import Image

from haar_extractor import extract_haar_features, predict_single_image
from face_detector import FullImageFaceDetector, draw_bounding_boxes, nms_boxes

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "demo"

print("=== KIỂM THỬ ĐẦU CUỐI TOÀN DIỆN (END-TO-END DEMO TEST) ===")

# 1. Kiểm tra dữ liệu ảnh con và đặc trưng
images = np.load(DATA_DIR / "images_24x24.npy")
X = np.load(DATA_DIR / "haar_features.npy")
y = np.load(DATA_DIR / "labels.npy")
splits = np.load(DATA_DIR / "split_indices.npz")
train_idx = splits["train"]

print(f"1. Dữ liệu: {len(images)} ảnh, {len(train_idx)} mẫu train.")

# 2. Nạp toàn bộ 7 mô hình
models = {}
expected_models = [
    ("AdaBoost", "adaboost.joblib"),
    ("SVM RBF", "svm_rbf.joblib"),
    ("Random Forest", "random_forest.joblib"),
    ("Logistic Regression", "logistic_regression.joblib"),
    ("k-NN", "knn.joblib"),
    ("Decision Tree", "decision_tree.joblib"),
    ("DummyClassifier", "dummy_classifier.joblib")
]

for m_name, f_name in expected_models:
    p = MODELS_DIR / f_name
    assert p.exists(), f"Thiếu tệp mô hình: {p}"
    models[m_name] = joblib.load(p)

print(f"2. Nạp thành công {len(models)} mô hình từ models/demo/.")

# 3. Kiểm tra trích xuất đặc trưng trên 50 mẫu ảnh con
max_diff = 0.0
for idx in train_idx[:50]:
    f = extract_haar_features(images[idx])
    diff = np.max(np.abs(f[0] - X[idx]))
    if diff > max_diff:
        max_diff = diff
assert max_diff <= 1e-5
print(f"3. Sai số trích xuất đặc trưng lớn nhất: {max_diff:.2e} (<= 1e-5).")

# 4. Thử nghiệm bộ phát hiện khuôn mặt trên ảnh đầy đủ thật (LFW Face Image)
lfw_face_path = BASE_DIR / "data" / "raw" / "faces" / "lfw_cache" / "lfw_home" / "lfw_funneled" / "Aaron_Eckhart" / "Aaron_Eckhart_0001.jpg"
assert lfw_face_path.exists(), f"Thiếu ảnh LFW: {lfw_face_path}"
pil_face = Image.open(lfw_face_path)

detector = FullImageFaceDetector(
    pipeline=models["AdaBoost"],
    scale_factor=1.2,
    stride=6,
    min_face_size=36,
    max_face_size=200,
    iou_threshold=0.3,
    score_threshold=0.25,
    use_threshold=True
)

res_face = detector.detect(pil_face)
print(f"\n4. Phát hiện trên ảnh mặt đầy đủ ({pil_face.size}):")
print(f"   - Tổng cửa sổ quét: {res_face['total_windows_scanned']}")
print(f"   - Thời gian xử lý: {res_face['time_total_seconds']:.3f} s")
print(f"   - Số mặt phát hiện: {res_face['faces_detected']}")
assert res_face['faces_detected'] >= 1, "Phát hiện khuôn mặt thất bại!"

for i, det in enumerate(res_face["detections"], 1):
    print(f"     Khung #{i}: x={det['x']}, y={det['y']}, w={det['w']}, h={det['h']}, score={det['score']:+.4f}")
    assert 0 <= det['x'] < pil_face.width
    assert 0 <= det['y'] < pil_face.height
    assert det['patch_24x24'].shape == (24, 24)

# 5. Thử nghiệm trên ảnh nền không có mặt (Caltech Background)
bg_path = BASE_DIR / "data" / "raw" / "backgrounds" / "caltech101" / "BACKGROUND_Google" / "image_0003.jpg"
assert bg_path.exists(), f"Thiếu ảnh nền: {bg_path}"
pil_bg = Image.open(bg_path)

res_bg = detector.detect(pil_bg)
print(f"\n5. Phát hiện trên ảnh nền ({pil_bg.size}):")
print(f"   - Tổng cửa sổ quét: {res_bg['total_windows_scanned']}")
print(f"   - Số mặt phát hiện: {res_bg['faces_detected']}")

# 6. Kiểm tra vẽ khung bao
annotated = draw_bounding_boxes(pil_face, res_face["detections"])
assert annotated.size == pil_face.size
print("\n6. Vẽ khung bao lên ảnh thành công.")

print("\n=== TOÀN BỘ KIỂM THỬ ĐẦU CUỐI ĐỀU THÀNH CÔNG 100%! ===")
