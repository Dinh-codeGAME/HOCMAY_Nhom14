"""
train_export_demo.py - Huấn luyện và xuất các mô hình học máy cho ứng dụng Demo.
Dự án: Phát hiện khuôn mặt người trong ảnh (Tuần 6)
Thư mục: FaceDetection_Week4

Mục tiêu:
1. Huấn luyện độc lập 6 Pipeline học máy và 1 Pipeline mốc cơ sở (DummyClassifier)
   trên toàn bộ 2.511 mẫu của tập Train.
2. Bộ chuẩn hóa StandardScaler nằm trong Pipeline, chỉ fit trên tập Train.
3. Giữ nguyên toàn bộ siêu tham số và random_state=42 đã thống nhất trong tuần 6.
4. Xuất các mô hình đã fit ra thư mục models/demo/ dưới định dạng joblib.
5. Lưu kèm tệp thông tin mô tả chi tiết demo_metadata.json.
6. Thẩm định kiểm tra nạp lại và đối chiếu tính nhất quán của kết quả dự đoán.
"""

from datetime import datetime, timezone
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from haar_extractor import extract_haar_features

import sklearn

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "demo"
REPORTS_DIR = BASE_DIR / "reports" / "tables"


def load_train_data():
    """Nạp dữ liệu tập Train từ các tệp chuẩn hóa của dự án."""
    print("-> Đang kiểm tra và nạp dữ liệu từ data/processed/...")
    
    images_path = DATA_DIR / "images_24x24.npy"
    haar_path = DATA_DIR / "haar_features.npy"
    labels_path = DATA_DIR / "labels.npy"
    metadata_path = DATA_DIR / "metadata.csv"
    splits_path = DATA_DIR / "split_indices.npz"

    for p in [images_path, haar_path, labels_path, metadata_path, splits_path]:
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp bắt buộc: {p}")

    images = np.load(images_path)
    X = np.load(haar_path)
    y = np.load(labels_path)
    meta = pd.read_csv(metadata_path)
    splits = np.load(splits_path)

    train_idx = splits["train"]
    assert len(train_idx) == 2511, f"Sai số mẫu tập train: {len(train_idx)}"
    assert X.shape == (3589, 258), f"Sai kích thước đặc trưng: {X.shape}"
    assert len(images) == 3589, f"Sai kích thước ảnh: {len(images)}"

    X_train = X[train_idx]
    y_train = y[train_idx]
    images_train = images[train_idx]
    meta_train = meta.iloc[train_idx].copy().reset_index(drop=True)

    n_face = int((y_train == 1).sum())
    n_nonface = int((y_train == 0).sum())
    print(f"   Tập Train: {len(X_train)} mẫu (Face: {n_face}, Non-face: {n_nonface})")
    assert n_face == 1277 and n_nonface == 1234, "Tỷ lệ nhãn train không khớp mốc đối chiếu!"

    return {
        "train_idx": train_idx,
        "X_train": X_train,
        "y_train": y_train,
        "images_train": images_train,
        "meta_train": meta_train
    }


def verify_feature_extractor(images_train, X_train):
    """Kiểm tra đối chiếu hàm trích xuất Haar trên 2.511 ảnh train."""
    print("-> Đang đối chiếu hàm trích xuất haar_extractor trên 2.511 mẫu train...")
    X_calc = extract_haar_features(images_train)
    max_diff = float(np.max(np.abs(X_calc - X_train)))
    mean_diff = float(np.mean(np.abs(X_calc - X_train)))
    print(f"   Sai số tuyệt đối lớn nhất: {max_diff:.2e}")
    print(f"   Sai số tuyệt đối trung bình: {mean_diff:.2e}")
    
    tol = 1e-5
    if max_diff > tol:
        raise AssertionError(f"Sai số trích xuất {max_diff} vượt quá ngưỡng dung sai {tol}")
    print("   [OK] Kiểm tra trích xuất đặc trưng ĐẠT (sai số <= 1e-5).")
    return max_diff


def define_pipelines(random_state=42):
    """Khởi tạo 7 pipeline với đúng cấu hình và siêu tham số của Tuần 6."""
    return {
        "AdaBoost": {
            "filename": "adaboost.joblib",
            "is_baseline": False,
            "pipeline": Pipeline([
                ("preprocess", "passthrough"),
                ("clf", AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state),
                    n_estimators=100,
                    learning_rate=1.0,
                    random_state=random_state
                ))
            ]),
            "score_method": "decision_function",
            "score_label": "Điểm quyết định",
            "default_rule": "Pipeline.predict()",
            "selected_threshold": 0.026787453434552827,
            "decision_rule": "score >= 0.026787",
            "description": "AdaBoost (100 Decision Stumps) - Thuật toán trọng tâm đề tài"
        },
        "SVM RBF": {
            "filename": "svm_rbf.joblib",
            "is_baseline": False,
            "pipeline": Pipeline([
                ("preprocess", StandardScaler()),
                ("clf", SVC(
                    C=1.0,
                    kernel="rbf",
                    gamma="scale",
                    probability=False
                ))
            ]),
            "score_method": "decision_function",
            "score_label": "Điểm quyết định",
            "default_rule": "Pipeline.predict()",
            "selected_threshold": -0.018045134113052197,
            "decision_rule": "score >= -0.018045",
            "description": "Support Vector Machine với nhân RBF, chuẩn hóa StandardScaler"
        },
        "Random Forest": {
            "filename": "random_forest.joblib",
            "is_baseline": False,
            "pipeline": Pipeline([
                ("preprocess", "passthrough"),
                ("clf", RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=1
                ))
            ]),
            "score_method": "predict_proba[:, 1]",
            "score_label": "Xác suất ước lượng lớp face",
            "default_rule": "Pipeline.predict()",
            "selected_threshold": 0.5061666666666668,
            "decision_rule": "score >= 0.506167",
            "description": "Random Forest (200 cây quyết định, min_samples_leaf=2)"
        },
        "Logistic Regression": {
            "filename": "logistic_regression.joblib",
            "is_baseline": False,
            "pipeline": Pipeline([
                ("preprocess", StandardScaler()),
                ("clf", LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=random_state
                ))
            ]),
            "score_method": "decision_function",
            "score_label": "Điểm quyết định",
            "default_rule": "Pipeline.predict()",
            "selected_threshold": 0.5389142036437988,
            "decision_rule": "score >= 0.538914",
            "description": "Logistic Regression (lbfgs, C=1.0), chuẩn hóa StandardScaler"
        },
        "k-NN": {
            "filename": "knn.joblib",
            "is_baseline": False,
            "pipeline": Pipeline([
                ("preprocess", StandardScaler()),
                ("clf", KNeighborsClassifier(
                    n_neighbors=5,
                    weights="uniform",
                    metric="minkowski",
                    p=2,
                    n_jobs=1
                ))
            ]),
            "score_method": "predict_proba[:, 1]",
            "score_label": "Xác suất ước lượng lớp face",
            "default_rule": "Pipeline.predict()",
            "selected_threshold": None,
            "decision_rule": "keep_default (predict)",
            "description": "k-Nearest Neighbors (k=5, khoảng cách Minkowski p=2)"
        },
        "Decision Tree": {
            "filename": "decision_tree.joblib",
            "is_baseline": False,
            "pipeline": Pipeline([
                ("preprocess", "passthrough"),
                ("clf", DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=2,
                    random_state=random_state
                ))
            ]),
            "score_method": "predict_proba[:, 1]",
            "score_label": "Xác suất ước lượng lớp face",
            "default_rule": "Pipeline.predict()",
            "selected_threshold": None,
            "decision_rule": "keep_default (predict)",
            "description": "Cây quyết định đơn (max_depth=5, min_samples_leaf=2)"
        },
        "DummyClassifier": {
            "filename": "dummy_classifier.joblib",
            "is_baseline": True,
            "pipeline": Pipeline([
                ("preprocess", "passthrough"),
                ("clf", DummyClassifier(strategy="most_frequent"))
            ]),
            "score_method": "predict_proba[:, 1]",
            "score_label": "Xác suất ước lượng lớp face",
            "default_rule": "Pipeline.predict() [most_frequent]",
            "selected_threshold": None,
            "decision_rule": "keep_default (baseline)",
            "description": "Mô hình mốc cơ sở: luôn dự đoán lớp đa số trong tập huấn luyện (face = 1)"
        }
    }


def train_and_export():
    """Thực thi huấn luyện và xuất toàn bộ 7 mô hình."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    data = load_train_data()
    X_train = data["X_train"]
    y_train = data["y_train"]
    images_train = data["images_train"]

    # 1. Thẩm định hàm trích xuất
    max_diff = verify_feature_extractor(images_train, X_train)

    # 2. Khởi tạo pipelines
    model_defs = define_pipelines(random_state=42)

    metadata_info = {
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "library_versions": {
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__
        },
        "dataset_scope": {
            "train_samples": len(X_train),
            "face_count": int((y_train == 1).sum()),
            "nonface_count": int((y_train == 0).sum()),
            "features_count": 258,
            "window_size": [24, 24],
            "feature_max_diff_train": max_diff
        },
        "models": {}
    }

    # Bảng lưu kết quả kiểm tra suy luận trước/sau khi lưu
    sanity_results = []

    print("\n-> Bắt đầu huấn luyện và lưu từng Pipeline trên 2.511 mẫu train:")
    for name, info in model_defs.items():
        pipe = info["pipeline"]
        fname = info["filename"]
        out_path = MODELS_DIR / fname

        print(f"   [+] Huấn luyện {name}...")
        pipe.fit(X_train, y_train)

        # Lưu mô hình
        joblib.dump(pipe, out_path)
        file_size = out_path.stat().st_size
        print(f"       -> Đã lưu {fname} ({file_size:,} bytes)")

        # Dự đoán mẫu trước khi lưu
        preds_before = pipe.predict(X_train[:100])

        # Nạp lại từ tệp để kiểm tra tính toàn vẹn
        reloaded_pipe = joblib.load(out_path)
        preds_after = reloaded_pipe.predict(X_train[:100])

        match_count = int(np.sum(preds_before == preds_after))
        assert match_count == 100, f"Mô hình {name} sau khi nạp không khớp 100% dự đoán ban đầu!"

        metadata_info["models"][name] = {
            "filename": fname,
            "is_baseline": info["is_baseline"],
            "score_method": info["score_method"],
            "score_label": info["score_label"],
            "default_rule": info["default_rule"],
            "selected_threshold": info["selected_threshold"],
            "decision_rule": info["decision_rule"],
            "description": info["description"],
            "file_size_bytes": file_size
        }

        sanity_results.append({
            "model": name,
            "file": fname,
            "file_size_bytes": file_size,
            "sanity_check": "PASSED (100% match on 100 samples)"
        })

    # Lưu tệp metadata json
    meta_json_path = MODELS_DIR / "demo_metadata.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_info, f, indent=2, ensure_ascii=False)
    print(f"\n-> Đã lưu tệp thông tin mô tả: {meta_json_path}")

    # Báo cáo tổng kết
    print("\n=== TỔNG KẾT XUẤT MÔ HÌNH DEMO ===")
    df_res = pd.DataFrame(sanity_results)
    print(df_res.to_string(index=False))
    print("\nToàn bộ 7 mô hình đã được huấn luyện và xuất thành công!")


if __name__ == "__main__":
    train_and_export()
