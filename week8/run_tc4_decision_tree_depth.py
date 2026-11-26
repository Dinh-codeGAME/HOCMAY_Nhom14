"""
run_tc4_decision_tree_depth.py - Thí nghiệm Tiêu chí 4 Tuần 8:
Kiểm tra xem giảm độ sâu tối đa của Decision Tree (max_depth=5 -> max_depth=3)
có thu hẹp chênh lệch F1 huấn luyện–kiểm định hay không.

Cấu hình so sánh:
- DecisionTree_depth5 (Tuần 6): max_depth=5, min_samples_leaf=2, random_state=42, 258 đặc trưng
- DecisionTree_depth3 (Tiêu chí 4): max_depth=3, min_samples_leaf=2, random_state=42, 258 đặc trưng

Nguyên tắc:
- Chỉ dùng đúng 2.511 mẫu train và đúng 5 fold từ week6_cv_assignments.csv.
- Để dành validation giữ riêng (540) và test (538) cho bước đánh giá sau.
- Tổng cộng 10 lượt fit (2 cấu hình x 5 fold).
- Dùng .predict() cho nhãn dự đoán và .predict_proba() lớp 1 cho ROC-AUC.
- Ghi nhận độ sâu thực tế (get_depth) và số lá thực tế (get_n_leaves) ở từng fold.
- Tổng hợp trung bình ± độ lệch chuẩn với ddof=0 và đối chiếu cấu hình depth=5 với Tuần 6.
"""

from datetime import datetime, timezone
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
TABLES_DIR = BASE_DIR / "reports" / "tables"
WEEK8_DIR = BASE_DIR / "week8"
WEEK8_DIR.mkdir(parents=True, exist_ok=True)


def load_and_verify_train_cv():
    """Nạp 2.511 mẫu train và đối chiếu toàn vẹn mẫu, nhãn, nhóm với week6_cv_assignments.csv."""
    X_all = np.load(DATA_DIR / "haar_features.npy")
    y_all = np.load(DATA_DIR / "labels.npy")
    meta_all = pd.read_csv(DATA_DIR / "metadata.csv")
    splits = np.load(DATA_DIR / "split_indices.npz")
    haar_splits = np.load(DATA_DIR / "haar_dataset_splits.npz")

    train_idx = splits["train"]
    assert len(train_idx) == 2511
    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    assert np.array_equal(X_train, haar_splits["X_train"])
    assert np.array_equal(y_train, haar_splits["y_train"])
    assert X_train.shape == (2511, 258)

    train_meta = meta_all.iloc[train_idx].copy().reset_index(drop=False)
    train_meta.rename(columns={"index": "global_row_index"}, inplace=True)

    cv_assign = pd.read_csv(TABLES_DIR / "week6_cv_assignments.csv")
    assert len(cv_assign) == 2511
    for col in ["source_id", "group", "label", "person_id", "source_image_id", "split"]:
        assert (train_meta[col].fillna("") == cv_assign[col].fillna("")).all(), f"Lệch cột {col}"
    assert (train_meta["label"].to_numpy() == y_train).all()

    cv_folds = cv_assign["cv_fold"].to_numpy()
    cv_splits = []
    for fold_num in range(1, 6):
        tr_pos = np.where(cv_folds != fold_num)[0]
        val_pos = np.where(cv_folds == fold_num)[0]
        assert len(set(tr_pos) & set(val_pos)) == 0
        assert len(set(train_meta.iloc[tr_pos]["group"]) & set(train_meta.iloc[val_pos]["group"])) == 0
        assert len(np.unique(y_train[tr_pos])) == 2 and len(np.unique(y_train[val_pos])) == 2
        cv_splits.append((tr_pos, val_pos))

    print("[OK] Đã xác minh 2.511 mẫu train (258 đặc trưng) và 5 fold không giao thoa nhóm.")
    return X_train, y_train, train_meta, cv_splits


def build_dt_configs(random_state: int = 42):
    return {
        "DecisionTree_depth5": Pipeline([
            ("preprocess", "passthrough"),
            ("clf", DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=2,
                random_state=random_state,
            )),
        ]),
        "DecisionTree_depth3": Pipeline([
            ("preprocess", "passthrough"),
            ("clf", DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=2,
                random_state=random_state,
            )),
        ]),
    }


def compute_metrics(y_true, y_pred, prob_pos):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
    auc = roc_auc_score(y_true, prob_pos)
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
        "roc_auc": float(auc),
    }


def main():
    X_train, y_train, train_meta, cv_splits = load_and_verify_train_cv()
    configs = build_dt_configs(random_state=42)

    fold_records = []
    oof_records = []
    total_fits = 0

    for config_name, base_pipe in configs.items():
        print(f"\n-> Đang chạy 5-fold CV cho cấu hình: {config_name}")
        for fold_num, (tr_pos, val_pos) in enumerate(cv_splits, start=1):
            X_tr_f, y_tr_f = X_train[tr_pos], y_train[tr_pos]
            X_val_f, y_val_f = X_train[val_pos], y_train[val_pos]

            pipe = clone(base_pipe)

            t0 = time.perf_counter()
            pipe.fit(X_tr_f, y_tr_f)
            fit_time = time.perf_counter() - t0
            total_fits += 1

            clf = pipe.named_steps["clf"]
            actual_depth = int(clf.get_depth())
            actual_n_leaves = int(clf.get_n_leaves())
            c1_idx = list(clf.classes_).index(1)
            score_method = f"predict_proba[:, {c1_idx}]"

            t1 = time.perf_counter()
            val_pred = pipe.predict(X_val_f)
            predict_val_time = time.perf_counter() - t1
            val_prob1 = pipe.predict_proba(X_val_f)[:, c1_idx]

            tr_pred = pipe.predict(X_tr_f)
            tr_prob1 = pipe.predict_proba(X_tr_f)[:, c1_idx]

            m_val = compute_metrics(y_val_f, val_pred, val_prob1)
            m_tr = compute_metrics(y_tr_f, tr_pred, tr_prob1)
            delta_f1 = m_tr["f1"] - m_val["f1"]

            fold_records.append({
                "config": config_name,
                "max_depth_param": int(clf.max_depth),
                "min_samples_leaf_param": int(clf.min_samples_leaf),
                "fold": fold_num,
                "train_samples": int(len(tr_pos)),
                "val_samples": int(len(val_pos)),
                "group_overlap": 0,
                "actual_depth": actual_depth,
                "actual_n_leaves": actual_n_leaves,
                "val_tn": m_val["tn"],
                "val_fp": m_val["fp"],
                "val_fn": m_val["fn"],
                "val_tp": m_val["tp"],
                "val_accuracy": m_val["accuracy"],
                "val_precision": m_val["precision"],
                "val_recall": m_val["recall"],
                "val_f1": m_val["f1"],
                "val_fpr": m_val["fpr"],
                "val_roc_auc": m_val["roc_auc"],
                "train_tn": m_tr["tn"],
                "train_fp": m_tr["fp"],
                "train_fn": m_tr["fn"],
                "train_tp": m_tr["tp"],
                "train_accuracy": m_tr["accuracy"],
                "train_precision": m_tr["precision"],
                "train_recall": m_tr["recall"],
                "train_f1": m_tr["f1"],
                "train_fpr": m_tr["fpr"],
                "train_roc_auc": m_tr["roc_auc"],
                "delta_f1_train_minus_val": float(delta_f1),
                "fit_time": float(fit_time),
                "predict_val_time": float(predict_val_time),
            })

            for local_i, tr_row_idx in enumerate(val_pos):
                oof_records.append({
                    "config": config_name,
                    "cv_fold": fold_num,
                    "train_row_index": int(tr_row_idx),
                    "global_row_index": int(train_meta.loc[tr_row_idx, "global_row_index"]),
                    "source_id": train_meta.loc[tr_row_idx, "source_id"],
                    "group": train_meta.loc[tr_row_idx, "group"],
                    "y_true": int(y_val_f[local_i]),
                    "y_pred": int(val_pred[local_i]),
                    "score": float(val_prob1[local_i]),
                    "score_method": score_method,
                })

            print(
                f"   Fold {fold_num}: depth={actual_depth}, leaves={actual_n_leaves}, "
                f"train_F1={m_tr['f1']:.4f}, val_F1={m_val['f1']:.4f}, delta_F1={delta_f1:.4f}, "
                f"Rec={m_val['recall']:.4f}, FPR={m_val['fpr']:.4f}, AUC={m_val['roc_auc']:.4f}, "
                f"FP={m_val['fp']}, FN={m_val['fn']}, fit={fit_time:.3f}s"
            )

    assert total_fits == 10, f"Tổng lượt fit phải là 10, thực tế={total_fits}"
    print(f"\n[OK] Hoàn tất đúng {total_fits} lượt fit (5 fold x 2 cấu hình).")

    df_folds = pd.DataFrame(fold_records)
    df_oof = pd.DataFrame(oof_records)

    # Tổng hợp Mean ± Std (ddof=0)
    summary_rows = []
    val_metrics = ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "fp", "fn", "actual_depth", "actual_n_leaves", "delta_f1_train_minus_val", "fit_time", "predict_val_time"]
    train_metrics = ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "fp", "fn"]

    for cfg in configs.keys():
        sub_f = df_folds[df_folds["config"] == cfg]
        for m in val_metrics:
            col = m if m in ["actual_depth", "actual_n_leaves", "delta_f1_train_minus_val", "fit_time", "predict_val_time"] else f"val_{m}"
            vals = sub_f[col].to_numpy(dtype=float)
            summary_rows.append({
                "config": cfg,
                "split": "validation_cv",
                "metric": m,
                "mean": float(np.mean(vals)),
                "std_ddof0": float(np.std(vals, ddof=0)),
                "sum": float(np.sum(vals)),
            })
        for m in train_metrics:
            vals = sub_f[f"train_{m}"].to_numpy(dtype=float)
            summary_rows.append({
                "config": cfg,
                "split": "train_cv",
                "metric": m,
                "mean": float(np.mean(vals)),
                "std_ddof0": float(np.std(vals, ddof=0)),
                "sum": float(np.sum(vals)),
            })

    df_summary = pd.DataFrame(summary_rows)

    folds_path = WEEK8_DIR / "week8_tc4_dt_folds.csv"
    summary_path = WEEK8_DIR / "week8_tc4_dt_summary.csv"
    oof_path = WEEK8_DIR / "week8_tc4_dt_oof_predictions.csv"

    df_folds.to_csv(folds_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    df_oof.to_csv(oof_path, index=False, encoding="utf-8-sig", float_format="%.17g")

    # Đối chiếu cấu hình max_depth=5 với kết quả Tuần 6 đã lưu
    w6_folds = pd.read_csv(TABLES_DIR / "week6_models_cv_folds.csv")
    w6_dt_folds = w6_folds[w6_folds["model"] == "Decision Tree"].sort_values("fold").reset_index(drop=True)
    w8_d5_folds = df_folds[df_folds["config"] == "DecisionTree_depth5"].sort_values("fold").reset_index(drop=True)

    w6_oof = pd.read_csv(TABLES_DIR / "week6_models_oof_predictions.csv", float_precision="round_trip")
    w6_dt_oof = w6_oof[w6_oof["model"] == "Decision Tree"].sort_values("train_row_index").reset_index(drop=True)
    w8_d5_oof = df_oof[df_oof["config"] == "DecisionTree_depth5"].sort_values("train_row_index").reset_index(drop=True)

    assert (w6_dt_oof["train_row_index"].values == w8_d5_oof["train_row_index"].values).all()
    assert (w6_dt_oof["global_row_index"].values == w8_d5_oof["global_row_index"].values).all()
    assert (w6_dt_oof["cv_fold"].values == w8_d5_oof["cv_fold"].values).all()

    pred_mismatches = int(np.sum(w6_dt_oof["y_pred"].values != w8_d5_oof["y_pred"].values))
    max_score_abs_diff = float(np.max(np.abs(w6_dt_oof["score"].values - w8_d5_oof["score"].values)))

    fold_diffs = {}
    for col in ["val_tn", "val_fp", "val_fn", "val_tp", "val_f1", "val_recall", "val_fpr", "val_roc_auc", "train_f1"]:
        fold_diffs[col] = float(np.max(np.abs(w6_dt_folds[col].values - w8_d5_folds[col].values)))

    reconciliation = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_fits_executed": total_fits,
        "oof_samples_compared": int(len(w8_d5_oof)),
        "oof_y_pred_mismatches": pred_mismatches,
        "oof_prob1_score_max_abs_diff": max_score_abs_diff,
        "fold_metric_max_abs_diffs": fold_diffs,
        "artifacts": {
            "folds_csv": str(folds_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "summary_csv": str(summary_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "oof_csv": str(oof_path.relative_to(BASE_DIR)).replace("\\", "/"),
        },
    }

    reconcil_path = WEEK8_DIR / "week8_tc4_week6_reconciliation.json"
    with open(reconcil_path, "w", encoding="utf-8") as f:
        json.dump(reconciliation, f, indent=2, ensure_ascii=False)

    print("\n=== ĐỐI CHIẾU DECISION TREE (max_depth=5) VỚI TUẦN 6 ===")
    print(f"  OOF y_pred mismatches : {pred_mismatches} / {len(w8_d5_oof)}")
    print(f"  Max abs diff (score)  : {max_score_abs_diff:.2e}")
    print(f"  Fold metric max diffs : {fold_diffs}")

    print("\n=== BẢNG TỔNG HỢP TIÊU CHÍ 4 (MEAN ± STD ddof=0) ===")
    for cfg in configs.keys():
        print(f"\n[{cfg}]")
        sub_v = df_summary[(df_summary["config"] == cfg) & (df_summary["split"] == "validation_cv")].set_index("metric")
        sub_t = df_summary[(df_summary["config"] == cfg) & (df_summary["split"] == "train_cv")].set_index("metric")
        print(f"  train_f1        : {sub_t.loc['f1', 'mean']:.6f} ± {sub_t.loc['f1', 'std_ddof0']:.6f}")
        print(f"  val_f1          : {sub_v.loc['f1', 'mean']:.6f} ± {sub_v.loc['f1', 'std_ddof0']:.6f}")
        print(f"  delta_f1        : {sub_v.loc['delta_f1_train_minus_val', 'mean']:.6f} ± {sub_v.loc['delta_f1_train_minus_val', 'std_ddof0']:.6f}")
        for m in ["recall", "fpr", "roc_auc", "fp", "fn", "actual_depth", "actual_n_leaves", "fit_time"]:
            extra = f" (tổng={int(sub_v.loc[m, 'sum'])})" if m in ["fp", "fn"] else ""
            print(f"  {m:16s}: {sub_v.loc[m, 'mean']:.6f} ± {sub_v.loc[m, 'std_ddof0']:.6f}{extra}")


if __name__ == "__main__":
    main()
