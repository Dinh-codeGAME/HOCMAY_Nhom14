"""
run_tc3_adaboost_selectkbest.py - Thí nghiệm đầu tiên cho Tiêu chí 3 Tuần 8:
So sánh AdaBoost (258 đặc trưng Haar-like) với AdaBoost + SelectKBest(f_classif, k=128)
trên đúng 2.511 mẫu train và đúng 5 fold từ reports/tables/week6_cv_assignments.csv.

Nguyên tắc thực nghiệm:
- Không sử dụng tập validation giữ riêng (540) hoặc tập test (538).
- SelectKBest nằm bên trong Pipeline, chỉ fit trên phần huấn luyện của từng fold.
- Cố định k=128 và toàn bộ siêu tham số AdaBoost như Tuần 6 (random_state=42).
- Dùng .predict() cho nhãn dự đoán và .decision_function() cho ROC-AUC.
- Tổng hợp trung bình ± độ lệch chuẩn với ddof=0.
- Đối chiếu kết quả cấu hình 258 đặc trưng với kết quả đã lưu của Tuần 6.
"""

from datetime import datetime, timezone
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
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
    """Nạp dữ liệu train (2.511 mẫu) và đối chiếu 100% thứ tự mẫu, nhãn, nhóm với week6_cv_assignments.csv."""
    X_all = np.load(DATA_DIR / "haar_features.npy")
    y_all = np.load(DATA_DIR / "labels.npy")
    meta_all = pd.read_csv(DATA_DIR / "metadata.csv")
    splits = np.load(DATA_DIR / "split_indices.npz")
    haar_splits = np.load(DATA_DIR / "haar_dataset_splits.npz")
    feature_dict = pd.read_csv(DATA_DIR / "data_dictionary_haar.csv")

    train_idx = splits["train"]
    assert len(train_idx) == 2511, f"Sai số mẫu train: {len(train_idx)}"

    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    assert np.array_equal(X_train, haar_splits["X_train"]), "X_train không khớp haar_dataset_splits.npz!"
    assert np.array_equal(y_train, haar_splits["y_train"]), "y_train không khớp haar_dataset_splits.npz!"
    assert X_train.shape == (2511, 258)
    assert len(feature_dict) == 258

    train_meta = meta_all.iloc[train_idx].copy().reset_index(drop=False)
    train_meta.rename(columns={"index": "global_row_index"}, inplace=True)

    cv_assign = pd.read_csv(TABLES_DIR / "week6_cv_assignments.csv")
    assert len(cv_assign) == len(train_meta) == 2511

    for col in ["source_id", "group", "label", "person_id", "source_image_id", "split"]:
        s_meta = train_meta[col].fillna("")
        s_cv = cv_assign[col].fillna("")
        assert (s_meta == s_cv).all(), f"Cột {col} không khớp giữa train_meta và week6_cv_assignments.csv!"

    assert (train_meta["label"].to_numpy() == y_train).all(), "Nhãn y_train không khớp metadata!"

    cv_folds = cv_assign["cv_fold"].to_numpy()
    cv_splits = []
    for fold_num in range(1, 6):
        tr_pos = np.where(cv_folds != fold_num)[0]
        val_pos = np.where(cv_folds == fold_num)[0]
        assert len(set(tr_pos) & set(val_pos)) == 0
        tr_groups = set(train_meta.iloc[tr_pos]["group"])
        val_groups = set(train_meta.iloc[val_pos]["group"])
        assert len(tr_groups & val_groups) == 0, f"Fold {fold_num} bị giao thoa nhóm!"
        assert len(np.unique(y_train[tr_pos])) == 2 and len(np.unique(y_train[val_pos])) == 2
        cv_splits.append((tr_pos, val_pos))

    print("[OK] Đã đối chiếu toàn vẹn 2.511 mẫu train (thứ tự, nhãn, nhóm) và 5 fold từ week6_cv_assignments.csv.")
    return X_train, y_train, train_meta, train_idx, feature_dict, cv_splits


def build_adaboost_configs(random_state: int = 42):
    """Khởi tạo đúng 2 Pipeline AdaBoost cần so sánh."""
    base_estimator = lambda: AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state),
        n_estimators=100,
        learning_rate=1.0,
        random_state=random_state,
    )
    return {
        "AdaBoost_258": Pipeline([
            ("preprocess", "passthrough"),
            ("clf", base_estimator()),
        ]),
        "AdaBoost_SelectKBest_128": Pipeline([
            ("preprocess", "passthrough"),
            ("select", SelectKBest(score_func=f_classif, k=128)),
            ("clf", base_estimator()),
        ]),
    }


def compute_metrics(y_true, y_pred, scores):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
    auc = roc_auc_score(y_true, scores)
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
    X_train, y_train, train_meta, train_idx, feature_dict, cv_splits = load_and_verify_train_cv()
    configs = build_adaboost_configs(random_state=42)

    fold_records = []
    oof_records = []
    selected_feature_records = []
    total_fits = 0

    for config_name, base_pipe in configs.items():
        print(f"\n-> Đang chạy 5-fold CV cho cấu hình: {config_name}")
        for fold_num, (tr_pos, val_pos) in enumerate(cv_splits, start=1):
            X_tr_f, y_tr_f = X_train[tr_pos], y_train[tr_pos]
            X_val_f, y_val_f = X_train[val_pos], y_train[val_pos]

            pipe = clone(base_pipe)

            t_fit_0 = time.perf_counter()
            pipe.fit(X_tr_f, y_tr_f)
            fit_time = time.perf_counter() - t_fit_0
            total_fits += 1

            t_pred_0 = time.perf_counter()
            val_pred = pipe.predict(X_val_f)
            predict_val_time = time.perf_counter() - t_pred_0
            val_score = pipe.decision_function(X_val_f)

            tr_pred = pipe.predict(X_tr_f)
            tr_score = pipe.decision_function(X_tr_f)

            m_val = compute_metrics(y_val_f, val_pred, val_score)
            m_tr = compute_metrics(y_tr_f, tr_pred, tr_score)

            n_selected = int(X_tr_f.shape[1])
            if "select" in pipe.named_steps:
                sel = pipe.named_steps["select"]
                mask = sel.get_support()
                n_selected = int(mask.sum())
                assert n_selected == 128, f"Số đặc trưng được chọn khác 128: {n_selected}"
                scores_f = sel.scores_
                pvals_f = sel.pvalues_
                for f_idx in np.where(mask)[0]:
                    selected_feature_records.append({
                        "config": config_name,
                        "fold": fold_num,
                        "feature_index": int(f_idx),
                        "feature_name": feature_dict.loc[f_idx, "feature"],
                        "feature_type": feature_dict.loc[f_idx, "type"],
                        "block": int(feature_dict.loc[f_idx, "block"]),
                        "anova_f_score": float(scores_f[f_idx]),
                        "anova_p_value": float(pvals_f[f_idx]),
                    })

            fold_records.append({
                "config": config_name,
                "fold": fold_num,
                "train_samples": int(len(tr_pos)),
                "val_samples": int(len(val_pos)),
                "num_features_used": n_selected,
                "group_overlap": 0,
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
                    "score": float(val_score[local_i]),
                    "score_method": "decision_function",
                })

            print(
                f"   Fold {fold_num}: val_F1={m_val['f1']:.4f}, val_Rec={m_val['recall']:.4f}, "
                f"val_FPR={m_val['fpr']:.4f}, val_AUC={m_val['roc_auc']:.4f}, "
                f"FP={m_val['fp']}, FN={m_val['fn']}, fit_time={fit_time:.3f}s"
            )

    assert total_fits == 10, f"Tổng lượt fit phải là 10, thực tế={total_fits}"
    print(f"\n[OK] Hoàn tất đúng {total_fits} lượt fit (5 fold x 2 cấu hình).")

    df_folds = pd.DataFrame(fold_records)
    df_oof = pd.DataFrame(oof_records)
    df_sel_feats = pd.DataFrame(selected_feature_records)

    # Tổng hợp trung bình ± độ lệch chuẩn (ddof=0)
    metrics_list = ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "fp", "fn", "fit_time", "predict_val_time"]
    summary_rows = []
    for config_name in configs.keys():
        sub_f = df_folds[df_folds["config"] == config_name]
        for split_key, split_label in [("val", "validation_cv"), ("train", "train_cv")]:
            for m in metrics_list:
                if m in ["fit_time", "predict_val_time"]:
                    if split_key != "val":
                        continue
                    vals = sub_f[m].to_numpy(dtype=float)
                else:
                    vals = sub_f[f"{split_key}_{m}"].to_numpy(dtype=float)
                summary_rows.append({
                    "config": config_name,
                    "split": split_label,
                    "metric": m,
                    "mean": float(np.mean(vals)),
                    "std_ddof0": float(np.std(vals, ddof=0)),
                })

    df_summary = pd.DataFrame(summary_rows)

    # Lưu CSV kết quả vào week8/
    folds_path = WEEK8_DIR / "week8_tc3_adaboost_folds.csv"
    summary_path = WEEK8_DIR / "week8_tc3_adaboost_summary.csv"
    oof_path = WEEK8_DIR / "week8_tc3_adaboost_oof_predictions.csv"
    sel_feats_path = WEEK8_DIR / "week8_tc3_selected_features_by_fold.csv"

    df_folds.to_csv(folds_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    df_oof.to_csv(oof_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    df_sel_feats.to_csv(sel_feats_path, index=False, encoding="utf-8-sig", float_format="%.17g")

    # Đối chiếu cấu hình 258 đặc trưng với kết quả Tuần 6 đã lưu
    w6_folds = pd.read_csv(TABLES_DIR / "week6_models_cv_folds.csv")
    w6_ada_folds = w6_folds[w6_folds["model"] == "AdaBoost"].sort_values("fold").reset_index(drop=True)
    w8_258_folds = df_folds[df_folds["config"] == "AdaBoost_258"].sort_values("fold").reset_index(drop=True)

    w6_oof = pd.read_csv(TABLES_DIR / "week6_models_oof_predictions.csv", float_precision="round_trip")
    w6_ada_oof = w6_oof[w6_oof["model"] == "AdaBoost"].sort_values("train_row_index").reset_index(drop=True)
    w8_258_oof = df_oof[df_oof["config"] == "AdaBoost_258"].sort_values("train_row_index").reset_index(drop=True)

    assert (w6_ada_oof["train_row_index"].values == w8_258_oof["train_row_index"].values).all()
    assert (w6_ada_oof["global_row_index"].values == w8_258_oof["global_row_index"].values).all()
    assert (w6_ada_oof["cv_fold"].values == w8_258_oof["cv_fold"].values).all()

    pred_mismatches = int(np.sum(w6_ada_oof["y_pred"].values != w8_258_oof["y_pred"].values))
    max_score_abs_diff = float(np.max(np.abs(w6_ada_oof["score"].values - w8_258_oof["score"].values)))

    fold_diffs = {}
    for col in ["val_tn", "val_fp", "val_fn", "val_tp", "val_f1", "val_recall", "val_fpr", "val_roc_auc"]:
        fold_diffs[col] = float(np.max(np.abs(w6_ada_folds[col].values - w8_258_folds[col].values)))

    fit_time_diffs = [
        {
            "fold": int(f),
            "week6_fit_time_s": float(w6_t),
            "week8_fit_time_s": float(w8_t),
            "diff_s": float(w8_t - w6_t),
        }
        for f, w6_t, w8_t in zip(w6_ada_folds["fold"], w6_ada_folds["fit_time"], w8_258_folds["fit_time"])
    ]

    # Thống kê số đặc trưng ổn định qua 5 fold của k=128
    freq = df_sel_feats.groupby("feature_name")["fold"].count()
    stable_5of5 = int((freq == 5).sum())
    unique_selected = int(len(freq))

    reconciliation = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_fits_executed": total_fits,
        "oof_samples_compared": int(len(w8_258_oof)),
        "oof_y_pred_mismatches": pred_mismatches,
        "oof_decision_score_max_abs_diff": max_score_abs_diff,
        "fold_metric_max_abs_diffs": fold_diffs,
        "fit_time_comparison_258": fit_time_diffs,
        "selectkbest_k128_stability": {
            "unique_features_selected_across_5_folds": unique_selected,
            "features_selected_in_all_5_folds": stable_5of5,
        },
        "artifacts": {
            "folds_csv": str(folds_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "summary_csv": str(summary_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "oof_csv": str(oof_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "selected_features_csv": str(sel_feats_path.relative_to(BASE_DIR)).replace("\\", "/"),
        },
    }

    reconcil_path = WEEK8_DIR / "week8_tc3_week6_reconciliation.json"
    with open(reconcil_path, "w", encoding="utf-8") as f:
        json.dump(reconciliation, f, indent=2, ensure_ascii=False)

    print("\n=== ĐỐI CHIẾU ADABOOST 258 ĐẶC TRƯNG VỚI TUẦN 6 ===")
    print(f"  OOF y_pred mismatches : {pred_mismatches} / {len(w8_258_oof)}")
    print(f"  Max abs diff (score)  : {max_score_abs_diff:.2e}")
    print(f"  Fold metric max diffs : {fold_diffs}")
    print(f"  SelectKBest stability : {stable_5of5}/128 đặc trưng xuất hiện ở cả 5 fold (tổng {unique_selected} đặc trưng duy nhất)")

    print("\n=== BẢNG TỔNG HỢP KẾT QUẢ (VALIDATION CV, MEAN ± STD ddof=0) ===")
    val_sum = df_summary[df_summary["split"] == "validation_cv"].pivot(index="config", columns="metric", values="mean")
    val_std = df_summary[df_summary["split"] == "validation_cv"].pivot(index="config", columns="metric", values="std_ddof0")
    for cfg in configs.keys():
        print(f"\n[{cfg}]")
        for m in ["f1", "recall", "fpr", "roc_auc", "fp", "fn", "fit_time", "predict_val_time"]:
            print(f"  {m:16s}: {val_sum.loc[cfg, m]:.6f} ± {val_std.loc[cfg, m]:.6f}")

    # In F1 train_cv để xem Delta F1
    tr_sum = df_summary[(df_summary["split"] == "train_cv") & (df_summary["metric"] == "f1")].set_index("config")
    print("\n[Train CV F1 (Mean ± Std ddof=0)]")
    for cfg in configs.keys():
        f1_tr_m = tr_sum.loc[cfg, "mean"]
        f1_tr_s = tr_sum.loc[cfg, "std_ddof0"]
        delta = f1_tr_m - val_sum.loc[cfg, "f1"]
        print(f"  {cfg:26s}: train_F1={f1_tr_m:.6f} ± {f1_tr_s:.6f} | delta_F1={delta:.6f}")


if __name__ == "__main__":
    main()
