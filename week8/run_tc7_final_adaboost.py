"""
run_tc7_final_adaboost.py - Thực hiện Tiêu chí 7 Tuần 8:
Chốt mô hình AdaBoost (258 đặc trưng Haar-like) trước bước đánh giá test
vì có Recall CV trung bình cao nhất trong số các mô hình đạt cả 3 mục tiêu CV.
(Ghi rõ SVM RBF có F1 cao hơn và FPR thấp hơn; lựa chọn AdaBoost ưu tiên Recall.)

Quy tắc đánh giá:
- Dùng đúng Pipeline AdaBoost tuần 6, fit trên toàn bộ 2.511 mẫu train.
- Đánh giá lại trên cùng 538 mẫu tập test giữ riêng đã dùng ở Tuần 6 bằng Pipeline.predict().
- Tính ROC-AUC từ decision_function().
- Giữ nguyên ngưỡng và toàn bộ kết quả Tuần 6.
- Lập bảng CV (Mean ± Std ddof=0) và Test cho F1, Recall, FPR, ROC-AUC, kèm FP/FN.
- Nhận xét độ ổn định qua 5 fold và đối chiếu kết quả test với dòng AdaBoost mặc định Tuần 6.
"""

from datetime import datetime, timezone
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
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
    # 1. Nạp và đối chiếu dữ liệu Train (2.511) và Test giữ riêng (538)
    X_all = np.load(DATA_DIR / "haar_features.npy")
    y_all = np.load(DATA_DIR / "labels.npy")
    meta_all = pd.read_csv(DATA_DIR / "metadata.csv")
    splits = np.load(DATA_DIR / "split_indices.npz")
    haar_splits = np.load(DATA_DIR / "haar_dataset_splits.npz")

    train_idx = splits["train"]
    test_idx = splits["test"]
    assert len(train_idx) == 2511
    assert len(test_idx) == 538

    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    assert np.array_equal(X_train, haar_splits["X_train"])
    assert np.array_equal(y_train, haar_splits["y_train"])
    assert np.array_equal(X_test, haar_splits["X_test"])
    assert np.array_equal(y_test, haar_splits["y_test"])

    meta_test = meta_all.iloc[test_idx].copy().reset_index(drop=False)
    meta_test.rename(columns={"index": "global_row_index"}, inplace=True)

    # Kiểm tra không giao thoa nhóm train-test
    g_tr = set(meta_all.iloc[train_idx]["group"])
    g_te = set(meta_all.iloc[test_idx]["group"])
    assert len(g_tr & g_te) == 0, "Phát hiện giao thoa nhóm giữa Train và Test!"

    # 2. Tổng hợp bằng chứng CV (từ week6_models_cv_folds.csv & week6_models_cv_summary.csv)
    cv_folds_all = pd.read_csv(TABLES_DIR / "week6_models_cv_folds.csv")
    ada_cv_folds = cv_folds_all[cv_folds_all["model"] == "AdaBoost"].sort_values("fold").reset_index(drop=True)
    assert len(ada_cv_folds) == 5

    # Tính thống kê ổn định qua 5 fold (ddof=0)
    stability_rows = []
    for _, r in ada_cv_folds.iterrows():
        stability_rows.append({
            "row_type": "fold",
            "fold": str(int(r["fold"])),
            "val_samples": int(r["val_samples"]),
            "val_fp": int(r["val_fp"]),
            "val_fn": int(r["val_fn"]),
            "val_f1": float(r["val_f1"]),
            "val_recall": float(r["val_recall"]),
            "val_fpr": float(r["val_fpr"]),
            "val_roc_auc": float(r["val_roc_auc"]),
            "fit_time": float(r["fit_time"]),
        })

    for stat_name, fn_agg in [
        ("mean", lambda s: float(np.mean(s))),
        ("std_ddof0", lambda s: float(np.std(s, ddof=0))),
        ("min", lambda s: float(np.min(s))),
        ("max", lambda s: float(np.max(s))),
        ("range_max_minus_min", lambda s: float(np.max(s) - np.min(s))),
    ]:
        stability_rows.append({
            "row_type": "summary",
            "fold": stat_name,
            "val_samples": int(np.sum(ada_cv_folds["val_samples"])) if stat_name == "mean" else 0,
            "val_fp": fn_agg(ada_cv_folds["val_fp"].to_numpy(dtype=float)),
            "val_fn": fn_agg(ada_cv_folds["val_fn"].to_numpy(dtype=float)),
            "val_f1": fn_agg(ada_cv_folds["val_f1"].to_numpy(dtype=float)),
            "val_recall": fn_agg(ada_cv_folds["val_recall"].to_numpy(dtype=float)),
            "val_fpr": fn_agg(ada_cv_folds["val_fpr"].to_numpy(dtype=float)),
            "val_roc_auc": fn_agg(ada_cv_folds["val_roc_auc"].to_numpy(dtype=float)),
            "fit_time": fn_agg(ada_cv_folds["fit_time"].to_numpy(dtype=float)),
        })

    df_stability = pd.DataFrame(stability_rows)
    stability_path = WEEK8_DIR / "week8_tc7_adaboost_cv_stability.csv"
    df_stability.to_csv(stability_path, index=False, encoding="utf-8-sig", float_format="%.17g")

    # 3. Huấn luyện lại đúng Pipeline AdaBoost tuần 6 trên 2.511 mẫu train & đánh giá trên 538 mẫu test
    ada_pipeline = Pipeline([
        ("preprocess", "passthrough"),
        ("clf", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=100,
            learning_rate=1.0,
            random_state=42,
        )),
    ])

    t0 = time.perf_counter()
    ada_pipeline.fit(X_train, y_train)
    fit_full_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_test_pred = ada_pipeline.predict(X_test)
    predict_test_time = time.perf_counter() - t1
    test_scores = ada_pipeline.decision_function(X_test)

    m_test = compute_metrics(y_test, y_test_pred, test_scores)

    # Lưu 538 dự đoán trên tập Test
    test_pred_rows = []
    for i in range(len(test_idx)):
        test_pred_rows.append({
            "model": "AdaBoost_258_Final",
            "test_row_index": i,
            "global_row_index": int(meta_test.loc[i, "global_row_index"]),
            "source_id": meta_test.loc[i, "source_id"],
            "group": meta_test.loc[i, "group"],
            "y_true": int(y_test[i]),
            "y_pred": int(y_test_pred[i]),
            "score": float(test_scores[i]),
            "score_method": "decision_function",
            "decision_rule": "Pipeline.predict()",
        })
    df_test_preds = pd.DataFrame(test_pred_rows)
    test_preds_path = WEEK8_DIR / "week8_tc7_test_predictions.csv"
    df_test_preds.to_csv(test_preds_path, index=False, encoding="utf-8-sig", float_format="%.17g")

    # 4. Lập bảng CV (Mean ± Std ddof=0) và Kết quả Test
    cv_f1_m = float(np.mean(ada_cv_folds["val_f1"]))
    cv_f1_s = float(np.std(ada_cv_folds["val_f1"], ddof=0))
    cv_rec_m = float(np.mean(ada_cv_folds["val_recall"]))
    cv_rec_s = float(np.std(ada_cv_folds["val_recall"], ddof=0))
    cv_fpr_m = float(np.mean(ada_cv_folds["val_fpr"]))
    cv_fpr_s = float(np.std(ada_cv_folds["val_fpr"], ddof=0))
    cv_auc_m = float(np.mean(ada_cv_folds["val_roc_auc"]))
    cv_auc_s = float(np.std(ada_cv_folds["val_roc_auc"], ddof=0))
    cv_fp_sum = int(np.sum(ada_cv_folds["val_fp"]))
    cv_fn_sum = int(np.sum(ada_cv_folds["val_fn"]))
    cv_fp_m = float(np.mean(ada_cv_folds["val_fp"]))
    cv_fp_s = float(np.std(ada_cv_folds["val_fp"], ddof=0))
    cv_fn_m = float(np.mean(ada_cv_folds["val_fn"]))
    cv_fn_s = float(np.std(ada_cv_folds["val_fn"], ddof=0))
    cv_fit_m = float(np.mean(ada_cv_folds["fit_time"]))
    cv_fit_s = float(np.std(ada_cv_folds["fit_time"], ddof=0))

    comparison_rows = [
        {
            "stage": "CV_5Fold_Train_OOF",
            "dataset_note": "2.511 mẫu train (5 fold theo nhóm, ngoài fold)",
            "decision_rule": "Pipeline.predict()",
            "f1_mean_or_val": cv_f1_m,
            "f1_std_ddof0": cv_f1_s,
            "recall_mean_or_val": cv_rec_m,
            "recall_std_ddof0": cv_rec_s,
            "fpr_mean_or_val": cv_fpr_m,
            "fpr_std_ddof0": cv_fpr_s,
            "roc_auc_mean_or_val": cv_auc_m,
            "roc_auc_std_ddof0": cv_auc_s,
            "fp_total": cv_fp_sum,
            "fn_total": cv_fn_sum,
            "fp_per_fold_mean": cv_fp_m,
            "fp_per_fold_std_ddof0": cv_fp_s,
            "fn_per_fold_mean": cv_fn_m,
            "fn_per_fold_std_ddof0": cv_fn_s,
            "fit_time_s": cv_fit_m,
            "fit_time_std_ddof0": cv_fit_s,
        },
        {
            "stage": "Test_Heldout_Reevaluation",
            "dataset_note": "538 mẫu test giữ riêng (đánh giá lại trên cùng tập test đã dùng ở Tuần 6)",
            "decision_rule": "Pipeline.predict()",
            "f1_mean_or_val": m_test["f1"],
            "f1_std_ddof0": np.nan,
            "recall_mean_or_val": m_test["recall"],
            "recall_std_ddof0": np.nan,
            "fpr_mean_or_val": m_test["fpr"],
            "fpr_std_ddof0": np.nan,
            "roc_auc_mean_or_val": m_test["roc_auc"],
            "roc_auc_std_ddof0": np.nan,
            "fp_total": m_test["fp"],
            "fn_total": m_test["fn"],
            "fp_per_fold_mean": np.nan,
            "fp_per_fold_std_ddof0": np.nan,
            "fn_per_fold_mean": np.nan,
            "fn_per_fold_std_ddof0": np.nan,
            "fit_time_s": fit_full_time,
            "fit_time_std_ddof0": np.nan,
        },
    ]
    df_comp = pd.DataFrame(comparison_rows)
    comp_path = WEEK8_DIR / "week8_tc7_cv_vs_test_comparison.csv"
    df_comp.to_csv(comp_path, index=False, encoding="utf-8-sig", float_format="%.17g")

    # 5. Đối chiếu với dòng AdaBoost mặc định của Tuần 6 (week6_test_results.csv & week6_test_predictions.csv)
    w6_test_res = pd.read_csv(TABLES_DIR / "week6_test_results.csv")
    w6_ada_def = w6_test_res[(w6_test_res["model"] == "AdaBoost") & (w6_test_res["mode"] == "default")].iloc[0]

    w6_test_preds = pd.read_csv(TABLES_DIR / "week6_test_predictions.csv", float_precision="round_trip")
    w6_ada_preds = w6_test_preds[w6_test_preds["model"] == "AdaBoost"].sort_values("test_row_index").reset_index(drop=True)
    w8_ada_preds = df_test_preds.sort_values("test_row_index").reset_index(drop=True)

    assert (w6_ada_preds["global_row_index"].values == w8_ada_preds["global_row_index"].values).all()
    test_pred_mismatches = int(np.sum(w6_ada_preds["y_pred_default"].values != w8_ada_preds["y_pred"].values))
    test_score_max_diff = float(np.max(np.abs(w6_ada_preds["score"].values - w8_ada_preds["score"].values)))

    # Tham chiếu thêm SVM RBF và Random Forest ở CV để làm rõ lý do chọn AdaBoost
    cv_sum_w6 = pd.read_csv(TABLES_DIR / "week6_models_cv_summary.csv")
    cv_selection_table = {}
    for model_name in ["AdaBoost", "SVM RBF", "Random Forest"]:
        sub = cv_sum_w6[(cv_sum_w6["model"] == model_name) & (cv_sum_w6["split"] == "validation_cv")].set_index("metric")
        cv_selection_table[model_name] = {
            "recall_mean": float(sub.loc["recall", "mean"]),
            "recall_std": float(sub.loc["recall", "std"]),
            "fpr_mean": float(sub.loc["fpr", "mean"]),
            "fpr_std": float(sub.loc["fpr", "std"]),
            "f1_mean": float(sub.loc["f1", "mean"]),
            "f1_std": float(sub.loc["f1", "std"]),
            "roc_auc_mean": float(sub.loc["roc_auc", "mean"]),
            "roc_auc_std": float(sub.loc["roc_auc", "std"]),
        }

    reconciliation = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rationale": (
            "Chốt AdaBoost (258 đặc trưng) trước bước đánh giá test vì có Recall CV trung bình cao nhất (0.9702) "
            "trong số các mô hình đạt cả 3 mục tiêu CV (Recall >= 0.90, FPR <= 0.05, ROC-AUC). "
            "SVM RBF có F1 CV cao hơn (0.9703 vs 0.9656) và FPR CV thấp hơn (0.0227 vs 0.0405), "
            "còn lựa chọn AdaBoost ưu tiên Recall theo mục tiêu không bỏ sót khuôn mặt của đề tài."
        ),
        "test_set_usage_disclosure": "Đánh giá lại trên cùng 538 mẫu tập test giữ riêng đã dùng ở Tuần 6 bằng Pipeline.predict(), giữ nguyên ngưỡng và kết quả Tuần 6.",
        "cv_models_meeting_all_3_goals": cv_selection_table,
        "test_metrics_exact": m_test,
        "fit_full_train_time_s": float(fit_full_time),
        "predict_test_time_s": float(predict_test_time),
        "week6_default_test_comparison": {
            "week6_row": {
                "tn": int(w6_ada_def["tn"]),
                "fp": int(w6_ada_def["fp"]),
                "fn": int(w6_ada_def["fn"]),
                "tp": int(w6_ada_def["tp"]),
                "f1_rounded_4": float(w6_ada_def["f1"]),
                "recall_rounded_4": float(w6_ada_def["recall"]),
                "fpr_rounded_4": float(w6_ada_def["fpr"]),
                "roc_auc_rounded_4": float(w6_ada_def["roc_auc"]),
            },
            "week8_row_rounded_4": {
                "tn": m_test["tn"],
                "fp": m_test["fp"],
                "fn": m_test["fn"],
                "tp": m_test["tp"],
                "f1_rounded_4": round(m_test["f1"], 4),
                "recall_rounded_4": round(m_test["recall"], 4),
                "fpr_rounded_4": round(m_test["fpr"], 4),
                "roc_auc_rounded_4": round(m_test["roc_auc"], 4),
            },
            "test_y_pred_mismatches": test_pred_mismatches,
            "test_decision_score_max_abs_diff": test_score_max_diff,
        },
    }

    reconcil_path = WEEK8_DIR / "week8_tc7_reconciliation.json"
    with open(reconcil_path, "w", encoding="utf-8") as f:
        json.dump(reconciliation, f, indent=2, ensure_ascii=False)

    print("=== KẾT QUẢ TIÊU CHÍ 7 (ADABOOST 258 ĐẶC TRƯNG) ===")
    print(f"CV (5 fold): F1={cv_f1_m:.4f} ± {cv_f1_s:.4f}, Recall={cv_rec_m:.4f} ± {cv_rec_s:.4f}, FPR={cv_fpr_m:.4f} ± {cv_fpr_s:.4f}, AUC={cv_auc_m:.4f} ± {cv_auc_s:.4f}, OOF FP={cv_fp_sum}, OOF FN={cv_fn_sum}")
    print(f"Test (538 mẫu, Pipeline.predict()): TN={m_test['tn']}, FP={m_test['fp']}, FN={m_test['fn']}, TP={m_test['tp']}")
    print(f"  F1={m_test['f1']:.4f}, Recall={m_test['recall']:.4f}, FPR={m_test['fpr']:.4f}, ROC-AUC={m_test['roc_auc']:.4f}, Acc={m_test['accuracy']:.4f}")
    print(f"  Fit train (2.511 mẫu): {fit_full_time:.3f}s | Predict test (538 mẫu): {predict_test_time:.4f}s")
    print(f"Đối chiếu với AdaBoost mặc định Tuần 6: mismatches={test_pred_mismatches}/538, max_score_diff={test_score_max_diff:.2e}")


if __name__ == "__main__":
    main()
