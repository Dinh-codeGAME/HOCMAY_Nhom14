"""
run_tc5_dt_nested_gridsearch.py - Thí nghiệm Tiêu chí 5 Tuần 8:
Nested Group Cross-Validation (5 outer folds x 3 inner folds) với GridSearchCV
cho Decision Tree trên đủ 258 đặc trưng Haar-like và random_state=42.

Lưới tìm kiếm (9 tổ hợp):
- clf__max_depth: [3, 5, 7]
- clf__min_samples_leaf: [2, 5, 10]

Giao thức chống rò rỉ:
- Vòng ngoài (5 folds): dùng đúng week6_cv_assignments.csv trên 2.511 mẫu train.
- Vòng trong (3 folds): StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
  trên phần train của từng fold ngoài, kiểm tra giao thoa nhóm ở cả 2 vòng.
- Chọn tham số theo F1 trung bình vòng trong (scoring='f1'), refit=True.
- Tổng số lượt fit: 5 * (9 * 3 + 1) = 140 lượt.
- Không chạm tập validation giữ riêng (540) và tập test (538).
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
TABLES_DIR = BASE_DIR / "reports" / "tables"
WEEK8_DIR = BASE_DIR / "week8"
WEEK8_DIR.mkdir(parents=True, exist_ok=True)


def load_and_verify_outer_cv():
    """Nạp 2.511 mẫu train và đối chiếu toàn vẹn 5 outer folds từ week6_cv_assignments.csv."""
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

    outer_folds_col = cv_assign["cv_fold"].to_numpy()
    outer_splits = []
    for fold_num in range(1, 6):
        tr_pos = np.where(outer_folds_col != fold_num)[0]
        val_pos = np.where(outer_folds_col == fold_num)[0]
        assert len(set(tr_pos) & set(val_pos)) == 0
        assert len(set(train_meta.iloc[tr_pos]["group"]) & set(train_meta.iloc[val_pos]["group"])) == 0
        assert len(np.unique(y_train[tr_pos])) == 2 and len(np.unique(y_train[val_pos])) == 2
        outer_splits.append((tr_pos, val_pos))

    print("[OK] Đã xác minh 2.511 mẫu train (258 đặc trưng) và 5 outer fold (0 giao thoa nhóm).")
    return X_train, y_train, train_meta, outer_splits


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
    X_train, y_train, train_meta, outer_splits = load_and_verify_outer_cv()

    param_grid = {
        "clf__max_depth": [3, 5, 7],
        "clf__min_samples_leaf": [2, 5, 10],
    }
    n_combos = len(param_grid["clf__max_depth"]) * len(param_grid["clf__min_samples_leaf"])
    assert n_combos == 9

    inner_assign_records = []
    grid_records = []
    outer_fold_records = []
    oof_records = []
    total_fits = 0

    for outer_fold, (tr_pos, val_pos) in enumerate(outer_splits, start=1):
        X_tr_out, y_tr_out = X_train[tr_pos], y_train[tr_pos]
        X_val_out, y_val_out = X_train[val_pos], y_train[val_pos]
        groups_tr_out = train_meta.iloc[tr_pos]["group"].to_numpy()

        # Tạo và thẩm định 3 fold trong bằng StratifiedGroupKFold
        sgkf_inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splits = []
        inner_fold_labels = np.zeros(len(tr_pos), dtype=int)

        for in_f, (in_tr_idx, in_val_idx) in enumerate(sgkf_inner.split(X_tr_out, y_tr_out, groups=groups_tr_out), start=1):
            assert len(set(in_tr_idx) & set(in_val_idx)) == 0, f"Outer {outer_fold} Inner {in_f}: trùng mẫu!"
            g_in_tr = set(groups_tr_out[in_tr_idx])
            g_in_val = set(groups_tr_out[in_val_idx])
            assert len(g_in_tr & g_in_val) == 0, f"Outer {outer_fold} Inner {in_f}: giao thoa nhóm!"
            assert len(np.unique(y_tr_out[in_tr_idx])) == 2 and len(np.unique(y_tr_out[in_val_idx])) == 2
            inner_splits.append((in_tr_idx, in_val_idx))
            inner_fold_labels[in_val_idx] = in_f

        assert (inner_fold_labels > 0).all(), f"Outer {outer_fold}: còn mẫu chưa gán inner fold!"

        for local_i, tr_row_idx in enumerate(tr_pos):
            inner_assign_records.append({
                "outer_fold": outer_fold,
                "outer_train_local_idx": local_i,
                "train_row_index": int(tr_row_idx),
                "global_row_index": int(train_meta.loc[tr_row_idx, "global_row_index"]),
                "source_id": train_meta.loc[tr_row_idx, "source_id"],
                "group": train_meta.loc[tr_row_idx, "group"],
                "label": int(y_tr_out[local_i]),
                "inner_fold": int(inner_fold_labels[local_i]),
            })

        # Thiết lập Pipeline và GridSearchCV
        pipe = Pipeline([
            ("preprocess", "passthrough"),
            ("clf", DecisionTreeClassifier(random_state=42)),
        ])

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="f1",
            cv=inner_splits,
            refit=True,
            n_jobs=1,
            return_train_score=True,
        )

        t0 = time.perf_counter()
        grid.fit(X_tr_out, y_tr_out)
        search_and_refit_time = time.perf_counter() - t0

        # Kiểm đếm chính xác số lượt fit: 9 tổ hợp x 3 fold + 1 refit = 28
        n_cv_fits = len(grid.cv_results_["params"]) * len(inner_splits)
        total_fits += n_cv_fits + 1

        best_params = grid.best_params_
        best_inner_f1 = float(grid.best_score_)
        best_pipe = grid.best_estimator_
        best_clf = best_pipe.named_steps["clf"]
        actual_depth = int(best_clf.get_depth())
        actual_n_leaves = int(best_clf.get_n_leaves())
        c1_idx = list(best_clf.classes_).index(1)
        score_method = f"predict_proba[:, {c1_idx}]"

        # Lưu kết quả của cả 9 tổ hợp ở vòng trong
        cv_res = grid.cv_results_
        for cand_i, params in enumerate(cv_res["params"]):
            is_sel = (
                params["clf__max_depth"] == best_params["clf__max_depth"]
                and params["clf__min_samples_leaf"] == best_params["clf__min_samples_leaf"]
            )
            grid_records.append({
                "outer_fold": outer_fold,
                "clf__max_depth": int(params["clf__max_depth"]),
                "clf__min_samples_leaf": int(params["clf__min_samples_leaf"]),
                "inner_mean_val_f1": float(cv_res["mean_test_score"][cand_i]),
                "inner_std_val_f1": float(cv_res["std_test_score"][cand_i]),
                "inner_mean_train_f1": float(cv_res["mean_train_score"][cand_i]),
                "inner_std_train_f1": float(cv_res["std_train_score"][cand_i]),
                "inner_fold1_val_f1": float(cv_res["split0_test_score"][cand_i]),
                "inner_fold2_val_f1": float(cv_res["split1_test_score"][cand_i]),
                "inner_fold3_val_f1": float(cv_res["split2_test_score"][cand_i]),
                "rank_inner_val_f1": int(cv_res["rank_test_score"][cand_i]),
                "inner_mean_fit_time_s": float(cv_res["mean_fit_time"][cand_i]),
                "is_selected": bool(is_sel),
            })

        # Đánh giá trên phần kiểm định của fold ngoài
        t_pred0 = time.perf_counter()
        val_pred = best_pipe.predict(X_val_out)
        predict_val_time = time.perf_counter() - t_pred0
        val_prob1 = best_pipe.predict_proba(X_val_out)[:, c1_idx]

        tr_pred = best_pipe.predict(X_tr_out)
        tr_prob1 = best_pipe.predict_proba(X_tr_out)[:, c1_idx]

        m_val = compute_metrics(y_val_out, val_pred, val_prob1)
        m_tr = compute_metrics(y_tr_out, tr_pred, tr_prob1)
        delta_f1 = m_tr["f1"] - m_val["f1"]

        outer_fold_records.append({
            "outer_fold": outer_fold,
            "outer_train_samples": int(len(tr_pos)),
            "outer_val_samples": int(len(val_pos)),
            "outer_group_overlap": 0,
            "inner_folds_count": len(inner_splits),
            "inner_max_group_overlap": 0,
            "selected_max_depth": int(best_params["clf__max_depth"]),
            "selected_min_samples_leaf": int(best_params["clf__min_samples_leaf"]),
            "inner_best_cv_f1": best_inner_f1,
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
            "train_f1": m_tr["f1"],
            "train_recall": m_tr["recall"],
            "train_fpr": m_tr["fpr"],
            "train_roc_auc": m_tr["roc_auc"],
            "delta_f1_train_minus_val": float(delta_f1),
            "search_and_refit_time_s": float(search_and_refit_time),
            "refit_time_s": float(grid.refit_time_),
            "predict_val_time_s": float(predict_val_time),
        })

        for local_i, tr_row_idx in enumerate(val_pos):
            oof_records.append({
                "model": "DecisionTree_NestedGridSearchCV",
                "outer_fold": outer_fold,
                "train_row_index": int(tr_row_idx),
                "global_row_index": int(train_meta.loc[tr_row_idx, "global_row_index"]),
                "source_id": train_meta.loc[tr_row_idx, "source_id"],
                "group": train_meta.loc[tr_row_idx, "group"],
                "selected_max_depth": int(best_params["clf__max_depth"]),
                "selected_min_samples_leaf": int(best_params["clf__min_samples_leaf"]),
                "y_true": int(y_val_out[local_i]),
                "y_pred": int(val_pred[local_i]),
                "score": float(val_prob1[local_i]),
                "score_method": score_method,
            })

        print(
            f"Outer Fold {outer_fold}: selected=(max_depth={best_params['clf__max_depth']}, "
            f"min_samples_leaf={best_params['clf__min_samples_leaf']}) | "
            f"inner_F1={best_inner_f1:.4f} | outer_val_F1={m_val['f1']:.4f} | "
            f"Rec={m_val['recall']:.4f}, FPR={m_val['fpr']:.4f}, AUC={m_val['roc_auc']:.4f}, "
            f"FP={m_val['fp']}, FN={m_val['fn']}, depth={actual_depth}, leaves={actual_n_leaves}, "
            f"time={search_and_refit_time:.2f}s"
        )

    assert total_fits == 140, f"Tổng số lượt fit phải bằng 140, thực tế={total_fits}"
    print(f"\n[OK] Hoàn tất đúng {total_fits} lượt fit (5 outer x [9 tổ hợp x 3 inner + 1 refit]).")

    df_inner = pd.DataFrame(inner_assign_records)
    df_grid = pd.DataFrame(grid_records)
    df_outer = pd.DataFrame(outer_fold_records)
    df_oof = pd.DataFrame(oof_records)

    assert len(df_grid) == 45
    assert len(df_outer) == 5
    assert len(df_oof) == 2511
    assert df_oof["train_row_index"].nunique() == 2511

    # Tổng hợp Mean ± Std với ddof=0 trên 5 outer folds
    summary_metrics = [
        ("outer_val", "f1", "val_f1"),
        ("outer_val", "recall", "val_recall"),
        ("outer_val", "fpr", "val_fpr"),
        ("outer_val", "roc_auc", "val_roc_auc"),
        ("outer_val", "accuracy", "val_accuracy"),
        ("outer_val", "precision", "val_precision"),
        ("outer_val", "fp", "val_fp"),
        ("outer_val", "fn", "val_fn"),
        ("outer_train", "f1", "train_f1"),
        ("outer_gap", "delta_f1_train_minus_val", "delta_f1_train_minus_val"),
        ("inner_selection", "inner_best_cv_f1", "inner_best_cv_f1"),
        ("complexity", "actual_depth", "actual_depth"),
        ("complexity", "actual_n_leaves", "actual_n_leaves"),
        ("timing", "search_and_refit_time_s", "search_and_refit_time_s"),
        ("timing", "refit_time_s", "refit_time_s"),
        ("timing", "predict_val_time_s", "predict_val_time_s"),
    ]
    summary_rows = []
    for scope, m_name, col in summary_metrics:
        vals = df_outer[col].to_numpy(dtype=float)
        summary_rows.append({
            "model": "DecisionTree_NestedGridSearchCV",
            "scope": scope,
            "metric": m_name,
            "mean": float(np.mean(vals)),
            "std_ddof0": float(np.std(vals, ddof=0)),
            "sum": float(np.sum(vals)),
        })
    df_summary = pd.DataFrame(summary_rows)

    inner_path = WEEK8_DIR / "week8_tc5_inner_cv_assignments.csv"
    grid_path = WEEK8_DIR / "week8_tc5_grid_results.csv"
    outer_path = WEEK8_DIR / "week8_tc5_outer_folds.csv"
    summary_path = WEEK8_DIR / "week8_tc5_summary.csv"
    oof_path = WEEK8_DIR / "week8_tc5_oof_predictions.csv"

    df_inner.to_csv(inner_path, index=False, encoding="utf-8-sig")
    df_grid.to_csv(grid_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    df_outer.to_csv(outer_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    df_oof.to_csv(oof_path, index=False, encoding="utf-8-sig", float_format="%.17g")

    print("\n=== TỔNG HỢP KẾT QUẢ NESTED GROUP CV (MEAN ± STD ddof=0) ===")
    for _, r in df_summary.iterrows():
        extra = f" (tổng={int(r['sum'])})" if r["metric"] in ["fp", "fn"] else ""
        print(f"  [{r['scope']:15s}] {r['metric']:25s}: {r['mean']:.6f} ± {r['std_ddof0']:.6f}{extra}")


if __name__ == "__main__":
    main()
