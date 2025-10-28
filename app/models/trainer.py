import os, uuid, time
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from joblib import dump

from app.core.config import ARTIFACTS_DIR
from app.core.utils import now_ts
from app.models.artifacts import save_artifacts


def train_pipeline(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,          # 최종 검증(valid) 비율 (temp 중 절반은 calib로 분할)
    random_state: int = 42,
    feature_select_threshold: str = "median",
    learning_rate: float = 0.03,
    n_estimators: int = 3000,
    num_leaves: int = 63,
    calibrator: str = "none",        # 'isotonic' | 'sigmoid' | 'none'
    reg_lambda: float = 10.0,        # 약간 강한 정규화로 과출력 완화
    reg_alpha: float = 1.0,
    min_data_in_leaf: int = 50,
    min_sum_hessian_in_leaf: float = 5.0,
) -> Dict[str, Any]:

    print("\n================= [TRAIN START] =================")
    print(f"[CONFIG] target='{target}', test_size={test_size}, random_state={random_state}")
    print(f"[CONFIG] feature_select_threshold='{feature_select_threshold}', calibrator='{calibrator}'")
    print(f"[CONFIG] params: lr={learning_rate}, n_estimators={n_estimators}, num_leaves={num_leaves}, "
          f"lambda={reg_lambda}, alpha={reg_alpha}, min_data_in_leaf={min_data_in_leaf}, "
          f"min_sum_hessian_in_leaf={min_sum_hessian_in_leaf}")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    print(f"[DATA] df shape={df.shape}, columns={list(df.columns)}")
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    base_rate = float(y.mean())
    print(f"[DATA] base_rate(positive ratio)={base_rate:.6f}")

    # ---------- 1) 세 분할: Train / Calib / Valid ----------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_calib, X_valid, y_calib, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )
    print(f"[SPLIT] train={len(y_train)}, calib={len(y_calib)}, valid={len(y_valid)} "
          f"(total={len(y)})")

    # ---------- 2) 인코더 ----------
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    print(f"[COLS] num_cols({len(num_cols)}): {num_cols}")
    print(f"[COLS] cat_cols({len(cat_cols)}): {cat_cols}")

    encoder = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    X_train_enc = encoder.fit_transform(X_train)
    X_calib_enc = encoder.transform(X_calib)
    X_valid_enc = encoder.transform(X_valid)
    enc_feature_names = num_cols + cat_cols

    print(f"[ENC] X_train_enc shape={getattr(X_train_enc,'shape',None)}, "
          f"X_calib_enc shape={getattr(X_calib_enc,'shape',None)}, "
          f"X_valid_enc shape={getattr(X_valid_enc,'shape',None)}")
    print(f"[ENC] enc_feature_names({len(enc_feature_names)}) ok")

    # ---------- 3) 1차 모델 ----------
    base = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        min_data_in_leaf=min_data_in_leaf,
        min_sum_hessian_in_leaf=min_sum_hessian_in_leaf,
        random_state=random_state,
        metric="auc",
    )

    print("[FIT:BASE] fitting base model with early_stopping on VALID...")
    base.fit(
        X_train_enc, y_train,
        eval_set=[(X_valid_enc, y_valid)],
        callbacks=[early_stopping(stopping_rounds=200), log_evaluation(period=0)],
    )
    auc_all = roc_auc_score(y_valid, base.predict_proba(X_valid_enc)[:, 1])
    print(f"[METRIC:BASE] AUC(all features)={auc_all:.6f}")

    # importance 상위 10개 출력(참고)
    try:
        imp = base.feature_importances_
        idx_top10 = np.argsort(-imp)[:10]
        top10 = [(enc_feature_names[i], int(imp[i])) for i in idx_top10]
        print(f"[IMP:BASE] top10={top10}")
    except Exception as e:
        print(f"[IMP:BASE] importance print failed: {e}")

    # ---------- 4) 특성 선택 ----------
    print(f"[SEL] SelectFromModel threshold='{feature_select_threshold}'")
    selector = SelectFromModel(estimator=base, threshold=feature_select_threshold, prefit=True)
    X_train_sel = selector.transform(X_train_enc)
    X_calib_sel = selector.transform(X_calib_enc)
    X_valid_sel = selector.transform(X_valid_enc)

    selected_mask = selector.get_support()
    selected_features = [f for f, keep in zip(enc_feature_names, selected_mask) if keep]
    print(f"[SEL] selected_features({len(selected_features)}): {selected_features}")
    print(f"[SEL] shapes → train={X_train_sel.shape}, calib={X_calib_sel.shape}, valid={X_valid_sel.shape}")

    if X_train_sel.shape[1] == 0:
        print("[WARN] No features selected! Fallback to all encoded features.")
        X_train_sel, X_calib_sel, X_valid_sel = X_train_enc, X_calib_enc, X_valid_enc
        selected_features = enc_feature_names

    # ---------- 5) 최종 트리 ----------
    final = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        min_data_in_leaf=min_data_in_leaf,
        min_sum_hessian_in_leaf=min_sum_hessian_in_leaf,
        random_state=random_state,
        metric="auc",
    )

    print("[FIT:FINAL] fitting final tree on selected features...")
    final.fit(
        X_train_sel, y_train,
        eval_set=[(X_valid_sel, y_valid)],
        callbacks=[early_stopping(stopping_rounds=200), log_evaluation(period=0)],
    )

    # ---------- 6) 캘리브레이션 ----------
    print(f"[CALIB] calibrator='{calibrator}' (cv='prefit' on CALIB split)")
    if calibrator == "none":
        calib = None
        pd_valid = final.predict_proba(X_valid_sel)[:, 1]
    elif calibrator == "isotonic":
        calib = CalibratedClassifierCV(final, method="isotonic", cv="prefit")
        calib.fit(X_calib_sel, y_calib)
        pd_valid = calib.predict_proba(X_valid_sel)[:, 1]
    elif calibrator == "sigmoid":
        calib = CalibratedClassifierCV(final, method="sigmoid", cv="prefit")
        calib.fit(X_calib_sel, y_calib)
        pd_valid = calib.predict_proba(X_valid_sel)[:, 1]
    else:
        raise ValueError("calibrator must be one of: 'isotonic' | 'sigmoid' | 'none'")

    auc_sel = roc_auc_score(y_valid, pd_valid)
    brier = brier_score_loss(y_valid, pd_valid)
    print(f"[METRIC:FINAL] AUC(selected)={auc_sel:.6f}, Brier={brier:.6f}")
    try:
        print(f"[CHECK] pd_valid stats → min={pd_valid.min():.6f}, max={pd_valid.max():.6f}, "
              f"mean={pd_valid.mean():.6f}, std={pd_valid.std():.6f}")
    except Exception:
        pass

    # ---------- 7) 저장 ----------
    version = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    save_dir = os.path.join(ARTIFACTS_DIR, version)
    os.makedirs(save_dir, exist_ok=True)

    dump(final, os.path.join(save_dir, "tree_model.joblib"))
    print(f"[SAVE] tree_model.joblib saved → {os.path.join(save_dir, 'tree_model.joblib')}")

    meta = {
        "version": version,
        "trained_at": now_ts(),
        "target": target,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "enc_feature_names": enc_feature_names,
        "selected_features": selected_features,
        "auc_all": float(auc_all),
        "auc_selected": float(auc_sel),
        "brier": float(brier),
        "base_rate": base_rate,
        "calibrator": calibrator,
        "split": {
            "train": int(len(y_train)),
            "calib": int(len(y_calib)),
            "valid": int(len(y_valid)),
            "test_size": test_size
        },
        "params": {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "feature_select_threshold": feature_select_threshold,
            "random_state": random_state,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "min_data_in_leaf": min_data_in_leaf,
            "min_sum_hessian_in_leaf": min_sum_hessian_in_leaf,
        },
    }

    model_to_save = calib if calibrator != "none" else final
    save_artifacts(save_dir, encoder, selector, model_to_save, meta)
    print(f"[SAVE] artifacts saved under → {save_dir}")
    print(f"[SAVE] meta.json includes selected_features({len(selected_features)}) & metrics")
    print("================= [TRAIN END] =================\n")

    return {
        "message": "trained",
        "version": version,
        "metrics": {"auc_all": auc_all, "auc_selected": auc_sel, "brier": brier},
        "selected_features": selected_features,
        "artifact_dir": save_dir,
    }
