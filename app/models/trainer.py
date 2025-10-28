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
    calibrator: str = "isotonic",     # 'isotonic' | 'sigmoid' | 'none'
    reg_lambda: float = 10.0,         # 약간 강한 정규화로 과출력 완화
    reg_alpha: float = 1.0,
    min_data_in_leaf: int = 50,
    min_sum_hessian_in_leaf: float = 5.0,
) -> Dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    y = df[target].astype(int)
    X = df.drop(columns=[target])
    base_rate = float(y.mean())

    # ---------- 1) 세 분할: Train / Calib / Valid ----------
    # 먼저 Train vs Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Temp을 다시 Calib/Valid로 50:50
    X_calib, X_valid, y_calib, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    # ---------- 2) 인코더 ----------
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

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

    base.fit(
        X_train_enc, y_train,
        eval_set=[(X_valid_enc, y_valid)],   # 얼리스탑은 Valid만 참조
        callbacks=[
            early_stopping(stopping_rounds=200),
            log_evaluation(period=0),
        ],
    )

    auc_all = roc_auc_score(y_valid, base.predict_proba(X_valid_enc)[:, 1])

    # ---------- 4) 특성 선택 ----------
    selector = SelectFromModel(estimator=base, threshold=feature_select_threshold, prefit=True)
    X_train_sel = selector.transform(X_train_enc)
    X_calib_sel = selector.transform(X_calib_enc)
    X_valid_sel = selector.transform(X_valid_enc)

    selected_mask = selector.get_support()
    selected_features = [f for f, keep in zip(enc_feature_names, selected_mask) if keep]

    # 피처가 0개로 떨어지는 극단 방어
    if X_train_sel.shape[1] == 0:
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

    final.fit(
        X_train_sel, y_train,
        eval_set=[(X_valid_sel, y_valid)],   # 얼리스탑은 계속 Valid
        callbacks=[
            early_stopping(stopping_rounds=200),
            log_evaluation(period=0),
        ],
    )

    # ---------- 6) 캘리브레이션 ----------
    if calibrator == "none":
        calib = None
        pd_valid = final.predict_proba(X_valid_sel)[:, 1]
    elif calibrator == "isotonic":
        # Holdout Calib만 사용 (과적합/절벽화 방지)
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

    # ---------- 7) 저장 ----------
    version = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    save_dir = os.path.join(ARTIFACTS_DIR, version)
    os.makedirs(save_dir, exist_ok=True)

    # 원 트리 모델도 별도 저장(원 확률/SHAP 비교용)
    dump(final, os.path.join(save_dir, "tree_model.joblib"))

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

    # 캘리브레이터가 있으면 그걸 저장, 없으면 최종 트리 저장(후방 호환)
    model_to_save = calib if calibrator != "none" else final
    save_artifacts(save_dir, encoder, selector, model_to_save, meta)

    return {
        "message": "trained",
        "version": version,
        "metrics": {"auc_all": auc_all, "auc_selected": auc_sel, "brier": brier},
        "selected_features": selected_features,
        "artifact_dir": save_dir,
    }
