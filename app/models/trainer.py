import os, uuid, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from lightgbm import LGBMClassifier, early_stopping, log_evaluation


from app.core.config import ARTIFACTS_DIR
from app.core.utils import now_ts
from app.models.artifacts import save_artifacts




def train_pipeline(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_select_threshold: str = "median",
    learning_rate: float = 0.03,
    n_estimators: int = 3000,
    num_leaves: int = 63,
):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")


    y = df[target].astype(int)
    X = df.drop(columns=[target])


    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()


    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


    encoder = ColumnTransformer(
        transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


    X_train_enc = encoder.fit_transform(X_train)
    X_valid_enc = encoder.transform(X_valid)


    enc_feature_names = num_cols + cat_cols


    base = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=random_state,
        metric="auc",            # ← 여기로 이동
    )

    base.fit(
        X_train_enc, y_train,
        eval_set=[(X_valid_enc, y_valid)],
        callbacks=[
            early_stopping(stopping_rounds=200),
            log_evaluation(period=0)   # 로그 숨김(원하면 50 등으로 변경)
        ],
    )


    auc_all = roc_auc_score(y_valid, base.predict_proba(X_valid_enc)[:, 1])


    selector = SelectFromModel(estimator=base, threshold=feature_select_threshold, prefit=True)
    X_train_sel = selector.transform(X_train_enc)
    X_valid_sel = selector.transform(X_valid_enc)


    selected_mask = selector.get_support()
    selected_features = [f for f, keep in zip(enc_feature_names, selected_mask) if keep]


    final = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=random_state,
        metric="auc",
    )

    final.fit(
        X_train_sel, y_train,
        eval_set=[(X_valid_sel, y_valid)],
        callbacks=[
            early_stopping(stopping_rounds=200),
            log_evaluation(period=0)
        ],
    )


    calib = CalibratedClassifierCV(final, method="sigmoid", cv=5)
    calib.fit(X_train_sel, y_train)


    pd_valid = calib.predict_proba(X_valid_sel)[:, 1]
    auc_sel = roc_auc_score(y_valid, pd_valid)
    brier = brier_score_loss(y_valid, pd_valid)


    version = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    save_dir = os.path.join(ARTIFACTS_DIR, version)


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
        "params": {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "feature_select_threshold": feature_select_threshold,
            "test_size": test_size,
            "random_state": random_state,
        },
    }


    from app.models.artifacts import save_artifacts
    save_artifacts(save_dir, encoder, selector, calib, meta)


    return {
        "message": "trained",
        "version": version,
        "metrics": {"auc_all": auc_all, "auc_selected": auc_sel, "brier": brier},
        "selected_features": selected_features,
        "artifact_dir": save_dir,
    }