import pandas as pd
import numpy as np
from typing import Dict, Any
from fastapi import HTTPException
from app.models.schemas import PredictPayload, ExplainPayload
from app.models.artifacts import load_latest_artifacts
from app.core.utils import prob_to_score
import shap

class Predictor:
    def __init__(self, bundle: Dict[str, Any]):
        self.encoder = bundle["encoder"]
        self.selector = bundle["selector"]
        self.model = bundle["model"]  # CalibratedClassifierCV
        self.meta = bundle["meta"]

    @classmethod
    def load_latest(cls):
        bundle = load_latest_artifacts()
        return cls(bundle)

    def _prep(self, records):
        cols = self.meta["num_cols"] + self.meta["cat_cols"]
        df = pd.DataFrame(records)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
        df = df[cols]
        X_enc = self.encoder.transform(df)
        X_sel = self.selector.transform(X_enc)
        return X_sel

    def predict(self, payload: PredictPayload) -> Dict[str, Any]:
        X_sel = self._prep(payload.records)
        pd_hat = self.model.predict_proba(X_sel)[:, 1]
        scores = prob_to_score(pd_hat)
        result: Dict[str, Any] = {
            "version": self.meta["version"],
            "n": len(payload.records),
            "predictions": [
                {"pd": float(p), "score": float(s)} for p, s in zip(pd_hat, scores)
            ],
        }
        if payload.return_reasons:
            try:
                base_est = self.model.base_estimator_ if hasattr(self.model, "base_estimator_") else None
                if base_est is None:
                    raise RuntimeError("Base estimator not available for SHAP.")
                explainer = shap.TreeExplainer(base_est)
                shap_vals = explainer.shap_values(X_sel)
                if isinstance(shap_vals, list):
                    shap_arr = shap_vals[1]
                else:
                    shap_arr = shap_vals
                feat_names = self.meta["selected_features"]
                topn = int(payload.top_n_reasons)
                reasons = []
                for i in range(min(len(payload.records), 200)):
                    sv = np.abs(shap_arr[i])
                    top_idx = np.argsort(-sv)[:topn]
                    reasons.append([
                        {"feature": feat_names[j], "abs_contrib": float(sv[j])} for j in top_idx
                    ])
                result["reasons"] = reasons
            except Exception as e:
                result["reasons_error"] = str(e)
        return result

    def explain(self, payload: ExplainPayload) -> Dict[str, Any]:
        X_sel = self._prep(payload.records)
        base_est = self.model.base_estimator_ if hasattr(self.model, "base_estimator_") else None
        if base_est is None:
            raise HTTPException(status_code=500, detail="Base estimator not available for SHAP.")
        explainer = shap.TreeExplainer(base_est)
        maxn = min(payload.max_records, X_sel.shape[0])
        shap_vals = explainer.shap_values(X_sel[:maxn])
        if isinstance(shap_vals, list):
            shap_arr = shap_vals[1]
        else:
            shap_arr = shap_vals
        feat_names = self.meta["selected_features"]
        abs_mean = np.mean(np.abs(shap_arr), axis=0)
        global_imp = [
            {"feature": f, "abs_mean_contrib": float(v)}
            for f, v in sorted(zip(feat_names, abs_mean), key=lambda x: -x[1])
        ]
        return {"version": self.meta["version"], "n_used": int(maxn), "global_importance": global_imp[:50]}