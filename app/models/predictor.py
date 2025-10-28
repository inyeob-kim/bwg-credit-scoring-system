import os
import pandas as pd
import numpy as np
from typing import Dict, Any
from fastapi import HTTPException
from app.models.schemas import PredictPayload, ExplainPayload
from app.models.artifacts import load_latest_artifacts
from app.core.utils import prob_to_score
import shap
from dotenv import load_dotenv

load_dotenv()

# 👇 추가: OpenAI 클라이언트
from openai import OpenAI
LLM_MODEL = os.getenv("EXPLAIN_MODEL", "gpt-4o-mini")
_openai_client = None
try:
    _openai_client = OpenAI()  # OPENAI_API_KEY가 환경변수로 있어야 함
except Exception:
    _openai_client = None


class Predictor:
    def __init__(self, bundle: Dict[str, Any]):
        self.encoder = bundle["encoder"]
        self.selector = bundle["selector"]
        self.model   = bundle["model"]            # Calibrated or Final
        self.meta    = bundle["meta"]
        self.tree_model = bundle.get("tree_model", None)

    @classmethod
    def load_latest(cls):
        bundle = load_latest_artifacts()
        return cls(bundle)

    def _prep(self, records):
        cols = self.meta["num_cols"] + self.meta["cat_cols"]
        df = pd.DataFrame(records)

        # 누락 컬럼 자동 생성
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

        df = df[cols].replace({"": np.nan})
        for c in self.meta["num_cols"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        X_enc = self.encoder.transform(df)
        X_sel = self.selector.transform(X_enc)
        return X_sel

    def _extract_tree_estimator(self):
        if self.tree_model is not None:
            return self.tree_model

        m = self.model
        if hasattr(m, "calibrated_classifiers_") and m.calibrated_classifiers_:
            cc0 = m.calibrated_classifiers_[0]
            if hasattr(cc0, "estimator"):
                return cc0.estimator
            if hasattr(cc0, "base_estimator"):
                return cc0.base_estimator
        if hasattr(m, "base_estimator_"):
            return m.base_estimator_
        if hasattr(m, "base_estimator"):
            return m.base_estimator
        return None

    # 👇 추가: 고객 친화 설명 생성기 (GPT 요약)
    def _build_gpt_description(self, record: Dict[str, Any], pd_val: float, score: float, reasons_for_one: list) -> str:
        """
        record: 원 입력 한 건
        pd_val, score: 해당 고객의 예측 결과
        reasons_for_one: [{"feature": "...", "abs_contrib": ...}, ...]
        """
        if _openai_client is None:
            return "OpenAI client is None"  # LLM 미설정 시 설명 생략(조용히 실패)

        print("[DEBUG] API KEY =", os.getenv("OPENAI_API_KEY"))

        # 읽기 쉬운 키 라벨(원하면 확장)
        label_map = {
            "utilization": "신용한도 사용률",
            "credit_utilization": "신용한도 사용률(원)",
            "debt_ratio": "부채비율",
            "credit_history_months": "신용이력 개월",
            "n_late_30d": "30일 이상 연체 횟수",
            "income": "소득",
            "loan_amount": "대출금액",
            "num_of_loans": "대출건수",
            "employment_length": "현 직장 근속연수",
            "employment_type": "고용형태",
            "region": "지역",
            "gender": "성별",
            "channel": "채널",
        }

        # 상위 요인 텍스트 구성 (방향 정보가 없다면 영향도가 큰 순서로만 요약)
        reasons_lines = []
        if reasons_for_one:
            for r in reasons_for_one:
                feat = r.get("feature", "")
                nm = label_map.get(feat, feat)
                reasons_lines.append(f"- {nm} (영향도: {r.get('abs_contrib', 0):.3f})")
        reasons_text = "\n".join(reasons_lines) if reasons_lines else "- 상위 영향 요인 정보 없음"

        # 프롬프트 구성
        user_prompt = f"""
                너는 금융 상담사다. 아래 고객의 신용평가 예측 결과를 일반 고객이 이해하기 쉽게 3~5줄로 요약해라.
                전문용어는 풀어서 쓰고, 점수/부도확률의 의미를 함께 설명해라.

                [고객 입력 요약]
                { {k: record.get(k) for k in list(record.keys())[:12]} }  # 주요 항목만

                [예측 결과]
                - 예상 부도확률(PD): {pd_val:.5f}
                - 신용점수(Score): {score:.0f}

                [상위 영향 요인]
                {reasons_text}

                설명 시 '왜 이렇게 나왔는지'를 핵심 요인 중심으로 풀어서 말하고,
                개선 여지가 있는 항목(예: 한도 사용률, 부채비율)이 높으면 관리 팁 한 줄을 덧붙여라.
                """

        try:
            resp = _openai_client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.2,
                max_tokens=220,
                messages=[
                    {"role": "system", "content": "너는 친절한 금융 상담사다. 고객 눈높이에 맞게 간결하고 정확히 설명해라."},
                    {"role": "user", "content": user_prompt}
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return ""  # 실패 시 조용히 패스

    def predict(self, payload: PredictPayload) -> Dict[str, Any]:
        X_sel = self._prep(payload.records)

        use_raw = bool(getattr(payload, "use_raw", False))
        if use_raw:
            base_est = self._extract_tree_estimator()
            if base_est is None:
                raise HTTPException(status_code=500, detail="Underlying tree estimator not found for raw proba.")
            pd_hat = base_est.predict_proba(X_sel)[:, 1]
        else:
            pd_hat = self.model.predict_proba(X_sel)[:, 1]

        # 수치 안정성: 0/1 확률 방지
        pd_hat = np.clip(pd_hat, 1e-5, 1 - 1e-5)

        scores = prob_to_score(pd_hat)

        # 기본 결과 틀
        result: Dict[str, Any] = {
            "version": self.meta["version"],
            "n": len(payload.records),
            "predictions": [],
        }

        # reasons를 먼저 계산 (설명에 활용)
        reasons_batch = None
        if payload.return_reasons:
            try:
                base_est = self._extract_tree_estimator()
                if base_est is None:
                    raise RuntimeError("Underlying tree estimator not found for SHAP.")
                explainer = shap.TreeExplainer(base_est)
                shap_vals = explainer.shap_values(X_sel)
                if isinstance(shap_vals, list):
                    class_idx = 1 if len(shap_vals) > 1 else 0
                    shap_arr = shap_vals[class_idx]
                else:
                    shap_arr = shap_vals

                feat_names = self.meta["selected_features"]
                if len(feat_names) != X_sel.shape[1]:
                    raise RuntimeError(f"Selected feature count mismatch: meta={len(feat_names)} vs X={X_sel.shape[1]}")
                topn = max(1, min(int(payload.top_n_reasons), len(feat_names)))

                reasons_batch = []
                for i in range(min(len(payload.records), 200)):
                    sv = np.abs(shap_arr[i])
                    top_idx = np.argsort(-sv)[:topn]
                    reasons_one = [{"feature": feat_names[j], "abs_contrib": float(sv[j])} for j in top_idx]
                    reasons_batch.append(reasons_one)
            except Exception as e:
                result["reasons_error"] = str(e)
                reasons_batch = [None] * len(payload.records)

        # 각 레코드별 description 생성 + predictions 채우기
        for i, (p, s) in enumerate(zip(pd_hat, scores)):
            reasons_for_one = reasons_batch[i] if reasons_batch else None
            desc = self._build_gpt_description(
                record=payload.records[i],
                pd_val=float(p),
                score=float(s),
                reasons_for_one=reasons_for_one or []
            )
            item = {
                "pd": float(p),
                "score": float(s),
                "description": desc  # 👈 고객용 설명 추가
            }
            # 요청이 있으면 reasons도 같이 포함
            if reasons_for_one is not None:
                item["reasons"] = reasons_for_one
            result["predictions"].append(item)

        print("Result = ", result)
        return result

    def explain(self, payload: ExplainPayload) -> Dict[str, Any]:
        X_sel = self._prep(payload.records)
        base_est = self._extract_tree_estimator()
        if base_est is None:
            raise HTTPException(status_code=500, detail="Underlying tree estimator not found for SHAP.")
        explainer = shap.TreeExplainer(base_est)
        maxn = min(payload.max_records, X_sel.shape[0])
        shap_vals = explainer.shap_values(X_sel[:maxn])
        if isinstance(shap_vals, list):
            class_idx = 1 if len(shap_vals) > 1 else 0
            shap_arr = shap_vals[class_idx]
        else:
            shap_arr = shap_vals
        feat_names = self.meta["selected_features"]
        if len(feat_names) != X_sel.shape[1]:
            raise HTTPException(status_code=500, detail=f"Selected feature count mismatch: meta={len(feat_names)} vs X={X_sel.shape[1]}")
        abs_mean = np.mean(np.abs(shap_arr), axis=0)
        global_imp = [{"feature": f, "abs_mean_contrib": float(v)}
                      for f, v in sorted(zip(feat_names, abs_mean), key=lambda x: -x[1])][:50]
        return {"version": self.meta["version"], "n_used": int(maxn), "global_importance": global_imp}
