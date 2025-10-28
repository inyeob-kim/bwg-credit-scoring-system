import os
import json
from joblib import dump, load
from typing import Dict, Any
from app.core.config import ARTIFACTS_DIR
import os, json
from typing import Dict, Any, List
from joblib import load

REQUIRED_FILES = ["encoder.joblib", "selector.joblib", "model_calibrated.joblib", "meta.json"]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_artifacts(save_dir: str, encoder, selector, calibrated_model, meta: Dict[str, Any]):
    """
    모델 학습 후 encoder, selector, model, meta 저장
    """
    # ✅ 경로를 중첩시키지 않도록 보정
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(ARTIFACTS_DIR, os.path.basename(save_dir))

    ensure_dir(save_dir)

    dump(encoder, os.path.join(save_dir, "encoder.joblib"))
    dump(selector, os.path.join(save_dir, "selector.joblib"))
    dump(calibrated_model, os.path.join(save_dir, "model_calibrated.joblib"))

    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 최신 버전을 가리키는 심볼릭 링크(또는 fallback)
    latest = os.path.join(ARTIFACTS_DIR, "latest")
    try:
        if os.path.exists(latest) or os.path.islink(latest):
            os.remove(latest)
        os.symlink(save_dir, latest)
    except Exception:
        # 윈도우에서는 symlink 실패 → latest.txt로 대체
        with open(os.path.join(ARTIFACTS_DIR, "latest.txt"), "w", encoding="utf-8") as f:
            f.write(save_dir)


def _has_complete_artifacts(dirpath: str) -> bool:
    try:
        for f in REQUIRED_FILES:
            p = os.path.join(dirpath, f)
            if not os.path.exists(p) or os.path.getsize(p) <= 0:
                return False
        return True
    except Exception:
        return False

def _list_subdirs(path: str) -> List[str]:
    try:
        return [
            os.path.join(path, d)
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]
    except FileNotFoundError:
        return []

def load_latest_artifacts() -> Dict[str, Any]:
    """
    최신 아티팩트 로드 (우선순위)
    1) artifacts/latest (symlink 또는 디렉터리) - 완전한지 검증
    2) artifacts/latest.txt (포인터 파일) - 완전한지 검증
    3) artifacts 하위 폴더들 중 mtime 최신 & 완전한 폴더
    4) 실패 시, 어떤 후보들을 검사했는지 상세 에러 제공
    """
    # 0) ARTIFACTS_DIR 절대경로화 (상대경로로 생기는 실행 위치 문제 방지)
    from app.core.config import ARTIFACTS_DIR
    base = os.path.abspath(ARTIFACTS_DIR)

    checked = []  # 디버깅용: 어딜 검사했는지 기록

    # 1) latest (symlink 또는 디렉터리)
    latest = os.path.join(base, "latest")
    if os.path.islink(latest) or os.path.isdir(latest):
        cand = os.path.realpath(latest)
        checked.append(("latest", cand))
        if os.path.exists(cand) and _has_complete_artifacts(cand):
            model_dir = cand
            # 로드
            encoder = load(os.path.join(model_dir, "encoder.joblib"))
            selector = load(os.path.join(model_dir, "selector.joblib"))
            model = load(os.path.join(model_dir, "model_calibrated.joblib"))
            with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            return {"dir": model_dir, "encoder": encoder, "selector": selector, "model": model, "meta": meta}

    # 2) latest.txt 포인터 (Windows symlink 실패 대비)
    latest_txt = os.path.join(base, "latest.txt")
    if os.path.exists(latest_txt):
        try:
            cand = open(latest_txt, "r", encoding="utf-8").read().strip()
        except Exception:
            cand = ""
        checked.append(("latest.txt", cand))
        if cand and os.path.exists(cand) and _has_complete_artifacts(cand):
            model_dir = cand
            encoder = load(os.path.join(model_dir, "encoder.joblib"))
            selector = load(os.path.join(model_dir, "selector.joblib"))
            model = load(os.path.join(model_dir, "model_calibrated.joblib"))
            with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            return {"dir": model_dir, "encoder": encoder, "selector": selector, "model": model, "meta": meta}

    # 3) 가장 최근 하위 폴더 검색 (latest/ latest.txt가 깨진 경우)
    subdirs = _list_subdirs(base)
    # 'latest' 디렉터리 자체가 하위로 잡히지 않게 필터링
    subdirs = [d for d in subdirs if os.path.basename(d).lower() != "latest"]
    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    for cand in subdirs:
        checked.append(("subdir", cand))
        if _has_complete_artifacts(cand):
            model_dir = cand
            encoder = load(os.path.join(model_dir, "encoder.joblib"))
            selector = load(os.path.join(model_dir, "selector.joblib"))
            model = load(os.path.join(model_dir, "model_calibrated.joblib"))
            with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            return {"dir": model_dir, "encoder": encoder, "selector": selector, "model": model, "meta": meta}

    # 4) 실패: 상세 원인 메시지
    hint_lines = [f"- ARTIFACTS_DIR: {base}"]
    for kind, path in checked:
        ok = os.path.exists(path)
        files = ", ".join(REQUIRED_FILES) if ok else "(path not found)"
        hint_lines.append(f"  • checked {kind}: {path}  -> exists={ok}")
        if ok:
            missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(path, f))]
            empties = [f for f in REQUIRED_FILES if os.path.exists(os.path.join(path, f)) and os.path.getsize(os.path.join(path, f)) <= 0]
            if missing:
                hint_lines.append(f"      missing: {missing}")
            if empties:
                hint_lines.append(f"      empty:   {empties}")

    raise FileNotFoundError(
        "No trained artifacts found. Run /train first.\n"
        + "\n".join(hint_lines)
    )
