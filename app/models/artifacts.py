import os, json
from joblib import dump, load
from typing import Dict, Any
from app.core.config import ARTIFACTS_DIR




def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)




def save_artifacts(save_dir: str, encoder, selector, calibrated_model, meta: Dict[str, Any]):
    ensure_dir(save_dir)
    dump(encoder, os.path.join(save_dir, "encoder.joblib"))
    dump(selector, os.path.join(save_dir, "selector.joblib"))
    dump(calibrated_model, os.path.join(save_dir, "model_calibrated.joblib"))
    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


    latest = os.path.join(ARTIFACTS_DIR, "latest")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            try:
                os.remove(latest)
            except OSError:
                pass
        os.symlink(save_dir, latest)
    except Exception:
        # Windows etc.
        pass




def load_latest_artifacts() -> Dict[str, Any]:
    latest = os.path.join(ARTIFACTS_DIR, "latest")
    model_dir = None

    # 1) latest 심볼릭/디렉터리 우선
    if os.path.islink(latest) or os.path.isdir(latest):
        model_dir = os.path.realpath(latest)
    else:
        # 2) 버전 하위폴더 중 mtime 최신 선택
        subdirs = [
            os.path.join(ARTIFACTS_DIR, d)
            for d in os.listdir(ARTIFACTS_DIR)
            if os.path.isdir(os.path.join(ARTIFACTS_DIR, d))
        ]
        if subdirs:
            model_dir = max(subdirs, key=os.path.getmtime)

    # 3) 폴더를 못 찾았거나(=하위폴더 없음) Windows에서 링크 실패한 경우
    #    루트에 파일이 직접 있는 케이스를 폴백 지원
    if model_dir is None:
        root_files = [
            os.path.join(ARTIFACTS_DIR, f)
            for f in ["encoder.joblib", "selector.joblib", "model_calibrated.joblib", "meta.json"]
        ]
        if all(os.path.exists(p) for p in root_files):
            model_dir = ARTIFACTS_DIR

    if model_dir is None:
        raise FileNotFoundError("No trained artifacts found. Run /train first.")

    encoder_path = os.path.join(model_dir, "encoder.joblib")
    selector_path = os.path.join(model_dir, "selector.joblib")
    model_path   = os.path.join(model_dir, "model_calibrated.joblib")
    meta_path    = os.path.join(model_dir, "meta.json")

    if not all(os.path.exists(p) for p in [encoder_path, selector_path, model_path, meta_path]):
        raise FileNotFoundError(f"Artifacts are incomplete under: {model_dir}")

    encoder = load(encoder_path)
    selector = load(selector_path)
    model = load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {"dir": model_dir, "encoder": encoder, "selector": selector, "model": model, "meta": meta}
