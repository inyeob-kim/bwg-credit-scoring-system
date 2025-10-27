```md
# Credit Scoring API (Modularized)

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Endpoints
- `GET /health`
- `POST /train` (CSV upload: `file`, `target`, etc.)
- `POST /predict` (JSON: `{ records: [...], return_reasons, top_n_reasons }`)
- `POST /explain` (JSON: `{ records: [...], max_records }`)

## Artifacts
Saved under `./artifacts/<version>/` and `./artifacts/latest` symlink.
```

---

## .env.example
```dotenv
# Optional overrides
ARTIFACTS_DIR=./artifacts
PDO=50
S0=600
O0=20
```
