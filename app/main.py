from fastapi import FastAPI
from app.api.routes_health import router as health_router
from app.api.routes_train import router as train_router
from app.api.routes_predict import router as predict_router
from app.api.routes_explain import router as explain_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Credit Scoring API", version="1.0.0")

# 프론트 도메인에 맞게 조정 (개발 중이면 아래 두 개면 충분)
ORIGINS = [
    "http://localhost:7000",
    "http://127.0.0.1:7000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,      # 모두 허용하려면 ["*"]
    allow_credentials=True,
    allow_methods=["*"],        # OPTIONS 포함 전부 허용
    allow_headers=["*"],        # Content-Type 등 커스텀 헤더 허용
)

app.include_router(health_router)
app.include_router(train_router)
app.include_router(predict_router)
app.include_router(explain_router)


@app.get("/")
async def root():
    return {
    "name": "Credit Scoring API",
    "version": "1.0.0",
    "health": "/health",
    "train": "/train",
    "predict": "/predict",
    "explain": "/explain"
    }