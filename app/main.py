from fastapi import FastAPI
from app.api.routes_health import router as health_router
from app.api.routes_train import router as train_router
from app.api.routes_predict import router as predict_router
from app.api.routes_explain import router as explain_router


app = FastAPI(title="Credit Scoring API", version="1.0.0")


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