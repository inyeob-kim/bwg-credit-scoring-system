from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.models.schemas import PredictPayload
from app.models.predictor import Predictor


router = APIRouter()


@router.post("/predict")
async def predict(payload: PredictPayload):
    try:
        pred = Predictor.load_latest()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


    result = pred.predict(payload)
    return JSONResponse(result)