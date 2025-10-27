from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.models.schemas import ExplainPayload
from app.models.predictor import Predictor


router = APIRouter()


@router.post("/explain")
async def explain(payload: ExplainPayload):
    try:
        pred = Predictor.load_latest()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


    result = pred.explain(payload)
    return JSONResponse(result)