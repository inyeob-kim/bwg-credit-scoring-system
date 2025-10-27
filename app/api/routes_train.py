from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import io
import pandas as pd


from app.models.trainer import train_pipeline


router = APIRouter()


@router.post("/train")
async def train(
    file: UploadFile = File(..., description="CSV file with features + target column"),
    target: str = Form(...),
    test_size: float = Form(0.2),
    random_state: int = Form(42),
    feature_select_threshold: str = Form("median"),
    learning_rate: float = Form(0.03),
    n_estimators: int = Form(3000),
    num_leaves: int = Form(63),
):
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")


    result = train_pipeline(
        df=df,
        target=target,
        test_size=test_size,
        random_state=random_state,
        feature_select_threshold=feature_select_threshold,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
    )

    return JSONResponse(result)