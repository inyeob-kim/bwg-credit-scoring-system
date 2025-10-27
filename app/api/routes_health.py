from fastapi import APIRouter
from app.core.utils import now_ts


router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "time": now_ts()}