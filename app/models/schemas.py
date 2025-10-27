from pydantic import BaseModel, Field
from typing import List, Dict, Any


class PredictPayload(BaseModel):
    records: List[Dict[str, Any]] = Field(...)
    return_reasons: bool = False
    top_n_reasons: int = 5


class ExplainPayload(BaseModel):
    records: List[Dict[str, Any]]
    max_records: int = 50