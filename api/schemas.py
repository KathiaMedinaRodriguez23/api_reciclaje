from pydantic import BaseModel
from typing import Dict, List

class PredictResponse(BaseModel):
    detected: List[dict]
    probs: Dict[str, float]
    meta: Dict[str, int]
