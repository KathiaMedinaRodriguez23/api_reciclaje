from pydantic import BaseModel
from typing import Dict, List, Optional


class PredictionOut(BaseModel):
    id: str
    label: str
    category: str
    dateIso: str
    thumbnail: str


class PredictionsResponse(BaseModel):
    items: List[PredictionOut]
    totalsByLabel: Dict[str, int]
    totalItemsInPage: int
    nextStartAfterIso: Optional[str] = None