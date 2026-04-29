from pydantic import BaseModel

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    category: str
    confidence: float
    all_scores: dict[str, float]
    processing_ms: float