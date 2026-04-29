from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PredictRequest, PredictResponse
from app.model import ComplaintClassifier

MODEL_PATH = './model_weights'
classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    print("Loading model...")
    classifier = ComplaintClassifier(MODEL_PATH)
    print("Model ready")
    yield
    print("Shutting down")

app = FastAPI(
    title="Complaint Classifier API",
    description="Classifies financial complaints into categories",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model_loaded': classifier is not None,
        'classes': list(classifier.label_mapping.values())
    }

@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(request.text) < 20:
        raise HTTPException(status_code=400, detail="Text too short to classify")

    result = classifier.predict(request.text)
    return result

@app.post('/predict/batch')
def predict_batch(requests: list[PredictRequest]):
    if len(requests) > 32:
        raise HTTPException(status_code=400, detail="Max batch size is 32")

    return [classifier.predict(r.text) for r in requests]