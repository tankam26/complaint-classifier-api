# Complaint Classifier API

A REST API that classifies financial complaints into 7 categories using a fine-tuned DistilBERT model.

## Categories
`bank_account` · `credit_card` · `credit_reporting` · `debt_collection` · `money_transfer` · `mortgage` · `student_loan`

## Model
- Base: `distilbert-base-multilingual-cased`
- Fine-tuned on 28k samples from the CFPB Consumer Complaint Database
- Test accuracy: **87%** · Weighted F1: **0.87**

## Project Structure

```
complaint-api/
├── app/
│   ├── main.py        # FastAPI app
│   ├── model.py       # Model loading + inference
│   └── schemas.py     # Pydantic schemas
├── index.html         # Frontend UI
├── Dockerfile
└── requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `index.html` in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API status + loaded classes |
| POST | `/predict` | Classify a single complaint |
| POST | `/predict/batch` | Classify up to 32 complaints |

## Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice for my mortgage payment and the bank refused to refund me"}'
```

```json
{
  "category": "mortgage",
  "confidence": 0.94,
  "all_scores": { "mortgage": 0.94, "bank_account": 0.02 },
  "processing_ms": 320.5
}
```