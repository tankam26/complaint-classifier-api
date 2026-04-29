import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ComplaintClassifier:
    def __init__(self, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        with open(f'{model_path}/label_mapping.json') as f:
            self.label_mapping = json.load(f)

        print(f"Model loaded on {self.device}")
        print(f"Classes: {list(self.label_mapping.values())}")

    def predict(self, text: str) -> dict:
        start = time.time()

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()

        all_scores = {
            self.label_mapping[str(i)]: round(prob, 4)
            for i, prob in enumerate(probs)
        }

        top_idx = int(torch.argmax(logits, dim=-1).item())
        category = self.label_mapping[str(top_idx)]
        confidence = round(probs[top_idx], 4)
        processing_ms = round((time.time() - start) * 1000, 2)

        return {
            'category': category,
            'confidence': confidence,
            'all_scores': all_scores,
            'processing_ms': processing_ms
        }