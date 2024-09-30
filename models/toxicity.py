import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ToxicityModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = 'cointegrated/rubert-tiny-toxicity'
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(self.device)

    def detect_toxicity(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        toxicity_probability = probabilities[0][1].item()
        return toxicity_probability >= 0.5
