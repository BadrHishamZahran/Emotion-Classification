import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .model import RecurrentClassifier

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
FALLBACK_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]

class EmotionPredictor:
    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache = {}

    def available_models(self):
        names = []
        for name in ["LSTM", "GRU", "BiLSTM_Attention", "DistilBERT"]:
            folder = self.models_dir / name
            if name == "DistilBERT":
                if folder.exists() and (folder / "config.json").exists():
                    names.append(name)
            elif folder.exists() and all(
                (folder / filename).exists()
                for filename in ["model.pth", "config.json", "vocab.json", "label_mapping.json"]
            ):
                names.append(name)
        return names

    def _load(self, model_name):
        if model_name in self.cache:
            return self.cache[model_name]

        folder = self.models_dir / model_name
        if model_name == "DistilBERT":
            model = AutoModelForSequenceClassification.from_pretrained(folder)
            tokenizer = AutoTokenizer.from_pretrained(folder)
            model.to(self.device).eval()
            with open(folder / "label_mapping.json", encoding="utf-8") as f:
                mapping = json.load(f)
            labels = mapping.get("id2label", {})
            labels = {int(k): v for k, v in labels.items()} if labels else {
                i: label for i, label in enumerate(FALLBACK_LABELS)
            }
            loaded = {"type": "transformer", "model": model, "tokenizer": tokenizer, "labels": labels}
        else:
            with open(folder / "config.json", encoding="utf-8") as f:
                config = json.load(f)
            with open(folder / "vocab.json", encoding="utf-8") as f:
                vocab = json.load(f)
            with open(folder / "label_mapping.json", encoding="utf-8") as f:
                mapping = json.load(f)
            labels = mapping.get("id2label", {})
            labels = {int(k): v for k, v in labels.items()} if labels else {
                i: label for i, label in enumerate(FALLBACK_LABELS)
            }
            model = RecurrentClassifier(
                kind=config["kind"],
                vocab_size=config["vocab_size"],
                emb_dim=config["emb_dim"],
                hidden_dim=config["hidden_dim"],
                num_classes=config["num_classes"],
                dropout=config["dropout"],
            )
            state = torch.load(folder / "model.pth", map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device).eval()
            loaded = {"type": "recurrent", "model": model, "vocab": vocab,
                      "labels": labels, "max_len": config.get("max_len", 128)}

        self.cache[model_name] = loaded
        return loaded

    @staticmethod
    def _scores(logits, labels):
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        return {labels.get(i, FALLBACK_LABELS[i] if i < len(FALLBACK_LABELS) else str(i)):
                float(probabilities[i]) for i in range(len(probabilities))}

    def predict(self, text, model_name):
        bundle = self._load(model_name)
        if bundle["type"] == "transformer":
            encoded = bundle["tokenizer"](
                text, return_tensors="pt", truncation=True, max_length=128
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = bundle["model"](**encoded).logits
            scores = self._scores(logits, bundle["labels"])
            return {"emotion": max(scores, key=scores.get), "scores": scores, "attention": None}

        tokens = TOKEN_RE.findall(text.lower())[:bundle["max_len"]]
        if not tokens:
            tokens = ["<unk>"]
        ids = [bundle["vocab"].get(token, bundle["vocab"].get("<unk>", 1)) for token in tokens]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(ids)], dtype=torch.long, device=self.device)

        with torch.no_grad():
            if model_name == "BiLSTM_Attention":
                logits, weights = bundle["model"](
                    input_ids, lengths, return_attention=True
                )
                attention = [
                    {"token": token, "weight": float(weight)}
                    for token, weight in zip(tokens, weights.squeeze(0).cpu().tolist())
                ]
            else:
                logits = bundle["model"](input_ids, lengths)
                attention = None

        scores = self._scores(logits, bundle["labels"])
        return {"emotion": max(scores, key=scores.get), "scores": scores, "attention": attention}
