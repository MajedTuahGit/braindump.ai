"""
Tier 3: Fine-tuned BERT Teacher — local oracle (no API, no internet).

Phase 1 (now):   Stub — not loaded. Arbitrator skips it.
Phase 6 (later): After running train_teacher.py, models/teacher_bert/ is populated
                 and this class loads it automatically on the next restart.

Speed target (when trained): ~280ms per thought (only called for ambiguous inputs).
"""
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/teacher_bert")
CATEGORIES = ["PERSONAL", "FINANCIAL", "PROJECTS", "ADMIN", "AUTOMATION"]


class Tier3BERT:
    def __init__(self):
        self.model     = None
        self.tokenizer = None
        self.available = False

    def load(self) -> bool:
        if not MODEL_PATH.exists():
            logger.warning(
                "Tier 3: BERT teacher not found at %s. "
                "Run: python -m ml.slm.01_teacher.train_teacher", MODEL_PATH
            )
            return False
        try:
            from transformers import BertForSequenceClassification, BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH))
            self.model     = BertForSequenceClassification.from_pretrained(str(MODEL_PATH))
            self.model.eval()
            self.available = True
            logger.info("Tier 3 (BERT Teacher) loaded — full-precision local oracle ready")
            return True
        except Exception as exc:
            logger.error("Tier 3 load failed: %s", exc)
            return False

    def predict(self, text: str) -> Tuple[str, float]:
        if not self.available:
            return "PERSONAL", 0.0
        try:
            import torch
            inputs = self.tokenizer(
                text, return_tensors="pt",
                max_length=128, padding=True, truncation=True
            )
            with torch.no_grad():
                probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
            idx = torch.argmax(probs).item()
            return CATEGORIES[idx], float(probs[idx])
        except Exception as exc:
            logger.error("Tier 3 predict error: %s", exc)
            return "PERSONAL", 0.0

    def is_available(self) -> bool:
        return self.available
