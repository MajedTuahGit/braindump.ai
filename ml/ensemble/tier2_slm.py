"""
Tier 2: Quantized MiniTransformer (our compressed SLM).

Phase 1 (now):   Stub — always returns (category, 0.0) so the arbitrator skips it.
Phase 9 (later): After running the full SLM pipeline (distil → prune → quantize),
                 models/student_quantized.pt is placed here and this class loads it
                 automatically on the next server restart.

Speed target (when trained): ~8ms per thought.
"""
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

MODEL_PATH     = Path("models/student_quantized.pt")
TOKENIZER_PATH = Path("models/student_tokenizer")
CATEGORIES     = ["PERSONAL", "FINANCIAL", "PROJECTS", "ADMIN", "AUTOMATION"]


class Tier2SLM:
    def __init__(self):
        self.model     = None
        self.tokenizer = None
        self.available = False

    def load(self) -> bool:
        if not MODEL_PATH.exists():
            logger.warning(
                "Tier 2: quantized model not found at %s. "
                "Complete the SLM pipeline first (Phases 6-9).", MODEL_PATH
            )
            return False
        try:
            import torch
            from transformers import AutoTokenizer

            self.model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            self.model.eval()

            tok_src        = str(TOKENIZER_PATH) if TOKENIZER_PATH.exists() else "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
            self.available = True

            size_mb = MODEL_PATH.stat().st_size / 1e6
            logger.info("Tier 2 (Quantized MiniTransformer) loaded — %.1f MB", size_mb)
            return True
        except Exception as exc:
            logger.error("Tier 2 load failed: %s", exc)
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
                out    = self.model(**inputs)
                logits = out.logits if hasattr(out, "logits") else out[0]
                probs  = torch.softmax(logits, dim=-1)[0]
            idx = torch.argmax(probs).item()
            return CATEGORIES[idx], float(probs[idx])
        except Exception as exc:
            logger.error("Tier 2 predict error: %s", exc)
            return "PERSONAL", 0.0

    def is_available(self) -> bool:
        return self.available
