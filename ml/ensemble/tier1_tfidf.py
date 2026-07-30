"""
Tier 1: TF-IDF + SVM Classifier.

Phase 1 (now): Ships with a keyword-based fallback so the app works immediately
               without any training. Confidence is capped at 0.75 to signal
               "rule-based, not ML" — the arbitrator will still use it if the
               SVM hasn't been trained yet.

Phase 5 (later): run `python training/train_tier1.py` → the .pkl is saved to
                 models/tier1_svm.pkl → this class auto-loads it on next restart
                 and uses real SVM probabilities instead of keyword matching.

Speed target: < 1ms per thought.
"""
import re
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

CATEGORIES  = ["PERSONAL", "FINANCIAL", "PROJECTS", "ADMIN", "AUTOMATION"]

# ── Keyword Fallback Rules ────────────────────────────────────────────────────
# Used when no trained model exists. Ordered by specificity.
KEYWORD_RULES: dict[str, set[str]] = {
    "AUTOMATION": {
        "automate", "automation", "script", "bot", "cron", "zapier", "n8n",
        "workflow", "trigger", "webhook", "pipeline", "auto", "automatically",
        "schedule", "integrat", "make scenario", "github actions",
    },
    "FINANCIAL": {
        "invest", "money", "budget", "savings", "expense", "pay", "bill",
        "salary", "income", "tax", "fund", "stock", "etf", "crypto", "debt",
        "loan", "insurance", "subscription", "fee", "afford", "wealth",
        "portfolio", "ibkr", "syfe", "dividend", "interest", "retire",
    },
    "PROJECTS": {
        "build", "create", "design", "develop", "implement", "launch",
        "website", "app", "feature", "mvp", "ship", "release", "deploy",
        "refactor", "document", "api", "prototype", "v2", "product",
        "portfolio", "chrome extension", "saas", "newsletter",
    },
    "ADMIN": {
        "reply", "send", "submit", "schedule", "meeting", "invoice", "email",
        "report", "follow up", "renew", "register", "book", "confirm",
        "respond", "archive", "update", "notify", "approve", "sign",
        "timesheet", "slack", "expense report",
    },
}
# Anything not matching above defaults to PERSONAL


class Tier1Classifier:
    """
    Wraps either a trained scikit-learn Pipeline (fast, ML-based)
    or a keyword fallback (rule-based, works from day 1).
    """

    def __init__(self):
        self.pipeline       = None   # set when model loads
        self.available      = False
        self.active_version = None   # filename of active model

    def load(self, specific_version: str = None) -> bool:
        """
        Scan 'models/' and load either the latest versioned pipeline,
        or a specific version file name if specified.
        """
        models_dir = Path("models")
        target_path = None

        if specific_version:
            target_path = models_dir / specific_version
            if not target_path.exists():
                logger.error("Tier 1 load failed: specific version %s does not exist", specific_version)
                return False
        else:
            # Auto-detect latest timestamped model (e.g. tier1_svm_YYYYMMDD_HHMMSS.pkl)
            if models_dir.exists():
                versions = sorted(list(models_dir.glob("tier1_svm_*.pkl")))
                if versions:
                    target_path = versions[-1] # latest file sorted alphabetically (chronologically)
                else:
                    legacy_path = models_dir / "tier1_svm.pkl"
                    if legacy_path.exists():
                        target_path = legacy_path

        if not target_path or not target_path.exists():
            logger.warning(
                "Tier 1: no trained model found in %s/ — using keyword fallback. "
                "Run: python training/train_tier1.py", models_dir
            )
            self.pipeline = None
            self.available = False
            self.active_version = None
            return False

        try:
            import joblib
            self.pipeline       = joblib.load(target_path)
            self.available      = True
            self.active_version = target_path.name
            logger.info("Tier 1 (TF-IDF + SVM) loaded version: %s", self.active_version)
            return True
        except Exception as exc:
            logger.error("Tier 1 load failed: %s", exc)
            return False

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Returns (category, confidence).

        If the SVM is loaded: uses decision-function → softmax for probability.
        If not:               uses keyword matching, confidence capped at 0.75.
        """
        if self.available and self.pipeline is not None:
            return self._predict_svm(text)
        return self._predict_keywords(text)

    def is_available(self) -> bool:
        """Always True — keyword fallback means Tier 1 is always usable."""
        return True

    # ── Private ───────────────────────────────────────────────────────────────

    def _predict_svm(self, text: str) -> Tuple[str, float]:
        import numpy as np
        try:
            scores   = self.pipeline.decision_function([text])[0]
            exp_s    = np.exp(scores - np.max(scores))
            probs    = exp_s / exp_s.sum()
            best_idx = int(np.argmax(probs))
            return CATEGORIES[best_idx], float(probs[best_idx])
        except Exception as exc:
            logger.error("SVM predict error: %s", exc)
            return self._predict_keywords(text)

    def _predict_keywords(self, text: str) -> Tuple[str, float]:
        """Simple keyword overlap scoring — works with zero training data."""
        words = set(re.findall(r'\b\w+\b', text.lower()))
        scores: dict[str, int] = {}

        for category, keywords in KEYWORD_RULES.items():
            overlap = sum(1 for kw in keywords if kw in text.lower())
            if overlap:
                scores[category] = overlap

        if not scores:
            return "PERSONAL", 0.50   # default

        best      = max(scores, key=scores.get)
        # cap at 0.75 so the arbitrator knows this is rule-based, not ML
        confidence = min(scores[best] / 3.0, 0.75)
        return best, confidence
