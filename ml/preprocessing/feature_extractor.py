"""
FeatureExtractor — detects signal scores from a thought's text and spaCy entities.

Output dict feeds two things:
  1. The action_engine (for context-aware suggestions)
  2. Future ML features beyond bag-of-words

All logic is pure Python — no external dependencies.
"""
import re
from typing import Dict, Any

# Urgency signals
_URGENCY = {
    "urgent", "asap", "immediately", "deadline", "overdue", "critical",
    "today", "tonight", "now", "emergency", "priority", "must", "need to",
    "have to", "should have", "been putting off", "procrastinating",
}

class FeatureExtractor:
    def extract(self, text: str, entities: Dict) -> Dict[str, Any]:
        text_lower = text.lower()
        words      = set(re.findall(r"\b\w+\b", text_lower))

        return {
            "is_question":     (
                text.rstrip().endswith("?") or
                any(text_lower.startswith(w) for w in ("what", "why", "how", "when", "where", "who"))
            ),
            "urgency_score":   self._score(text_lower, words, _URGENCY),
            "has_money":       bool(entities.get("money")),
            "has_date":        bool(entities.get("dates")),
            "has_person":      bool(entities.get("names")),
            "word_count":      len(text.split()),
            "verbs":           [],   # populated by cleaner.get_verbs() in Phase 4
        }

    # ── Private ───────────────────────────────────────────────────────────────
    @staticmethod
    def _score(text_lower: str, words: set, signals: set) -> float:
        phrase_hits = sum(1 for p in signals if " " in p and p in text_lower)
        word_hits   = len(words.intersection(signals))
        return min((phrase_hits + word_hits) / 3.0, 1.0)
