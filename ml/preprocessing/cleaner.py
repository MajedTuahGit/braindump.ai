"""
TextCleaner — spaCy-powered NER, lemmatization, and verb extraction.

Phase 1 (now): Minimal no-spaCy fallback so the server boots without spaCy.
               Returns empty entities + tokenized text.

Phase 4 (later): `python -m spacy download en_core_web_sm` then restart the server.
                 The class auto-detects spaCy and switches to full NLP processing.
"""
import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Try to load spaCy — silently fall back if not installed/downloaded yet
try:
    # pyrefly: ignore [missing-import]
    import spacy
    _nlp = spacy.load("en_core_web_sm") 
    _SPACY_AVAILABLE = True
    logger.info("TextCleaner: spaCy loaded ✓")
except Exception:
    _SPACY_AVAILABLE = False
    logger.warning(
        "TextCleaner: spaCy not available — using regex fallback. "
        "Run: python -m spacy download en_core_web_sm"
    )


class TextCleaner:
    """NER + lemmatization wrapper. Degrades gracefully without spaCy."""

    def clean(self, text: str) -> str:
        """Return lemmatized, stop-word-free version for TF-IDF input."""
        if _SPACY_AVAILABLE:
            doc = _nlp(text.lower())
            return " ".join(
                t.lemma_ for t in doc
                if not t.is_stop and not t.is_punct and len(t.text) > 1
            )
        # Fallback: lowercase + basic punctuation strip
        return re.sub(r"[^\w\s]", " ", text.lower()).strip()

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Return named entities grouped by type."""
        entities: Dict[str, List[str]] = {
            "money": [], "dates": [], "names": [], "organizations": [], "products": []
        }
        if not _SPACY_AVAILABLE:
            # Minimal regex fallback for money + dates
            entities["money"] = re.findall(r"\$[\d,]+(?:\.\d+)?", text)
            entities["dates"]  = re.findall(
                r"\b(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|"
                r"saturday|sunday|next week|this week|tonight|\d{1,2}/\d{1,2})\b",
                text, re.IGNORECASE
            )
            return entities

        doc = _nlp(text)
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                entities["money"].append(ent.text)
            elif ent.label_ in ("DATE", "TIME"):
                entities["dates"].append(ent.text)
            elif ent.label_ == "PERSON":
                entities["names"].append(ent.text)
            elif ent.label_ == "ORG":
                entities["organizations"].append(ent.text)
            elif ent.label_ == "PRODUCT":
                entities["products"].append(ent.text)
        return entities

    def get_verbs(self, text: str) -> List[str]:
        """Extract root action verbs."""
        if not _SPACY_AVAILABLE:
            return []
        doc = _nlp(text)
        return [t.lemma_ for t in doc if t.pos_ == "VERB"]

    def process(self, text: str) -> Dict[str, Any]:
        """Full pipeline — returns everything in one call."""
        return {
            "cleaned":   self.clean(text),
            "entities":  self.extract_entities(text),
            "verbs":     self.get_verbs(text),
            "original":  text,
        }
