"""
BlobSplitter — converts raw brain-dump text into individual thought strings.

Strategy (in order):
  1. Split on newlines — each line is treated as one thought candidate
  2. If a line is very long (>150 chars), try splitting on sentence boundaries
     or semicolons
  3. Strip leading bullet/dash/arrow characters
  4. Filter out fragments shorter than 5 characters
  5. Deduplicate while preserving original order

Phase 1 (now): Plain Python, no external dependencies — works immediately.
Phase 4 (later): cleaner.py adds spaCy sentence segmentation as an upgrade.
"""
import re
from typing import List

MIN_LENGTH  = 5    # chars — shorter = noise
MAX_LINE    = 150  # chars — above this, try further splitting


class BlobSplitter:
    def split(self, text: str) -> List[str]:
        """Split a raw brain dump blob into individual thought strings."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")

        thoughts: List[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if len(line) > MAX_LINE:
                thoughts.extend(self._split_long(line))
            else:
                thoughts.append(line)

        # Clean, filter, deduplicate
        thoughts = [self._clean(t) for t in thoughts]
        thoughts = [t for t in thoughts if len(t) >= MIN_LENGTH]

        seen:   set   = set()
        result: List  = []
        for t in thoughts:
            key = t.lower().strip()
            if key not in seen:
                seen.add(key)
                result.append(t)

        return result

    # ── Private ───────────────────────────────────────────────────────────────

    def _split_long(self, line: str) -> List[str]:
        """Try to further split an unusually long line."""
        # Sentence boundary: lowercase char → period → space → uppercase
        parts = re.split(r"(?<=[a-z])\.\s+(?=[A-Z])", line)
        if len(parts) > 1:
            return parts
        # Semicolons
        parts = [p.strip() for p in line.split(";") if p.strip()]
        if len(parts) > 1:
            return parts
        return [line]

    def _clean(self, text: str) -> str:
        """Strip leading bullets, normalize whitespace."""
        text = re.sub(r"^[\-\*\•\>\→\►\▶\d+\.]+\s*", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
