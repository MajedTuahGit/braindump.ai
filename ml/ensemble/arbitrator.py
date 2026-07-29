"""
Arbitrator — routes each thought to the right tier based on confidence.

Routing rules (evaluated top-to-bottom):
  1. tier1.confidence >= 0.90  → return tier1 immediately (speed path)
  2. tier2.confidence >= 0.85  → return tier2 (quality path)
  3. both uncertain OR disagree → defer to tier3 oracle + majority vote
  4. nothing available         → return best available result

The arbitrator also tracks per-tier usage counts, which appear in /api/model-stats.
"""
import time
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

TIER1_THRESHOLD     = 0.90
TIER2_THRESHOLD     = 0.85
DISAGREEMENT_MARGIN = 0.10


@dataclass
class PredictionResult:
    category:       str
    confidence:     float
    model_used:     str
    latency_ms:     float
    tier_breakdown: Dict[str, int] = field(default_factory=dict)


class Arbitrator:
    def __init__(self, tier1, tier2, tier3):
        self.tier1 = tier1
        self.tier2 = tier2
        self.tier3 = tier3
        self._usage = {"tier1": 0, "tier2": 0, "tier3": 0}

    def predict(self, text: str) -> PredictionResult:
        t_start   = time.perf_counter()
        breakdown = {"tier1_used": 0, "tier2_used": 0, "tier3_used": 0}

        # ── Tier 1 (always runs — near-zero cost) ─────────────────────────
        cat1, conf1 = self.tier1.predict(text)
        breakdown["tier1_used"] = 1

        # Rule 1: Tier 1 highly confident → instant return
        if conf1 >= TIER1_THRESHOLD:
            self._usage["tier1"] += 1
            return PredictionResult(
                category=cat1, confidence=conf1,
                model_used="tier1_tfidf",
                latency_ms=(time.perf_counter() - t_start) * 1000,
                tier_breakdown=breakdown,
            )

        # ── Tier 2 (run if available) ──────────────────────────────────────
        cat2, conf2 = "PERSONAL", 0.0
        if self.tier2.is_available():
            cat2, conf2 = self.tier2.predict(text)
            breakdown["tier2_used"] = 1

            # Rule 2: Tier 2 confident enough → return it
            if conf2 >= TIER2_THRESHOLD:
                # Bonus: average when both agree
                eff_conf = (conf2 + conf1) / 2 if cat1 == cat2 else conf2
                self._usage["tier2"] += 1
                return PredictionResult(
                    category=cat2, confidence=eff_conf,
                    model_used="tier2_slm",
                    latency_ms=(time.perf_counter() - t_start) * 1000,
                    tier_breakdown=breakdown,
                )

        # ── Tier 3 (oracle fallback) ───────────────────────────────────────
        cat3, conf3 = "PERSONAL", 0.0
        if self.tier3.is_available():
            cat3, conf3 = self.tier3.predict(text)
            breakdown["tier3_used"] = 1

        # Majority vote across all available tiers
        votes = []
        if breakdown["tier1_used"]:  votes.append(cat1)
        if breakdown["tier2_used"]:  votes.append(cat2)
        if breakdown["tier3_used"]:  votes.append(cat3)

        if votes:
            majority = Counter(votes).most_common(1)[0][0]
            # Weight: tier3 > tier2 > tier1 (higher accuracy wins)
            if breakdown["tier3_used"] and conf3 > 0:
                final_cat, final_conf = cat3, conf3
            elif breakdown["tier2_used"] and conf2 > 0:
                final_cat, final_conf = cat2, conf2
            else:
                final_cat, final_conf = majority, max(conf1, conf2, conf3)
        else:
            final_cat, final_conf = cat1, conf1

        model_tag = (
            "tier3_bert" if breakdown["tier3_used"]
            else "tier2_slm" if breakdown["tier2_used"]
            else "tier1_tfidf"
        )
        self._usage[model_tag.split("_")[0]] += 1

        return PredictionResult(
            category=final_cat, confidence=final_conf,
            model_used=model_tag,
            latency_ms=(time.perf_counter() - t_start) * 1000,
            tier_breakdown=breakdown,
        )

    @property
    def usage_stats(self) -> Dict[str, int]:
        return self._usage.copy()
