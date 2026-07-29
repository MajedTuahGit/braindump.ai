"""
POST /api/process-dump
The core endpoint — accepts raw brain dump text, returns categorized thoughts.

Pipeline per thought:
  BlobSplitter → TextCleaner → FeatureExtractor → Arbitrator → ActionEngine → SQLite
"""
import uuid
import time
import logging
from datetime import date

from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session

from api.schemas import (
    ProcessDumpRequest, ProcessDumpResponse,
    ThoughtResult, TierBreakdown
)
from ml.preprocessing.splitter       import BlobSplitter
from ml.preprocessing.cleaner        import TextCleaner
from ml.preprocessing.feature_extractor import FeatureExtractor
from ml.ensemble.arbitrator          import Arbitrator
from ml.ensemble.action_engine       import get_action
from db.database                     import get_db, Thought, ModelMetric

logger   = logging.getLogger(__name__)
router   = APIRouter()

# Module-level singletons (shared across requests)
_splitter   = BlobSplitter()
_cleaner    = TextCleaner()
_features   = FeatureExtractor()


def _get_arbitrator(request: Request) -> Arbitrator:
    m = request.app.state.models
    return Arbitrator(tier1=m["tier1"], tier2=m["tier2"], tier3=m["tier3"])


@router.post("/process-dump", response_model=ProcessDumpResponse)
async def process_dump(
    body:    ProcessDumpRequest,
    request: Request,
    db:      Session = Depends(get_db),
):
    overall_start = time.perf_counter()

    # ── 1. Split raw text ─────────────────────────────────────────────────────
    raw_thoughts = _splitter.split(body.text)
    if not raw_thoughts:
        return ProcessDumpResponse(
            thoughts=[], processing_time_ms=0,
            tier_breakdown=TierBreakdown(tier1_used=0, tier2_used=0, tier3_used=0)
        )

    arb       = _get_arbitrator(request)
    results   = []
    breakdown = {"tier1_used": 0, "tier2_used": 0, "tier3_used": 0}
    today     = date.today().isoformat()

    for raw_text in raw_thoughts:
        # ── 2. NLP preprocessing ───────────────────────────────────────────
        processed = _cleaner.process(raw_text)
        features  = _features.extract(raw_text, processed["entities"])
        # Wire verbs from cleaner into features (used by action engine)
        features["verbs"] = processed.get("verbs", [])

        # ── 3. Classify via ensemble ───────────────────────────────────────
        prediction = arb.predict(raw_text)

        # ── 4. Generate action suggestion ──────────────────────────────────
        action = get_action(
            category=prediction.category,
            features=features,
            entities=processed["entities"],
            original_text=raw_text,
        )

        thought_id = f"BD-{uuid.uuid4().hex[:6].upper()}"

        results.append(ThoughtResult(
            id=thought_id,
            thought=raw_text,
            category=prediction.category,
            status="NEW",
            confidence=round(prediction.confidence, 4),
            suggested_action=action,
            model_used=prediction.model_used,
            date=today,
        ))

        for k in breakdown:
            breakdown[k] += prediction.tier_breakdown.get(k, 0)

        # ── 5. Persist ─────────────────────────────────────────────────────
        db.add(Thought(
            id=thought_id, thought=raw_text,
            category=prediction.category, status="NEW",
            confidence=prediction.confidence,
            suggested_action=action, model_used=prediction.model_used,
        ))
        db.add(ModelMetric(
            tier=prediction.model_used,
            latency_ms=prediction.latency_ms,
            category_predicted=prediction.category,
            confidence=prediction.confidence,
        ))

    db.commit()

    return ProcessDumpResponse(
        thoughts=results,
        processing_time_ms=round((time.perf_counter() - overall_start) * 1000, 2),
        tier_breakdown=TierBreakdown(**breakdown),
    )
