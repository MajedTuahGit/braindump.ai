"""
BrainDump.AI — Pydantic schemas (request/response contracts).

Every API endpoint uses these models for validation and serialization.
Pydantic enforces types at runtime — if the client sends wrong data, FastAPI
returns a 422 automatically before your code even runs.
"""
from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from enum import Enum


# ── Domain Enums ──────────────────────────────────────────────────────────────

class Category(str, Enum):
    PERSONAL   = "PERSONAL"
    FINANCIAL  = "FINANCIAL"
    PROJECTS   = "PROJECTS"
    ADMIN      = "ADMIN"
    AUTOMATION = "AUTOMATION"


class Status(str, Enum):
    NEW      = "NEW"
    ONGOING  = "ONGOING"
    DONE     = "DONE"
    PINNABLE = "PINNABLE"


class ModelTier(str, Enum):
    TIER1_TFIDF = "tier1_tfidf"
    TIER2_SLM   = "tier2_slm"
    TIER3_BERT  = "tier3_bert"
    KEYWORD     = "keyword_fallback"


# ── Request Models ────────────────────────────────────────────────────────────

class ProcessDumpRequest(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "need to call dentist\ninvest in ETFs\nbuild discord bot for morning tasks"
            }
        }
    }


class UpdateThoughtRequest(BaseModel):
    status:   Optional[Status]   = None
    category: Optional[Category] = None


class FeedbackRequest(BaseModel):
    thought_id:       str
    correct_category: Category


# ── Response Models ───────────────────────────────────────────────────────────


class ThoughtResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())   # allows 'model_used' field

    id:               str
    thought:          str
    category:         str
    status:           str
    confidence:       float
    suggested_action: str
    model_used:       str
    date:             str


class TierBreakdown(BaseModel):
    tier1_used: int
    tier2_used: int
    tier3_used: int


class ProcessDumpResponse(BaseModel):
    thoughts:          List[ThoughtResult]
    processing_time_ms: float
    tier_breakdown:    TierBreakdown


class TierStats(BaseModel):
    name:            str
    available:       bool
    avg_latency_ms:  float
    size_mb:         Optional[float] = None
    requests:        int


class ModelStatsResponse(BaseModel):
    tiers:                    dict
    total_thoughts_processed: int
    feedback_corrections:     int
    compression:              dict
