"""
BrainDump.AI — FastAPI application entry point.

Startup sequence (lifespan):
  1. init_db()        — creates SQLite tables if not exist
  2. Load Tier 1      — always ready (keyword fallback if not trained)
  3. Load Tier 2      — loads quantized SLM if models/student_quantized.pt exists
  4. Load Tier 3      — loads BERT teacher if models/teacher_bert/ exists

All three models are stored in app.state.models so every request has access
without circular imports.

Run:
  uvicorn api.main:app --reload --port 8000
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from db.database                 import init_db
from ml.ensemble.tier1_tfidf     import Tier1Classifier
from ml.ensemble.tier2_slm       import Tier2SLM
from ml.ensemble.tier3_bert      import Tier3BERT
from api.routers                 import process, thoughts, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("braindump")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup; release on shutdown."""
    logger.info("=" * 55)
    logger.info("  BrainDump.AI  — starting up")
    logger.info("=" * 55)

    init_db()
    logger.info("Database initialised ✓")

    models = {}

    # Tier 1 — always available
    models["tier1"] = Tier1Classifier()
    models["tier1"].load()

    # Tier 2 — loads only when quantized model exists
    models["tier2"] = Tier2SLM()
    models["tier2"].load()

    # Tier 3 — loads only when fine-tuned teacher exists
    models["tier3"] = Tier3BERT()
    models["tier3"].load()

    t2 = "✅" if models["tier2"].is_available() else "⏳ (train SLM pipeline)"
    t3 = "✅" if models["tier3"].is_available() else "⏳ (run train_teacher.py)"
    logger.info("Tier 1 (TF-IDF/Keyword):  ✅")
    logger.info("Tier 2 (Quantized SLM):   %s", t2)
    logger.info("Tier 3 (BERT Teacher):    %s", t3)
    logger.info("Ready → http://localhost:8000")
    logger.info("=" * 55)

    app.state.models = models
    yield

    logger.info("BrainDump.AI shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "BrainDump.AI API",
    description = "Fully local cascading ensemble classifier — no LLM, no API keys",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

app.include_router(process.router,  prefix="/api", tags=["Core"])
app.include_router(thoughts.router, prefix="/api", tags=["Thoughts"])
app.include_router(health.router,   prefix="/api", tags=["Health"])

# Serve the frontend from the /frontend directory
_frontend = Path("frontend")
if _frontend.exists():
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
