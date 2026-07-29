"""
BrainDump.AI — SQLite database schema via SQLAlchemy.

Three tables:
  thoughts     → every processed/stored thought
  feedback     → user corrections (drives retraining)
  model_metrics → per-request latency + tier used (for /api/model-stats)
"""
import os
import enum
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, String, Float,
    DateTime, Integer, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ── Engine ────────────────────────────────────────────────────────────────────
DATABASE_URL = "sqlite:///./db/braindump.db"

engine = SessionLocal = None   # populated in init_db()
Base = declarative_base()


# ── ORM Models ────────────────────────────────────────────────────────────────

class Thought(Base):
    """One categorized thought — the primary app entity."""
    __tablename__ = "thoughts"

    id               = Column(String, primary_key=True)
    thought          = Column(Text,   nullable=False)
    category         = Column(String, nullable=False)
    status           = Column(String, default="NEW")
    confidence       = Column(Float)
    suggested_action = Column(Text)
    model_used       = Column(String)
    created_at       = Column(DateTime, default=datetime.utcnow)


class Feedback(Base):
    """User correction: 'this was wrong, the real category is X'."""
    __tablename__ = "feedback"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    thought_id        = Column(String)
    thought_text      = Column(Text)
    original_category = Column(String)
    correct_category  = Column(String)
    created_at        = Column(DateTime, default=datetime.utcnow)


class ModelMetric(Base):
    """Per-request performance snapshot for the live stats dashboard."""
    __tablename__ = "model_metrics"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    tier               = Column(String)
    latency_ms         = Column(Float)
    category_predicted = Column(String)
    confidence         = Column(Float)
    created_at         = Column(DateTime, default=datetime.utcnow)


# ── Initialisation ────────────────────────────────────────────────────────────

def init_db():
    """Create the db/ directory and all tables if they don't exist."""
    global engine, SessionLocal

    os.makedirs("db", exist_ok=True)

    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}   # required for SQLite + FastAPI
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session, closes it when done."""
    if SessionLocal is None:
        init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
