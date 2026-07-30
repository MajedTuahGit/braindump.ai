"""
CRUD operations for stored thoughts.

GET    /api/thoughts              — list all (optional ?category= ?status= filter)
PATCH  /api/thoughts/{id}         — update status or category
DELETE /api/thoughts/{id}         — remove a thought
POST   /api/feedback              — submit a wrong-category correction
"""
import logging
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.schemas import UpdateThoughtRequest, FeedbackRequest
from db.database import get_db, Thought, Feedback

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/thoughts")
async def get_thoughts(
    category: Optional[str] = None,
    status:   Optional[str] = None,
    db:       Session       = Depends(get_db),
):
    query = db.query(Thought)
    if category and category.upper() != "ALL":
        query = query.filter(Thought.category == category.upper())
    if status and status.upper() != "ALL":
        query = query.filter(Thought.status == status.upper())

    thoughts = query.order_by(Thought.created_at.desc()).all()

    return {
        "thoughts": [
            {
                "id":               t.id,
                "thought":          t.thought,
                "category":         t.category,
                "status":           t.status,
                "confidence":       t.confidence,
                "suggested_action": t.suggested_action,
                "model_used":       t.model_used,
                "date":             t.created_at.date().isoformat() if t.created_at else None,
            }
            for t in thoughts
        ],
        "total": len(thoughts),
    }


@router.patch("/thoughts/{thought_id}")
async def update_thought(
    thought_id: str,
    update:     UpdateThoughtRequest,
    db:         Session = Depends(get_db),
):
    thought = db.query(Thought).filter(Thought.id == thought_id).first()
    if not thought:
        raise HTTPException(status_code=404, detail="Thought not found")

    if update.status:
        thought.status   = update.status
    if update.category:
        thought.category = update.category

    db.commit()
    return {"success": True, "id": thought_id}


@router.delete("/thoughts/{thought_id}")
async def delete_thought(
    thought_id: str,
    db:         Session = Depends(get_db),
):
    thought = db.query(Thought).filter(Thought.id == thought_id).first()
    if not thought:
        raise HTTPException(status_code=404, detail="Thought not found")

    db.delete(thought)
    db.commit()
    return {"success": True}


@router.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    db:       Session = Depends(get_db),
):
    """Record a user correction — updates the DB and the training dataset."""
    thought = db.query(Thought).filter(Thought.id == feedback.thought_id).first()
    if not thought:
        raise HTTPException(status_code=404, detail="Thought not found")

    original = thought.category
    thought.category = feedback.correct_category

    db.add(Feedback(
        thought_id=feedback.thought_id,
        thought_text=thought.thought,
        original_category=original,
        correct_category=feedback.correct_category,
    ))
    
    # ── Update seed_data.json training set ──────────────────────────────────
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        seed_data_path = project_root / "training" / "data" / "seed_data.json"
        
        if seed_data_path.exists():
            with open(seed_data_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            
            # Look for exact matching text in the dataset
            matched = False
            for entry in dataset:
                if entry["text"].strip().lower() == thought.thought.strip().lower():
                    entry["label"] = feedback.correct_category
                    matched = True
                    break
            
            if not matched:
                # Add new example
                dataset.append({
                    "text": thought.thought,
                    "label": feedback.correct_category
                })
                
            with open(seed_data_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2)
            logger.info("[API] Corrected category written to seed_data.json: %r -> %s", thought.thought, feedback.correct_category)
        else:
            logger.warning("[API] seed_data.json not found at %s. Correction not written to disk.", seed_data_path)
            
    except Exception as exc:
        logger.error("[API] Failed to update seed_data.json: %s", exc)

    db.commit()

    return {
        "success": True,
        "message": f"Correction recorded: {original} → {feedback.correct_category}",
    }
