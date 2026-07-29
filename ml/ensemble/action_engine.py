"""
Action Suggestion Engine — 100% rule-based, zero external calls.

Uses:
  - Predicted category
  - Extracted entities (money amounts, dates, names) from spaCy (Phase 4)
  - Feature scores (urgency, signal words)
  - Raw thought text for context

Each category has a pool of templated action strings. Entity-aware overrides
fire first when entities are detected (e.g., a money entity in FINANCIAL gives
a more specific suggestion than the generic pool).
"""
import random
from typing import Dict, Any, List

# ── Action Template Pools ─────────────────────────────────────────────────────
_POOLS: Dict[str, List[str]] = {
    "PERSONAL": [
        "Block 30 min in your calendar this week to handle this",
        "Set a phone reminder for tonight — if not now, when?",
        "Reach out and make it happen — stop overthinking",
        "Add this to your weekly review so it doesn't vanish",
        "This has been on your list — do it first thing tomorrow",
        "Track this in your habit log, make it non-negotiable",
    ],
    "FINANCIAL": [
        "Log this in your expense tracker right now",
        "Research 3 options, shortlist, then decide — no impulse",
        "Calculate total cost: upfront + recurring + hidden fees",
        "Open your banking app and take action on this today",
        "Set a budget cap before spending a single cent",
        "Check if there's a cheaper alternative first",
    ],
    "PROJECTS": [
        "Define 'done' first — what does finished look like?",
        "Break into 3 subtasks — what's the first 30-min action?",
        "Open your IDE and start a blank file — beginning is everything",
        "Block 2 hours this week, ship a rough first version",
        "Talk to one potential user before building anything",
        "Write a one-sentence problem statement before coding",
    ],
    "ADMIN": [
        "Do this now — it'll take less than 5 minutes",
        "Schedule a specific time, or it will never happen",
        "Draft and send — done is better than perfect",
        "Set a deadline reminder so this doesn't get buried",
        "Batch with other admin tasks on a fixed day each week",
        "Delegate this if possible — your time is worth more",
    ],
    "AUTOMATION": [
        "Write the manual steps first, then automate one by one",
        "Check if Make / Zapier / n8n already does this — don't rebuild",
        "Start with a simple Python script, get it working first",
        "Define: trigger → action → output before writing code",
        "This could save you hours per week — prioritize it",
        "Look for an existing library or API before building from scratch",
    ],
}


def get_action(
    category: str,
    features: Dict[str, Any],
    entities: Dict[str, list],
    original_text: str,
) -> str:
    """
    Return a context-aware action suggestion.

    Entity-aware overrides fire first; generic pool is the fallback.
    """
    text_lower = original_text.lower()

    # ── Entity-aware overrides ────────────────────────────────────────────────
    if category == "FINANCIAL":
        if entities.get("money"):
            return f"Track {entities['money'][0]} — categorise it in your budget now"
        if entities.get("dates"):
            return f"Mark this financial action on your calendar: {entities['dates'][0]}"
        if "tax" in text_lower:
            return "Prepare documents now — taxes are never urgent until they are"
        if any(w in text_lower for w in ("etf", "stock", "crypto", "invest")):
            return "Research options, understand the risk profile, start small"

    if category == "PERSONAL":
        if entities.get("names"):
            return f"Reach out to {entities['names'][0]} — don't let this drift further"
        if entities.get("dates"):
            return f"Lock this in for {entities['dates'][0]} — add to calendar now"
        if features.get("urgency_score", 0) > 0.5:
            return "This is urgent — handle it before anything else today"

    if category == "AUTOMATION":
        verbs = features.get("verbs", [])
        if verbs:
            return f"Start by automating the '{verbs[0]}' step — build small, iterate"

    if category == "PROJECTS":
        if features.get("urgency_score", 0) > 0.5:
            return "High-priority project item — block time in your calendar this week"

    if category == "ADMIN":
        if features.get("is_question"):
            return "Find the answer, action it, close the loop — don't let it hang"

    # ── Generic pool fallback ─────────────────────────────────────────────────
    pool = _POOLS.get(category, _POOLS["PERSONAL"])
    return random.choice(pool)
