"""
Router for triggering in-memory model retraining and model version control.
"""
import logging
import time
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import functions from our training script
from training.train_tier1 import (
    load_data, clean_texts, build_pipeline, train,
    save_plots, save_model, DATA_PATH, get_timestamped_model_path, CATEGORIES
)

logger = logging.getLogger("api.routers.train")
router = APIRouter()


class ActivateModelRequest(BaseModel):
    version: str


@router.post("/train/tier1")
async def retrain_tier1(request: Request):
    """
    Retrains the Tier 1 SVM model using current dataset in seed_data.json,
    saves the model with a unique timestamp, and hot-swaps it in-memory.
    """
    start_time = time.perf_counter()
    logger.info("[API] Triggered Tier 1 SVM model retraining...")

    try:
        # 1. Load the updated seed data
        if not DATA_PATH.exists():
            raise HTTPException(status_code=400, detail=f"Training seed data not found at {DATA_PATH}")
            
        texts, labels = load_data(DATA_PATH)
        if len(texts) < 10:
            raise HTTPException(status_code=400, detail="Too few training samples to train (minimum 10 required).")

        # 2. Preprocess text
        cleaned = clean_texts(texts)

        # 3. Train/Test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            cleaned, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )

        # 4. Build and train pipeline
        pipe = build_pipeline()
        pipe = train(pipe, X_train, y_train)

        # 5. Evaluate accuracy
        y_pred = pipe.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, target_names=CATEGORIES, zero_division=0)

        # 6. Generate path and save static charts for the frontend
        target_path = get_timestamped_model_path()
        ts_suffix = target_path.name.replace("tier1_svm_", "").replace(".pkl", "")
        save_plots(pipe, texts, labels, X_test, y_test, suffix=ts_suffix)

        # 7. Retrain on full dataset and save to disk
        logger.info("[API] Retraining on all %d samples before exporting to %s...", len(cleaned), target_path)
        pipe.fit(cleaned, labels)
        save_model(pipe, target_path)

        # 8. Dynamic in-memory reload
        t1_model = request.app.state.models.get("tier1")
        if t1_model:
            t1_model.load(specific_version=target_path.name)
            logger.info("[API] In-memory hot-swap of Tier 1 SVM complete!")
        else:
            logger.warning("[API] Tier 1 model instance not found in app state. Skipping hot-swap.")

        duration = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info("[API] Retraining finished in %.2f ms. Accuracy: %.2f%%", duration, accuracy * 100)

        return {
            "success": True,
            "accuracy": accuracy,
            "samples": len(texts),
            "duration_ms": duration,
            "report": report,
            "active_version": target_path.name,
        }

    except Exception as exc:
        logger.error("[API] Retraining failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/train/versions")
async def get_model_versions(request: Request):
    """
    Scans the models/ directory and returns list of all saved SVM model versions,
    identifying which version is currently loaded/active in RAM.
    """
    models_dir = Path("models")
    versions = []

    if models_dir.exists():
        # Match versioned models: tier1_svm_*.pkl
        timestamped_files = sorted([f.name for f in models_dir.glob("tier1_svm_*.pkl")], reverse=True)
        versions.extend(timestamped_files)

        # Legacy fallback
        legacy_path = models_dir / "tier1_svm.pkl"
        if legacy_path.exists():
            versions.append("tier1_svm.pkl")

    t1_model = request.app.state.models.get("tier1")
    active_version = t1_model.active_version if t1_model else None

    return {
        "active_version": active_version,
        "versions": versions
    }


@router.post("/train/activate")
async def activate_model_version(body: ActivateModelRequest, request: Request):
    """
    Switches the active in-memory model to the requested version file name.
    """
    t1_model = request.app.state.models.get("tier1")
    if not t1_model:
        raise HTTPException(status_code=500, detail="Tier 1 model wrapper is not initialized in server state.")

    # Clean version name string (in case the client sent extra comments/tags)
    version_filename = body.version.strip()
    
    logger.info("[API] Request to switch Tier 1 model to version: %s", version_filename)
    success = t1_model.load(specific_version=version_filename)
    
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to load model version '{version_filename}'. Check filename and server logs.")
        
    return {
        "success": True,
        "active_version": t1_model.active_version
    }

