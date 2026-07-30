"""
train_tier1.py — Train the TF-IDF + LinearSVC pipeline for Tier 1.

Usage (from project root):
    python training/train_tier1.py

Output:
    models/tier1_svm.pkl   ← auto-loaded by Tier1Classifier on next restart

What happens:
  1. Loads labeled data from training/data/seed_data.json
  2. Cleans text with TextCleaner (spaCy if available, regex fallback)
  3. Trains a TF-IDF + LinearSVC sklearn Pipeline
  4. Evaluates on a held-out split and prints a classification report
  5. Saves the pipeline to models/tier1_svm.pkl with joblib
"""

import json
import sys
import logging
import datetime
from pathlib import Path

# ── Make sure project root is on sys.path so `ml` imports work ────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_tier1")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = ROOT / "training" / "data" / "seed_data.json"

def get_timestamped_model_path() -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "models" / f"tier1_svm_{ts}.pkl"

# ── Category order must match tier1_tfidf.py ─────────────────────────────────
CATEGORIES = ["PERSONAL", "FINANCIAL", "PROJECTS", "ADMIN", "AUTOMATION"]


def load_data(path: Path):
    """Load [{text, label}, ...] and return (texts, labels) lists."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    texts, labels = [], []
    skipped = 0
    for r in records:
        label = r["label"].upper().strip()
        if label not in CATEGORIES:
            logger.warning("Skipping unknown label '%s' — text: %s", label, r["text"][:60])
            skipped += 1
            continue
        texts.append(r["text"])
        labels.append(label)

    logger.info("Loaded %d samples (%d skipped). Distribution:", len(texts), skipped)
    from collections import Counter
    for cat, count in sorted(Counter(labels).items()):
        logger.info("  %-12s %d", cat, count)

    return texts, labels


def clean_texts(texts):
    """Run TextCleaner on all samples. Falls back to regex if spaCy is absent."""
    try:
        from ml.preprocessing.cleaner import TextCleaner
        cleaner = TextCleaner()
        logger.info("TextCleaner ready (spaCy: see above for availability)")
        cleaned = [cleaner.clean(t) for t in texts]
    except Exception as exc:
        logger.warning("TextCleaner import failed (%s) — using raw text", exc)
        cleaned = texts
    return cleaned


def build_pipeline():
    """Return an untrained TF-IDF + LinearSVC sklearn Pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    # CalibratedClassifierCV wraps LinearSVC so we can call predict_proba later
    # (tier1_tfidf.py uses decision_function + manual softmax, but having
    #  predict_proba available is handy for evaluation / future use)
    svc = LinearSVC(C=1.0, max_iter=3000, class_weight="balanced")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30_000,
            sublinear_tf=True,      # log(1+tf) — helps with short texts
            min_df=1,               # keep rare tokens (small dataset)
            strip_accents="unicode",
        )),
        ("svm", svc),
    ])
    return pipe


def train(pipe, X_train, y_train):
    logger.info("Training TF-IDF + LinearSVC on %d samples…", len(X_train))
    pipe.fit(X_train, y_train)
    logger.info("Training complete.")
    return pipe


def evaluate(pipe, X_test, y_test):
    from sklearn.metrics import classification_report, accuracy_score
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Test accuracy: %.1f%%", acc * 100)
    print("\n── Classification Report ─────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=CATEGORIES, zero_division=0))


def save_model(pipe, path: Path):
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    size_kb = path.stat().st_size / 1024
    logger.info("Model saved → %s  (%.1f KB)", path, size_kb)


def save_plots(pipe, texts, labels, X_test, y_test, suffix: str = ""):
    """Generate and save visual evaluation plots for the frontend."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # headless execution inside Docker
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        plots_dir = ROOT / "frontend" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        dist_filename = f"class_distribution_{suffix}.png" if suffix else "class_distribution.png"
        cm_filename = f"confusion_matrix_{suffix}.png" if suffix else "confusion_matrix.png"
        feat_filename = f"feature_importance_{suffix}.png" if suffix else "feature_importance.png"
        
        logger.info("Generating evaluation plots in %s with filenames: %s, %s, %s...", 
                    plots_dir, dist_filename, cm_filename, feat_filename)

        # 1. Class Distribution
        plt.figure(figsize=(7, 4))
        df = pd.DataFrame({"label": labels})
        counts = df['label'].value_counts()
        colors = ['#3b82f6', '#22c55e', '#06b6d4', '#a855f7', '#f59e0b']
        bars = plt.bar(counts.index, counts.values, color=colors[:len(counts)])
        plt.title('Training Data Distribution')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, str(yval), ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(plots_dir / dist_filename, dpi=120)
        plt.close()

        # 2. Confusion Matrix
        fig, ax = plt.subplots(figsize=(6, 6))
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=CATEGORIES)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CATEGORIES)
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
        plt.title('Confusion Matrix')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(plots_dir / cm_filename, dpi=120)
        plt.close()

        # 3. Feature Importance (Coefficients)
        vectorizer = pipe.named_steps['tfidf']
        classifier = pipe.named_steps['svm']
        feature_names = np.array(vectorizer.get_feature_names_out())

        plt.figure(figsize=(12, 8))
        for i, category in enumerate(classifier.classes_):
            coefs = classifier.coef_[i] if len(classifier.classes_) > 2 else classifier.coef_[0]
            if len(classifier.classes_) == 2 and i == 0:
                coefs = -coefs

            top_indices = np.argsort(coefs)[-6:]
            top_features = feature_names[top_indices]
            top_coefs = coefs[top_indices]

            plt.subplot(3, 2, i + 1)
            y_pos = np.arange(len(top_features))
            plt.barh(y_pos, top_coefs, align='center', color=colors[i % len(colors)], alpha=0.8)
            plt.yticks(y_pos, top_features)
            plt.title(f'{category} Top Words', fontsize=10, fontweight='bold')
            plt.grid(axis='x', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(plots_dir / feat_filename, dpi=120)
        plt.close()
        logger.info("Successfully generated class distribution, confusion matrix, and feature importance plots.")
    except Exception as exc:
        logger.error("Plot generation failed: %s", exc)


def main():
    # ── 1. Load ───────────────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        logger.error("Training data not found: %s", DATA_PATH)
        sys.exit(1)

    texts, labels = load_data(DATA_PATH)

    if len(texts) < 10:
        logger.error("Too few samples (%d). Add more to seed_data.json.", len(texts))
        sys.exit(1)

    # ── 2. Clean ──────────────────────────────────────────────────────────────
    cleaned = clean_texts(texts)

    # ── 3. Train / test split ─────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    logger.info("Split: %d train / %d test", len(X_train), len(X_test))

    # ── 4. Build + train ──────────────────────────────────────────────────────
    pipe = build_pipeline()
    pipe = train(pipe, X_train, y_train)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    evaluate(pipe, X_test, y_test)

    target_path = get_timestamped_model_path()
    ts_suffix = target_path.name.replace("tier1_svm_", "").replace(".pkl", "")

    # ── 5.5 Save Plots ────────────────────────────────────────────────────────
    save_plots(pipe, texts, labels, X_test, y_test, suffix=ts_suffix)

    # ── 6. Retrain on full dataset then save ──────────────────────────────────
    logger.info("Retraining on full dataset (%d samples) before saving…", len(cleaned))
    pipe.fit(cleaned, labels)
    save_model(pipe, target_path)

    # ── 7. Quick smoke test ───────────────────────────────────────────────────
    print("\n── Smoke Test (5 new phrases) ────────────────────────────")
    test_phrases = [
        "I need to call my mum",
        "buy more ETF shares this month",
        "ship the MVP by Friday",
        "reply to the client invoice email",
        "set up a cron job to backup the database nightly",
    ]
    import numpy as np
    for phrase in test_phrases:
        scores = pipe.decision_function([phrase])[0]
        exp_s  = np.exp(scores - np.max(scores))
        probs  = exp_s / exp_s.sum()
        best   = CATEGORIES[int(np.argmax(probs))]
        conf   = float(probs[np.argmax(probs)])
        print(f"  [{best:<12} {conf:.0%}]  {phrase}")

    print("\n✅ Tier 1 training complete. Restart the server to load the new model.")


if __name__ == "__main__":
    main()
