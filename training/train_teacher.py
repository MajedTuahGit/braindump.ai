"""
BrainDump.AI — Phase 6 BERT Teacher Model Training.
Fine-tunes bert-base-uncased on the seed dataset and exports it to models/teacher_bert.
"""
import sys
import json
import logging
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_teacher")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "training" / "data" / "seed_data.json"
MODEL_PATH = ROOT / "models" / "teacher_bert"
CATEGORIES = ["PERSONAL", "FINANCIAL", "PROJECTS", "ADMIN", "AUTOMATION"]


class ThoughtsDataset(torch.utils.data.Dataset):
    """Simple wrapper for PyTorch model inputs."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_data(path: Path):
    """Loads seed text and categories."""
    logger.info("Loading training dataset from %s...", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels


def main():
    if not DATA_PATH.exists():
        logger.error("Dataset not found at %s. Please compile seed data first.", DATA_PATH)
        sys.exit(1)

    # 1. Load data
    texts, labels = load_data(DATA_PATH)
    logger.info("Loaded %d thoughts for fine-tuning.", len(texts))

    # Map labels to category indices
    label_map = {cat: i for i, cat in enumerate(CATEGORIES)}
    label_ids = [label_map[l] for l in labels]

    # 2. Train/Test split for evaluation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, label_ids,
        test_size=0.2,
        random_state=42,
        stratify=label_ids,
    )
    logger.info("Split: %d train / %d validation samples.", len(train_texts), len(val_texts))

    # 3. Tokenize
    logger.info("Loading bert-base-uncased tokenizer from Hugging Face...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    logger.info("Tokenizing inputs...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)

    # 4. Wrap in PyTorch Dataset
    train_dataset = ThoughtsDataset(train_encodings, train_labels)
    val_dataset = ThoughtsDataset(val_encodings, val_labels)

    # 5. Load model architecture
    logger.info("Loading pre-trained bert-base-uncased model...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

    # 6. Training parameters optimized for CPU training speed inside Docker
    logger.info("Setting up Trainer arguments (optimised for CPU)...")
    training_args = TrainingArguments(
        output_dir=str(ROOT / "training" / "results"),
        num_train_epochs=3,                # 3 epochs is enough for seed convergence
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir=str(ROOT / "training" / "logs"),
        logging_steps=5,
        evaluation_strategy="epoch",       # evaluate metrics at the end of each epoch
        save_strategy="no",                # don't save checkpoints (saves disk space)
        no_cuda=True,                      # force CPU-only execution inside Docker container
        report_to="none"                   # disable third-party telemetry integrations
    )

    # 7. Initialize Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # 8. Train the model
    logger.info("Starting fine-tuning sequence (will run on CPU, ~1 minute)...")
    trainer.train()

    # 9. Save finalized model assets
    logger.info("Saving trained teacher model to %s...", MODEL_PATH)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(MODEL_PATH))
    model.save_pretrained(str(MODEL_PATH))

    logger.info("✅ BERT Teacher fine-tuning complete! Restart the Docker container to activate Tier 3.")


if __name__ == "__main__":
    main()
