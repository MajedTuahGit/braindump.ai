# ────────────────────────────────────────────────────────────
#  BrainDump.AI — Dockerfile
#  Base: python:3.11-slim  (keeps image lean)
#  PyTorch: CPU-only build (~800MB vs ~3GB CUDA build)
# ────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# ── System libs needed by spaCy + PyTorch CPU ─────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps: split into two layers for smart caching ──────
# Layer 1 — PyTorch CPU (installed separately to avoid the 3GB CUDA download)
# If requirements.txt changes, this layer is still cached → fast rebuild
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.3.0 \
        --index-url https://download.pytorch.org/whl/cpu

# Layer 2 — everything else
# torch is already satisfied at the right version, pip skips it
RUN pip install --no-cache-dir -r requirements.txt

# ── NLP model data (cached layer — only re-runs if pip layer above changes) ──
RUN python -m spacy download en_core_web_sm && \
    python -c "\
import nltk; \
nltk.download('wordnet',  quiet=True); \
nltk.download('punkt',    quiet=True); \
nltk.download('averaged_perceptron_tagger', quiet=True)"

# ── Application code (last — code changes only invalidate this layer) ─────────
COPY . .

# Create directories that need to exist at runtime
RUN mkdir -p models db

EXPOSE 8000

# --host 0.0.0.0 is required inside Docker so the port is accessible from outside
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload"]
