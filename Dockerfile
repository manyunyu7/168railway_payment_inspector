# ─── Payment Inspector — production image ──────────────────────────
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/.torch \
    OMP_NUM_THREADS=2

# System deps: tesseract OCR + language data + libgl (torchvision needs)
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-eng \
      libgl1 libglib2.0-0 \
      curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (CPU-only torch to shave ~2GB)
COPY requirements.prod.txt /app/
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2 \
    && pip install --no-cache-dir -r requirements.prod.txt

# Copy app
COPY api/ /app/api/
COPY models/stage_a_best.pt /app/models/stage_a_best.pt

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -sf http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
