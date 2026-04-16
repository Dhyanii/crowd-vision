# ── Stage 1: base image ─────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 2: install Python deps ─────────────────────────────────────────────
FROM base AS deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n weights so container starts instantly
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ── Stage 3: final image ──────────────────────────────────────────────────────
FROM deps AS final
COPY . .

# Create output directory
RUN mkdir -p outputs

# Environment variables (overridable at runtime)
ENV YOLO_MODEL=yolov8n.pt
ENV YOLO_CONF=0.35
ENV YOLO_DEVICE=cpu
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]