# CrowdVision 🛰️

> **Human detection & density estimation** from road images and drone videos.  
> Powered by YOLOv8 · FastAPI · Vanilla JS

---

## Project Structure

```
crowd-vision/
├── detector.py          # Core YOLOv8 detection logic
├── server.py            # FastAPI backend (REST API)
├── static/
│   └── index.html       # Single-file frontend UI
├── outputs/             # Auto-created – stores annotated files
├── requirements.txt
├── Dockerfile
├── render.yaml          # Render.com IaC config
└── README.md
```

---

## Quick Start (Local)

### 1. Prerequisites

- Python 3.10 or 3.11
- pip

### 2. Clone & install

```bash
git clone <your-repo-url> crowd-vision
cd crowd-vision

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the server

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.  
The YOLOv8n weights (~6 MB) are downloaded automatically on first run.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Model readiness check |
| POST | `/analyze/image` | Detect humans in an image |
| POST | `/analyze/video` | Detect humans in a video |
| GET | `/outputs/{file}` | Download annotated output |

### POST `/analyze/image`

**Form fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | — | JPEG / PNG / BMP image |
| `area_sq_km` | float | 0.1 | Ground area covered by the image (sq km) |

**Response**

```json
{
  "type": "image",
  "people_count": 14,
  "density_per_sq_km": 140.0,
  "area_sq_km": 0.1,
  "confidence_scores": [0.91, 0.87, ...],
  "annotated_image_b64": "<base64>",
  "annotated_image_url": "/outputs/abc123.jpg"
}
```

### POST `/analyze/video`

**Form fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | — | MP4 / AVI / MOV video |
| `area_sq_km` | float | 0.1 | Ground area per frame (sq km) |
| `frame_skip` | int | 5 | Process every N-th frame |

**Response**

```json
{
  "type": "video",
  "frames_processed": 120,
  "average_count": 8.3,
  "peak_count": 22,
  "peak_timestamp_sec": 14.2,
  "average_density_per_sq_km": 83.0,
  "peak_density_per_sq_km": 220.0,
  "area_sq_km": 0.1,
  "frame_counts": [6, 8, 10, ...],
  "timestamps": [0.0, 0.2, 0.4, ...],
  "annotated_video_url": "/outputs/annotated_output.mp4"
}
```

---

## Configuration

All settings can be passed as environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL` | `yolov8n.pt` | Model variant (n/s/m/l/x) |
| `YOLO_CONF` | `0.35` | Detection confidence threshold |
| `YOLO_DEVICE` | `cpu` | `cpu`, `cuda`, or `mps` |
| `PORT` | `8000` | Server port |

**Model accuracy vs speed trade-off:**

| Model | Size | Speed (CPU) | mAP |
|-------|------|-------------|-----|
| yolov8n | 6 MB | ⚡⚡⚡ | Good |
| yolov8s | 22 MB | ⚡⚡ | Better |
| yolov8m | 50 MB | ⚡ | Best (practical) |

---

## Deployment

### Option 1 — Render.com (Easiest)

1. Push your project to a GitHub repository.
2. Go to [render.com](https://render.com) → **New → Blueprint**.
3. Connect your repo. Render detects `render.yaml` automatically.
4. Click **Apply** — Render builds the Docker image and deploys.
5. Visit the generated `*.onrender.com` URL.

> **Note:** Render's free plan has limited RAM. Use **Standard ($7/mo)** for video processing.

---

### Option 2 — AWS (EC2 + Docker)

```bash
# 1. Launch an EC2 instance (Ubuntu 22.04, t3.medium recommended)
# 2. SSH in and install Docker
sudo apt-get update && sudo apt-get install -y docker.io
sudo systemctl start docker

# 3. Build & run
git clone <your-repo-url> crowd-vision && cd crowd-vision
docker build -t crowdvision .
docker run -d -p 80:8000 --name cv crowdvision

# 4. Open port 80 in your EC2 Security Group
```

For GPU inference, use a `g4dn.xlarge` instance and add `--gpus all` to the docker run command, setting `YOLO_DEVICE=cuda`.

---

### Option 3 — GCP Cloud Run

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build & push to Artifact Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/crowdvision

# Deploy
gcloud run deploy crowdvision \
  --image gcr.io/YOUR_PROJECT_ID/crowdvision \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300
```

---

### Option 4 — Docker Compose (self-hosted)

```yaml
# docker-compose.yml
version: "3.9"
services:
  crowdvision:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./outputs:/app/outputs
    environment:
      - YOLO_MODEL=yolov8n.pt
      - YOLO_CONF=0.35
      - YOLO_DEVICE=cpu
    restart: unless-stopped
```

```bash
docker-compose up -d
```

---

## Performance Tips

- Use `frame_skip=10` or higher for long drone videos — the graph still captures trends.
- Switch to `yolov8n.pt` (default) for CPU deployments; `yolov8s.pt` gives ~15% accuracy gain at 2× cost.
- For GPU cloud instances, set `YOLO_DEVICE=cuda` and use a `yolov8m.pt` model for excellent accuracy.
- Large video files are streamed to disk (temp file) to avoid OOM errors.

---

## License

MIT