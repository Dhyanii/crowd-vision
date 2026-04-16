"""
server.py - FastAPI backend for CrowdVision.
Exposes endpoints for image and video analysis.
"""

import io
import os
import uuid
import logging
import tempfile
import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from detector import HumanDetector, DetectionResult, VideoResult

# ──────────────────────────────────────────────
# Logging & App Setup
# ──────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CrowdVision API",
    description="Human detection and density estimation from images and drone videos.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Output directory (persists annotated files) ──
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Serve static output files ──
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# ── Serve the frontend ──
STATIC_DIR = Path("static")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Singleton detector (loads model once on startup) ──
MODEL_NAME  = os.getenv("YOLO_MODEL", "yolov8n.pt")
CONFIDENCE  = float(os.getenv("YOLO_CONF", "0.35"))
DEVICE      = os.getenv("YOLO_DEVICE", "cpu")
detector: Optional[HumanDetector] = None


@app.on_event("startup")
async def load_model():
    global detector
    logger.info("Initialising HumanDetector …")
    detector = HumanDetector(model_name=MODEL_NAME, confidence=CONFIDENCE, device=DEVICE)
    logger.info("Model ready.")


# ──────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────

def _ndarray_to_b64(img: np.ndarray) -> str:
    """Encode a BGR numpy image as base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def _save_annotated_image(img: np.ndarray) -> str:
    """Save annotated image to OUTPUT_DIR and return its filename."""
    fname = f"{uuid.uuid4().hex}.jpg"
    fpath = OUTPUT_DIR / fname
    cv2.imwrite(str(fpath), img)
    return fname


def _cleanup_file(path: str):
    """Background task: delete a temporary file."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend HTML if present, else redirect to docs."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return JSONResponse({"message": "CrowdVision API is running. Visit /docs"})


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(..., description="Image file (JPEG / PNG / BMP)"),
    area_sq_km: float = Form(0.1, description="Ground area covered by image in sq km"),
):
    """
    Detect humans in an uploaded image.

    Returns count, density, confidence scores, and base64-encoded annotated image.
    """
    if detector is None:
        raise HTTPException(503, "Model not yet loaded")

    # ── Read upload into numpy array ──
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image. Ensure it is a valid JPEG/PNG/BMP.")

    result: DetectionResult = detector.process_image(image, area_sq_km=area_sq_km)

    # Save and return
    fname = _save_annotated_image(result.annotated_image)

    return {
        "type": "image",
        "people_count": result.count,
        "density_per_sq_km": result.density,
        "area_sq_km": result.area_sq_km,
        "confidence_scores": [round(s, 3) for s in result.confidence_scores],
        "annotated_image_b64": _ndarray_to_b64(result.annotated_image),
        "annotated_image_url": f"/outputs/{fname}",
    }


@app.post("/analyze/video")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file (MP4 / AVI / MOV)"),
    area_sq_km: float = Form(0.1, description="Ground area per frame in sq km"),
    frame_skip: int = Form(5, description="Process every N-th frame (higher = faster)"),
):
    """
    Detect humans in an uploaded video.

    Returns per-frame counts, peak/average stats, density estimates, and
    a URL to download the annotated output video.
    """
    if detector is None:
        raise HTTPException(503, "Model not yet loaded")

    # ── Stream upload to temp file ──
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()

        result: VideoResult = detector.process_video(
            video_path=tmp.name,
            area_sq_km=area_sq_km,
            frame_skip=max(1, frame_skip),
            output_dir=str(OUTPUT_DIR),
        )
    except Exception as exc:
        background_tasks.add_task(_cleanup_file, tmp.name)
        logger.exception("Video processing failed")
        raise HTTPException(500, f"Video processing error: {exc}") from exc

    background_tasks.add_task(_cleanup_file, tmp.name)

    # Build relative URL for annotated video
    video_fname = Path(result.annotated_video_path).name
    video_url = f"/outputs/{video_fname}"

    return {
        "type": "video",
        "frames_processed": result.total_frames_processed,
        "average_count": result.average_count,
        "peak_count": result.peak_count,
        "peak_timestamp_sec": result.peak_timestamp,
        "average_density_per_sq_km": result.average_density,
        "peak_density_per_sq_km": result.peak_density,
        "area_sq_km": result.area_sq_km,
        "frame_counts": result.frame_counts,
        "timestamps": result.timestamps,
        "annotated_video_url": video_url,
    }