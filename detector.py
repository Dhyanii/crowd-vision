"""
detector.py - Core YOLO-based human detection module.
Handles both image and video processing with density estimation.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import tempfile
import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Data classes for structured results
# ──────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Result from processing a single image."""
    count: int
    density: float                  # people / sq km
    annotated_image: np.ndarray     # BGR image with bounding boxes
    confidence_scores: list[float] = field(default_factory=list)
    area_sq_km: float = 0.0


@dataclass
class VideoResult:
    """Aggregated result from processing a video file."""
    total_frames_processed: int
    frame_counts: list[int]         # person count per processed frame
    timestamps: list[float]         # seconds for each frame
    average_count: float
    peak_count: int
    peak_timestamp: float
    average_density: float          # people / sq km
    peak_density: float
    area_sq_km: float
    annotated_video_path: str       # path to output MP4


# ──────────────────────────────────────────────
# Detector class
# ──────────────────────────────────────────────

class HumanDetector:
    """
    Wraps YOLOv8 for human-only detection.
    
    Args:
        model_name: YOLO model variant (yolov8n / yolov8s / yolov8m …)
        confidence:  Minimum confidence threshold (0–1).
        device:      'cpu', 'cuda', or 'mps'.
    """

    PERSON_CLASS_ID = 0  # COCO class index for 'person'

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.35,
        device: str = "cpu",
    ):
        self.confidence = confidence
        self.device = device
        logger.info(f"Loading YOLO model: {model_name} on {device}")
        self.model = YOLO(model_name)
        self.model.to(device)

    # ── helpers ──────────────────────────────

    def _area_to_sq_km(self, area_m2: float) -> float:
        """Convert square metres to square kilometres."""
        return area_m2 / 1_000_000.0

    def _calculate_density(self, count: int, area_sq_km: float) -> float:
        """People per square kilometre; avoids division by zero."""
        if area_sq_km <= 0:
            return 0.0
        return round(count / area_sq_km, 2)

    def _run_inference(self, frame: np.ndarray):
        """
        Run YOLO inference on a single BGR frame.
        Returns (boxes, scores) for person class only.
        """
        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=[self.PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else np.array([])
        scores = result.boxes.conf.cpu().numpy() if result.boxes else np.array([])
        return boxes, scores

    def _draw_annotations(
        self, frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, count: int
    ) -> np.ndarray:
        """
        Draw bounding boxes, labels, and HUD overlay on the frame.
        Returns a copy of the frame with annotations.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # ── bounding boxes ──
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 100), 2)
            label = f"{score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 100), -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # ── HUD: count badge ──
        badge_text = f"Persons: {count}"
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cv2.rectangle(annotated, (10, 10), (bw + 20, bh + 20), (20, 20, 20), -1)
        cv2.putText(annotated, badge_text, (15, bh + 14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 100), 2, cv2.LINE_AA)
        return annotated

    # ── public API ───────────────────────────

    def process_image(
        self,
        image: np.ndarray,
        area_sq_km: float = 0.1,
    ) -> DetectionResult:
        """
        Detect humans in a single image.

        Args:
            image:      BGR numpy array (as returned by cv2.imread).
            area_sq_km: Ground area the image covers (sq km).

        Returns:
            DetectionResult with count, density, and annotated image.
        """
        boxes, scores = self._run_inference(image)
        count = len(boxes)
        density = self._calculate_density(count, area_sq_km)
        annotated = self._draw_annotations(image, boxes, scores, count)

        return DetectionResult(
            count=count,
            density=density,
            annotated_image=annotated,
            confidence_scores=scores.tolist(),
            area_sq_km=area_sq_km,
        )

    def process_video(
        self,
        video_path: str,
        area_sq_km: float = 0.1,
        frame_skip: int = 5,          # process every Nth frame for speed
        output_dir: Optional[str] = None,
        progress_callback=None,       # callback(progress_fraction)
    ) -> VideoResult:
        """
        Process a video file frame-by-frame.

        Args:
            video_path:        Path to source video.
            area_sq_km:        Ground area per frame (sq km).
            frame_skip:        Process every N-th frame (1 = every frame).
            output_dir:        Directory for annotated output video.
            progress_callback: Optional callable(float) for progress updates.

        Returns:
            VideoResult with per-frame counts and paths to output video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── set up output writer ──
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "annotated_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps / frame_skip, (width, height))

        frame_counts: list[int]   = []
        timestamps:   list[float] = []
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_skip == 0:
                boxes, scores = self._run_inference(frame)
                count = len(boxes)
                ts = frame_index / fps

                frame_counts.append(count)
                timestamps.append(round(ts, 2))

                annotated = self._draw_annotations(frame, boxes, scores, count)

                # Burn timestamp onto frame
                ts_label = f"t={ts:.1f}s"
                cv2.putText(annotated, ts_label, (width - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
                writer.write(annotated)

                if progress_callback and total_frames > 0:
                    progress_callback(frame_index / total_frames)

            frame_index += 1

        cap.release()
        writer.release()

        if not frame_counts:
            raise ValueError("No frames were processed from the video.")

        avg_count  = round(np.mean(frame_counts), 2)
        peak_count = int(np.max(frame_counts))
        peak_idx   = int(np.argmax(frame_counts))

        return VideoResult(
            total_frames_processed=len(frame_counts),
            frame_counts=frame_counts,
            timestamps=timestamps,
            average_count=avg_count,
            peak_count=peak_count,
            peak_timestamp=timestamps[peak_idx],
            average_density=self._calculate_density(avg_count, area_sq_km),
            peak_density=self._calculate_density(peak_count, area_sq_km),
            area_sq_km=area_sq_km,
            annotated_video_path=out_path,
        )