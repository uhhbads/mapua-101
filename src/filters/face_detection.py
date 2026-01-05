"""
Face detection utilities using MediaPipe Tasks API.
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import urllib.request
import os


# Model download URL
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MODEL_FILENAME = "blaze_face_short_range.tflite"


@dataclass
class FaceDetection:
    """Represents a detected face."""
    x: int  # Top-left x
    y: int  # Top-left y
    width: int
    height: int
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        """Get center point of face box."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Get bounding box as (x, y, w, h)."""
        return (self.x, self.y, self.width, self.height)


def _ensure_model_downloaded(models_dir: Path) -> Path:
    """Download model if not present."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILENAME
    
    if not model_path.exists():
        print(f"Downloading face detection model...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"Model downloaded to {model_path}")
    
    return model_path


class FaceDetector:
    """MediaPipe Tasks-based face detector."""

    def __init__(self, min_confidence: float = 0.5, model_selection: int = 0):
        """
        Initialize face detector.
        
        Args:
            min_confidence: Minimum detection confidence (0.0-1.0)
            model_selection: 0 for short range, 1 for full range (not used in new API)
        """
        self.min_confidence = min_confidence
        self._detector: Optional[vision.FaceDetector] = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize MediaPipe face detection."""
        # Get model path (in assets folder relative to project root)
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "assets" / "models"
        model_path = _ensure_model_downloaded(models_dir)
        
        # Create detector options
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self.min_confidence,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        """
        Detect faces in a frame.
        
        Args:
            frame: BGR image (OpenCV format)
            
        Returns:
            List of FaceDetection objects
        """
        if self._detector is None:
            return []

        h, w = frame.shape[:2]

        # Convert BGR to RGB for MediaPipe
        rgb_frame = frame[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect faces
        result = self._detector.detect(mp_image)

        faces = []
        for detection in result.detections:
            bbox = detection.bounding_box
            
            # Get absolute pixel coordinates
            x = bbox.origin_x
            y = bbox.origin_y
            width = bbox.width
            height = bbox.height
            
            # Clamp to frame bounds
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)

            # Get confidence score
            confidence = detection.categories[0].score if detection.categories else 0.5

            faces.append(FaceDetection(
                x=int(x),
                y=int(y),
                width=int(width),
                height=int(height),
                confidence=confidence,
            ))

        return faces

    def release(self) -> None:
        """Release detector resources."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None
