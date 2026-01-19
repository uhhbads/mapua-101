"""
Face detection utilities using OpenCV YuNet.
"""

import cv2
import numpy as np
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import urllib.request


# YuNet model download URL (OpenCV Zoo)
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"


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
    """Download YuNet model if not present."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILENAME
    
    if not model_path.exists():
        print(f"Downloading YuNet face detection model...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"Model downloaded to {model_path}")
    
    return model_path


class FaceDetector:
    """OpenCV YuNet-based face detector."""

    def __init__(self, min_confidence: float = 0.5, max_faces: int = 8):
        """
        Initialize face detector.
        
        Args:
            min_confidence: Minimum detection confidence (0.0-1.0)
            max_faces: Maximum number of faces to detect (default: 8 for group shots)
        """
        self.min_confidence = min_confidence
        self.max_faces = max_faces
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._current_size: tuple[int, int] = (0, 0)
        self._model_path: Optional[Path] = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize YuNet face detection."""
        # Get model path (in assets folder relative to project root)
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "assets" / "models"
        self._model_path = _ensure_model_downloaded(models_dir)
        
        # Detector will be created on first detect() call when we know the frame size
        self._detector = None

    def _create_detector(self, width: int, height: int) -> None:
        """Create or recreate detector for given frame size."""
        if self._model_path is None:
            return
            
        self._detector = cv2.FaceDetectorYN.create(
            str(self._model_path),
            "",  # No config file needed
            (width, height),
            self.min_confidence,
            0.3,  # NMS threshold
            self.max_faces
        )
        self._current_size = (width, height)

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        """
        Detect faces in a frame.
        
        Args:
            frame: BGR image (OpenCV format)
            
        Returns:
            List of FaceDetection objects
        """
        h, w = frame.shape[:2]
        
        # Create or resize detector if needed
        if self._detector is None or self._current_size != (w, h):
            self._create_detector(w, h)
        
        if self._detector is None:
            return []

        # Detect faces - YuNet works directly with BGR
        _, detections = self._detector.detect(frame)

        faces = []
        if detections is not None:
            for det in detections:
                # YuNet returns: x, y, w, h, right_eye_x, right_eye_y, left_eye_x, left_eye_y,
                # nose_x, nose_y, right_mouth_x, right_mouth_y, left_mouth_x, left_mouth_y, confidence
                x = int(det[0])
                y = int(det[1])
                width = int(det[2])
                height = int(det[3])
                confidence = float(det[14])
                
                # Clamp to frame bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                if width > 0 and height > 0:
                    faces.append(FaceDetection(
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        confidence=confidence,
                    ))

        return faces

    def release(self) -> None:
        """Release detector resources."""
        self._detector = None
