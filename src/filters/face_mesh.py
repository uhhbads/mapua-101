"""
Face mesh/landmarks utilities using MediaPipe Tasks API.
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import urllib.request


# Model download URL
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_FILENAME = "face_landmarker.task"


@dataclass
class FaceLandmarks:
    """Represents face landmarks for a detected face."""
    landmarks: np.ndarray  # Shape (478, 3) - x, y, z normalized coords
    frame_width: int
    frame_height: int

    def get_point(self, index: int) -> tuple[int, int]:
        """Get a landmark point in pixel coordinates."""
        lm = self.landmarks[index]
        return (int(lm[0] * self.frame_width), int(lm[1] * self.frame_height))

    # Key landmark indices for dog filter
    # Forehead/top of head area
    @property
    def forehead_center(self) -> tuple[int, int]:
        return self.get_point(10)  # Top of forehead

    @property
    def left_ear_anchor(self) -> tuple[int, int]:
        """Left side of forehead for left ear placement."""
        return self.get_point(71)  # Left forehead

    @property
    def right_ear_anchor(self) -> tuple[int, int]:
        """Right side of forehead for right ear placement."""
        return self.get_point(301)  # Right forehead

    @property
    def nose_tip(self) -> tuple[int, int]:
        return self.get_point(4)  # Nose tip

    @property
    def left_eye_center(self) -> tuple[int, int]:
        return self.get_point(468)  # Left eye center (iris)

    @property
    def right_eye_center(self) -> tuple[int, int]:
        return self.get_point(473)  # Right eye center (iris)

    @property
    def face_width(self) -> int:
        """Approximate face width in pixels."""
        left = self.get_point(234)  # Left cheek
        right = self.get_point(454)  # Right cheek
        return abs(right[0] - left[0])

    @property
    def face_height(self) -> int:
        """Approximate face height in pixels."""
        top = self.get_point(10)  # Forehead
        bottom = self.get_point(152)  # Chin
        return abs(bottom[1] - top[1])


def _ensure_model_downloaded(models_dir: Path) -> Path:
    """Download model if not present."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILENAME
    
    if not model_path.exists():
        print(f"Downloading face landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"Model downloaded to {model_path}")
    
    return model_path


class FaceMesh:
    """MediaPipe Tasks-based face mesh/landmarker."""

    def __init__(self, min_confidence: float = 0.5, max_faces: int = 4):
        """
        Initialize face mesh.
        
        Args:
            min_confidence: Minimum detection confidence (0.0-1.0)
            max_faces: Maximum number of faces to detect
        """
        self.min_confidence = min_confidence
        self.max_faces = max_faces
        self._landmarker: Optional[vision.FaceLandmarker] = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize MediaPipe face landmarker."""
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "assets" / "models"
        model_path = _ensure_model_downloaded(models_dir)
        
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self.max_faces,
            min_face_detection_confidence=self.min_confidence,
            min_face_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> list[FaceLandmarks]:
        """
        Detect face landmarks in a frame.
        
        Args:
            frame: BGR image (OpenCV format)
            
        Returns:
            List of FaceLandmarks objects
        """
        if self._landmarker is None:
            return []

        h, w = frame.shape[:2]

        # Convert BGR to RGB
        rgb_frame = frame[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = self._landmarker.detect(mp_image)

        faces = []
        for face_landmarks in result.face_landmarks:
            # Convert to numpy array
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in face_landmarks
            ])
            
            faces.append(FaceLandmarks(
                landmarks=landmarks,
                frame_width=w,
                frame_height=h,
            ))

        return faces

    def release(self) -> None:
        """Release resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
