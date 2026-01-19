"""
Dog Ear Filter: Snapchat-style dog ears and nose overlay.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from .base import BaseFilter
from .face_mesh import FaceMesh, FaceLandmarks


class DogEarFilter(BaseFilter):
    """
    Overlays dog ears and nose on detected faces using landmarks.
    """

    name = "Dog Filter"

    def __init__(self):
        """Initialize dog ear filter for group shots (5-7+ people)."""
        self.face_mesh = FaceMesh(min_confidence=0.5, max_faces=8)
        
        # Load or generate assets
        self._left_ear: Optional[np.ndarray] = None
        self._right_ear: Optional[np.ndarray] = None
        self._nose: Optional[np.ndarray] = None
        self._load_assets()

    def _load_assets(self) -> None:
        """Load or generate dog ear/nose assets."""
        assets_dir = Path(__file__).parent.parent.parent / "assets" / "overlays"
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        left_ear_path = assets_dir / "dog_ear_left.png"
        right_ear_path = assets_dir / "dog_ear_right.png"
        nose_path = assets_dir / "dog_nose.png"
        
        # Try to load existing assets, otherwise generate simple ones
        if left_ear_path.exists():
            self._left_ear = cv2.imread(str(left_ear_path), cv2.IMREAD_UNCHANGED)
        else:
            self._left_ear = self._generate_ear(flip=False)
            cv2.imwrite(str(left_ear_path), self._left_ear)
            
        if right_ear_path.exists():
            self._right_ear = cv2.imread(str(right_ear_path), cv2.IMREAD_UNCHANGED)
        else:
            self._right_ear = self._generate_ear(flip=True)
            cv2.imwrite(str(right_ear_path), self._right_ear)
            
        if nose_path.exists():
            self._nose = cv2.imread(str(nose_path), cv2.IMREAD_UNCHANGED)
        else:
            self._nose = self._generate_nose()
            cv2.imwrite(str(nose_path), self._nose)

    def _generate_ear(self, flip: bool = False) -> np.ndarray:
        """Generate a simple dog ear shape (BGRA)."""
        size = 150
        ear = np.zeros((size, size, 4), dtype=np.uint8)
        
        # Draw a floppy ear shape
        # Main ear color (brown)
        color = (60, 90, 140, 255)  # BGRA - brownish
        inner_color = (80, 120, 180, 255)  # Lighter inner
        
        # Draw ear outline (rounded triangle-ish)
        pts = np.array([
            [size // 2, 10],      # Top point
            [size - 20, size - 30],  # Bottom right
            [20, size - 30],      # Bottom left
        ], np.int32)
        
        cv2.fillPoly(ear, [pts], color)
        
        # Inner ear
        inner_pts = np.array([
            [size // 2, 30],
            [size - 40, size - 50],
            [40, size - 50],
        ], np.int32)
        cv2.fillPoly(ear, [inner_pts], inner_color)
        
        if flip:
            ear = cv2.flip(ear, 1)
            
        return ear

    def _generate_nose(self) -> np.ndarray:
        """Generate a simple dog nose shape (BGRA)."""
        size = 80
        nose = np.zeros((size, size, 4), dtype=np.uint8)
        
        # Black nose
        center = (size // 2, size // 2)
        
        # Main nose oval
        cv2.ellipse(nose, center, (size // 3, size // 4), 0, 0, 360, (20, 20, 20, 255), -1)
        
        # Nostrils
        cv2.ellipse(nose, (center[0] - 12, center[1] + 5), (8, 6), 0, 0, 360, (0, 0, 0, 255), -1)
        cv2.ellipse(nose, (center[0] + 12, center[1] + 5), (8, 6), 0, 0, 360, (0, 0, 0, 255), -1)
        
        # Highlight
        cv2.ellipse(nose, (center[0], center[1] - 8), (10, 6), 0, 0, 360, (60, 60, 60, 255), -1)
        
        return nose

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process frame: detect faces and overlay dog features."""
        result = frame.copy()
        faces = self.face_mesh.detect(frame)

        for face in faces:
            self._apply_dog_overlay(result, face)

        # Draw filter label
        self._draw_filter_label(result)

        return result

    def _apply_dog_overlay(self, frame: np.ndarray, face: FaceLandmarks) -> None:
        """Apply dog ears and nose to a face."""
        # Calculate scale based on face size
        face_width = face.face_width
        scale = face_width / 200.0  # Base scale factor

        # Left ear
        if self._left_ear is not None:
            left_anchor = face.left_ear_anchor
            ear_size = int(100 * scale)
            self._overlay_image(
                frame, 
                self._left_ear, 
                left_anchor[0] - ear_size // 2,
                left_anchor[1] - ear_size,
                ear_size,
                ear_size,
            )

        # Right ear
        if self._right_ear is not None:
            right_anchor = face.right_ear_anchor
            ear_size = int(100 * scale)
            self._overlay_image(
                frame,
                self._right_ear,
                right_anchor[0] - ear_size // 2,
                right_anchor[1] - ear_size,
                ear_size,
                ear_size,
            )

        # Nose
        if self._nose is not None:
            nose_pos = face.nose_tip
            nose_size = int(50 * scale)
            self._overlay_image(
                frame,
                self._nose,
                nose_pos[0] - nose_size // 2,
                nose_pos[1] - nose_size // 3,
                nose_size,
                nose_size,
            )

    def _overlay_image(
        self,
        background: np.ndarray,
        overlay: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        """Overlay an RGBA image onto the background at specified position."""
        if overlay is None or width <= 0 or height <= 0:
            return

        h, w = background.shape[:2]
        
        # Resize overlay
        resized = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_AREA)
        
        # Calculate valid region
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + width), min(h, y + height)
        
        if x1 >= x2 or y1 >= y2:
            return
        
        # Crop overlay to valid region
        ox1, oy1 = x1 - x, y1 - y
        ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
        
        if ox1 < 0 or oy1 < 0 or ox2 > width or oy2 > height:
            return
            
        overlay_crop = resized[oy1:oy2, ox1:ox2]
        
        if overlay_crop.shape[2] == 4:
            # Has alpha channel
            alpha = overlay_crop[:, :, 3:4] / 255.0
            rgb = overlay_crop[:, :, :3]
            
            bg_region = background[y1:y2, x1:x2]
            blended = (rgb * alpha + bg_region * (1 - alpha)).astype(np.uint8)
            background[y1:y2, x1:x2] = blended
        else:
            background[y1:y2, x1:x2] = overlay_crop[:, :, :3]

    def _draw_filter_label(self, frame: np.ndarray) -> None:
        """Draw filter name label."""
        h, w = frame.shape[:2]
        text = f"Filter: {self.name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        x = 10
        y = h - 20

        cv2.putText(frame, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    def release(self) -> None:
        """Release resources."""
        self.face_mesh.release()
