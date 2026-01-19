"""
GPA Filter: Face boxes with stable GPA overlay per person.
"""

import cv2
import random
import time
import numpy as np
from typing import Optional

from .base import BaseFilter
from .face_detection import FaceDetector, FaceDetection


class GPAFilter(BaseFilter):
    """
    Draws bounding boxes around faces with stable GPA labels per person.
    """

    name = "GPA Scanner"

    # GPA range
    GPA_MIN = 1.0
    GPA_MAX = 4.0

    # Colors (BGR)
    BOX_COLOR = (0, 255, 0)  # Green
    TEXT_COLOR = (255, 255, 255)  # White
    TEXT_BG_COLOR = (0, 100, 0)  # Dark green

    def __init__(self, gpa_refresh_interval: float = 3.0):
        """
        Initialize GPA filter for detecting 5-7+ people.
        
        Args:
            gpa_refresh_interval: Not used anymore, kept for compatibility
        """
        self.detector = FaceDetector(min_confidence=0.5, max_faces=8)
        
        # Track GPAs per face (by approximate position)
        # center -> (gpa, last_seen_time)
        self._face_gpas: dict[tuple[int, int], tuple[float, float]] = {}
        self._position_tolerance = 120  # Pixels to consider same face
        self._stale_timeout = 5.0  # Remove face after 5 seconds of not seeing it

    def _get_gpa_for_face(self, face: FaceDetection) -> float:
        """Get or generate stable GPA for a face based on position."""
        center = face.center
        current_time = time.time()

        # Find matching face by position
        best_match_center = None
        best_distance = float('inf')
        
        for stored_center in self._face_gpas.keys():
            dx = abs(center[0] - stored_center[0])
            dy = abs(center[1] - stored_center[1])
            distance = (dx * dx + dy * dy) ** 0.5
            
            if distance < self._position_tolerance and distance < best_distance:
                best_distance = distance
                best_match_center = stored_center

        if best_match_center is not None:
            # Found existing face - update position and return same GPA
            gpa, _ = self._face_gpas[best_match_center]
            # Update with new center position and timestamp
            del self._face_gpas[best_match_center]
            self._face_gpas[center] = (gpa, current_time)
            return gpa

        # New face - generate GPA that persists
        gpa = round(random.uniform(self.GPA_MIN, self.GPA_MAX), 2)
        self._face_gpas[center] = (gpa, current_time)
        
        # Cleanup old entries
        self._cleanup_stale_faces(current_time)
        
        return gpa

    def _cleanup_stale_faces(self, current_time: float) -> None:
        """Remove faces not seen for a while."""
        to_remove = [
            center for center, (_, last_seen) in self._face_gpas.items()
            if current_time - last_seen > self._stale_timeout
        ]
        for center in to_remove:
            del self._face_gpas[center]

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process frame: detect faces and draw GPA boxes."""
        result = frame.copy()
        faces = self.detector.detect(frame)

        for face in faces:
            gpa = self._get_gpa_for_face(face)
            self._draw_face_box(result, face, gpa)

        # Draw filter label
        self._draw_filter_label(result)

        return result

    def _draw_face_box(self, frame: np.ndarray, face: FaceDetection, gpa: float) -> None:
        """Draw bounding box and GPA label for a face."""
        x, y, w, h = face.bbox

        # Draw box with thickness based on confidence
        thickness = max(2, int(face.confidence * 4))
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.BOX_COLOR, thickness)

        # Prepare GPA text
        gpa_text = f"GPA: {gpa:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        text_thickness = 2

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(gpa_text, font, font_scale, text_thickness)

        # Position text above face box
        text_x = x + (w - text_w) // 2
        text_y = y - 10

        # Ensure text is within frame
        if text_y - text_h < 0:
            text_y = y + h + text_h + 10

        # Draw background rectangle for text
        padding = 5
        cv2.rectangle(
            frame,
            (text_x - padding, text_y - text_h - padding),
            (text_x + text_w + padding, text_y + baseline + padding),
            self.TEXT_BG_COLOR,
            -1,  # Filled
        )

        # Draw text
        cv2.putText(
            frame,
            gpa_text,
            (text_x, text_y),
            font,
            font_scale,
            self.TEXT_COLOR,
            text_thickness,
        )

        # Draw confidence below
        conf_text = f"{face.confidence:.0%}"
        conf_scale = 0.5
        cv2.putText(
            frame,
            conf_text,
            (x, y + h + 20),
            font,
            conf_scale,
            self.BOX_COLOR,
            1,
        )

    def _draw_filter_label(self, frame: np.ndarray) -> None:
        """Draw filter name label."""
        h, w = frame.shape[:2]
        text = f"Filter: {self.name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Bottom-left corner
        x = 10
        y = h - 20

        # Shadow
        cv2.putText(frame, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 1)
        # Text
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    def release(self) -> None:
        """Release resources."""
        self.detector.release()
        self._face_gpas.clear()
