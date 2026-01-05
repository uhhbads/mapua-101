"""
Camera capture handler for the AR booth application.
"""

import cv2
import time
from typing import Optional
import numpy as np


class Camera:
    """Handles camera capture and frame retrieval."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 30,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time = 0.0
        self._fps = 0.0
        self._frame_count = 0
        self._fps_update_time = time.time()

    def open(self) -> bool:
        """Open the camera and configure settings."""
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Fallback without DirectShow
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            return False

        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

        return True

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera."""
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if ret:
            # Mirror the frame for natural interaction
            frame = cv2.flip(frame, 1)
            self._update_fps()

        return ret, frame

    def _update_fps(self) -> None:
        """Update FPS calculation."""
        self._frame_count += 1
        current_time = time.time()
        elapsed = current_time - self._fps_update_time

        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_update_time = current_time

    @property
    def fps(self) -> float:
        """Get current FPS."""
        return self._fps

    def release(self) -> None:
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.cap is not None and self.cap.isOpened()
