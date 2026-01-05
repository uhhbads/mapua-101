"""
Custom Frame Filter: Overlay a custom PNG frame on the video feed.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from .base import BaseFilter


class CustomFrameFilter(BaseFilter):
    """
    Overlays a custom PNG frame with transparency over the video feed.
    """

    name = "Custom Frame"

    def __init__(self, frame_name: str = "custom_frame.png"):
        """
        Initialize custom frame filter.
        
        Args:
            frame_name: Filename of the frame PNG in assets/frames/
        """
        self.frame_name = frame_name
        self._frame: Optional[np.ndarray] = None
        self._frame_cache: dict[tuple[int, int], np.ndarray] = {}
        self._load_frame()

    def _load_frame(self) -> None:
        """Load or generate custom frame."""
        assets_dir = Path(__file__).parent.parent.parent / "assets" / "frames"
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        frame_path = assets_dir / self.frame_name
        
        if frame_path.exists():
            self._frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
            print(f"Loaded custom frame: {frame_path}")
        else:
            # Generate a default custom frame
            self._frame = self._generate_default_frame()
            cv2.imwrite(str(frame_path), self._frame)
            print(f"Generated default custom frame: {frame_path}")

    def _generate_default_frame(self) -> np.ndarray:
        """Generate a default custom frame (BGRA)."""
        w, h = 1280, 720
        frame = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Modern, clean frame design
        border = 30
        inner_border = 35
        
        # Colors
        primary = (200, 150, 50, 255)      # Blue-ish
        secondary = (255, 200, 100, 255)   # Lighter blue
        accent = (100, 255, 255, 255)      # Yellow/gold
        white = (255, 255, 255, 255)
        
        # Outer border
        cv2.rectangle(frame, (0, 0), (w-1, h-1), primary, border)
        
        # Inner accent line
        cv2.rectangle(frame, (border, border), (w-border-1, h-border-1), secondary, 2)
        
        # Corner brackets
        bracket_len = 60
        bracket_thick = 4
        corners = [
            # Top-left
            ((inner_border, inner_border), (inner_border + bracket_len, inner_border), (inner_border, inner_border + bracket_len)),
            # Top-right
            ((w - inner_border, inner_border), (w - inner_border - bracket_len, inner_border), (w - inner_border, inner_border + bracket_len)),
            # Bottom-left
            ((inner_border, h - inner_border), (inner_border + bracket_len, h - inner_border), (inner_border, h - inner_border - bracket_len)),
            # Bottom-right
            ((w - inner_border, h - inner_border), (w - inner_border - bracket_len, h - inner_border), (w - inner_border, h - inner_border - bracket_len)),
        ]
        
        for corner, h_end, v_end in corners:
            cv2.line(frame, corner, h_end, accent, bracket_thick)
            cv2.line(frame, corner, v_end, accent, bracket_thick)
        
        # Title text area (top center)
        title = "CAMPUS BOOTH"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        (text_w, text_h), baseline = cv2.getTextSize(title, font, font_scale, thickness)
        text_x = (w - text_w) // 2
        text_y = border - 8
        
        # Text background
        cv2.rectangle(
            frame,
            (text_x - 15, text_y - text_h - 5),
            (text_x + text_w + 15, text_y + baseline + 5),
            primary,
            -1
        )
        cv2.putText(frame, title, (text_x, text_y), font, font_scale, white, thickness)
        
        # Bottom info bar
        info_text = "Say Cheese! :)"
        info_scale = 0.6
        (info_w, info_h), _ = cv2.getTextSize(info_text, font, info_scale, 1)
        info_x = (w - info_w) // 2
        info_y = h - border + 20
        
        cv2.rectangle(
            frame,
            (info_x - 10, info_y - info_h - 3),
            (info_x + info_w + 10, info_y + 5),
            primary,
            -1
        )
        cv2.putText(frame, info_text, (info_x, info_y), font, info_scale, accent, 1)
        
        # Decorative dots
        dot_positions = [
            (border // 2, h // 2),
            (w - border // 2, h // 2),
        ]
        for dx, dy in dot_positions:
            cv2.circle(frame, (dx, dy), 6, accent, -1)
            cv2.circle(frame, (dx, dy), 4, primary, -1)
        
        return frame

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply custom frame overlay."""
        if self._frame is None:
            return frame
            
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Get or create resized frame
        key = (h, w)
        if key not in self._frame_cache:
            resized = cv2.resize(self._frame, (w, h), interpolation=cv2.INTER_AREA)
            self._frame_cache[key] = resized
        
        overlay = self._frame_cache[key]
        
        # Blend using alpha channel
        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3:4] / 255.0
            rgb = overlay[:, :, :3]
            result = (rgb * alpha + result * (1 - alpha)).astype(np.uint8)
        
        # Draw filter label
        self._draw_filter_label(result)
        
        return result

    def _draw_filter_label(self, frame: np.ndarray) -> None:
        """Draw filter name label."""
        h, w = frame.shape[:2]
        text = f"Filter: {self.name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        x = 10
        y = h - 60  # Avoid frame overlap

        cv2.putText(frame, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    def reload_frame(self) -> None:
        """Reload the frame from disk (useful for hot-reloading custom designs)."""
        self._frame_cache.clear()
        self._load_frame()

    def release(self) -> None:
        """Release resources."""
        self._frame_cache.clear()
