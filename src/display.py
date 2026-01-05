"""
Display handler for the AR booth application.
"""

import cv2
import numpy as np
from typing import Optional


class Display:
    """Handles window display and rendering."""

    WINDOW_NAME = "Campus Booth AR Camera"

    def __init__(
        self,
        fullscreen: bool = True,
        show_fps: bool = True,
        show_instructions: bool = True,
    ):
        self.fullscreen = fullscreen
        self.show_fps = show_fps
        self.show_instructions = show_instructions
        self._window_created = False
        self._instruction_alpha = 1.0  # For fade effect
        self._frames_since_start = 0
        self._fade_start_frame = 150  # Start fading after 5 seconds at 30fps
        self._fade_duration = 60  # Fade over 2 seconds

    def create_window(self) -> None:
        """Create and configure the display window."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

        if self.fullscreen:
            cv2.setWindowProperty(
                self.WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )

        self._window_created = True

    def render(
        self,
        frame: np.ndarray,
        fps: float = 0.0,
        filter_name: str = "",
        filter_index: int = 0,
        filter_count: int = 0,
        auto_rotate: bool = False,
    ) -> None:
        """Render frame to display with overlays."""
        if not self._window_created:
            self.create_window()

        display_frame = frame.copy()
        self._frames_since_start += 1

        # Draw FPS
        if self.show_fps and fps > 0:
            self._draw_fps(display_frame, fps)

        # Draw filter indicator
        if filter_name and filter_count > 0:
            self._draw_filter_indicator(display_frame, filter_name, filter_index, filter_count, auto_rotate)

        # Draw instructions (fades after a few seconds)
        if self.show_instructions and self._instruction_alpha > 0:
            self._draw_instructions(display_frame)
            self._update_instruction_fade()

        cv2.imshow(self.WINDOW_NAME, display_frame)

    def _draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Draw FPS counter on frame."""
        text = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 0)  # Green
        shadow_color = (0, 0, 0)  # Black shadow

        # Position in top-left corner
        position = (10, 30)

        # Draw shadow
        cv2.putText(
            frame,
            text,
            (position[0] + 2, position[1] + 2),
            font,
            font_scale,
            shadow_color,
            thickness + 1,
        )

        # Draw text
        cv2.putText(frame, text, position, font, font_scale, color, thickness)

    def _draw_filter_indicator(
        self,
        frame: np.ndarray,
        filter_name: str,
        filter_index: int,
        filter_count: int,
        auto_rotate: bool,
    ) -> None:
        """Draw current filter indicator at top-right."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Filter name and index
        text = f"{filter_index + 1}/{filter_count}: {filter_name}"
        if auto_rotate:
            text += " [AUTO]"
        
        font_scale = 0.7
        thickness = 2
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        x = w - text_w - 15
        y = 30
        
        # Background
        cv2.rectangle(
            frame,
            (x - 10, y - text_h - 5),
            (x + text_w + 10, y + baseline + 5),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            frame,
            (x - 10, y - text_h - 5),
            (x + text_w + 10, y + baseline + 5),
            (100, 100, 100),
            1,
        )
        
        # Text
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)

        # Draw filter dots
        dot_y = y + 20
        dot_spacing = 20
        total_width = (filter_count - 1) * dot_spacing
        start_x = w - 15 - total_width // 2 - (text_w // 2)
        
        for i in range(filter_count):
            dot_x = start_x + i * dot_spacing
            color = (0, 255, 255) if i == filter_index else (100, 100, 100)
            cv2.circle(frame, (dot_x, dot_y), 5, color, -1)

    def _draw_instructions(self, frame: np.ndarray) -> None:
        """Draw usage instructions with fade effect."""
        if self._instruction_alpha <= 0:
            return
            
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        instructions = [
            "SPACE - Next Filter",
            "1-4 - Select Filter",
            "Q/ESC - Quit",
        ]
        
        font_scale = 0.6
        thickness = 1
        line_height = 25
        
        # Calculate box size
        max_text_w = 0
        for inst in instructions:
            (tw, _), _ = cv2.getTextSize(inst, font, font_scale, thickness)
            max_text_w = max(max_text_w, tw)
        
        box_w = max_text_w + 30
        box_h = len(instructions) * line_height + 20
        
        # Position at bottom-right
        box_x = w - box_w - 20
        box_y = h - box_h - 80  # Above filter label
        
        # Apply alpha
        alpha = self._instruction_alpha
        
        # Create overlay for transparency
        overlay = frame.copy()
        
        # Background
        cv2.rectangle(
            overlay,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            (40, 40, 40),
            -1,
        )
        cv2.rectangle(
            overlay,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            (100, 100, 100),
            1,
        )
        
        # Draw instructions
        for i, inst in enumerate(instructions):
            text_y = box_y + 20 + i * line_height
            cv2.putText(overlay, inst, (box_x + 15, text_y), font, font_scale, (200, 200, 200), thickness)
        
        # Blend with alpha
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _update_instruction_fade(self) -> None:
        """Update instruction fade based on frame count."""
        if self._frames_since_start > self._fade_start_frame:
            fade_progress = (self._frames_since_start - self._fade_start_frame) / self._fade_duration
            self._instruction_alpha = max(0, 1 - fade_progress)

    def reset_instructions(self) -> None:
        """Reset instruction visibility (call on filter change)."""
        self._instruction_alpha = 1.0
        self._frames_since_start = 0

    def check_key(self, delay: int = 1) -> int:
        """Check for key press. Returns key code or -1."""
        return cv2.waitKey(delay) & 0xFF

    def destroy(self) -> None:
        """Destroy the display window."""
        if self._window_created:
            cv2.destroyWindow(self.WINDOW_NAME)
            self._window_created = False
