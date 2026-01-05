"""
Y2K Filter: Retro low-fi aesthetic with frame, noise, scanlines, and color grading.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from .base import BaseFilter


class Y2KFilter(BaseFilter):
    """
    Applies Y2K/CRT aesthetic: low resolution, noise, scanlines,
    chromatic aberration, and vintage color grading.
    """

    name = "Y2K Retro"

    def __init__(
        self,
        downsample_factor: float = 0.4,
        noise_intensity: float = 25.0,
        scanline_intensity: float = 0.3,
        chroma_shift: int = 3,
        vignette_strength: float = 0.4,
    ):
        """
        Initialize Y2K filter.
        
        Args:
            downsample_factor: How much to reduce resolution (0.0-1.0)
            noise_intensity: Strength of noise overlay
            scanline_intensity: Darkness of scanlines (0.0-1.0)
            chroma_shift: Pixels to shift for chromatic aberration
            vignette_strength: Strength of corner darkening (0.0-1.0)
        """
        self.downsample_factor = downsample_factor
        self.noise_intensity = noise_intensity
        self.scanline_intensity = scanline_intensity
        self.chroma_shift = chroma_shift
        self.vignette_strength = vignette_strength
        
        # Frame overlay
        self._frame: Optional[np.ndarray] = None
        self._frame_cache: dict[tuple[int, int], np.ndarray] = {}
        
        # Precomputed masks
        self._scanline_cache: dict[tuple[int, int], np.ndarray] = {}
        self._vignette_cache: dict[tuple[int, int], np.ndarray] = {}
        
        self._load_frame()

    def _load_frame(self) -> None:
        """Load or generate Y2K frame."""
        assets_dir = Path(__file__).parent.parent.parent / "assets" / "frames"
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        frame_path = assets_dir / "y2k_frame.png"
        
        if frame_path.exists():
            self._frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        else:
            self._frame = self._generate_frame()
            cv2.imwrite(str(frame_path), self._frame)

    def _generate_frame(self) -> np.ndarray:
        """Generate a Y2K-style frame (BGRA)."""
        # Create at a base size, will be resized on use
        w, h = 1280, 720
        frame = np.zeros((h, w, 4), dtype=np.uint8)
        
        border = 40
        corner_radius = 20
        
        # Outer border - purple/pink Y2K gradient feel
        color_outer = (180, 50, 200, 255)  # Purple-ish
        color_inner = (255, 100, 150, 255)  # Pink-ish
        
        # Draw outer rectangle
        cv2.rectangle(frame, (0, 0), (w-1, h-1), color_outer, border)
        
        # Inner accent line
        cv2.rectangle(frame, (border-5, border-5), (w-border+5, h-border+5), color_inner, 3)
        
        # Corner decorations - Y2K style circles
        corners = [(border, border), (w-border, border), (border, h-border), (w-border, h-border)]
        for cx, cy in corners:
            cv2.circle(frame, (cx, cy), 15, color_inner, -1)
            cv2.circle(frame, (cx, cy), 10, color_outer, -1)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255, 255), -1)
        
        # Add some Y2K text elements
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "2000", (w//2 - 40, border - 10), font, 0.7, (255, 255, 255, 255), 2)
        cv2.putText(frame, "CAM", (w - border - 50, h - 10), font, 0.5, color_inner, 1)
        
        # Star decorations
        star_positions = [(100, 50), (w-100, 50), (100, h-50), (w-100, h-50)]
        for sx, sy in star_positions:
            self._draw_star(frame, sx, sy, 8, (255, 255, 255, 200))
        
        return frame

    def _draw_star(self, img: np.ndarray, cx: int, cy: int, size: int, color: tuple) -> None:
        """Draw a simple 4-point star."""
        pts = [
            (cx, cy - size),
            (cx + size//3, cy - size//3),
            (cx + size, cy),
            (cx + size//3, cy + size//3),
            (cx, cy + size),
            (cx - size//3, cy + size//3),
            (cx - size, cy),
            (cx - size//3, cy - size//3),
        ]
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], color)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply Y2K effect pipeline."""
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # 1. Downsample and upsample for pixelated look
        result = self._apply_pixelation(result)
        
        # 2. Apply chromatic aberration
        result = self._apply_chromatic_aberration(result)
        
        # 3. Apply vintage color grading
        result = self._apply_color_grading(result)
        
        # 4. Add noise
        result = self._apply_noise(result)
        
        # 5. Add scanlines
        result = self._apply_scanlines(result)
        
        # 6. Add vignette
        result = self._apply_vignette(result)
        
        # 7. Overlay frame
        result = self._apply_frame(result)
        
        # Draw filter label
        self._draw_filter_label(result)
        
        return result

    def _apply_pixelation(self, frame: np.ndarray) -> np.ndarray:
        """Downsample then upsample for retro pixelated look."""
        h, w = frame.shape[:2]
        small_h = int(h * self.downsample_factor)
        small_w = int(w * self.downsample_factor)
        
        # Downsample
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        # Upsample with nearest neighbor for blocky pixels
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def _apply_chromatic_aberration(self, frame: np.ndarray) -> np.ndarray:
        """Shift color channels for chromatic aberration effect."""
        if self.chroma_shift <= 0:
            return frame
            
        b, g, r = cv2.split(frame)
        
        # Shift red channel right, blue channel left
        rows, cols = r.shape
        M_right = np.float32([[1, 0, self.chroma_shift], [0, 1, 0]])
        M_left = np.float32([[1, 0, -self.chroma_shift], [0, 1, 0]])
        
        r_shifted = cv2.warpAffine(r, M_right, (cols, rows))
        b_shifted = cv2.warpAffine(b, M_left, (cols, rows))
        
        return cv2.merge([b_shifted, g, r_shifted])

    def _apply_color_grading(self, frame: np.ndarray) -> np.ndarray:
        """Apply Y2K color grading - warm tint, boosted contrast."""
        # Convert to float for processing
        result = frame.astype(np.float32)
        
        # Warm tint - boost red/yellow, reduce blue
        result[:, :, 0] *= 0.85  # Reduce blue
        result[:, :, 1] *= 1.0   # Keep green
        result[:, :, 2] *= 1.1   # Boost red
        
        # Slight contrast boost
        result = (result - 128) * 1.1 + 128
        
        # Add slight sepia/warm overlay
        sepia_r = result[:, :, 2] * 0.95 + result[:, :, 1] * 0.05
        sepia_g = result[:, :, 1] * 0.9 + result[:, :, 2] * 0.1
        sepia_b = result[:, :, 0] * 0.85 + result[:, :, 1] * 0.1
        
        result[:, :, 0] = sepia_b
        result[:, :, 1] = sepia_g
        result[:, :, 2] = sepia_r
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_noise(self, frame: np.ndarray) -> np.ndarray:
        """Add film grain/noise."""
        if self.noise_intensity <= 0:
            return frame
            
        h, w = frame.shape[:2]
        noise = np.random.normal(0, self.noise_intensity, (h, w)).astype(np.float32)
        noise = np.stack([noise] * 3, axis=-1)
        
        result = frame.astype(np.float32) + noise
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_scanlines(self, frame: np.ndarray) -> np.ndarray:
        """Add CRT-style scanlines."""
        if self.scanline_intensity <= 0:
            return frame
            
        h, w = frame.shape[:2]
        key = (h, w)
        
        if key not in self._scanline_cache:
            # Create scanline pattern
            scanlines = np.ones((h, w), dtype=np.float32)
            scanlines[::2, :] = 1.0 - self.scanline_intensity
            self._scanline_cache[key] = scanlines
        
        scanlines = self._scanline_cache[key]
        result = frame.astype(np.float32)
        
        for c in range(3):
            result[:, :, c] *= scanlines
            
        return result.astype(np.uint8)

    def _apply_vignette(self, frame: np.ndarray) -> np.ndarray:
        """Add vignette (darkened corners)."""
        if self.vignette_strength <= 0:
            return frame
            
        h, w = frame.shape[:2]
        key = (h, w)
        
        if key not in self._vignette_cache:
            # Create vignette mask
            Y, X = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            
            # Distance from center, normalized
            dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            dist = dist / max_dist
            
            # Vignette falloff
            vignette = 1 - (dist ** 2) * self.vignette_strength
            vignette = np.clip(vignette, 0, 1).astype(np.float32)
            self._vignette_cache[key] = vignette
        
        vignette = self._vignette_cache[key]
        result = frame.astype(np.float32)
        
        for c in range(3):
            result[:, :, c] *= vignette
            
        return result.astype(np.uint8)

    def _apply_frame(self, frame: np.ndarray) -> np.ndarray:
        """Overlay Y2K frame."""
        if self._frame is None:
            return frame
            
        h, w = frame.shape[:2]
        key = (h, w)
        
        if key not in self._frame_cache:
            # Resize frame to match
            resized = cv2.resize(self._frame, (w, h), interpolation=cv2.INTER_AREA)
            self._frame_cache[key] = resized
        
        overlay = self._frame_cache[key]
        
        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3:4] / 255.0
            rgb = overlay[:, :, :3]
            result = (rgb * alpha + frame * (1 - alpha)).astype(np.uint8)
            return result
        
        return frame

    def _draw_filter_label(self, frame: np.ndarray) -> None:
        """Draw filter name label."""
        h, w = frame.shape[:2]
        text = f"Filter: {self.name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        x = 10
        y = h - 60  # Move up to avoid frame overlap

        cv2.putText(frame, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 200, 255), thickness)

    def release(self) -> None:
        """Release resources."""
        self._scanline_cache.clear()
        self._vignette_cache.clear()
        self._frame_cache.clear()
