"""
Configuration loader for the AR booth application.
"""

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration."""
    camera_index: int = 0
    capture_width: int = 1280
    capture_height: int = 720
    display_fullscreen: bool = True
    target_fps: int = 30
    filter_auto_rotate: bool = True
    filter_rotate_interval_seconds: int = 10
    gpa_refresh_interval_seconds: int = 3
    show_fps: bool = True
    show_instructions: bool = True

    @classmethod
    def load(cls, path: Path | str = "config.json") -> "Config":
        """Load configuration from JSON file."""
        path = Path(path)
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            return cls(**data)
        return cls()

    def save(self, path: Path | str = "config.json") -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
