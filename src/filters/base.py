"""
Base filter class and filter manager for the AR booth application.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseFilter(ABC):
    """Abstract base class for all filters."""

    name: str = "Base Filter"

    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame and return the filtered result."""
        pass

    def release(self) -> None:
        """Release any resources held by the filter."""
        pass


class FilterManager:
    """Manages multiple filters and handles switching."""

    def __init__(self):
        self.filters: list[BaseFilter] = []
        self.current_index: int = 0

    def add_filter(self, filter_instance: BaseFilter) -> None:
        """Add a filter to the manager."""
        self.filters.append(filter_instance)

    def get_current_filter(self) -> Optional[BaseFilter]:
        """Get the currently active filter."""
        if not self.filters:
            return None
        return self.filters[self.current_index]

    def next_filter(self) -> Optional[BaseFilter]:
        """Switch to the next filter."""
        if not self.filters:
            return None
        self.current_index = (self.current_index + 1) % len(self.filters)
        return self.get_current_filter()

    def previous_filter(self) -> Optional[BaseFilter]:
        """Switch to the previous filter."""
        if not self.filters:
            return None
        self.current_index = (self.current_index - 1) % len(self.filters)
        return self.get_current_filter()

    def set_filter(self, index: int) -> Optional[BaseFilter]:
        """Set a specific filter by index."""
        if not self.filters or index < 0 or index >= len(self.filters):
            return None
        self.current_index = index
        return self.get_current_filter()

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with current filter."""
        current = self.get_current_filter()
        if current is None:
            return frame
        return current.process(frame)

    def release_all(self) -> None:
        """Release all filters."""
        for f in self.filters:
            f.release()
        self.filters.clear()

    @property
    def current_name(self) -> str:
        """Get current filter name."""
        current = self.get_current_filter()
        return current.name if current else "None"

    @property
    def count(self) -> int:
        """Get number of filters."""
        return len(self.filters)
