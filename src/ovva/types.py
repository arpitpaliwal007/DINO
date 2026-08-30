from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Detection:
    """One open-vocabulary detection in pixel xyxy coordinates."""
    xyxy: tuple[float, float, float, float]
    confidence: float
    label: str
    mask: Any | None = None
    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
