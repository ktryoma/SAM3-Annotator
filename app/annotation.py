from __future__ import annotations
import uuid
from dataclasses import dataclass, field


@dataclass
class BoundingBox:
    """Normalized [0,1] bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""
    confidence: float = 0.0
    bb_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    is_auto: bool = True

    def normalized_xywh(self):
        xc = (self.x1 + self.x2) / 2
        yc = (self.y1 + self.y2) / 2
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        return xc, yc, w, h

    def to_pixel(self, img_w: int, img_h: int):
        return (
            self.x1 * img_w,
            self.y1 * img_h,
            self.x2 * img_w,
            self.y2 * img_h,
        )

    @classmethod
    def from_xyxy_pixel(cls, x1, y1, x2, y2, img_w, img_h, **kwargs):
        return cls(
            x1=x1 / img_w, y1=y1 / img_h,
            x2=x2 / img_w, y2=y2 / img_h,
            **kwargs
        )
