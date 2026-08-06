
"""
Phase A mock detector.
 
Returns hardcoded but realistically-shaped detections so the whole app works
end to end before any ML exists. The SHAPE here (cell_type + confidence +
normalized center-xywh) is exactly what real YOLO will return later, so when you
swap the mock for the model, nothing downstream — your endpoint, your frontend
overlay — has to change.
 
Coordinates are normalized: x/y is the CENTER of the box, width/height are the
box size, all as fractions of the image (0.0-1.0). That means the frontend can
draw the image at any size and multiply by the rendered width/height — no
resolution bugs.
"""
 
import random
from enum import Enum
from pydantic import BaseModel, Field
 
 
class CellType(str, Enum):
    WBC = "WBC"
    RBC = "RBC"
    PLATELETS = "Platelets"
 
 
class Detection(BaseModel):
    cell_type: CellType
    confidence: float = Field(ge=0.0, le=1.0)
    x: float = Field(ge=0.0, le=1.0)       # box center X, normalized
    y: float = Field(ge=0.0, le=1.0)       # box center Y, normalized
    width: float = Field(ge=0.0, le=1.0)   # box width, normalized
    height: float = Field(ge=0.0, le=1.0)  # box height, normalized
 
 
# Deterministic so the boxes don't jump around on every request while you build
# the overlay. Swap to random.Random() (no seed) if you want variety.
_rng = random.Random(42)
 
 
def _box(cell_type: CellType, size: float, conf_range: tuple[float, float]) -> Detection:
    half = size / 2
    return Detection(
        cell_type=cell_type,
        confidence=round(_rng.uniform(*conf_range), 3),
        x=round(_rng.uniform(half, 1 - half), 4),
        y=round(_rng.uniform(half, 1 - half), 4),
        width=round(size * _rng.uniform(0.85, 1.15), 4),
        height=round(size * _rng.uniform(0.85, 1.15), 4),
    )
 
 
def predict(image_bytes: bytes) -> list[Detection]:
    """Ignores the image, returns a realistic spread of a blood-smear field."""
    detections: list[Detection] = []
    for _ in range(28):  # many RBCs filling the field
        detections.append(_box(CellType.RBC, 0.09, (0.80, 0.97)))
    for _ in range(3):   # a few larger, high-confidence WBCs
        detections.append(_box(CellType.WBC, 0.16, (0.85, 0.99)))
    for _ in range(9):   # small, noisier platelets
        detections.append(_box(CellType.PLATELETS, 0.04, (0.55, 0.90)))
    return detections