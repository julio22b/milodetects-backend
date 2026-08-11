"""Cell detection: the data types plus the swappable inference engines.

`Detection` is the single shape every engine returns and everything downstream
consumes: cell_type + confidence + normalized center-xywh. Coordinates are
normalized (x/y is the box CENTER, width/height the box size, all fractions of the
image 0.0-1.0), so the frontend can draw the image at any size and multiply by the
rendered dimensions — no resolution bugs, and boxes render correctly over the
downscaled stored copy.

Two engines implement one interface (`DetectionEngine.predict`):
- `MockEngine`  — hardcoded, realistically-shaped detections. Needs no ML deps, so
  the app boots and tests run without torch. Kept for local frontend work + tests.
- `YoloEngine`  — loads the trained YOLOv8 weights and runs real detection. torch/
  ultralytics are imported lazily inside it so this module imports without them.

`get_engine(name)` selects one; `app.main` builds it once at startup and keeps the
model resident for the process lifetime.
"""

import io
import logging
import math
import random
import threading
from enum import Enum
from typing import Any, Protocol, cast

from pydantic import BaseModel, Field

from app import config

logger = logging.getLogger("milodetects")


class CellType(str, Enum):
    WBC = "WBC"
    RBC = "RBC"
    PLATELETS = "Platelets"  # plural — matches the YOLO model's class name exactly


class Detection(BaseModel):
    cell_type: CellType
    confidence: float = Field(ge=0.0, le=1.0)
    x: float = Field(ge=0.0, le=1.0)       # box center X, normalized
    y: float = Field(ge=0.0, le=1.0)       # box center Y, normalized
    width: float = Field(ge=0.0, le=1.0)   # box width, normalized
    height: float = Field(ge=0.0, le=1.0)  # box height, normalized


class DetectionEngine(Protocol):
    def predict(self, image_bytes: bytes) -> list[Detection]:
        """Detect cells in one image. Blocking/CPU-bound — call via run_in_executor."""
        ...


# The trained model emits {0:'bccd', 1:'Platelets', 2:'RBC', 3:'WBC'}. 'bccd' is a
# COCO-export parent category with zero annotated boxes and is never predicted, so
# it's deliberately absent here. The three real classes map 1:1 onto CellType.
_CLASS_MAP: dict[str, CellType] = {
    "WBC": CellType.WBC,
    "RBC": CellType.RBC,
    "Platelets": CellType.PLATELETS,
}


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _to_detection(
    raw_name: str, confidence: float, x: float, y: float, width: float, height: float
) -> Detection | None:
    """Map a raw YOLO class name + box to a Detection, or None (and log) if the name
    isn't one we recognise.

    An unmapped name is the worst failure mode here: it silently drops every
    detection of that class, so it's logged loudly rather than ignored. Finite
    values are clamped to [0,1] so a boundary float out of NMS can't trip the Field
    validators; a non-finite (NaN/inf) value drops just that one box rather than
    letting the ValidationError bubble up and fail the whole image.
    """
    cell_type = _CLASS_MAP.get(raw_name)
    if cell_type is None:
        logger.warning("YOLO produced unmapped class %r; dropping detection", raw_name)
        return None
    values = (confidence, x, y, width, height)
    if not all(math.isfinite(v) for v in values):
        logger.warning(
            "YOLO produced non-finite box %r for class %r; dropping detection",
            values,
            raw_name,
        )
        return None
    return Detection(
        cell_type=cell_type,
        confidence=_clamp01(confidence),
        x=_clamp01(x),
        y=_clamp01(y),
        width=_clamp01(width),
        height=_clamp01(height),
    )


class MockEngine:
    """Ignores the image; returns a realistic spread of a blood-smear field.

    Deterministic (seeded rng) so boxes don't jump around while building the
    overlay. The rng lives on the instance, so two engines never share state.
    """

    def __init__(self) -> None:
        self._rng = random.Random(42)

    def _box(
        self, cell_type: CellType, size: float, conf_range: tuple[float, float]
    ) -> Detection:
        half = size / 2
        return Detection(
            cell_type=cell_type,
            confidence=round(self._rng.uniform(*conf_range), 3),
            x=round(self._rng.uniform(half, 1 - half), 4),
            y=round(self._rng.uniform(half, 1 - half), 4),
            width=round(size * self._rng.uniform(0.85, 1.15), 4),
            height=round(size * self._rng.uniform(0.85, 1.15), 4),
        )

    def predict(self, image_bytes: bytes) -> list[Detection]:
        detections: list[Detection] = []
        for _ in range(28):  # many RBCs filling the field
            detections.append(self._box(CellType.RBC, 0.09, (0.80, 0.97)))
        for _ in range(3):   # a few larger, high-confidence WBCs
            detections.append(self._box(CellType.WBC, 0.16, (0.85, 0.99)))
        for _ in range(9):   # small, noisier platelets
            detections.append(self._box(CellType.PLATELETS, 0.04, (0.55, 0.90)))
        return detections


class YoloEngine:
    """Runs the trained YOLOv8 model. torch/ultralytics are imported lazily here so
    importing this module never requires them (the mock path stays usable without
    them installed). The model is loaded once and kept resident on the instance.
    """

    def __init__(self, weights_path: str, confidence: float, iou: float) -> None:
        import torch  # lazy: heavy, optional dependency
        from ultralytics import YOLO

        # Cap torch's CPU thread pool before inference if configured — avoids thread
        # thrashing on fractional-CPU hosts. Set once at startup, before predict().
        if config.TORCH_NUM_THREADS is not None:
            torch.set_num_threads(config.TORCH_NUM_THREADS)
            logger.info("torch intra-op threads capped at %d", config.TORCH_NUM_THREADS)

        try:
            self._model = YOLO(weights_path)
        except Exception as exc:  # missing/corrupt weights → fail loud, name the path
            raise RuntimeError(
                f"Failed to load YOLO weights at {weights_path!r}: {exc}"
            ) from exc
        self._confidence = confidence
        self._iou = iou
        # ultralytics reuses mutable per-model state across predict() calls, so the
        # shared resident model can't be run from two executor threads at once.
        self._lock = threading.Lock()

    def predict(self, image_bytes: bytes) -> list[Detection]:
        from PIL import Image

        # Orientation is already baked into the pixels upstream and the EXIF tag
        # cleared, so ultralytics won't re-rotate. RGB is what the model expects.
        with Image.open(io.BytesIO(image_bytes)) as opened:
            image = opened.convert("RGB")

        detections: list[Detection] = []
        # Serialize inference + result extraction: the model and its Results share
        # mutable state that concurrent executor threads would corrupt.
        with self._lock:
            results = self._model.predict(
                image, conf=self._confidence, iou=self._iou, verbose=False
            )
            # ultralytics' predict() return is loosely typed to the checker; cast so
            # we can iterate its Results (each has .names / .boxes) without spurious
            # errors.
            for result in cast(list[Any], results):
                names = result.names  # {index: class_name}
                boxes = result.boxes
                if boxes is None:
                    continue
                # xywhn is already normalized center-xywh — the exact Detection
                # shape, no coordinate conversion needed.
                for xywhn, conf, cls in zip(boxes.xywhn, boxes.conf, boxes.cls):
                    x, y, w, h = xywhn.tolist()
                    detection = _to_detection(
                        names[int(cls)], float(conf), x, y, w, h
                    )
                    if detection is not None:
                        detections.append(detection)
        return detections


def get_engine(name: str) -> DetectionEngine:
    """Build the inference engine named by config.INFERENCE_ENGINE."""
    if name == "mock":
        return MockEngine()
    if name == "yolo":
        return YoloEngine(
            config.YOLO_WEIGHTS_PATH,
            config.CONFIDENCE_THRESHOLD,
            config.IOU_THRESHOLD,
        )
    raise ValueError(
        f"Unknown INFERENCE_ENGINE {name!r} (expected 'mock' or 'yolo')"
    )
