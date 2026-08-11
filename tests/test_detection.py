import importlib.util
import io
import logging
from collections import Counter

import pytest
from PIL import Image

from app import config
from app.detection import (
    CellType,
    Detection,
    MockEngine,
    _to_detection,
    get_engine,
)

# --- class-name mapping (the "silent drop" guard) ---------------------------


@pytest.mark.parametrize(
    "raw_name,expected",
    [("WBC", CellType.WBC), ("RBC", CellType.RBC), ("Platelets", CellType.PLATELETS)],
)
def test_to_detection_maps_known_classes(raw_name, expected):
    det = _to_detection(raw_name, 0.9, 0.5, 0.5, 0.1, 0.1)
    assert det is not None
    assert det.cell_type is expected  # Platelets (plural) must land on PLATELETS


@pytest.mark.parametrize("bad_name", ["bccd", "nope", "platelets", "wbc", ""])
def test_to_detection_drops_and_logs_unmapped(bad_name, caplog):
    """Unmapped names must be dropped AND logged — a silent drop would wipe out
    every detection of a class without a trace."""
    with caplog.at_level(logging.WARNING, logger="milodetects"):
        result = _to_detection(bad_name, 0.9, 0.5, 0.5, 0.1, 0.1)
    assert result is None
    assert "unmapped class" in caplog.text
    assert repr(bad_name) in caplog.text


def test_to_detection_clamps_out_of_range_values():
    """A boundary float from NMS shouldn't fail an image on the Field validators."""
    det = _to_detection("RBC", 1.2, -0.1, 1.05, 0.5, 0.5)
    assert det is not None
    assert det.confidence == 1.0
    assert det.x == 0.0
    assert det.y == 1.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_to_detection_drops_non_finite_box(bad, caplog):
    """A non-finite value drops just that box (and logs) rather than raising a
    ValidationError that would fail the whole image."""
    with caplog.at_level(logging.WARNING, logger="milodetects"):
        result = _to_detection("RBC", 0.9, bad, 0.5, 0.1, 0.1)
    assert result is None
    assert "non-finite" in caplog.text


# --- engine factory ---------------------------------------------------------


def test_get_engine_returns_mock():
    assert isinstance(get_engine("mock"), MockEngine)


def test_get_engine_unknown_name_raises():
    with pytest.raises(ValueError):
        get_engine("bogus")


def test_mock_engine_returns_the_expected_spread():
    detections = MockEngine().predict(b"ignored")
    counts = Counter(d.cell_type for d in detections)
    assert len(detections) == 40
    assert counts[CellType.RBC] == 28
    assert counts[CellType.WBC] == 3
    assert counts[CellType.PLATELETS] == 9


def test_yolo_engine_forwards_thresholds_to_predict():
    """The configured conf/iou must actually reach model.predict() — a definition-of-
    done requirement. Bypasses __init__ so it needs no weights or torch."""
    import threading

    from app.detection import YoloEngine

    recorded = {}

    class _RecordingModel:
        def predict(self, image, conf, iou, imgsz, verbose):
            recorded.update(conf=conf, iou=iou, imgsz=imgsz, verbose=verbose)
            return []  # no Results → predict() returns an empty list

    engine = YoloEngine.__new__(YoloEngine)  # skip real weight loading
    engine._model = _RecordingModel()  # type: ignore[assignment]
    engine._confidence = 0.33
    engine._iou = 0.55
    engine._imgsz = 512
    engine._lock = threading.Lock()

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")

    assert engine.predict(buffer.getvalue()) == []
    assert recorded == {"conf": 0.33, "iou": 0.55, "imgsz": 512, "verbose": False}


# --- real inference smoke test (skipped when torch/ultralytics absent) -------

_ULTRALYTICS_AVAILABLE = importlib.util.find_spec("ultralytics") is not None


@pytest.mark.skipif(
    not _ULTRALYTICS_AVAILABLE, reason="ultralytics/torch not installed"
)
def test_yolo_engine_runs_and_returns_valid_detections():
    from app.detection import YoloEngine

    engine = YoloEngine(config.YOLO_WEIGHTS_PATH, 0.25, 0.45, 320)
    buffer = io.BytesIO()
    Image.new("RGB", (640, 640), "white").save(buffer, format="JPEG")

    detections = engine.predict(buffer.getvalue())

    assert isinstance(detections, list)
    for det in detections:
        assert isinstance(det, Detection)
        assert det.cell_type in set(CellType)  # only mapped classes come through
        for coord in (det.x, det.y, det.width, det.height, det.confidence):
            assert 0.0 <= coord <= 1.0
