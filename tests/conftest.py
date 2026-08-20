import pytest

from app import main
from app.detection import MockEngine


class FakePersistence:
    """In-memory stand-in for SupabasePersistence — no network, no disk."""

    def __init__(self):
        self.saved = []

    def save_analysis(self, *, analysis_id, batch_id, sample, image_bytes,
                      content_type, extension, summary, detections, magnification):
        self.saved.append(
            {
                "analysis_id": analysis_id,
                "batch_id": batch_id,
                "sample": sample,
                "content_type": content_type,
                "extension": extension,
                "summary": summary,
                "detections": detections,
                "magnification": magnification,
            }
        )
        return f"analyses/{analysis_id}{extension}"

    def public_url(self, image_path):
        return f"https://fake.supabase.co/storage/v1/object/public/milodetects-smears/{image_path}"

    def list_batches(self, limit):
        return []

    def get_batch(self, batch_id):
        return None


@pytest.fixture(autouse=True)
def fake_persist(monkeypatch):
    """Every test runs against an in-memory persistence backend, so the suite is
    hermetic and needs no Supabase config. Request this fixture to inspect what
    was saved; override app.main.persist directly for custom read data.

    Also pins the inference engine to the mock, so the suite is deterministic and
    hermetic regardless of INFERENCE_ENGINE in the developer's .env (yolo would run
    the real model on the tiny test images and return nothing).
    """
    fake = FakePersistence()
    monkeypatch.setattr(main, "persist", fake)
    monkeypatch.setattr(main, "engine", MockEngine())
    return fake
