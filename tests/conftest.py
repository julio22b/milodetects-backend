import pytest

from app import main


class FakePersistence:
    """In-memory stand-in for SupabasePersistence — no network, no disk."""

    def __init__(self):
        self.saved = []

    def save_analysis(self, *, analysis_id, batch_id, sample, image_bytes,
                      content_type, extension, summary, detections):
        self.saved.append(
            {
                "analysis_id": analysis_id,
                "batch_id": batch_id,
                "sample": sample,
                "content_type": content_type,
                "extension": extension,
                "summary": summary,
                "detections": detections,
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

    Also clears the lazily-built inference engine so each test rebuilds it from the
    current config (default: the mock) instead of inheriting a prior test's engine.
    """
    fake = FakePersistence()
    monkeypatch.setattr(main, "persist", fake)
    monkeypatch.setattr(main, "engine", None)
    return fake
