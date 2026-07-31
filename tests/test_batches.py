from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

CANNED_BATCHES = [
    {
        "batch_id": "batch-1",
        "created_at": "2026-07-30T10:00:00Z",
        "image_count": 2,
        "summary": {"WBC": 6, "RBC": 56, "Platelet": 18},
        "images": [
            {"id": "a", "image_url": "https://x/a.png", "status": "completed",
             "summary": {"WBC": 3, "RBC": 28, "Platelet": 9}},
            {"id": "b", "image_url": "https://x/b.png", "status": "completed",
             "summary": {"WBC": 3, "RBC": 28, "Platelet": 9}},
        ],
    }
]

CANNED_BATCH = {
    "batch_id": "known-batch",
    "created_at": "2026-07-30T10:00:00Z",
    "images": [
        {
            "id": "a",
            "image_url": "https://x/a.png",
            "content_type": "image/png",
            "status": "completed",
            "summary": {"WBC": 3, "RBC": 28, "Platelet": 9},
            "detections": [
                {"cell_type": "RBC", "confidence": 0.9,
                 "x": 0.5, "y": 0.5, "width": 0.1, "height": 0.1}
            ],
        }
    ],
}


class FakeReadPersistence:
    def list_batches(self, limit):
        return CANNED_BATCHES

    def get_batch(self, batch_id):
        return CANNED_BATCH if batch_id == "known-batch" else None


def test_list_batches_returns_grouped_batches(monkeypatch):
    monkeypatch.setattr("app.main.persist", FakeReadPersistence())
    response = client.get("/batches")
    assert response.status_code == 200
    assert response.json() == {"batches": CANNED_BATCHES}


def test_get_batch_returns_images_with_detections(monkeypatch):
    monkeypatch.setattr("app.main.persist", FakeReadPersistence())
    response = client.get("/batches/known-batch")
    assert response.status_code == 200
    assert response.json() == CANNED_BATCH


def test_get_unknown_batch_is_404(monkeypatch):
    monkeypatch.setattr("app.main.persist", FakeReadPersistence())
    response = client.get("/batches/does-not-exist")
    assert response.status_code == 404
