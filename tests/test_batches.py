from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

CANNED_BATCHES = [
    {
        "batch_id": "batch-1",
        "sample": "S-100",
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
    "sample": "S-100",
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

    def delete_batch(self, batch_id):
        return 2 if batch_id == "known-batch" else 0


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


def test_delete_batch_removes_and_returns_count(monkeypatch):
    monkeypatch.setattr("app.main.persist", FakeReadPersistence())
    response = client.delete("/batches/known-batch")
    assert response.status_code == 200
    assert response.json() == {"batch_id": "known-batch", "deleted_images": 2}


def test_delete_unknown_batch_is_404(monkeypatch):
    monkeypatch.setattr("app.main.persist", FakeReadPersistence())
    response = client.delete("/batches/does-not-exist")
    assert response.status_code == 404
