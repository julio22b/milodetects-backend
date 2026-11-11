from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

KNOWN_BATCH = "3f2b8c1e-5d4a-4e6f-9a7b-1c2d3e4f5a6b"
UNKNOWN_BATCH = "00000000-0000-4000-8000-000000000000"

CANNED_BATCHES = [
    {
        "batch_id": "batch-1",
        "sample": "S-100",
        "created_at": "2026-07-30T10:00:00Z",
        "image_count": 2,
        "summary": {"WBC": 6, "RBC": 56, "Platelets": 18},
        "images": [
            {"id": "a", "image_url": "https://x/a.png", "status": "completed",
             "summary": {"WBC": 3, "RBC": 28, "Platelets": 9}},
            {"id": "b", "image_url": "https://x/b.png", "status": "completed",
             "summary": {"WBC": 3, "RBC": 28, "Platelets": 9}},
        ],
    }
]

CANNED_BATCH = {
    "batch_id": KNOWN_BATCH,
    "sample": "S-100",
    "created_at": "2026-07-30T10:00:00Z",
    "images": [
        {
            "id": "a",
            "image_url": "https://x/a.png",
            "content_type": "image/png",
            "status": "completed",
            "summary": {"WBC": 3, "RBC": 28, "Platelets": 9},
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
        return CANNED_BATCH if batch_id == KNOWN_BATCH else None

    def delete_batch(self, batch_id):
        return 2 if batch_id == KNOWN_BATCH else 0


def test_list_batches_returns_grouped_batches(monkeypatch):
    monkeypatch.setattr("app.main.persist", FakeReadPersistence())
    response = client.get("/batches")
    assert response.status_code == 200
    assert response.json() == {"batches": CANNED_BATCHES}


def test_get_batch_returns_images_with_detections(monkeypatch):
    monkeypatch.setattr("app.main.persist", FakeReadPersistence())
    response = client.get(f"/batches/{KNOWN_BATCH}")
    assert response.status_code == 200
    assert response.json() == CANNED_BATCH


def test_get_unknown_batch_is_404(monkeypatch):
    monkeypatch.setattr("app.main.persist", FakeReadPersistence())
    response = client.get(f"/batches/{UNKNOWN_BATCH}")
    assert response.status_code == 404


def test_delete_batch_removes_and_returns_count(monkeypatch):
    monkeypatch.setattr("app.main.persist", FakeReadPersistence())
    response = client.delete(f"/batches/{KNOWN_BATCH}")
    assert response.status_code == 200
    assert response.json() == {"batch_id": KNOWN_BATCH, "deleted_images": 2}


def test_delete_unknown_batch_is_404(monkeypatch):
    monkeypatch.setattr("app.main.persist", FakeReadPersistence())
    response = client.delete(f"/batches/{UNKNOWN_BATCH}")
    assert response.status_code == 404


def test_get_batch_with_non_uuid_id_is_422(monkeypatch):
    # A non-UUID id must be rejected at the route, never reach the uuid column
    # (where Postgres would raise and the request would 500).
    reached = []

    class Recording(FakeReadPersistence):
        def get_batch(self, batch_id):
            reached.append(batch_id)
            return None

    monkeypatch.setattr("app.main.persist", Recording())
    response = client.get("/batches/not-a-uuid")
    assert response.status_code == 422
    assert reached == []


def test_delete_batch_with_non_uuid_id_is_422(monkeypatch):
    reached = []

    class Recording(FakeReadPersistence):
        def delete_batch(self, batch_id):
            reached.append(batch_id)
            return 0

    monkeypatch.setattr("app.main.persist", Recording())
    response = client.delete("/batches/not-a-uuid")
    assert response.status_code == 422
    assert reached == []


def test_batch_id_reaches_persistence_as_canonical_string(monkeypatch):
    reached = []

    class Recording(FakeReadPersistence):
        def get_batch(self, batch_id):
            reached.append(batch_id)
            return None

    monkeypatch.setattr("app.main.persist", Recording())
    client.get(f"/batches/{KNOWN_BATCH.upper()}")
    assert reached == [KNOWN_BATCH]
