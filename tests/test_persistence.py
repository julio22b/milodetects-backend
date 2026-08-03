"""Unit tests for SupabasePersistence.save_analysis using a hand-rolled fake
Supabase client — verifies the write sequence without a live project."""

from types import SimpleNamespace

import pytest

SAMPLE = "S-100"  # canonical valid sample id used across these tests


class FakeTable:
    def __init__(self, client, name):
        self.client = client
        self.name = name
        self._op = None

    def insert(self, json):
        self.client.calls.append((self.name, "insert", json))
        self._op = "insert"
        return self

    def update(self, json):
        self.client.calls.append((self.name, "update", json))
        self._op = "update"
        return self

    def eq(self, column, value):
        return self

    def execute(self):
        if self.name == self.client.fail_table and self._op == "insert":
            raise RuntimeError("insert boom")
        return SimpleNamespace(data=[])


class FakeBucket:
    def __init__(self, client):
        self.client = client

    def upload(self, path, file, file_options):
        self.client.calls.append(("upload", path, file_options))


class FakeStorage:
    def __init__(self, client):
        self.client = client

    def from_(self, bucket):
        self.client.calls.append(("from_", bucket))
        return FakeBucket(self.client)


class FakeClient:
    def __init__(self, fail_table=None):
        self.calls = []
        self.fail_table = fail_table
        self.storage = FakeStorage(self)

    def table(self, name):
        return FakeTable(self, name)


def make_persistence(monkeypatch, fail_table=None):
    fake = FakeClient(fail_table=fail_table)
    monkeypatch.setattr("supabase.create_client", lambda url, key: fake)
    monkeypatch.setattr("app.config.SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setattr("app.config.SUPABASE_SERVICE_KEY", "svc")
    from app.persistence import SupabasePersistence

    return SupabasePersistence(), fake


DETECTIONS = [
    {"cell_type": "RBC", "confidence": 0.9, "x": 0.5, "y": 0.5, "width": 0.1, "height": 0.1},
    {"cell_type": "WBC", "confidence": 0.95, "x": 0.2, "y": 0.3, "width": 0.2, "height": 0.2},
]


def save(persistence, **overrides):
    kwargs = dict(
        analysis_id="A",
        batch_id="B",
        sample=SAMPLE,
        image_bytes=b"img",
        content_type="image/png",
        extension=".png",
        summary={"WBC": 1, "RBC": 1, "Platelet": 0},
        detections=DETECTIONS,
    )
    kwargs.update(overrides)
    return persistence.save_analysis(**kwargs)


def test_save_analysis_happy_path_sequence(monkeypatch):
    persistence, fake = make_persistence(monkeypatch)

    key = save(persistence)

    assert key == "analyses/A.png"
    assert ("from_", "milodetects-smears") in fake.calls

    upload = next(c for c in fake.calls if c[0] == "upload")
    assert upload[1] == "analyses/A.png"
    assert upload[2]["content-type"] == "image/png"

    # analyses row inserted as 'processing' first...
    analyses_insert = next(
        c[2] for c in fake.calls if c[0] == "analyses" and c[1] == "insert"
    )
    assert analyses_insert["id"] == "A"
    assert analyses_insert["batch_id"] == "B"
    assert analyses_insert["sample"] == SAMPLE
    assert analyses_insert["status"] == "processing"

    # ...detections inserted as ONE bulk list, each stamped with analysis_id...
    detections_insert = next(
        c[2] for c in fake.calls if c[0] == "detections" and c[1] == "insert"
    )
    assert isinstance(detections_insert, list) and len(detections_insert) == 2
    assert all(row["analysis_id"] == "A" for row in detections_insert)

    # ...then flipped to 'completed'.
    updates = [c[2] for c in fake.calls if c[1] == "update"]
    assert {"status": "completed"} in updates


def test_save_analysis_marks_failed_when_detections_insert_fails(monkeypatch):
    persistence, fake = make_persistence(monkeypatch, fail_table="detections")

    with pytest.raises(RuntimeError):
        save(persistence)

    # The analysis is flipped to 'failed' with an error message, never 'completed'.
    failed = [c[2] for c in fake.calls if c[1] == "update" and c[2].get("status") == "failed"]
    assert failed and "error_message" in failed[0]
    assert {"status": "completed"} not in [c[2] for c in fake.calls if c[1] == "update"]


# --- list_batches: whole-batch grouping (fake read client) ------------------


class FakeReadTable:
    """Honors order/limit/in_ over a fixed list of analyses rows."""

    def __init__(self, rows):
        self._rows = rows
        self._desc = False
        self._limit = None
        self._in = None

    def select(self, columns):
        return self

    def order(self, column, *, desc=False):
        self._desc = desc
        return self

    def limit(self, size):
        self._limit = size
        return self

    def in_(self, column, values):
        self._in = set(values)
        return self

    def execute(self):
        rows = sorted(self._rows, key=lambda r: r["created_at"], reverse=self._desc)
        if self._in is not None:
            rows = [r for r in rows if r["batch_id"] in self._in]
        if self._limit is not None:
            rows = rows[: self._limit]
        return SimpleNamespace(data=rows)


class FakeReadClient:
    def __init__(self, rows):
        self._rows = rows

    def table(self, name):
        return FakeReadTable(self._rows)


def make_read_persistence(monkeypatch, rows):
    monkeypatch.setattr("supabase.create_client", lambda url, key: FakeReadClient(rows))
    monkeypatch.setattr("app.config.SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setattr("app.config.SUPABASE_SERVICE_KEY", "svc")
    from app.persistence import SupabasePersistence

    return SupabasePersistence()


def _row(batch_id, i, created_at):
    return {
        "id": f"{batch_id}-img{i}",
        "batch_id": batch_id,
        "sample": SAMPLE,
        "created_at": created_at,
        "image_path": f"analyses/{batch_id}-img{i}.jpg",
        "status": "completed",
        "summary": {"WBC": 1, "RBC": 2, "Platelet": 0},
    }


def test_list_batches_returns_whole_batches_not_truncated(monkeypatch):
    # B1 (older) and B2 (newer), 3 images each. With limit=1 the old row-limited
    # code would report a 1-image batch; the two-step query returns all 3.
    rows = [
        _row("B1", 0, "2026-07-30T09:00:00Z"),
        _row("B1", 1, "2026-07-30T09:00:01Z"),
        _row("B1", 2, "2026-07-30T09:00:02Z"),
        _row("B2", 0, "2026-07-30T10:00:00Z"),
        _row("B2", 1, "2026-07-30T10:00:01Z"),
        _row("B2", 2, "2026-07-30T10:00:02Z"),
    ]
    persistence = make_read_persistence(monkeypatch, rows)

    batches = persistence.list_batches(limit=1)

    assert len(batches) == 1
    batch = batches[0]
    assert batch["batch_id"] == "B2"  # most recent batch
    assert batch["sample"] == SAMPLE  # surfaced at batch level
    assert batch["image_count"] == 3  # whole batch, not truncated
    assert len(batch["images"]) == 3
    assert batch["summary"] == {"WBC": 3, "RBC": 6, "Platelet": 0}  # aggregated
    # Earliest image's timestamp — consistent with get_batch.
    assert batch["created_at"] == "2026-07-30T10:00:00Z"
    assert batch["images"][0]["image_url"].startswith("https://x.supabase.co/")


def test_list_batches_orders_batches_newest_first(monkeypatch):
    rows = [
        _row("B1", 0, "2026-07-30T09:00:00Z"),
        _row("B2", 0, "2026-07-30T10:00:00Z"),
        _row("B3", 0, "2026-07-30T11:00:00Z"),
    ]
    persistence = make_read_persistence(monkeypatch, rows)

    batches = persistence.list_batches(limit=10)

    assert [b["batch_id"] for b in batches] == ["B3", "B2", "B1"]
