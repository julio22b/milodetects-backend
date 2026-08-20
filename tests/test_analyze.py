import io
from uuid import UUID

from fastapi.testclient import TestClient
from PIL import Image

from app.main import (
    MAX_IMAGES,
    STORAGE_MAX_DIMENSION,
    _downscale_to_jpeg,
    _normalize_orientation,
    app,
)

client = TestClient(app)

SAMPLE = "S-100"  # a valid sample id (<= 8 chars); /analyze requires one


def make_png_bytes() -> bytes:
    """A tiny valid PNG we can upload without needing a real file on disk."""
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def png_uploads(count: int):
    """Build `count` multipart file parts, all under the field name 'files'."""
    return [
        ("files", (f"smear{i}", make_png_bytes(), "image/png")) for i in range(count)
    ]


def is_uuid(value: str) -> bool:
    try:
        UUID(value)
        return True
    except (ValueError, TypeError):
        return False


# --- /analyze ---------------------------------------------------------------


def test_root_says_hello():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_analyze_response_shape():
    response = client.post("/analyze", files=png_uploads(3), data={"sample": SAMPLE})
    assert response.status_code == 200

    body = response.json()
    assert body["count"] == 3
    assert is_uuid(body["batch_id"])
    assert body["sample"] == SAMPLE  # echoed back at the top level

    for result in body["results"]:
        assert result["status"] == "completed"
        assert is_uuid(result["id"])
        assert result["id"] in result["image_url"]  # one identifier throughout
        assert result["image_url"].endswith(".jpg")  # stored as downscaled JPEG
        assert result["content_type"] == "image/jpeg"
        assert "filename" not in result and "saved_path" not in result

    assert body["results"][0]["summary"] == {"WBC": 3, "RBC": 28, "Platelets": 9}


def test_analyze_persists_each_image(fake_persist):
    body = client.post("/analyze", files=png_uploads(2), data={"sample": SAMPLE}).json()

    assert body["count"] == 2
    assert len(fake_persist.saved) == 2  # save_analysis called once per image
    # Detections handed over as ONE bulk list of plain dicts.
    detections = fake_persist.saved[0]["detections"]
    assert isinstance(detections, list) and len(detections) == 40
    assert detections[0]["cell_type"] in {"WBC", "RBC", "Platelets"}

    # The stored copy is a downscaled JPEG, so content_type/extension describe it.
    assert fake_persist.saved[0]["content_type"] == "image/jpeg"
    assert fake_persist.saved[0]["extension"] == ".jpg"
    assert fake_persist.saved[0]["sample"] == SAMPLE  # batch-level id on every row

    first = body["results"][0]
    assert first["id"] == fake_persist.saved[0]["analysis_id"]  # id == analysis id
    assert first["image_url"].startswith("https://fake.supabase.co/")
    # batch_id constant across results and matches what save_analysis received.
    assert body["batch_id"] == fake_persist.saved[0]["batch_id"]
    assert body["batch_id"] == fake_persist.saved[1]["batch_id"]


def test_analyze_preserves_order_with_a_bad_image(fake_persist):
    files = [
        ("files", ("a", make_png_bytes(), "image/png")),
        ("files", ("b", b"not an image", "text/plain")),
        ("files", ("c", make_png_bytes(), "image/png")),
    ]

    body = client.post("/analyze", files=files, data={"sample": SAMPLE}).json()

    # Order-correlated: results[i] matches files[i].
    assert [r["status"] for r in body["results"]] == ["completed", "failed", "completed"]
    assert body["results"][1]["id"] is None
    assert len(fake_persist.saved) == 2  # only the two valid images were persisted


def test_analyze_error_result_when_persistence_fails(monkeypatch, fake_persist):
    def boom(**kwargs):
        raise RuntimeError("storage down")

    monkeypatch.setattr(fake_persist, "save_analysis", boom)

    body = client.post("/analyze", files=png_uploads(1), data={"sample": SAMPLE}).json()

    assert body["count"] == 1
    result = body["results"][0]
    assert result["status"] == "failed"
    assert result["id"] is None


def test_analyze_rejects_too_many_files():
    response = client.post(
        "/analyze", files=png_uploads(MAX_IMAGES + 1), data={"sample": SAMPLE}
    )
    assert response.status_code == 400


def test_analyze_defaults_sample_when_omitted(fake_persist):
    # `sample` is optional for now; omitting it falls back to the placeholder.
    body = client.post("/analyze", files=png_uploads(1)).json()
    assert body["sample"] == "TEST"
    assert fake_persist.saved[0]["sample"] == "TEST"


def test_analyze_blank_sample_falls_back_to_default(fake_persist):
    # Present but whitespace-only also uses the placeholder default.
    body = client.post("/analyze", files=png_uploads(1), data={"sample": "   "}).json()
    assert body["sample"] == "TEST"
    assert fake_persist.saved[0]["sample"] == "TEST"


def test_analyze_rejects_oversized_file(fake_persist, monkeypatch):
    monkeypatch.setattr("app.main.MAX_IMAGE_BYTES", 10)  # tiny cap for the test

    body = client.post("/analyze", files=png_uploads(1), data={"sample": SAMPLE}).json()

    result = body["results"][0]
    assert result["status"] == "failed"
    assert "too large" in result["error"].lower()
    assert result["id"] is None
    assert fake_persist.saved == []  # rejected before being read or persisted


# --- magnification (per-image, optional) ------------------------------------


def test_analyze_forwards_magnifications_per_image(fake_persist):
    body = client.post(
        "/analyze",
        files=png_uploads(2),
        data={"sample": SAMPLE, "magnifications": ["40x", "100x"]},
    ).json()

    # Paired to files by position, both persisted and echoed back per image.
    assert fake_persist.saved[0]["magnification"] == "40x"
    assert fake_persist.saved[1]["magnification"] == "100x"
    assert body["results"][0]["magnification"] == "40x"
    assert body["results"][1]["magnification"] == "100x"
    assert "magnification_error" not in body["results"][0]


def test_analyze_magnification_omitted_is_none(fake_persist):
    # Optional: no field sent → stored as None, echoed as null, no error note.
    body = client.post("/analyze", files=png_uploads(1), data={"sample": SAMPLE}).json()

    assert fake_persist.saved[0]["magnification"] is None
    assert body["results"][0]["magnification"] is None
    assert "magnification_error" not in body["results"][0]


def test_analyze_invalid_magnification_still_completes(fake_persist):
    # A bad value must NOT fail the analysis: the image is still detected and
    # persisted, the value is dropped to None, and a note explains why.
    body = client.post(
        "/analyze",
        files=png_uploads(1),
        data={"sample": SAMPLE, "magnifications": ["5x"]},
    ).json()

    result = body["results"][0]
    assert result["status"] == "completed"          # analysis went through
    assert result["magnification"] is None           # bad value dropped
    assert "5x" in result["magnification_error"]     # ...but reported
    assert len(fake_persist.saved) == 1              # still persisted
    assert fake_persist.saved[0]["magnification"] is None


def test_analyze_magnification_count_mismatch_pairs_by_index(fake_persist):
    # Fewer magnifications than images: pair what we can, leave the tail None,
    # never 400.
    body = client.post(
        "/analyze",
        files=png_uploads(2),
        data={"sample": SAMPLE, "magnifications": ["40x"]},
    ).json()

    assert [r["status"] for r in body["results"]] == ["completed", "completed"]
    assert fake_persist.saved[0]["magnification"] == "40x"
    assert fake_persist.saved[1]["magnification"] is None


# --- Downscaling for storage ------------------------------------------------


def test_downscale_shrinks_large_images_to_jpeg():
    source = io.BytesIO()
    # RGBA exercises the alpha->RGB conversion JPEG requires.
    Image.new("RGBA", (3000, 2000), (255, 0, 0, 128)).save(source, format="PNG")

    result = Image.open(io.BytesIO(_downscale_to_jpeg(source.getvalue())))

    assert result.format == "JPEG"
    assert result.mode == "RGB"
    assert max(result.size) == STORAGE_MAX_DIMENSION  # 1600 on the longest side
    assert min(result.size) < STORAGE_MAX_DIMENSION  # aspect ratio preserved


def test_downscale_does_not_upscale_small_images():
    source = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(source, format="PNG")

    result = Image.open(io.BytesIO(_downscale_to_jpeg(source.getvalue())))

    assert result.format == "JPEG"
    assert result.size == (100, 80)  # unchanged — never upscaled


# --- EXIF orientation normalization -----------------------------------------


def test_normalize_orientation_bakes_in_rotation():
    image = Image.new("RGB", (100, 60), "white")
    exif = image.getexif()
    exif[0x0112] = 6  # Orientation: rotate 90° when displaying
    source = io.BytesIO()
    image.save(source, format="JPEG", exif=exif)

    result = Image.open(io.BytesIO(_normalize_orientation(source.getvalue())))

    # Orientation 6 swaps width/height once baked into the pixels...
    assert result.size == (60, 100)
    # ...and the tag is cleared so viewers don't rotate a second time.
    assert result.getexif().get(0x0112, 1) == 1


def test_normalize_orientation_is_noop_when_upright():
    data = make_png_bytes()  # PNG, no orientation tag
    # Returns the exact same object — no wasteful re-encode.
    assert _normalize_orientation(data) is data
