import io
import asyncio
import logging
import time
from collections import Counter
from contextlib import asynccontextmanager
from functools import partial
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

from app import config, detection
from app.persistence import Persistence, get_persistence

logger = logging.getLogger("milodetects")
# uvicorn doesn't configure the root logger, so app-level INFO would otherwise be
# swallowed (only WARNING+ reaches stderr via logging's last-resort handler). Attach
# our own handler so the startup engine banner and rejected-upload warnings show in
# dev and on Render. propagate stays on so pytest's caplog still captures records.
if not logger.handlers:
    _log_handler = logging.StreamHandler()
    _log_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_log_handler)
    logger.setLevel(logging.INFO)

# Persistence backend, created lazily so importing this module never requires
# Supabase config (tests substitute a fake). The lifespan below forces it at
# startup so a misconfigured server fails fast instead of on the first request.
persist: Persistence | None = None


def _persistence() -> Persistence:
    global persist
    if persist is None:
        persist = get_persistence()
    return persist


# Inference engine, built once and kept resident (loading the model per request
# would add seconds to every analysis). Lazy like `persist` so importing this
# module needs no ML deps; the lifespan forces it at startup so a bad
# INFERENCE_ENGINE or missing weights fails the boot instead of the first request.
engine: detection.DetectionEngine | None = None


def _engine() -> detection.DetectionEngine:
    global engine
    if engine is None:
        # Logged before AND after the build: for yolo the load takes a moment, so if
        # the process OOMs/hangs mid-load you'll see "Loading..." with no "ready".
        logger.info("Loading inference engine %r...", config.INFERENCE_ENGINE)
        engine = detection.get_engine(config.INFERENCE_ENGINE)
        detail = (
            f" (weights={config.YOLO_WEIGHTS_PATH})"
            if config.INFERENCE_ENGINE == "yolo"
            else ""
        )
        logger.info("Inference engine ready: %s%s", config.INFERENCE_ENGINE, detail)
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    _persistence()  # fail fast if Supabase isn't configured
    _engine()  # fail fast (and load the model once) if inference is misconfigured
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_FORMATS = {"PNG", "JPEG"}
# Microscope objectives the frontend's dropdown offers. A per-image label sent
# parallel to `files`; anything outside this set is dropped (not stored) without
# failing the analysis. Domain constant, deliberately not env-configurable — same
# treatment as ALLOWED_FORMATS.
ALLOWED_MAGNIFICATIONS = {"4x", "10x", "40x", "100x"}
MAX_IMAGES = config.MAX_IMAGES
# Guards memory/DoS, NOT Supabase's Storage limit: the original is only held in
# memory for inference + downscaling, and the copy sent to Storage is the small
# downscaled JPEG. Phone cameras routinely emit 12-15 MB PNGs (Chrome mobile
# especially), so keep this generous or every phone upload is rejected.
MAX_IMAGE_BYTES = 25 * 1024 * 1024

# Stored images are downscaled to this longest-side size and re-encoded as JPEG
# to keep Storage small (~1 GB free tier). Detections are normalised (0..1), so
# the stored dimensions are irrelevant to the overlay renderer.
STORAGE_MAX_DIMENSION = 1600
STORED_CONTENT_TYPE = "image/jpeg"
STORED_EXTENSION = ".jpg"


class InvalidUpload(ValueError):
    """Raised when an uploaded file isn't an image we accept."""


def _validate_image(contents: bytes) -> None:
    """Raise InvalidUpload unless the bytes are an allowed image (PNG or JPEG)."""
    try:
        with Image.open(io.BytesIO(contents)) as image:
            image_format = image.format
            image.verify()
    except (OSError, Image.DecompressionBombError):
        raise InvalidUpload("File is not a valid image")

    if image_format not in ALLOWED_FORMATS:
        raise InvalidUpload(f"File type not allowed (detected {image_format})")


def _normalize_orientation(contents: bytes) -> bytes:
    """Bake EXIF rotation into the pixels so inference and the stored image share
    one grid — otherwise a phone photo (rotated via its EXIF tag) would be stored
    sideways, or, if only the stored copy were transposed, the boxes would land
    on the wrong grid. Returns the input unchanged when there's nothing to rotate
    (the common case: PNGs and already-upright photos), avoiding a re-encode.

    CPU-bound — call via run_in_executor.
    """
    with Image.open(io.BytesIO(contents)) as image:
        if image.getexif().get(0x0112, 1) == 1:  # 0x0112 = Orientation; 1 = none
            return contents
        source_format = image.format or "PNG"
        upright = ImageOps.exif_transpose(image)

    buffer = io.BytesIO()
    if source_format == "JPEG":
        upright.save(buffer, format="JPEG", quality=95)  # near-lossless re-encode
    else:
        upright.save(buffer, format=source_format)
    return buffer.getvalue()


def _downscale_to_jpeg(contents: bytes) -> bytes:
    """Resize (never upscale) to fit STORAGE_MAX_DIMENSION and encode as JPEG.

    CPU-bound — call via run_in_executor. Converts to RGB first because JPEG
    can't hold an alpha channel or a palette. `thumbnail` preserves aspect ratio
    in place and never enlarges an image already under the limit.
    """
    with Image.open(io.BytesIO(contents)) as image:
        rgb = image.convert("RGB")
    rgb.thumbnail((STORAGE_MAX_DIMENSION, STORAGE_MAX_DIMENSION))
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/analyze")
async def analyze(
    files: list[UploadFile] = File(...),
    sample: str = Form("TEST", max_length=8),
    magnifications: list[str] = Form(default=[]),
):
    # `magnifications` is optional and rides parallel to `files`, paired by
    # position (the frontend sends them in the same order). Repeated form fields
    # collect into a list; absent → []. One of ALLOWED_MAGNIFICATIONS per image —
    # an unrecognized or missing value is dropped (stored as None) and never fails
    # the analysis.
    # `sample` is temporarily optional: absent or blank/whitespace-only falls back
    # to a placeholder until the frontend always sends one.
    sample = sample.strip() or "TEST"

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files: {len(files)} (max {MAX_IMAGES})",
        )

    backend = _persistence()
    detector = _engine()
    loop = asyncio.get_running_loop()
    batch_id = str(uuid4())
    results = []

    # A count that doesn't line up with the images is a frontend bug worth
    # surfacing, but it must not fail the batch: pairing below is by index, so a
    # short list leaves the tail unpaired (None) and a long one ignores extras.
    if magnifications and len(magnifications) != len(files):
        logger.warning(
            "magnifications count (%d) != files count (%d); pairing by index",
            len(magnifications), len(files),
        )

    # Results are appended one-per-file in order; the frontend correlates each
    # result to the image it uploaded by position, so order must be preserved.
    for index, file in enumerate(files):
        if file.size is not None and file.size > MAX_IMAGE_BYTES:
            mb = MAX_IMAGE_BYTES // (1024 * 1024)
            logger.warning(
                "Rejected upload (too large): name=%s content_type=%s size=%s",
                file.filename, file.content_type, file.size,
            )
            results.append(
                {"id": None, "status": "failed", "error": f"File too large (max {mb} MB)"}
            )
            continue

        contents = await file.read()

        try:
            _validate_image(contents)  # ensure it's a supported image
        except InvalidUpload as exc:
            # Logged so the Render logs reveal what a client actually sent (e.g. a
            # WebP/HEIC the browser produced) — the reason never reaches the UI.
            logger.warning(
                "Rejected upload: name=%s content_type=%s size=%s reason=%s",
                file.filename, file.content_type, len(contents), exc,
            )
            results.append({"id": None, "status": "failed", "error": str(exc)})
            continue

        try:
            # Per-phase timing so a slow /analyze can be attributed to inference vs.
            # image work vs. the Supabase round-trip, instead of guessing.
            t0 = time.perf_counter()

            # Normalize EXIF orientation once so inference and the stored image
            # share one grid (keeps boxes aligned; fixes sideways phone photos).
            upright = await loop.run_in_executor(
                None, _normalize_orientation, contents
            )
            t_orient = time.perf_counter()

            # Inference runs on the upright, full-resolution bytes.
            detections = await loop.run_in_executor(None, detector.predict, upright)
            t_infer = time.perf_counter()
            detection_dicts = [
                {
                    "cell_type": item.cell_type.value,
                    "confidence": item.confidence,
                    "x": item.x,
                    "y": item.y,
                    "width": item.width,
                    "height": item.height,
                }
                for item in detections
            ]
            counts = Counter(item.cell_type for item in detections)
            summary = {ct.value: counts.get(ct, 0) for ct in detection.CellType}

            # Store a downscaled JPEG copy (of the same upright grid), never the
            # (multi-MB) original.
            image_bytes = await loop.run_in_executor(
                None, _downscale_to_jpeg, upright
            )
            t_downscale = time.perf_counter()

            # Resolve this image's magnification: keep it only if it's one we
            # recognise, else drop to None and note why. Never fails the analysis.
            raw_magnification = (
                magnifications[index].strip() if index < len(magnifications) else ""
            )
            if not raw_magnification:
                magnification = None
                magnification_error = None
            elif raw_magnification in ALLOWED_MAGNIFICATIONS:
                magnification = raw_magnification
                magnification_error = None
            else:
                magnification = None
                magnification_error = f"Unrecognized magnification {raw_magnification!r}"
                logger.warning(
                    "Invalid magnification %r for %s; storing none",
                    raw_magnification, file.filename,
                )

            analysis_id = str(uuid4())
            image_path = await loop.run_in_executor(
                None,
                partial(
                    backend.save_analysis,
                    analysis_id=analysis_id,
                    batch_id=batch_id,
                    sample=sample,
                    image_bytes=image_bytes,
                    content_type=STORED_CONTENT_TYPE,
                    extension=STORED_EXTENSION,
                    summary=summary,
                    detections=detection_dicts,
                    magnification=magnification,
                ),
            )
            t_save = time.perf_counter()
            logger.info(
                "analyze timing (s): orient=%.2f infer=%.2f downscale=%.2f "
                "save=%.2f total=%.2f",
                t_orient - t0,
                t_infer - t_orient,
                t_downscale - t_infer,
                t_save - t_downscale,
                t_save - t0,
            )
        except Exception:
            logger.exception("Failed to analyze/persist an uploaded image")
            results.append(
                {"id": None, "status": "failed", "error": "Failed to save analysis"}
            )
            continue

        result = {
            "id": analysis_id,
            "status": "completed",
            "content_type": STORED_CONTENT_TYPE,
            "image_url": backend.public_url(image_path),
            "detections": detection_dicts,
            "summary": summary,
            "magnification": magnification,
        }
        # Only present when a value was sent but not recognised — lets the frontend
        # flag it while still showing the (completed) analysis.
        if magnification_error:
            result["magnification_error"] = magnification_error
        results.append(result)

    return {
        "count": len(results),
        "batch_id": batch_id,
        "sample": sample,
        "results": results,
    }


@app.get("/batches")
async def list_batches(limit: int = 50):
    limit = max(1, min(limit, 100))
    loop = asyncio.get_running_loop()
    batches = await loop.run_in_executor(None, _persistence().list_batches, limit)
    return {"batches": batches}


@app.get("/batches/{batch_id}")
async def get_batch(batch_id: str):
    loop = asyncio.get_running_loop()
    batch = await loop.run_in_executor(None, _persistence().get_batch, batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    return batch


@app.delete("/batches/{batch_id}")
async def delete_batch(batch_id: str):
    loop = asyncio.get_running_loop()
    removed = await loop.run_in_executor(
        None, _persistence().delete_batch, batch_id
    )
    if removed == 0:
        raise HTTPException(status_code=404, detail="Batch not found")
    return {"batch_id": batch_id, "deleted_images": removed}
