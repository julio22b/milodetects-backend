"""Central configuration, loaded from the environment (and a local .env file).

`fastapi dev` does not auto-load .env, so we load it here at import time.
Supabase is required to run the app (see `get_persistence` in persistence.py).
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads .env in the working directory if present; no-op otherwise

# Max images accepted per /analyze request (a batch). Shared with persistence,
# which uses it to size the batch-listing query window.
MAX_IMAGES = 10


def _origins(raw: str) -> list[str]:
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


# Browser origins allowed to call the API. Defaults to the Vite dev server.
ALLOWED_ORIGINS = _origins(os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173"))

# Inference engine. "mock" (default) returns hardcoded detections and needs no ML
# deps installed, so the app boots and tests run out of the box; "yolo" loads the
# trained YOLOv8 weights and runs real detection. Production MUST set
# INFERENCE_ENGINE=yolo — the mock default means an unconfigured deploy would
# silently serve fake results.
INFERENCE_ENGINE = os.environ.get("INFERENCE_ENGINE", "mock")
# Passed straight to YOLO's predict(). CONFIDENCE_THRESHOLD drops low-score boxes;
# IOU_THRESHOLD tunes NMS duplicate-merging (the model double-boxes clustered RBCs).
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", "0.45"))
# Committed to the repo (~6 MB); never downloaded at runtime. Relative to the CWD
# the server runs from (backend/).
YOLO_WEIGHTS_PATH = os.environ.get("YOLO_WEIGHTS_PATH", "weights/best.pt")

# Supabase. SUPABASE_SERVICE_KEY must be the *service_role* key: it's trusted,
# bypasses RLS, and must never be exposed to the frontend.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
# Prefixed because the Supabase project is shared with another app (Miloscribe).
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "milodetects-smears")
