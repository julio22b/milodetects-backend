# MiloDetects — backend

FastAPI service for **MiloDetects**, an assistive blood smear analysis tool: it
takes a photographed microscope field and returns the blood cells detected in it
(WBC / RBC / Platelets) as labeled bounding boxes and per-field counts. Detection
runs a self-hosted, fine-tuned YOLOv8 model in-process.

> It detects and localizes cells.
> It does not perform a clinical differential or produce diagnostic values.

**Live demo:** <!-- TODO: link --> · **Frontend repo:** <!-- TODO: link --> ·
**Project write-up:** <!-- TODO: link to portfolio case study -->

The rest of this document is operational — how to run, configure, test, and deploy
the service. For the project overview, architecture, and model details, see the
write-up linked above.

---

## Layout

```
backend/
├── app/                 # application code (a Python package)
│   ├── __init__.py      # marks app/ as a package
│   ├── main.py          # FastAPI app + HTTP routes
│   ├── config.py        # environment/settings (loads .env)
│   ├── persistence.py   # Supabase Storage + Postgres persistence
│   └── detection.py     # Detection types + the mock and YOLO inference engines
├── weights/             # trained model
│   └── best.pt          # YOLOv8n weights (committed; loaded when INFERENCE_ENGINE=yolo)
├── tests/               # pytest tests
├── requirements.txt     # dependencies
└── pytest.ini           # test config
```

## Endpoints

- `POST /analyze` — upload up to 10 images with an optional `sample` form field (a
  text id, ≤8 chars, identifying the batch; defaults to `TEST` for now); persists
  each to Supabase and returns per-image detections plus the `batch_id` and `sample`.
- `GET /batches?limit=N` — the N most recent batches, newest first (default 50,
  max 100). e.g. `?limit=3` for a "recent analyses" section.
- `GET /batches/{batch_id}` — one batch's images + detections, for re-render.
- `DELETE /batches/{batch_id}` — delete a batch: its Storage images and rows
  (detections cascade). 404 if unknown.

## Setup

```bash
python3 -m venv .venv          # first time only
source .venv/bin/activate      # do this in every new terminal
pip install -r requirements.txt          # runtime deps
pip install -r requirements-dev.txt      # + test deps (for running pytest)
```

You know the virtualenv is active when `which fastapi` points inside `.venv/`.

## Run

```bash
fastapi dev app/main.py
```

Then open http://127.0.0.1:8000/docs for the interactive API.

## Test

```bash
pytest
```

## Configuration

Configuration comes from environment variables, loaded from a local `.env` file
(git-ignored) if present. **Supabase is required** — the server fails fast at
startup if `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` aren't set.

| Variable               | Default                 | Notes                                                                                                                                             |
| ---------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SUPABASE_URL`         | — (required)            | Project URL.                                                                                                                                      |
| `SUPABASE_SERVICE_KEY` | — (required)            | **service_role** key. Server-only — never ship it to the frontend.                                                                                |
| `SUPABASE_BUCKET`      | `milodetects-smears`    | Storage bucket for images.                                                                                                                        |
| `ALLOWED_ORIGINS`      | `http://localhost:5173` | Comma-separated browser origins for CORS.                                                                                                         |
| `INFERENCE_ENGINE`     | `mock`                  | `yolo` runs the trained model; `mock` returns fixed detections. **Set to `yolo` in production** — the default would otherwise serve mock results. |
| `CONFIDENCE_THRESHOLD` | `0.25`                  | Min detection score (yolo only).                                                                                                                  |
| `IOU_THRESHOLD`        | `0.45`                  | NMS overlap threshold (yolo only); lower merges duplicate boxes more aggressively.                                                                |
| `YOLO_WEIGHTS_PATH`    | `weights/best.pt`       | Path to the committed model weights.                                                                                                              |

### Inference

Detection runs behind one interface with two engines, chosen at startup by
`INFERENCE_ENGINE`. The `yolo` engine loads `weights/best.pt` **once** at startup
and keeps it resident; if the weights fail to load it fails the boot loudly (no
silent fallback to mock). `mock` needs no ML dependencies.

The `yolo` engine requires `torch` + `ultralytics` (in `requirements.txt`, CPU-only
build). These are large; a first request after a cold start is slow (process start

- model load). On a small host (e.g. Render's free ~512 MB tier) the resident model
  may exceed available memory — the fix is a larger host, not loading per request.

#### Testing the real (yolo) engine

Test **locally first** — the model is heavy and a small host may OOM.

```bash
# .env may already set INFERENCE_ENGINE=yolo; otherwise pass it inline:
INFERENCE_ENGINE=yolo fastapi dev app/main.py
```

At startup you'll see the engine banner in the logs (added so "is the real engine
live?" is never a guess):

```
INFO:milodetects:Loading inference engine 'yolo'...
INFO:milodetects:Inference engine ready: yolo (weights=weights/best.pt)
```

If you see `Loading...` with no `ready`, the model load hung or OOM'd. The `mock`
engine loads nothing and only logs `Inference engine ready: mock`.

Then upload a real smear via http://127.0.0.1:8000/docs (`POST /analyze`) or:

```bash
curl -F "files=@smear.jpg" -F "sample=BLD01" http://127.0.0.1:8000/analyze
```

Verify: detections come back (not all zeros); **platelet counts are non-zero on an
image that has platelets** (confirms the plural `Platelets` class maps correctly);
boxes visibly align with cells. Tune `CONFIDENCE_THRESHOLD` (default 0.25) and
`IOU_THRESHOLD` (default 0.45; lower it if RBCs get double-boxed) without code changes.

**On Render:** set `INFERENCE_ENGINE=yolo` in the dashboard env vars and redeploy.
Watch the logs for the banner and for any OOM/restart on the free tier.

> Tests always run against the `mock` engine (pinned in `tests/conftest.py`), so a
> local `.env` with `INFERENCE_ENGINE=yolo` doesn't make `pytest` invoke the model.

### Supabase setup (one-time)

> The Supabase project is **shared** with another app (Miloscribe), whose public
> anon key is exposed in a browser bundle. The steps below keep MiloDetects'
> tables locked to this backend.

1. **Storage** → create a bucket named `milodetects-smears` (public).
2. **SQL editor** → create the tables:

    ```sql
    create table analyses (
      id            uuid primary key default gen_random_uuid(),
      created_at    timestamptz not null default now(),
      batch_id      uuid,
      sample        text not null           -- user-entered batch id
                      check (char_length(sample) between 1 and 8),
      image_path    text not null,          -- Storage key, never a URL
      content_type  text not null,
      status        text not null default 'processing'
                      check (status in ('processing','completed','failed')),
      error_message text,
      smear_type    text not null default 'blood'
                      check (smear_type in ('blood','urine','feces')),
      notes         text,
      summary       jsonb,
      user_id       uuid
    );
    create index analyses_created_at_idx on analyses (created_at desc);
    create index analyses_batch_id_idx  on analyses (batch_id);

    create table detections (
      id          uuid primary key default gen_random_uuid(),
      analysis_id uuid not null references analyses(id) on delete cascade,
      cell_type   text not null check (cell_type in ('WBC','RBC','Platelets')),
      confidence  real not null check (confidence between 0 and 1),
      x           real not null check (x between 0 and 1),
      y           real not null check (y between 0 and 1),
      width       real not null check (width between 0 and 1),
      height      real not null check (height between 0 and 1)
    );
    create index detections_analysis_id_idx on detections (analysis_id);
    ```

    > **Existing project (adding `sample` to a table that already has rows):**
    > `NOT NULL` can't be added directly, so add → backfill → enforce:
    >
    > ```sql
    > alter table analyses add column sample text;
    > update analyses set sample = '0000' where sample is null;  -- 4-digit placeholder
    > alter table analyses
    >   alter column sample set not null,
    >   add constraint analyses_sample_len check (char_length(sample) between 1 and 8);
    > ```

3. **Enable RLS with NO policies** on both tables. With no policies, the shared
   project's anon key can read/write nothing here, while this backend's
   `service_role` key bypasses RLS and works unchanged:

    ```sql
    alter table analyses  enable row level security;
    alter table detections enable row level security;
    ```

4. Put the credentials in `.env`:

    ```
    SUPABASE_URL=https://<project>.supabase.co
    SUPABASE_SERVICE_KEY=<service_role key>
    ```

> **Troubleshooting:** if inserts appear to succeed but write nothing, the
> backend is using the anon key by mistake — confirm `SUPABASE_SERVICE_KEY`
> holds the **service_role** key and is loaded.
