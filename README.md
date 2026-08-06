# MiloDetects — backend

A small FastAPI service that accepts a blood-smear image and returns detected
cells (WBC / RBC / platelets). Detection is currently a **mock** with the same
response shape a real YOLO model will produce, so the app works end to end
before any ML exists.

## Layout

```
backend/
├── app/                 # application code (a Python package)
│   ├── __init__.py      # marks app/ as a package
│   ├── main.py          # FastAPI app + HTTP routes
│   ├── config.py        # environment/settings (loads .env)
│   ├── persistence.py   # Supabase Storage + Postgres persistence
│   └── detection.py     # detection models + predict() (the mock for now)
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

| Variable               | Default                 | Notes                                                              |
| ---------------------- | ----------------------- | ------------------------------------------------------------------ |
| `SUPABASE_URL`         | — (required)            | Project URL.                                                       |
| `SUPABASE_SERVICE_KEY` | — (required)            | **service_role** key. Server-only — never ship it to the frontend. |
| `SUPABASE_BUCKET`      | `milodetects-smears`    | Storage bucket for images.                                         |
| `ALLOWED_ORIGINS`      | `http://localhost:5173` | Comma-separated browser origins for CORS.                          |

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
