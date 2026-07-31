"""Persistence layer: where analyses, detections, and images are stored.

`SupabasePersistence` stores images in Supabase Storage and analyses + detections
in Postgres. `get_persistence()` returns it, or raises if Supabase isn't
configured. The `Persistence` protocol documents the interface (and lets tests
substitute a fake).

supabase-py is **synchronous**; every method here is blocking network I/O and is
meant to be called from the async routes via `run_in_executor`.
"""

from typing import Any, Protocol, cast

from app import config


def _aggregate_summary(summaries) -> dict[str, int]:
    """Sum per-cell-type counts across a batch's per-image summaries."""
    total: dict[str, int] = {}
    for summary in summaries:
        for cell_type, count in (summary or {}).items():
            total[cell_type] = total.get(cell_type, 0) + count
    return total


class Persistence(Protocol):
    def save_analysis(
        self,
        *,
        analysis_id: str,
        batch_id: str,
        image_bytes: bytes,
        content_type: str,
        extension: str,
        summary: dict[str, int],
        detections: list[dict],
    ) -> str:
        """Store the image (+ rows, in Supabase mode). Returns the image path/key."""
        ...

    def public_url(self, image_path: str) -> str:
        """Turn a stored image path/key into a URL a browser can load."""
        ...

    def list_batches(self, limit: int) -> list[dict]:
        """Recent batches, newest first, for the History screen."""
        ...

    def get_batch(self, batch_id: str) -> dict | None:
        """A single batch with every image and its detections, for re-render."""
        ...


class SupabasePersistence:
    """Images in Supabase Storage; analyses + detections in Postgres."""

    def __init__(self) -> None:
        # Imported lazily so the local fallback never needs supabase installed.
        from supabase import create_client

        self._client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        self._bucket = config.SUPABASE_BUCKET

    def save_analysis(
        self,
        *,
        analysis_id,
        batch_id,
        image_bytes,
        content_type,
        extension,
        summary,
        detections,
    ) -> str:
        key = f"analyses/{analysis_id}{extension}"

        # 1. Image first, so a 'completed' row can never point at a missing file.
        self._client.storage.from_(self._bucket).upload(
            path=key,
            file=image_bytes,
            file_options={"content-type": content_type, "upsert": "false"},
        )

        # 2. Row starts as 'processing'; a crash mid-write never leaves a
        #    'completed' analysis with zero detections.
        self._client.table("analyses").insert(
            {
                "id": analysis_id,
                "batch_id": batch_id,
                "image_path": key,
                "content_type": content_type,
                "status": "processing",
                "summary": summary,
            }
        ).execute()

        # 3. All detections in ONE insert, then flip to 'completed'.
        try:
            if detections:
                rows = [{"analysis_id": analysis_id, **d} for d in detections]
                self._client.table("detections").insert(rows).execute()
            self._client.table("analyses").update({"status": "completed"}).eq(
                "id", analysis_id
            ).execute()
        except Exception as exc:
            self._client.table("analyses").update(
                {"status": "failed", "error_message": str(exc)[:500]}
            ).eq("id", analysis_id).execute()
            raise

        return key

    def public_url(self, image_path: str) -> str:
        return f"{config.SUPABASE_URL}/storage/v1/object/public/{self._bucket}/{image_path}"

    def list_batches(self, limit: int) -> list[dict]:
        # Two-step so a batch is never truncated at a row-limit boundary.
        # Step 1: the most recent distinct batch_ids. A batch holds at most
        # MAX_IMAGES rows, so a window of limit*MAX_IMAGES recent rows is
        # guaranteed to contain at least `limit` whole batches.
        window = (
            self._client.table("analyses")
            .select("batch_id")
            .order("created_at", desc=True)
            .limit(limit * config.MAX_IMAGES)
            .execute()
        )
        batch_ids: list[str] = []
        seen: set[str] = set()
        for row in cast(list[dict[str, Any]], window.data or []):
            bid = row["batch_id"]
            if bid and bid not in seen:
                seen.add(bid)
                batch_ids.append(bid)
                if len(batch_ids) == limit:
                    break
        if not batch_ids:
            return []

        # Step 2: every row of those batches (never truncated), oldest-first so
        # the batch timestamp matches get_batch's (its earliest image).
        rows = cast(
            list[dict[str, Any]],
            (
                self._client.table("analyses")
                .select("id,batch_id,created_at,image_path,status,summary")
                .in_("batch_id", batch_ids)
                .order("created_at")
                .execute()
            ).data
            or [],
        )
        grouped: dict[str, list[dict]] = {bid: [] for bid in batch_ids}
        for row in rows:
            grouped[row["batch_id"]].append(row)

        batches = []
        for bid in batch_ids:  # recency order from step 1
            images = grouped[bid]
            if not images:  # deleted between the two queries
                continue
            batches.append(
                {
                    "batch_id": bid,
                    "created_at": images[0]["created_at"],
                    "image_count": len(images),
                    "summary": _aggregate_summary(img["summary"] for img in images),
                    "images": [
                        {
                            "id": img["id"],
                            "image_url": self.public_url(img["image_path"]),
                            "status": img["status"],
                            "summary": img["summary"],
                        }
                        for img in images
                    ],
                }
            )
        return batches

    def get_batch(self, batch_id: str) -> dict | None:
        response = (
            self._client.table("analyses")
            .select(
                "id,created_at,image_path,content_type,status,summary,"
                "detections(cell_type,confidence,x,y,width,height)"
            )
            .eq("batch_id", batch_id)
            .order("created_at")
            .execute()
        )
        rows = cast(list[dict[str, Any]], response.data or [])
        if not rows:
            return None

        return {
            "batch_id": batch_id,
            "created_at": rows[0]["created_at"],
            "images": [
                {
                    "id": row["id"],
                    "image_url": self.public_url(row["image_path"]),
                    "content_type": row["content_type"],
                    "status": row["status"],
                    "summary": row["summary"],
                    "detections": row.get("detections", []),
                }
                for row in rows
            ],
        }


def get_persistence() -> Persistence:
    """Return the Supabase persistence backend, or raise if it isn't configured."""
    if not (config.SUPABASE_URL and config.SUPABASE_SERVICE_KEY):
        raise RuntimeError(
            "Supabase is not configured — set SUPABASE_URL and "
            "SUPABASE_SERVICE_KEY (e.g. in a .env file)."
        )
    return SupabasePersistence()
