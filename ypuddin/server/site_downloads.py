"""Training images downloaded from Danbooru and Gelbooru.

A download searches a site for the tags given, keeps pictures of the chosen ratings and size, and adds
them to the version as a new dataset, or to one of its datasets, with the site's tags as captions in
that dataset's caption format. The files stay as the site serves them. Posts the version downloaded
before, including pictures deleted since, and posts its training or regularization images come from,
are never downloaded again.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import threading
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from ypuddin.data.index import iter_images

from .booru import (
    MEDIA_EXTS,
    RATINGS,
    BooruClient,
    Cancelled,
    Downloads,
    Post,
    Redirect,
    normalize,
    rating_terms,
)
from .db import new_id, now
from .errors import ApiError, NotFound

TERMINAL = {"completed", "failed", "cancelled"}
MAX_FILE_BYTES = 64 * 1024**2
# A download starts no further files past this; the few running ones may add a little more.
SOFT_BYTES = 3 * 1024**3
MAX_BYTES = 4 * 1024**3
MAX_PIXELS = 50_000_000
MAX_PAGES = 50
OWNER_FILE = ".site-download-owner"
# Never downloaded with any rating above general.
MINOR_TAGS = frozenset(
    {
        "loli",
        "shota",
        "child",
        "female child",
        "male child",
        "toddler",
        "baby",
        "aged down",
        "kindergarten uniform",
        "randoseru",
    }
)
# Tags that count the people in a picture; they lead a caption.
COUNT_TAG = re.compile(r"^(\d+\+?(girl|boy|other)s?|solo|solo focus|multiple (girls|boys|others)|no humans)$")
# Search operators both sites read before the colon; ratings, sort order and score have their own fields.
METATAGS = frozenset(
    "user approver commenter comm noter noteupdater artcomm commentaryupdater flagger appealer upvote "
    "downvote fav ordvote ordfav favgroup ordfavgroup reacted pool ordpool note comment commentary id "
    "rating source status filetype disapproved parent child search embedded md5 pixelhash width height "
    "mpixels ratio score upvotes downvotes favcount filesize date age order limit tagcount pixiv_id pixiv "
    "unaliased exif duration random is has ai sort".split()
)
FORMATS = {"JPEG": "jpg", "MPO": "jpg", "PNG": "png", "WEBP": "webp"}


def search_tags(text: str) -> list[str]:
    tags = text.split()
    if not tags or len(tags) > 12:
        raise ApiError("Enter 1 to 12 site tags separated by spaces", status=422, code="site_download.query")
    for tag in tags:
        operator = tag.split(":", 1)[0].lower() if ":" in tag[1:] else ""
        if (
            len(tag) > 100
            or tag[0] in "-~"
            or "*" in tag
            or tag.lower() in {"or", "and", "(", ")"}
            or operator in METATAGS
            or any(ord(char) < 32 for char in tag)
        ):
            raise ApiError(
                f"{tag} is not a plain site tag; set ratings, sort order and score with their own options",
                status=422,
                code="site_download.query",
                details={"tag": tag},
            )
    return tags


def excluded_tags(values) -> list[str]:
    # A leading minus is how the sites write an exclusion; this list is exclusions already.
    values = [value.strip().lstrip("-") for value in values]
    if any(not re.fullmatch(r"[^\s,*~][^,*]{0,99}", value) for value in values if value):
        raise ApiError("Excluded tags must be plain site tags", status=422, code="site_download.query")
    return list(dict.fromkeys(tag for tag in map(normalize, values) if tag))


def _label(tag: str) -> str:
    # Emoticons such as ^_^ keep their underscores; other tags read with spaces.
    return tag.replace("_", " ") if re.search(r"[^\W\d_]{2}", tag) else tag


def caption_groups(post: Post, *, anima: bool) -> dict[str, list[str]]:
    """The post's tags as caption text, grouped the way structured captions name them: people count,
    characters, works, artists (with Anima's @), then everything else in the site's order."""
    named = {*post.artists, *post.characters, *post.copyrights}
    general = [tag for tag in post.tags if tag not in named]
    return {
        "count": [_label(tag) for tag in general if COUNT_TAG.match(normalize(tag))],
        "character": [_label(tag) for tag in post.characters],
        "series": [_label(tag) for tag in post.copyrights],
        "artist": [("@" if anima else "") + _label(tag) for tag in post.artists],
        "tags": [_label(tag) for tag in general if not COUNT_TAG.match(normalize(tag))],
    }


def caption_text(groups: dict[str, list[str]], suffix: str) -> str:
    if suffix.lower() != ".json":
        return ", ".join(tag for tags in groups.values() for tag in tags)
    fixed = {key: ", ".join(groups[key]) for key in ("count", "character", "series", "artist") if groups[key]}
    return json.dumps({**fixed, "tags": groups["tags"]}, ensure_ascii=False, indent=2) + "\n"


def folder_name(name: str, tags: list[str]) -> str:
    base = name.strip() or "-".join(tags[:3])
    return re.sub(r"[^\w-]+", "-", base, flags=re.UNICODE).strip("-_")[:60] or "images"


def caption_path(image: Path, suffix: str) -> Path:
    return image.with_name(image.stem + (".txt" if suffix.lower() == "auto" else suffix))


def _post_ids(stem: str) -> set[tuple[str | None, str]]:
    if stem.isdigit():
        return {(None, stem)}
    found = re.fullmatch(r"(danbooru|gelbooru)[_-](\d+)", stem.lower())
    return {(found.group(1), found.group(2))} if found else set()


class _Batch(Downloads):
    """Posts of the allowed ratings and size, saved as the site serves them with a caption each."""

    def __init__(self, manager, oid, client, output, payload, *, taken, md5s, anima, suffix, cancelled):
        super().__init__(client, payload["count"], max_bytes=MAX_BYTES, cancelled=cancelled)
        self.manager, self.oid, self.output = manager, oid, output
        self.ratings = set(payload["ratings"])
        self.excluded = set(excluded_tags(payload["excluded_tags"]))
        self.min_side, self.min_score = payload["min_side"], payload["min_score"]
        self.taken, self.md5s, self.anima, self.suffix = taken, set(md5s), anima, suffix
        self.digests: set[str] = set()
        self.saved: list[dict] = []
        self.duplicates = 0

    def wanted(self, post: Post) -> bool:
        level = post.level
        return (
            level in self.ratings
            and post.ext in MEDIA_EXTS
            and bool(post.tags)
            and post.id not in self.seen
            and (self.source, post.id) not in self.taken
            and (None, post.id) not in self.taken
            and not (post.md5 and post.md5 in self.md5s)
            and not self.excluded & post.all_tags
            and not (level != "general" and MINOR_TAGS & post.all_tags)
            and "animated" not in post.all_tags
            and not (post.width and post.height and min(post.width, post.height) < self.min_side)
            and (self.min_score is None or post.score >= self.min_score)
        )

    def more(self) -> bool:
        return self.bytes < SOFT_BYTES

    def read(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > MAX_PIXELS or getattr(image, "is_animated", False):
                    raise ValueError("Not a still image within the pixel limit")
                image.load()  # Header-only checks accept truncated files that fail in training.
                if min(image.size) < max(16, self.min_side) or image.format not in FORMATS:
                    raise ValueError("Image too small or of an unsupported format")
                return data, FORMATS[image.format]

    def progress(self):
        self.manager._update(self.oid, phase="downloading", done=self.done)

    def keep(self, post, result, query):
        data, ext = result
        digest = hashlib.sha256(data).hexdigest()
        if digest in self.digests:
            self.duplicates += 1
            self.manager._update(self.oid, duplicates=self.duplicates)
            return False
        self.digests.add(digest)
        if post.md5:
            self.md5s.add(post.md5)
        image = self.output / f"{self.source}_{post.id}.{ext}"
        image.write_bytes(data)
        caption = caption_path(image, self.suffix)
        caption.write_text(
            caption_text(caption_groups(post, anima=self.anima), self.suffix), encoding="utf-8"
        )
        self.saved.append({"post_id": post.id, "md5": post.md5, "file": image.name, "caption": caption.name})
        return True


class SiteDownloadManager:
    def __init__(self, context, *, credentials, regularization=None, opener=None):
        self.c = context
        self.credentials = credentials
        self.regularization = regularization
        self.opener = opener  # Test seam; production always uses the provider-restricted opener.
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="site-download")
        self.cancel_events: dict[str, threading.Event] = {}
        self.previews: dict[tuple[str, str, str], BooruClient] = {}
        self.preview_lock = threading.Lock()
        self.closed = False
        self.c.db.execute("""CREATE TABLE IF NOT EXISTS site_downloads (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL, version_id TEXT NOT NULL,
            source TEXT NOT NULL, query TEXT NOT NULL, target_dataset_id TEXT,
            status TEXT NOT NULL, phase TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0,
            total INTEGER NOT NULL, request_json TEXT NOT NULL, logs_json TEXT NOT NULL DEFAULT '[]',
            error TEXT, dataset_id TEXT, path TEXT, images INTEGER NOT NULL DEFAULT 0,
            duplicates INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, finished_at REAL
        )""")
        self.c.db.execute("""CREATE TABLE IF NOT EXISTS site_download_posts (
            version_id TEXT NOT NULL, source TEXT NOT NULL, post_id TEXT NOT NULL,
            md5 TEXT NOT NULL DEFAULT '', file TEXT NOT NULL, dataset_id TEXT,
            operation_id TEXT NOT NULL, created_at REAL NOT NULL,
            PRIMARY KEY (version_id, source, post_id)
        )""")
        self._recover()

    # ------------------------------------------------------------------ records
    def _row(self, oid):
        row = self.c.db.fetchone("SELECT * FROM site_downloads WHERE id=?", (oid,))
        if not row:
            raise NotFound("Site download not found", code="site_download.not_found")
        return row

    def get(self, oid):
        row = self._row(oid)
        return {
            **{key: value for key, value in row.items() if key not in {"request_json", "logs_json"}},
            "logs": json.loads(row["logs_json"]),
            "can_cancel": row["status"] not in TERMINAL and row["phase"] != "publishing",
        }

    def snapshot(self, pid, vid):
        self.c.resolve_version(pid, vid)
        rows = self.c.db.fetchall(
            "SELECT id FROM site_downloads WHERE project_id=? AND version_id=? ORDER BY created_at DESC LIMIT 20",
            (pid, vid),
        )
        return {"operations": [self.get(row["id"]) for row in rows]}

    def _update(self, oid, *, message=None, **fields):
        with self.c.db.lock:
            if message:
                fields["logs_json"] = json.dumps([*json.loads(self._row(oid)["logs_json"]), message][-100:])
            self.c.db.update("site_downloads", oid, fields)
        self.c.bus.publish("site_download.changed", self.get(oid))

    def _staging(self, row) -> Path:
        return self.c.dataset_dir(row["project_id"], row["version_id"]) / f".site-{row['id']}"

    @staticmethod
    def _owned(path: Path, token: str | None) -> bool:
        marker = path / OWNER_FILE
        return bool(
            token
            and path.is_dir()
            and not path.is_symlink()
            and marker.is_file()
            and not marker.is_symlink()
            and marker.read_text(encoding="utf-8") == token
        )

    def _recover(self):
        for row in self.c.db.fetchall(
            "SELECT * FROM site_downloads WHERE status NOT IN ('completed','failed','cancelled')"
        ):
            token = json.loads(row["request_json"]).get("ownership_token")
            try:
                staging = self._staging(row)
                if self._owned(staging, token):
                    shutil.rmtree(staging)
                final = Path(row["path"]) if row["path"] else None
                # A new dataset folder only keeps its marker until the dataset is registered.
                if final and self._owned(final, token):
                    if self.c.db.fetchone("SELECT id FROM datasets WHERE path=?", (str(final),)):
                        (final / OWNER_FILE).unlink()
                    else:
                        shutil.rmtree(final)
            except Exception:  # noqa: BLE001 - a deleted version or folder leaves nothing to clean
                pass
            self.c.db.update(
                "site_downloads",
                row["id"],
                {
                    "status": "failed",
                    "phase": "failed",
                    "error": "Studio stopped during the download; nothing was added. Start it again.",
                    "finished_at": now(),
                },
            )

    def close(self):
        with self.c.db.lock:
            self.closed = True
            for event in self.cancel_events.values():
                event.set()
        self.pool.shutdown(wait=True)

    # ------------------------------------------------------------------ site access
    def _open_request(self, request, source, media):
        from .network import ProxyPolicy

        # The policy is read per request, so saved proxy settings apply to the next one.
        opener = self.opener or ProxyPolicy.from_context(self.c).opener(Redirect(source, media))
        return opener.open(request, timeout=15)

    def client(self, source, credentials, *, cancelled=None, notify=None):
        return BooruClient(
            source,
            *credentials,
            opener=lambda request, media: self._open_request(request, source, media),
            cancelled=cancelled,
            notify=notify,
            max_file_bytes=MAX_FILE_BYTES,
        )

    def _preview(self, source):
        """A long-lived client per site and account for previews, so their requests share one pace."""
        credentials = self.credentials.site(source)
        key = (source, *credentials)
        with self.preview_lock:
            client = self.previews.get(key)
            if client is None:
                self.previews = {other: value for other, value in self.previews.items() if other[0] != source}
                client = self.previews[key] = self.client(source, credentials)
        return client

    def _search(self, client, payload):
        conditions = rating_terms(payload["ratings"])
        if payload["min_score"] is not None:
            conditions.append(f"score:>={payload['min_score']}")
        sort = client.site.score_order if payload["order"] == "score" else None
        return client.fit(
            search_tags(payload["tags"]),
            excluded_tags(payload["excluded_tags"]),
            sort=sort,
            conditions=conditions,
        )

    def estimate(self, pid, vid, request):
        self.c.resolve_version(pid, vid)
        client = self._preview(request.source)
        search = self._search(client, request.model_dump())
        return {
            "source": request.source,
            "count": client.count(search.terms),
            "terms": search.terms,
            "local_exclusions": search.local,
            "tag_limit": client.tag_limit(),
            "sorted": search.sorted,
        }

    def suggest(self, source, query):
        term = query.strip()
        if len(term) < 2:
            return []
        return self._preview(source).suggest(term[:100])

    # ------------------------------------------------------------------ what a version has
    def _taken(self, pid, vid):
        """Posts this version must not download again, and the files it downloaded, by content."""
        from .routes_work import get_project_config

        ids, md5s = set(), set()
        for row in self.c.db.fetchall(
            "SELECT source, post_id, md5 FROM site_download_posts WHERE version_id=?", (vid,)
        ):
            ids.add((row["source"], row["post_id"]))
            if row["md5"]:
                md5s.add(row["md5"])
        config = get_project_config(pid, self.c, vid)
        roots = {
            Path(row["path"])
            for row in self.c.db.fetchall("SELECT path FROM datasets WHERE version_id=?", (vid,))
        } | {
            Path(source["path"]).expanduser()
            for source in [
                *config.get("dataset", {}).get("sources", []),
                *(config.get("validation") or {}).get("sources", []),
            ]
            if isinstance(source.get("path"), str) and source["path"].strip()
        }
        for root in roots:
            if not root.is_dir() or not self.c.is_allowed(root):
                continue
            for image in iter_images(root):
                ids |= _post_ids(image.stem)
        if self.regularization is not None:
            ids |= self.regularization.collected_posts(pid, vid)
        return ids, md5s

    # ------------------------------------------------------------------ tasks
    def start(self, pid, vid, request):
        from .routes_work import _get_dataset, _validate_caption_extension, get_project_config

        search_tags(request.tags)
        excluded_tags(request.excluded_tags)
        credentials = self.credentials.site(request.source)
        if request.source == "gelbooru" and not (credentials[0].isdigit() and credentials[1]):
            raise ApiError(
                "Configure the Gelbooru user ID and API key in Settings → Access keys",
                status=422,
                code="site_download.credentials",
            )
        version = self.c.resolve_version(pid, vid)
        target = None
        if request.dataset_id:
            target = _get_dataset(self.c, request.dataset_id)
            if target["version_id"] != version["id"]:
                raise ApiError(
                    "The dataset belongs to another version", status=409, code="site_download.dataset"
                )
            if target["is_reg"]:
                raise ApiError(
                    "Collect regularization images from the regularization panel",
                    status=409,
                    code="site_download.dataset",
                )
            from .source_roles import managed_source_role

            root = Path(target["path"])
            if (
                not managed_source_role(self.c, pid, version["id"], str(root))
                or root.absolute() != root.resolve()
                or not root.is_dir()
            ):
                raise ApiError("只能向当前版本内的数据集添加图片。", code="dataset.unmanaged", status=409)
            suffix = target["caption_ext"]
        else:
            suffix = request.caption_ext.strip() or "auto"
            _validate_caption_extension(suffix)
            family = get_project_config(pid, self.c, version["id"]).get("model", {}).get("family", "")
            if suffix.lower() == ".json":
                from ypuddin.data.caption_formats import family_caption_formats

                try:
                    formats = family_caption_formats(family)
                except Exception:  # noqa: BLE001 - an unknown family is checked when training starts
                    formats = ("txt", "json")
                if "json" not in formats:
                    raise ApiError(
                        "This model does not read JSON captions; choose TXT",
                        status=422,
                        code="site_download.caption",
                    )
        oid = new_id("sd")
        lease = self.c.versions.mutation(pid, version["id"])
        with self.c.db.lock:
            if self.closed:
                raise ApiError("Site downloads are stopping", status=503)
            version = lease.__enter__()
            try:
                config = get_project_config(pid, self.c, version["id"])
                root = self.c.dataset_dir(pid, version["id"])
                if target is None and any(
                    root.resolve().is_relative_to(Path(row["path"]).resolve())
                    for row in self.c.db.fetchall(
                        "SELECT path FROM datasets WHERE version_id=?", (version["id"],)
                    )
                ):
                    raise ApiError(
                        "A dataset of this version already covers its training folder; add the images to it",
                        status=409,
                        code="dataset.overlap",
                    )
                payload = request.model_dump(mode="json") | {
                    "ratings": [rating for rating in RATINGS if rating in request.ratings],
                    "caption_suffix": suffix,
                    "anima": config.get("model", {}).get("family") == "anima",
                    "ownership_token": uuid.uuid4().hex,
                }
                self.c.db.insert(
                    "site_downloads",
                    {
                        "id": oid,
                        "project_id": pid,
                        "version_id": version["id"],
                        "source": request.source,
                        "query": " ".join(request.tags.split()),
                        "target_dataset_id": target["id"] if target else None,
                        "status": "queued",
                        "phase": "queued",
                        "total": request.count,
                        "request_json": json.dumps(payload),
                        "created_at": now(),
                    },
                )
                event = threading.Event()
                self.cancel_events[oid] = event
                self.pool.submit(self._run, oid, payload, credentials, event, lease)
            except BaseException:
                self.cancel_events.pop(oid, None)
                if self.c.db.fetchone("SELECT id FROM site_downloads WHERE id=?", (oid,)):
                    self.c.db.update(
                        "site_downloads",
                        oid,
                        {
                            "status": "failed",
                            "phase": "failed",
                            "error": "Could not start the download",
                            "finished_at": now(),
                        },
                    )
                lease.__exit__(None, None, None)
                raise
        return self.get(oid)

    def cancel(self, oid):
        with self.c.db.lock:
            row = self._row(oid)
            if row["status"] in TERMINAL:
                return self.get(oid)
            if row["phase"] == "publishing":
                raise ApiError(
                    "The downloaded images are being added; wait for it to finish",
                    status=409,
                    code="site_download.publishing",
                )
            event = self.cancel_events.get(oid)
            if event:
                event.set()
            self._update(oid, status="cancelling", message="Cancellation requested")
        return self.get(oid)

    @staticmethod
    def _check(cancelled):
        if cancelled.is_set():
            raise Cancelled()

    def _run(self, oid, payload, credentials, cancelled, lease):
        row = self._row(oid)
        staging = None
        try:
            staging = self._staging(row)
            staging.parent.mkdir(parents=True, exist_ok=True)
            staging.mkdir()
            (staging / OWNER_FILE).write_text(payload["ownership_token"], encoding="utf-8")
            files = staging / "files"
            files.mkdir()
            self._update(oid, status="running", phase="searching", message="Preparing the download")
            batch = self._collect(oid, row, payload, credentials, files, cancelled)
            self._check(cancelled)
            with self.c.db.lock:
                self._check(cancelled)
                self._update(oid, phase="publishing", message=f"Adding {batch.done} images to the version")
                did, path, added = self._publish(row, payload, batch, files)
            from .routes_work import _index_dataset

            _index_dataset(self.c, did)
            self._update(
                oid,
                status="completed",
                phase="completed",
                images=added,
                finished_at=now(),
                message=f"Added {added} images to {Path(path).name}",
            )
        except Exception as exc:
            stopped = isinstance(exc, Cancelled) or cancelled.is_set()
            message = (
                None
                if stopped
                else exc.message
                if isinstance(exc, ApiError)
                else f"The download failed ({type(exc).__name__}); nothing was added"
            )
            self._update(
                oid,
                status="cancelled" if stopped else "failed",
                phase="cancelled" if stopped else "failed",
                error=message,
                finished_at=now(),
            )
        finally:
            if staging is not None and self._owned(staging, payload["ownership_token"]):
                shutil.rmtree(staging, ignore_errors=True)
            with self.c.db.lock:
                self.cancel_events.pop(oid, None)
                lease.__exit__(None, None, None)
            self.c.bus.publish(
                "version.changed", {"project_id": row["project_id"], "version_id": row["version_id"]}
            )

    def _collect(self, oid, row, payload, credentials, files, cancelled):
        client = self.client(
            row["source"],
            credentials,
            cancelled=cancelled,
            notify=lambda message: self._update(oid, message=message),
        )
        search = self._search(client, payload)
        if not search.sorted:
            self._update(
                oid,
                message="Sorting by score takes one more tag than this account may search; newest posts come first",
            )
        if search.local:
            self._update(
                oid, message=f"Checking {len(search.local)} excluded tags locally beyond the site's tag limit"
            )
        taken, md5s = self._taken(row["project_id"], row["version_id"])
        batch = _Batch(
            self,
            oid,
            client,
            files,
            payload,
            taken=taken,
            md5s=md5s,
            anima=payload["anima"],
            suffix=payload["caption_suffix"],
            cancelled=cancelled,
        )
        query = " ".join(search.terms)
        for page in range(1, MAX_PAGES + 1):
            self._check(cancelled)
            self._update(oid, message=f"Searching {row['source']}, page {page}")
            posts = client.search(search.terms, page)
            if not posts:
                break
            batch.take(posts, query=query)
            if batch.full or not batch.more():
                break
        if batch.skipped:
            self._update(oid, message=f"Skipped {batch.skipped} files that were not usable images")
        if not batch.done:
            raise ApiError(
                "No new matching images were found. Change the tags, ratings or filters; nothing was added.",
                code="site_download.empty",
            )
        if not batch.more():
            self._update(oid, message=f"Stopped at the 3 GiB download limit with {batch.done} images")
        elif batch.done < batch.count:
            self._update(oid, message=f"Found {batch.done} of {batch.count} matching images; adding them")
        return batch

    def _publish(self, row, payload, batch, files):
        """Move the downloaded files into their dataset and register what is new. Runs under the
        database lock and the version lease; returns the dataset id, its folder and the images added."""
        from .routes_work import DatasetBody, _get_dataset, _register_dataset

        pid, vid, token = row["project_id"], row["version_id"], payload["ownership_token"]
        moved: list[Path] = []
        final = None
        try:
            if row["target_dataset_id"]:
                dataset = _get_dataset(self.c, row["target_dataset_id"])
                folder = Path(dataset["path"])
                existing = {path.stem.casefold() for path in folder.iterdir()}
                for entry in batch.saved:
                    if Path(entry["file"]).stem.casefold() in existing:
                        continue  # Added to the folder by hand since the download started.
                    for name in (entry["file"], entry["caption"]):
                        (files / name).rename(folder / name)
                        moved.append(folder / name)
                added = len(moved) // 2
                did = dataset["id"]
                self.c.db.update("datasets", did, {"index_status": "indexing"})
            else:
                root = self.c.dataset_dir(pid, vid)
                base = folder_name(payload["name"], payload["tags"].split())
                occupied = {path.name.casefold() for path in root.iterdir()}
                name, number = base, 2
                while name.casefold() in occupied:
                    name, number = f"{base}-{number}", number + 1
                final = folder = root / name
                folder.mkdir()
                (folder / OWNER_FILE).write_text(token, encoding="utf-8")
                self._update(row["id"], path=str(folder))
                for entry in batch.saved:
                    for name in (entry["file"], entry["caption"]):
                        (files / name).rename(folder / name)
                added = len(batch.saved)
                did = _register_dataset(
                    self.c,
                    pid,
                    DatasetBody(
                        path=str(folder), repeats=payload["repeats"], caption_ext=payload["caption_suffix"]
                    ),
                    version_id=vid,
                    internal=True,
                )
                (folder / OWNER_FILE).unlink()
        except BaseException:
            for path in reversed(moved):
                path.unlink(missing_ok=True)
            if final is not None and self._owned(final, token):
                shutil.rmtree(final)
                self._update(row["id"], path=None)
            raise
        with self.c.db.transaction():
            for entry in batch.saved:
                self.c.db.execute(
                    "INSERT OR IGNORE INTO site_download_posts (version_id, source, post_id, md5, file, dataset_id, operation_id, created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        vid,
                        row["source"],
                        entry["post_id"],
                        entry["md5"],
                        entry["file"],
                        did,
                        row["id"],
                        now(),
                    ),
                )
        self._update(row["id"], dataset_id=did, path=str(folder))
        return did, str(folder), added
