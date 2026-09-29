"""Site regularization that follows the training set.

The captions say which tags the training images show and how often; the images say their shape. A
batch searches for the tags furthest below their share, keeps safe pictures of a similar shape, and
takes first those that close the most of the remaining gap. Images already in the version's
regularization folders count from the start, so every batch continues the set instead of repeating it.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ypuddin.data.caption_json import StructuredCaption
from ypuddin.data.captions import read_training_caption
from ypuddin.data.index import caption_target, iter_images

from .booru import META_TAG, Post, normalize, rating_terms
from .regularization_plan import training_sources

# Caption words no booru uses as tags: quality and period words, ratings, and Anima's @artist form.
NOT_SEARCHABLE = re.compile(
    r"^(@.+|score \d+( up)?|(masterpiece|best|high|good|normal|low|worst|amazing|great) quality|masterpiece"
    r"|(very )?aesthetic|(very )?displeasing|newest|recent|mid|early|old|year \d{4}"
    r"|safe|sensitive|nsfw|explicit|questionable|general)$"
)
# Looks that change what a picture teaches; skipped unless the training images have them too.
STYLE_TAGS = frozenset(
    {
        "monochrome",
        "greyscale",
        "comic",
        "4koma",
        "sketch",
        "lineart",
        "3d",
        "photo (medium)",
        "pixel art",
        "chibi",
    }
)
STYLE_SHARE = 0.05
# A tag the training images never show counts this much of a missing tag against a picture: pictures of
# other things lose to ones that look like the training set, without favouring pictures with few tags.
OFF_TARGET = 0.1
# Tags nearly every training image shows define the class (1girl, solo): every picture must show them,
# and they lead the searches when the tag limit leaves room.
CORE_SHARE = 0.9
TAG_SIZES = (10, 5, 3, 2, 1)
WINDOWS = 3  # Offsets tried into the list of tags furthest below their share.
PAGE_SIZE = 100
MAX_PAGES = 10
PICKS = 5  # Images taken per round before the shares are looked at again.
STALLED = 5  # Rounds in a row without a new image before the batch stops searching.
ROUNDS = 200


def searchable(key: str) -> bool:
    term = key.replace(" ", "_")
    return (
        0 < len(term) <= 60
        and key.count(" ") < 5  # Longer phrases are sentences, not tags.
        and not NOT_SEARCHABLE.match(key)
        and not META_TAG.match(term)
        and not re.match(r"^[a-z_]+:.", key)  # Would read as a metatag such as order:score.
        and term[0] not in "-~"
        and not {"*", ","} & set(term)
    )


def _caption_tags(path: Path, caption_ext: str, cache: dict) -> tuple[list[str], str] | None:
    """The caption's tags and its trigger, or None when it cannot be read."""
    try:
        raw = read_training_caption(caption_target(path, caption_ext, directory_cache=cache))
    except (OSError, UnicodeError, ValueError):
        return None
    if isinstance(raw, StructuredCaption):
        return [*raw.fixed, *raw.appearance, *raw.tags, *raw.environment], raw.trigger
    return [tag.strip() for tag in raw.split(",") if tag.strip()], ""


def _post_ids(stem: str) -> set[tuple[str | None, str]]:
    """Training images saved under a booru post id, so the batch never downloads them again."""
    if stem.isdigit():
        return {(None, stem)}
    found = re.fullmatch(r"(danbooru|gelbooru|e621|rule34)[_-](\d+)", stem.lower())
    return {(found.group(1), found.group(2))} if found else set()


@dataclass
class Profile:
    sources: list[dict]
    source_images: int = 0
    images: int = 0  # Captioned training images.
    missing_captions: int = 0
    invalid_captions: int = 0
    counts: Counter = field(default_factory=Counter)  # Tag: captioned training images with it.
    labels: dict[str, str] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)  # Searched tags: share of captioned images.
    unsearchable: list[str] = field(default_factory=list)
    aspect: tuple[float, float, float] = (0.5, 1.0, 2.0)  # Lowest, typical and highest width / height.
    size: tuple[float, float, float, float] = (1024.0, 1024.0, 0.0, 0.0)  # Median w, h and their spread.
    post_ids: set = field(default_factory=set)
    seed_images: int = 0
    seed: Counter = field(default_factory=Counter)  # Tag: regularization images with it already.
    triggers: set = field(default_factory=set)  # Always left out; not offered for exclusion.

    @property
    def min_short_side(self) -> float:
        return min(512.0, 0.5 * min(self.size[0], self.size[1]))

    def fits(self, post: Post) -> bool:
        if any(
            tag in STYLE_TAGS and self.counts[tag] < STYLE_SHARE * max(1, self.images)
            for tag in tag_keys(post)
        ):
            return False
        if not (post.width and post.height):
            return True  # The size is checked again once decoded.
        ratio = post.width / post.height
        return (
            self.aspect[0] <= ratio <= self.aspect[2] and min(post.width, post.height) >= self.min_short_side
        )

    def shape(self, post: Post) -> float:
        """How close the picture's shape and size are to the training images', from 0 to 1."""
        if not (post.width and post.height):
            return 0.5
        median = self.aspect[1]
        aspect = 1 / (1 + 10 * abs(post.width / post.height - median) / median)

        def near(value, typical, spread):
            if spread > 0:
                return 1 / (1 + abs(value - typical) / (2 * spread))
            return 1 / (1 + 10 * abs(value - typical) / typical)

        width, height, width_sd, height_sd = self.size
        return (
            0.6 * aspect
            + 0.4 * (near(post.width, width, width_sd) + near(post.height, height, height_sd)) / 2
        )

    def gain(self, tags, current: Counter, total: int) -> float:
        """How much closer one more picture with these tags brings the set's shares of the training
        tags to the training shares: the drop in squared distance, times total / 2. A training tag the
        set already shows enough counts against it, and so, lightly, does a tag the training images
        never show."""
        return sum(
            self.weights[tag] - current[tag] / total - 0.5 / total
            if tag in self.weights
            else -OFF_TARGET / total
            for tag in tags
        )

    def core(self, failed: set[str] = frozenset()) -> list[str]:
        return sorted(
            (tag for tag, weight in self.weights.items() if weight >= CORE_SHARE and tag not in failed),
            key=lambda tag: (-self.weights[tag], tag),
        )

    def order(self, current: Counter, picked: int, failed: set[str]) -> list[str]:
        """The training tags to search for, most behind first: those the pictures so far show least
        often compared with how often they should (share x pictures - pictures with it). Before any
        picture, and once every tag is on track, the most common tags lead."""
        live = [(tag, weight) for tag, weight in self.weights.items() if tag not in failed]
        behind = sorted(
            (
                (weight * picked - current[tag], weight, tag)
                for tag, weight in live
                if weight * picked - current[tag] > 0
            ),
            key=lambda item: (-item[0], -item[1], item[2]),
        )
        if picked and behind:
            return [tag for *_, tag in behind]
        return [tag for tag, _ in sorted(live, key=lambda item: (-item[1], item[0]))]

    def pick(self, candidates: list[Post], count: int, current: Counter, total: int) -> list[Post]:
        """The candidates to take, best first. Each pick counts before the next is scored; the tag
        gain is scaled across the candidates so it weighs 0.7 against the shape's 0.3."""
        pool, simulated, chosen = list(candidates), Counter(current), []
        while pool and len(chosen) < count:
            gains = [self.gain(tag_keys(post), simulated, total) for post in pool]
            low, high = min(gains), max(gains)
            scores = [
                0.7 * ((gain - low) / (high - low) if high > low else 1.0) + 0.3 * self.shape(post)
                for gain, post in zip(gains, pool, strict=True)
            ]
            best = max(range(len(pool)), key=lambda index: (scores[index], -index))
            post = pool.pop(best)
            chosen.append(post)
            simulated.update(tag_keys(post))
        return chosen


def tag_keys(post: Post) -> set[str]:
    return {normalize(tag) for tag in post.tags}


def build_profile(context, pid: str, vid: str, config: dict, source_ids=(), excluded=()) -> Profile:
    sources = training_sources(context, pid, vid, config, source_ids)
    chosen = set(source_ids)
    profile = Profile(sources=[{key: source[key] for key in ("id", "path", "name")} for source in sources])
    triggers = {normalize(config.get("dataset", {}).get("caption", {}).get("trigger_word") or "")}
    ratios, widths, heights = [], [], []
    cache: dict = {}
    for source in sources:
        if chosen and source["id"] not in chosen:
            continue
        root = Path(source["path"])
        if not root.is_dir():
            continue
        triggers.add(normalize(source.get("trigger_word") or ""))
        for path in sorted(iter_images(root)):
            profile.source_images += 1
            profile.post_ids |= _post_ids(path.stem)
            try:
                with Image.open(path) as image:
                    width, height = image.size
                if width > 0 and height > 0:
                    ratios.append(width / height)
                    widths.append(width)
                    heights.append(height)
            except (OSError, ValueError):
                pass
            read = _caption_tags(path, source["caption_ext"], cache)
            if read is None:
                profile.invalid_captions += 1
                continue
            tags, trigger = read
            triggers.add(normalize(trigger))
            keys = {}
            for tag in tags:
                keys.setdefault(normalize(tag), tag.strip())
            keys.pop("", None)
            if not keys:
                profile.missing_captions += 1
                continue
            profile.images += 1
            for key, label in keys.items():
                profile.counts[key] += 1
                profile.labels.setdefault(key, label)
    profile.triggers = triggers - {""}
    skip = {normalize(tag) for tag in excluded} | triggers
    for key, count in profile.counts.items():
        if key in skip:
            continue
        if searchable(key):
            profile.weights[key] = count / profile.images
        else:
            profile.unsearchable.append(key)
    profile.unsearchable.sort(key=lambda key: (-profile.counts[key], key))
    if len(ratios) >= 3:
        ordered = sorted(ratios)
        low = ordered[int(0.05 * (len(ordered) - 1))]
        high = ordered[round(0.95 * (len(ordered) - 1))]
        # A margin either side of the usual shapes; bucketing handles the rest.
        profile.aspect = (max(0.25, low / 1.2), statistics.median(ratios), min(4.0, high * 1.2))
    elif ratios:
        profile.aspect = (0.5, statistics.median(ratios), 2.0)
    if widths:
        profile.size = (
            float(statistics.median(widths)),
            float(statistics.median(heights)),
            statistics.pstdev(widths),
            statistics.pstdev(heights),
        )
    _seed(context, pid, vid, profile)
    return profile


def _seed(context, pid: str, vid: str, profile: Profile) -> None:
    """Count the images already in this version's regularization batches, and their tags."""
    root = context.reg_dir(pid, vid).resolve()
    for row in context.db.fetchall(
        "SELECT path FROM regularization_operations WHERE project_id=? AND version_id=? AND status='completed'",
        (pid, vid),
    ):
        batch = Path(row["path"] or "").resolve()
        if not batch.is_relative_to(root) or not batch.is_dir():
            continue
        for image in iter_images(batch):
            caption = image.with_suffix(".txt")
            try:
                text = caption.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                text = ""
            profile.seed_images += 1
            profile.seed.update({normalize(tag) for tag in text.split(",") if normalize(tag)})


def collect(profile: Profile, client, batch, count: int, notify) -> None:
    """Fill the batch with pictures that move the set's tags toward the training shares."""
    total = profile.seed_images + count
    current = Counter(profile.seed)
    limit = client.tag_limit()
    sizes = sorted({min(size, limit or size) for size in TAG_SIZES}, reverse=True)
    failed: set[str] = set()
    invalid: set[tuple[str, ...]] = set()
    pools: dict[tuple[str, ...], list[Post]] = {}
    pages: dict[tuple[str, ...], int] = {}
    exhausted: set[tuple[str, ...]] = set()

    def usable(post: Post) -> bool:
        return batch.wanted(post) and profile.fits(post) and set(core) <= tag_keys(post)

    def candidates(window: list[str]) -> list[Post]:
        key = tuple(sorted(window))
        pool = [post for post in pools.get(key, []) if usable(post)]
        while not pool and key not in exhausted and pages.get(key, 0) < MAX_PAGES:
            page = pages[key] = pages.get(key, 0) + 1
            terms = [tag.replace(" ", "_") for tag in window]
            notify(f"Searching {client.site.name} for {' '.join(terms)} (page {page})")
            posts = client.search([*terms, *rating_terms(["general"], client.site.name)], page, PAGE_SIZE)
            if not posts:
                exhausted.add(key)
            pool = [post for post in posts if usable(post)]
        pools[key] = pool
        return pool

    def windows(order: list[str]):
        """Search terms, widest first: the class tags lead, then the tags most behind, three ways."""
        if not order:
            yield core[: sizes[0]]
            return
        for size in sizes:
            lead = core[: size // 2]
            for offset in range(WINDOWS):
                window = lead + order[offset : offset + size - len(lead)]
                if len(window) < size:
                    break
                yield window

    stalled = 0
    core: list[str] = []
    for _ in range(ROUNDS):
        if batch.full:
            return
        core = profile.core(failed)
        order = [
            tag for tag in profile.order(current, profile.seed_images + batch.done, failed) if tag not in core
        ]
        if not order and not core:
            return  # The site has nothing usable for any training tag.
        found = None
        for window in windows(order):
            key = tuple(sorted(window))
            if key in invalid:
                continue
            pool = candidates(window)
            if pool:
                found = window, pool
                break
            invalid.add(key)
            if len(window) == 1:
                failed.add(window[0])  # The site has nothing usable for this tag.
        if not found:
            stalled += 1
            if stalled >= STALLED:
                return
            continue
        window, pool = found
        picks = profile.pick(pool, min(PICKS, count - batch.done), current, total)
        saved = batch.take(picks, query=" ".join(tag.replace(" ", "_") for tag in window), limit=len(picks))
        for post in saved:
            current.update(tag_keys(post))
        stalled = 0 if saved else stalled + 1
        if stalled >= STALLED:
            return


def plan(profile: Profile, *, tag_limit: int | None, tag_limit_known: bool, source: str) -> dict:
    top = sorted(profile.weights.items(), key=lambda item: (-item[1], item[0]))[:12]
    return {
        "source": source,
        "sources": profile.sources,
        "source_images": profile.source_images,
        "captioned_images": profile.images,
        "missing_captions": profile.missing_captions,
        "invalid_captions": profile.invalid_captions,
        "top_tags": [
            {"tag": profile.labels[key], "count": count}
            for key, count in sorted(profile.counts.items(), key=lambda item: (-item[1], item[0]))
            if key not in profile.triggers
        ][:200],
        "search_tags": [{"tag": profile.labels[key], "share": round(weight, 4)} for key, weight in top],
        "searchable_tags": len(profile.weights),
        "unsearchable_tags": [profile.labels[key] for key in profile.unsearchable[:20]],
        "aspect": {
            key: round(value, 3) for key, value in zip(("low", "median", "high"), profile.aspect, strict=True)
        },
        "size": {"width": round(profile.size[0]), "height": round(profile.size[1])},
        "existing_images": profile.seed_images,
        "suggested_count": max(0, min(200, profile.images - profile.seed_images)),
        "tag_limit": tag_limit,
        "tag_limit_known": tag_limit_known,
    }
