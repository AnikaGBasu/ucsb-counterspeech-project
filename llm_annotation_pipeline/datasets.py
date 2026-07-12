import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from annotation_app.app import convert_validation_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRIAL_ITEMS = PROJECT_ROOT / "annotation_app" / "data" / "items.json"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation_set_jsons"
FRINGE_PLATFORMS = ("4chan", "gab", "stormfront", "vanguard")
FORUM_PLATFORMS = {"stormfront", "vanguard"}
FORUM_QUOTE_PREFIX = re.compile(r"^Quote:\s*Originally Posted by\s+")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_trial_threads() -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in read_json(TRIAL_ITEMS):
        normalized = dict(item)
        normalized["platform"] = "twitter"
        normalized.setdefault("dataset", TRIAL_ITEMS.name)
        normalized.setdefault("image_urls", [])
        grouped[str(normalized["thread_id"])].append(normalized)
    return [grouped[key] for key in sorted(grouped)]


def sample_validation_threads(platform: str, *, count: int, seed: int) -> list[list[dict[str, Any]]]:
    path = VALIDATION_DIR / f"{platform}_validation_original_threads.json"
    threads = convert_validation_file(str(path))
    if len(threads) < count:
        raise ValueError(f"{platform} contains only {len(threads)} threads; requested {count}")
    return random.Random(f"{seed}:{platform}").sample(threads, count)


def split_forum_quotes(thread: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Separate a flattened forum quote from the target author's contribution."""
    processed: list[dict[str, Any]] = []
    prior_author_texts: list[str] = []

    for item in thread:
        normalized = dict(item)
        text = str(item.get("target_text") or "").strip()
        match = FORUM_QUOTE_PREFIX.match(text)
        quoted_text: str | None = None
        author_text = text

        quoted_parts: list[str] = []
        while match:
            remainder = author_text[match.end():]
            best: tuple[int, int, str] | None = None
            for candidate in prior_author_texts:
                probe = candidate[: min(40, len(candidate))]
                position = remainder.find(probe)
                if position < 0 or position > 80:
                    continue
                common_length = 0
                for actual, expected in zip(remainder[position:], candidate):
                    if actual != expected:
                        break
                    common_length += 1
                if common_length >= 8 and (best is None or common_length > best[0]):
                    best = (common_length, position, candidate)

            if not best:
                break
            common_length, position, _ = best
            quoted_parts.append(remainder[position:position + common_length].strip())
            author_text = remainder[position + common_length:].strip()
            match = FORUM_QUOTE_PREFIX.match(author_text)

        if quoted_parts:
            quoted_text = "\n\n".join(quoted_parts)

        normalized["target_post_author_text"] = author_text
        normalized["target_post_quoted_text"] = quoted_text
        processed.append(normalized)
        prior_author_texts.append(author_text)

    return processed


def select_items(*, samples_per_platform: int, seed: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for thread in load_trial_threads():
        selected.extend({**item, "selection_group": "trial"} for item in thread)
    for platform in FRINGE_PLATFORMS:
        for thread in sample_validation_threads(platform, count=samples_per_platform, seed=seed):
            if platform in FORUM_PLATFORMS:
                thread = split_forum_quotes(thread)
            selected.extend({**item, "selection_group": "validation_sample"} for item in thread)
    return selected
