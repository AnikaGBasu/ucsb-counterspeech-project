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
MISSING_GAB_TEXT = "No text found (Timeout/Not Found)."
MAX_QUOTE_AUTHOR_TOKENS = 12


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def skip_validation_item(item: dict[str, Any]) -> bool:
    """Return whether a converted validation item has no annotatable text."""
    platform = str(item.get("platform") or "").strip().lower()
    target_text = str(item.get("target_text") or "").strip()
    return (
        str(item.get("post_id")) == "gab-ad-comment"
        or target_text == MISSING_GAB_TEXT
        or (platform == "4chan" and not target_text)
    )


def remove_skipped_validation_items(
    threads: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    return [
        [item for item in thread if not skip_validation_item(item)]
        for thread in threads
    ]


def token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(), match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def match_leading_forum_quote(
    remainder: str,
    prior_author_texts: list[str],
) -> tuple[str, str] | None:
    """Match a leading flattened quote to any passage from an earlier post."""
    remainder_tokens = token_spans(remainder)
    if len(remainder_tokens) < 2:
        return None

    best: tuple[int, int, int, int] | None = None
    max_quote_start = min(MAX_QUOTE_AUTHOR_TOKENS, len(remainder_tokens) - 1)
    candidate_token_lists = [token_spans(candidate) for candidate in prior_author_texts]

    for quote_start in range(1, max_quote_start + 1):
        first_token = remainder_tokens[quote_start][0]
        for candidate_tokens in candidate_token_lists:
            for candidate_start, (token, _, _) in enumerate(candidate_tokens):
                if token != first_token:
                    continue
                matched = 0
                while (
                    quote_start + matched < len(remainder_tokens)
                    and candidate_start + matched < len(candidate_tokens)
                    and remainder_tokens[quote_start + matched][0]
                    == candidate_tokens[candidate_start + matched][0]
                ):
                    matched += 1
                if not matched:
                    continue
                start = remainder_tokens[quote_start][1]
                end = remainder_tokens[quote_start + matched - 1][2]
                matched_chars = end - start
                if matched < 2 and matched_chars < 3:
                    continue
                score = (matched, matched_chars, -quote_start, end)
                if best is None or score > best:
                    best = score

    if best is None:
        return None
    _, _, quote_start_score, quote_end = best
    quote_start = -quote_start_score
    quote_start_offset = remainder_tokens[quote_start][1]
    return remainder[quote_start_offset:quote_end].strip(), remainder[quote_end:].strip()


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
    threads = remove_skipped_validation_items(convert_validation_file(str(path)))
    if len(threads) < count:
        raise ValueError(f"{platform} contains only {len(threads)} threads; requested {count}")
    return random.Random(f"{seed}:{platform}").sample(threads, count)


def load_all_validation_threads(platform: str) -> list[list[dict[str, Any]]]:
    path = VALIDATION_DIR / f"{platform}_validation_original_threads.json"
    return remove_skipped_validation_items(convert_validation_file(str(path)))


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
            quote_match = match_leading_forum_quote(remainder, prior_author_texts)
            if quote_match is None:
                # The quoted source is absent from the scraped thread. Keep the
                # unresolved leading section out of the reply author text.
                quoted_parts.append(author_text)
                author_text = ""
                break
            quoted_part, author_text = quote_match
            quoted_parts.append(quoted_part)
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


def select_all_validation_items() -> list[dict[str, Any]]:
    """Select every post from every fringe-platform validation thread only."""
    selected: list[dict[str, Any]] = []
    for platform in FRINGE_PLATFORMS:
        for thread in load_all_validation_threads(platform):
            if platform in FORUM_PLATFORMS:
                thread = split_forum_quotes(thread)
            selected.extend({**item, "selection_group": "validation_all"} for item in thread)
    return selected
