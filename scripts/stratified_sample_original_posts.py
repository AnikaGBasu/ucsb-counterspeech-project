import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = PROJECT_ROOT / "data" / "full_data" / "0_cleaned_data.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "annotation_selection"
REVIEW_POOL_CSV = OUTPUT_DIR / "original_post_review_pool.csv"
ALL_RANDOM_CSV = OUTPUT_DIR / "random_0_cleaned_original_posts.csv"
ALL_RANDOM_JSON = OUTPUT_DIR / "random_0_cleaned_original_posts_items.json"
SELECTED_100_CSV = OUTPUT_DIR / "selected_100_original_posts.csv"
SELECTED_100_JSON = OUTPUT_DIR / "selected_100_original_posts_items.json"
CANDIDATES_CSV = OUTPUT_DIR / "original_post_candidates.csv"
CANDIDATES_JSON = OUTPUT_DIR / "original_post_candidates_items.json"
SUMMARY_JSON = OUTPUT_DIR / "selection_summary.json"
RANDOM_REPORT_JSON = OUTPUT_DIR / "random_0_cleaned_sampling_report.json"
SEED = 20260707
SELECTED_COUNT = 100
PLATFORM_NAME = "X"


def safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clean_text(text: Optional[str]) -> str:
    return (text or "").strip()


def has_media(post: Dict[str, Any]) -> bool:
    media = post.get("media") or {}
    return any(safe_int(media.get(key)) or 0 for key in ("numPhotos", "numVideos", "numAnimated"))


def looks_like_direct_reply(text: str) -> bool:
    return text.lstrip().startswith("@")


def collect_reply_ids(node: Dict[str, Any], reply_ids: set) -> None:
    tweet_id = node.get("tweet_id")
    if tweet_id is not None:
        reply_ids.add(str(tweet_id))
    for child in node.get("nested_replies", []) or []:
        collect_reply_ids(child, reply_ids)


def load_source_records() -> List[Dict[str, Any]]:
    with SOURCE_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("data", []) if isinstance(payload, dict) else payload


def load_original_posts_from_0() -> List[Dict[str, Any]]:
    records = load_source_records()
    nested_reply_ids = set()
    for record in records:
        for reply in (record or {}).get("replies", []) or []:
            collect_reply_ids(reply, nested_reply_ids)

    deduped: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(records):
        post = (record or {}).get("post") or {}
        tweet_id = post.get("tweet_id")
        text = clean_text(post.get("raw_content"))
        if tweet_id is None or not text:
            continue
        post_id = str(tweet_id)
        if post_id in nested_reply_ids or looks_like_direct_reply(text):
            continue
        if post_id in deduped:
            deduped[post_id]["duplicate_sources"].append(SOURCE_FILE.name)
            continue
        deduped[post_id] = {
            "tweet_id": post_id,
            "source_file": SOURCE_FILE.name,
            "source_index": index,
            "duplicate_sources": [],
            "url": post.get("url") or "",
            "date": post.get("date") or "",
            "username": post.get("username") or "",
            "raw_content": text,
            "word_count": len(text.split()),
            "reply_count": safe_int(post.get("reply_count")) or 0,
            "likes_count": safe_int(post.get("likes_count")) or 0,
            "retweet_count": safe_int(post.get("retweet_count")) or 0,
            "quote_count": safe_int(post.get("quote_count")) or 0,
            "num_links": safe_int(post.get("numLinks")) or 0,
            "has_media": has_media(post),
            "has_quoted_tweet": bool(post.get("quoted_tweet")),
        }
    return sorted(deduped.values(), key=lambda row: (int(row["source_index"]), row["tweet_id"]))


def random_order(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(SEED)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return [{**row, "sample_rank": rank} for rank, row in enumerate(shuffled, start=1)]


def to_app_item(row: Dict[str, Any]) -> Dict[str, Any]:
    post_id = str(row["tweet_id"])
    return {
        "item_id": f"random_0_original_post_{post_id}",
        "thread_id": post_id,
        "post_id": post_id,
        "parent_id": None,
        "platform": PLATFORM_NAME,
        "source": row["source_file"],
        "is_reply": False,
        "has_prior_hate_in_context": False,
        "context": [],
        "target_text": row["raw_content"],
        "url": row["url"],
        "date": row["date"],
        "username": row["username"],
        "lang": "",
        "selection_metadata": {
            "sampling_method": "simple_random_order_from_0_cleaned_data_only_no_classification",
            "sample_rank": row["sample_rank"],
            "random_seed": SEED,
        },
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_items(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump([to_app_item(row) for row in rows], f, ensure_ascii=False, indent=2)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = load_original_posts_from_0()
    randomly_ordered = random_order(pool)
    selected = randomly_ordered[:SELECTED_COUNT]
    candidates = randomly_ordered[SELECTED_COUNT:]

    pool_fields = [
        "tweet_id", "source_file", "source_index", "url", "date", "username",
        "raw_content", "word_count", "reply_count", "likes_count", "retweet_count",
        "quote_count", "num_links", "has_media", "has_quoted_tweet", "duplicate_sources",
    ]
    sample_fields = pool_fields + ["sample_rank"]
    write_csv(REVIEW_POOL_CSV, pool, pool_fields)
    write_csv(ALL_RANDOM_CSV, randomly_ordered, sample_fields)
    write_csv(SELECTED_100_CSV, selected, sample_fields)
    write_csv(CANDIDATES_CSV, candidates, sample_fields)
    write_items(ALL_RANDOM_JSON, randomly_ordered)
    write_items(SELECTED_100_JSON, selected)
    write_items(CANDIDATES_JSON, candidates)

    ids = [row["tweet_id"] for row in randomly_ordered]
    report = {
        "selection_method": "simple_random_order_from_0_cleaned_data_only_no_classification",
        "source_file": str(SOURCE_FILE.relative_to(PROJECT_ROOT)),
        "random_seed": SEED,
        "original_post_pool_count": len(pool),
        "random_order_count": len(randomly_ordered),
        "selected_100_count": len(selected),
        "candidate_count": len(candidates),
        "duplicate_count": len(ids) - len(set(ids)),
        "outputs": {
            "review_pool_csv": str(REVIEW_POOL_CSV.relative_to(PROJECT_ROOT)),
            "all_random_csv": str(ALL_RANDOM_CSV.relative_to(PROJECT_ROOT)),
            "all_random_items_json": str(ALL_RANDOM_JSON.relative_to(PROJECT_ROOT)),
            "selected_100_csv": str(SELECTED_100_CSV.relative_to(PROJECT_ROOT)),
            "selected_100_items_json": str(SELECTED_100_JSON.relative_to(PROJECT_ROOT)),
            "remaining_candidates_csv": str(CANDIDATES_CSV.relative_to(PROJECT_ROOT)),
            "remaining_candidates_items_json": str(CANDIDATES_JSON.relative_to(PROJECT_ROOT)),
            "sampling_report": str(RANDOM_REPORT_JSON.relative_to(PROJECT_ROOT)),
        },
    }
    with RANDOM_REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Loaded {len(pool)} top-level original posts from {SOURCE_FILE.name}")
    print(f"Randomly ordered {len(randomly_ordered)} posts with seed {SEED}")
    print(f"Wrote first {len(selected)} to {SELECTED_100_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote remaining {len(candidates)} to {CANDIDATES_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote all random rows to {ALL_RANDOM_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
