import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "full_data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "annotation_selection"
REVIEW_POOL_CSV = OUTPUT_DIR / "original_post_review_pool.csv"
CANDIDATE_CSV = OUTPUT_DIR / "original_post_candidates.csv"
SELECTED_CSV = OUTPUT_DIR / "selected_100_original_posts.csv"
SELECTED_JSON = OUTPUT_DIR / "selected_100_original_posts_items.json"
CANDIDATE_JSON = OUTPUT_DIR / "original_post_candidates_items.json"
SUMMARY_JSON = OUTPUT_DIR / "selection_summary.json"
MANUAL_CURATION_FILE = OUTPUT_DIR / "manual_curation_plan.json"
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
    return any(
        safe_int(media.get(key)) or 0
        for key in ("numPhotos", "numVideos", "numAnimated")
    )


def looks_like_direct_reply(text: str) -> bool:
    return text.lstrip().startswith("@")


def collect_reply_ids(node: Dict[str, Any], reply_ids: set) -> None:
    tweet_id = node.get("tweet_id")
    if tweet_id is not None:
        reply_ids.add(str(tweet_id))

    for child in node.get("nested_replies", []) or []:
        collect_reply_ids(child, reply_ids)


def load_source_records() -> List[Tuple[Path, List[Dict[str, Any]]]]:
    source_records = []
    for path in sorted(INPUT_DIR.glob("*_cleaned_data.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        records = payload.get("data", []) if isinstance(payload, dict) else payload
        source_records.append((path, records))
    return source_records


def load_outermost_posts() -> List[Dict[str, Any]]:
    source_records = load_source_records()
    nested_reply_ids = set()

    for _path, records in source_records:
        for record in records:
            for reply in (record or {}).get("replies", []) or []:
                collect_reply_ids(reply, nested_reply_ids)

    deduped: Dict[str, Dict[str, Any]] = {}

    for path, records in source_records:
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
                deduped[post_id]["duplicate_sources"].append(path.name)
                continue

            deduped[post_id] = {
                "tweet_id": post_id,
                "source_file": path.name,
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

    return list(deduped.values())


def load_manual_curation(
    pool: List[Dict[str, Any]],
    plan_key: str,
    expected: Dict[str, int],
    excluded_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    if not MANUAL_CURATION_FILE.exists():
        raise RuntimeError(
            f"Manual curation file is missing: {MANUAL_CURATION_FILE.relative_to(PROJECT_ROOT)}. "
            "Create it from the raw review pool; this script intentionally does not keyword-sample."
        )

    with MANUAL_CURATION_FILE.open("r", encoding="utf-8") as f:
        plan = json.load(f)

    by_id = {row["tweet_id"]: row for row in pool}
    selected = []
    label_counts = Counter()
    seen = set()
    excluded_ids = excluded_ids or set()

    for entry in plan.get(plan_key, []):
        tweet_id = str(entry["tweet_id"])
        if tweet_id in seen:
            raise RuntimeError(f"Duplicate tweet_id in manual curation plan {plan_key}: {tweet_id}")
        if tweet_id in excluded_ids:
            raise RuntimeError(f"Manual curation tweet_id appears in both selected and candidate sets: {tweet_id}")
        if tweet_id not in by_id:
            raise RuntimeError(f"Manual curation tweet_id is not in the outermost-post pool: {tweet_id}")

        label = entry["label"]
        if label not in {"hate_speech", "not_hate_speech"}:
            raise RuntimeError(f"Unsupported manual label for {tweet_id}: {label}")

        label_counts[label] += 1
        seen.add(tweet_id)
        selected.append({
            **by_id[tweet_id],
            "curated_label": label,
            "curation_bucket": entry.get("bucket", ""),
            "curation_notes": entry.get("notes", ""),
            "selection_rank": label_counts[label],
        })

    if dict(label_counts) != expected:
        raise RuntimeError(
            f"Manual curation label counts for {plan_key} are {dict(label_counts)}, expected {expected}"
        )

    return selected


def to_app_item(row: Dict[str, Any]) -> Dict[str, Any]:
    post_id = str(row["tweet_id"])
    return {
        "item_id": f"curated_original_post_{post_id}",
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
            "curated_label": row["curated_label"],
            "curation_bucket": row["curation_bucket"],
            "curation_notes": row["curation_notes"],
        },
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    pool: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "input_files": [path.name for path in sorted(INPUT_DIR.glob("*_cleaned_data.json"))],
        "selection_method": "manual_reasoned_curation_from_unlabeled_outermost_post_pool",
        "outermost_post_pool_count": len(pool),
        "selected_label_counts": Counter(row["curated_label"] for row in selected),
        "candidate_label_counts": Counter(row["curated_label"] for row in candidates),
        "selected_source_counts": Counter(row["source_file"] for row in selected),
        "candidate_source_counts": Counter(row["source_file"] for row in candidates),
        "selected_curation_bucket_counts": Counter(row["curation_bucket"] for row in selected),
        "candidate_curation_bucket_counts": Counter(row["curation_bucket"] for row in candidates),
        "outputs": {
            "review_pool_csv": str(REVIEW_POOL_CSV.relative_to(PROJECT_ROOT)),
            "candidate_csv": str(CANDIDATE_CSV.relative_to(PROJECT_ROOT)),
            "manual_curation_file": str(MANUAL_CURATION_FILE.relative_to(PROJECT_ROOT)),
            "selected_csv": str(SELECTED_CSV.relative_to(PROJECT_ROOT)),
            "selected_app_items_json": str(SELECTED_JSON.relative_to(PROJECT_ROOT)),
            "candidate_app_items_json": str(CANDIDATE_JSON.relative_to(PROJECT_ROOT)),
            "live_app_items_json": "annotation_app/data/items.json",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the manually curated, root-only original-post annotation set."
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Also copy the selected app-items JSON to annotation_app/data/items.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pool = load_outermost_posts()
    pool.sort(key=lambda row: (row["source_file"], int(row["source_index"]), row["tweet_id"]))

    pool_fields = [
        "tweet_id", "source_file", "source_index", "url", "date", "username",
        "raw_content", "word_count", "reply_count", "likes_count", "retweet_count",
        "quote_count", "num_links", "has_media", "has_quoted_tweet", "duplicate_sources",
    ]
    write_csv(REVIEW_POOL_CSV, pool, pool_fields)

    expected_balanced = {"hate_speech": 50, "not_hate_speech": 50}
    selected = load_manual_curation(pool, "entries", expected_balanced)
    selected_ids = {row["tweet_id"] for row in selected}
    candidates = load_manual_curation(pool, "candidate_entries", expected_balanced, excluded_ids=selected_ids)
    selected_fields = pool_fields + ["curated_label", "curation_bucket", "curation_notes", "selection_rank"]
    write_csv(SELECTED_CSV, selected, selected_fields)
    write_csv(CANDIDATE_CSV, candidates, selected_fields)

    app_items = [to_app_item(row) for row in selected]
    with SELECTED_JSON.open("w", encoding="utf-8") as f:
        json.dump(app_items, f, ensure_ascii=False, indent=2)

    candidate_items = [to_app_item(row) for row in candidates]
    with CANDIDATE_JSON.open("w", encoding="utf-8") as f:
        json.dump(candidate_items, f, ensure_ascii=False, indent=2)

    if args.install:
        live_items = PROJECT_ROOT / "annotation_app" / "data" / "items.json"
        live_items.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SELECTED_JSON, live_items)

    summary = summarize(pool, selected, candidates)
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Loaded {len(pool)} unlabeled outermost posts")
    print(f"Wrote raw review pool: {REVIEW_POOL_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote selected CSV: {SELECTED_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote app items JSON: {SELECTED_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Wrote candidate CSV: {CANDIDATE_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote candidate app items JSON: {CANDIDATE_JSON.relative_to(PROJECT_ROOT)}")
    if args.install:
        print("Installed selected items to annotation_app/data/items.json")
    print("Selected labels:")
    for label, count in summary["selected_label_counts"].items():
        print(f"  {label}: {count}")
    print("Candidate labels:")
    for label, count in summary["candidate_label_counts"].items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
