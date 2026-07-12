"""Build a human-readable, thread-centered annotation progress report."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from annotation_app.app import (
    REQUIRED_ANNOTATORS_PER_THREAD,
    load_full_threads,
    normalize_counterspeech,
    normalize_target_group,
    read_annotation_rows,
    row_is_complete,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "annotation_app" / "annotation_storage" / "annotations_by_thread.json"
)


def latest_rows_by_annotator_and_item(item_lookup: dict[str, dict]) -> dict[tuple[str, str], dict]:
    """Keep the newest saved version of each annotator/item pair."""
    latest: dict[tuple[str, str], dict] = {}
    for row in read_annotation_rows():
        annotator_id = (row.get("annotator_id") or "").strip()
        item_id = row.get("item_id") or ""
        if not annotator_id or item_id not in item_lookup:
            continue
        key = (annotator_id, item_id)
        if row.get("timestamp", "") >= latest.get(key, {}).get("timestamp", ""):
            latest[key] = row
    return latest


def clean_annotation(row: dict, item: dict) -> dict:
    return {
        "timestamp": row.get("timestamp", ""),
        "annotator_id": row.get("annotator_id", ""),
        "confirmation_code": row.get("confirmation_code", ""),
        "assignment_id": row.get("assignment_id", ""),
        "is_complete": row_is_complete(row, item),
        "identity_hate_speech": row.get("identity_hate_speech", ""),
        "hate_severity": row.get("hate_severity", ""),
        "hate_target_group": normalize_target_group(row.get("hate_target_group", "")),
        "abuse_level": row.get("abuse_level", ""),
        "abuse_target": row.get("abuse_target", ""),
        "counterspeech": normalize_counterspeech(row.get("counterspeech", "")),
        "confidence": row.get("confidence", ""),
        "flag_for_review": str(row.get("flag_for_review", "")).lower() == "true",
        "notes": row.get("notes", ""),
    }


def build_report() -> dict:
    threads = load_full_threads()
    thread_map = {items[0]["thread_key"]: items for items in threads if items}
    item_lookup = {item["item_id"]: item for items in thread_map.values() for item in items}
    latest = latest_rows_by_annotator_and_item(item_lookup)

    annotations_by_item: dict[str, list[dict]] = defaultdict(list)
    for (_, item_id), row in latest.items():
        annotations_by_item[item_id].append(clean_annotation(row, item_lookup[item_id]))

    thread_records = []
    status_counts: Counter[str] = Counter()
    complete_annotation_sets = 0
    annotated_posts = 0

    for thread_key in sorted(thread_map):
        items = thread_map[thread_key]
        required_item_ids = {item["item_id"] for item in items}
        complete_items_by_annotator: dict[str, set[str]] = defaultdict(set)
        annotators_with_any_saved = set()

        posts = []
        for item in items:
            annotations = sorted(
                annotations_by_item.get(item["item_id"], []),
                key=lambda annotation: annotation["annotator_id"],
            )
            if annotations:
                annotated_posts += 1
            for annotation in annotations:
                annotator_id = annotation["annotator_id"]
                annotators_with_any_saved.add(annotator_id)
                if annotation["is_complete"]:
                    complete_items_by_annotator[annotator_id].add(item["item_id"])

            posts.append(
                {
                    "post_id": item.get("post_id", ""),
                    "item_id": item["item_id"],
                    "parent_id": item.get("parent_id"),
                    "is_reply": bool(item.get("is_reply")),
                    "annotations": annotations,
                }
            )

        complete_annotators = sorted(
            annotator_id
            for annotator_id, item_ids in complete_items_by_annotator.items()
            if required_item_ids.issubset(item_ids)
        )
        partial_annotators = sorted(annotators_with_any_saved - set(complete_annotators))
        complete_count = len(complete_annotators)
        complete_annotation_sets += complete_count

        if complete_count >= REQUIRED_ANNOTATORS_PER_THREAD:
            status = "complete"
        elif annotators_with_any_saved:
            status = "in_progress"
        else:
            status = "not_started"
        status_counts[status] += 1

        thread_records.append(
            {
                "thread_key": thread_key,
                "thread_id": items[0].get("thread_id", ""),
                "platform": items[0].get("platform", ""),
                "dataset": items[0].get("dataset", items[0].get("source", "")),
                "status": status,
                "is_complete": status == "complete",
                "complete_annotators": complete_count,
                "required_annotators": REQUIRED_ANNOTATORS_PER_THREAD,
                "complete_annotator_ids": complete_annotators,
                "partial_annotator_ids": partial_annotators,
                "posts_annotated": sum(bool(post["annotations"]) for post in posts),
                "posts_total": len(posts),
                "posts": posts,
            }
        )

    return {
        "summary": {
            "generated_at": datetime.now().astimezone().isoformat(),
            "completion_definition": (
                "A thread is complete when at least "
                f"{REQUIRED_ANNOTATORS_PER_THREAD} annotators have complete required fields "
                "for every post in the thread."
            ),
            "threads_complete": status_counts["complete"],
            "threads_in_progress": status_counts["in_progress"],
            "threads_not_started": status_counts["not_started"],
            "threads_total": len(thread_records),
            "complete_annotation_sets": complete_annotation_sets,
            "posts_with_any_annotation": annotated_posts,
            "posts_total": len(item_lookup),
        },
        "threads": thread_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
