import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "sample" / "extracted_68_threads.json"
OUTPUT_FILE = PROJECT_ROOT / "annotation_app" / "data" / "items.json"
PLATFORM_NAME = "X"


def safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_text(text: Optional[str]) -> str:
    return text.strip() if text else ""


def make_context_entry(tweet: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "post_id": str(tweet.get("tweet_id", "")),
        "speaker": tweet.get("username", ""),
        "text": normalize_text(tweet.get("raw_content", ""))
    }


def walk_node(
    node: Dict[str, Any],
    root_tweet_id: str,
    source_name: str,
    target_ids: set,
    parent_id: Optional[str],
    ancestors: List[Dict[str, Any]],
    results: List[Dict[str, Any]]
) -> None:
    tweet_id_raw = node.get("tweet_id")
    tweet_id = str(tweet_id_raw) if tweet_id_raw is not None else ""

    if safe_int(tweet_id_raw) in target_ids:
        results.append({
            "item_id": f"{source_name}_{tweet_id}",
            "thread_id": str(root_tweet_id),
            "post_id": tweet_id,
            "parent_id": str(parent_id) if parent_id is not None else None,
            "platform": PLATFORM_NAME,
            "source": source_name,
            "is_reply": parent_id is not None,
            "has_prior_hate_in_context": len(ancestors) > 0,
            "context": ancestors,
            "target_text": normalize_text(node.get("raw_content", "")),
            "url": node.get("url"),
            "date": node.get("date"),
            "username": node.get("username"),
            "lang": node.get("lang")
        })

    new_ancestors = ancestors + [make_context_entry(node)]

    for child in node.get("nested_replies", []) or []:
        walk_node(
            node=child,
            root_tweet_id=root_tweet_id,
            source_name=source_name,
            target_ids=target_ids,
            parent_id=tweet_id,
            ancestors=new_ancestors,
            results=results
        )


def main() -> None:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_ids = {safe_int(x) for x in data.get("target_tweet_ids", [])}
    target_ids.discard(None)

    threads = data.get("threads", [])
    results: List[Dict[str, Any]] = []
    source_name = INPUT_FILE.stem

    for thread in threads:
        root = thread.get("post")
        if not root:
            continue

        root_id = str(root.get("tweet_id", ""))
        root_tid = safe_int(root.get("tweet_id"))

        if root_tid in target_ids:
            results.append({
                "item_id": f"{source_name}_{root_id}",
                "thread_id": root_id,
                "post_id": root_id,
                "parent_id": None,
                "platform": PLATFORM_NAME,
                "source": source_name,
                "is_reply": False,
                "has_prior_hate_in_context": False,
                "context": [],
                "target_text": normalize_text(root.get("raw_content", "")),
                "url": root.get("url"),
                "date": root.get("date"),
                "username": root.get("username"),
                "lang": root.get("lang")
            })

        root_context = [make_context_entry(root)]

        for reply in thread.get("replies", []) or []:
            walk_node(
                node=reply,
                root_tweet_id=root_id,
                source_name=source_name,
                target_ids=target_ids,
                parent_id=root_id,
                ancestors=root_context,
                results=results
            )

    seen = set()
    deduped = []
    for item in results:
        if item["post_id"] not in seen:
            seen.add(item["post_id"])
            deduped.append(item)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    found_ids = {safe_int(item["post_id"]) for item in deduped}
    missing_ids = sorted(target_ids - found_ids)

    print(f"Wrote {len(deduped)} items to {OUTPUT_FILE}")
    print(f"Matched {len(found_ids)} / {len(target_ids)} target tweet IDs")

    if missing_ids:
        print("\\nMissing target tweet IDs:")
        for tid in missing_ids:
            print(tid)


if __name__ == "__main__":
    main()