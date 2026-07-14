import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from llm_annotation_pipeline.datasets import PROJECT_ROOT, select_all_validation_items, select_items
from llm_annotation_pipeline.schemas import Annotation


PROMPT_FILES = {
    "twitter": "x_twitter_annotation_prompt.md", "x": "x_twitter_annotation_prompt.md",
    "gab": "gab_annotation_prompt.md", "4chan": "4chan_pol_annotation_prompt.md",
    "stormfront": "stormfront_annotation_prompt.md", "vanguard": "vanguard_annotation_prompt.md",
}
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def normalized_platform(value: str) -> str:
    platform = str(value).strip().lower()
    if platform not in PROMPT_FILES:
        raise ValueError(f"Unsupported platform: {value!r}")
    return "twitter" if platform == "x" else platform


def load_prompts() -> dict[str, str]:
    return {key: (PROMPT_DIR / filename).read_text(encoding="utf-8") for key, filename in PROMPT_FILES.items()}


def format_post(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return "null"
    return (
        f"POST_ID: {entry.get('post_id', '')}\n"
        f"SPEAKER: {entry.get('speaker', '') or 'unknown'}\n"
        f"TEXT: {entry.get('text', '') or ''}\n"
        f"IMAGE_URLS: {json.dumps(entry.get('image_urls') or [])}"
    )


def format_context(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "null"
    return "\n\n".join(
        f"[{index}] {format_post(entry)}"
        for index, entry in enumerate(entries, start=1)
    )


def item_input(item: dict[str, Any]) -> str:
    context = item.get("context") or []
    is_reply = bool(item.get("is_reply"))
    root_post = context[0] if is_reply and context else None
    parent_id = str(item.get("parent_id")) if item.get("parent_id") is not None else None
    direct_parent = next(
        (entry for entry in reversed(context) if str(entry.get("post_id")) == parent_id),
        None,
    )
    platform = normalized_platform(item["platform"])
    if platform in {"stormfront", "vanguard"}:
        target_fields = (
            f"TARGET_POST_AUTHOR_TEXT: {item.get('target_post_author_text', item.get('target_text', ''))}\n"
            f"TARGET_POST_QUOTED_TEXT: {item.get('target_post_quoted_text') or 'null'}\n"
        )
    else:
        target_fields = f"TARGET_POST: {item.get('target_text', '')}\n"
    return (
        f"PLATFORM: {platform}\nIS_REPLY: {str(is_reply).lower()}\n"
        f"ROOT_POST:\n{format_post(root_post)}\n"
        f"DIRECT_PARENT_POST:\n{format_post(direct_parent)}\n"
        f"CONVERSATION_CONTEXT:\n{format_context(context)}\n"
        f"{target_fields}"
        "OCR_TEXT: null (already merged into TARGET_POST when available)\n"
        f"MEDIA_DESCRIPTION: Image URLs only: {json.dumps(item.get('image_urls') or [])}"
    )


def annotate(client: OpenAI, *, model: str, reasoning_effort: str, prompt: str, content: str) -> tuple[Annotation, Any]:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = client.responses.parse(
                model=model,
                reasoning={"effort": reasoning_effort},
                input=[{"role": "system", "content": prompt}, {"role": "user", "content": content}],
                text_format=Annotation,
            )
            if response.output_parsed is None:
                raise ValueError(f"No parsed annotation in response {response.id}")
            return response.output_parsed, response
        except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError) as error:
            last_error = error
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable") from last_error


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                completed.add(str(json.loads(line)["item_id"]))
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"Cannot safely resume: invalid record at {path}:{line_number}") from error
    return completed


def write_manifest(path: Path, args: argparse.Namespace, items: list[dict[str, Any]]) -> None:
    threads = Counter((item["selection_group"], normalized_platform(item["platform"]), str(item["thread_id"])) for item in items)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "model": args.model,
        "reasoning_effort": args.reasoning_effort, "seed": args.seed,
        "all_validation": args.all_validation,
        "samples_per_fringe_platform": args.samples_per_platform, "item_count": len(items),
        "thread_count": len(threads),
        "selected_threads": [
            {"group": group, "platform": platform, "thread_id": thread_id, "item_count": count}
            for (group, platform, thread_id), count in sorted(threads.items())
        ],
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenAI prompts on trial and sampled validation threads.")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh"), default="medium")
    parser.add_argument("--samples-per-platform", type=int, default=2)
    parser.add_argument(
        "--all-validation",
        action="store_true",
        help="Annotate every post in every 4chan, Gab, Stormfront, and Vanguard validation thread; exclude trial items.",
    )
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_platform < 1:
        raise SystemExit("--samples-per-platform must be at least 1")
    load_dotenv(PROJECT_ROOT / ".env")
    items = select_all_validation_items() if args.all_validation else select_items(
        samples_per_platform=args.samples_per_platform,
        seed=args.seed,
    )
    if args.limit is not None:
        items = items[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    limit_suffix = f"_limit{args.limit}" if args.limit is not None else ""
    selection_suffix = "_all_validation" if args.all_validation else f"_seed{args.seed}"
    stem = f"annotations_{args.model.replace('/', '-')}_{args.reasoning_effort}_reasoning{selection_suffix}{limit_suffix}"
    results_path = args.output_dir / f"{stem}.jsonl"
    manifest_path = args.output_dir / f"{stem}.manifest.json"
    write_manifest(manifest_path, args, items)
    counts = Counter(normalized_platform(item["platform"]) for item in items)
    print(f"Selected {len(items)} posts: {dict(sorted(counts.items()))}")
    print(f"Manifest: {manifest_path}")
    if args.dry_run:
        return 0
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Put it in .env or export it.", file=sys.stderr)
        return 2

    prompts, client = load_prompts(), OpenAI()
    completed = existing_ids(results_path)
    remaining = [item for item in items if str(item["item_id"]) not in completed]
    print(f"Running {len(remaining)} API calls ({len(completed)} already completed).")
    with results_path.open("a", encoding="utf-8") as output:
        for index, item in enumerate(remaining, start=1):
            platform, started = normalized_platform(item["platform"]), time.monotonic()
            annotation, response = annotate(client, model=args.model, reasoning_effort=args.reasoning_effort, prompt=prompts[platform], content=item_input(item))
            usage = response.usage.model_dump(mode="json") if response.usage else {}
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(), "model": args.model,
                "reasoning_effort": args.reasoning_effort, "selection_group": item["selection_group"], "platform": platform,
                "dataset": item.get("dataset") or item.get("source"), "thread_id": str(item["thread_id"]),
                "item_id": str(item["item_id"]), "post_id": str(item["post_id"]),
                "parent_id": item.get("parent_id"), "is_reply": bool(item.get("is_reply")),
                "target_text": item.get("target_text", ""),
                "target_post_author_text": item.get("target_post_author_text"),
                "target_post_quoted_text": item.get("target_post_quoted_text"),
                "annotation": annotation.model_dump(mode="json"),
                "response_id": response.id, "usage": usage, "latency_seconds": round(time.monotonic() - started, 3),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            print(f"[{index}/{len(remaining)}] {platform} {item['item_id']}")
    print(f"Results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
