from flask import Flask, request, jsonify, render_template, redirect, url_for
import csv
import json
import os
import random
import re
import uuid
import threading
from collections import defaultdict
from datetime import datetime, timedelta

#uv run flask --app annotation_app.app run

app = Flask(__name__)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(APP_DIR, "data")
OUTPUT_DIR = os.path.join(APP_DIR, "annotation_storage")
ITEMS_FILE = os.path.join(DATA_DIR, "items.json")
VALIDATION_DIR = os.path.join(PROJECT_ROOT, "data", "validation_set_jsons")
ASSIGNMENT_STATE_FILE = os.path.join(OUTPUT_DIR, "thread_assignments.json")
ASSIGNMENT_LEASE_HOURS = 24
REQUIRED_ANNOTATORS_PER_THREAD = 3

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEX_ID_RE = re.compile(r"^[0-9a-fA-F]{10}$")

CSV_FIELDS = [
    "timestamp",
    "annotator_id",
    "confirmation_code",
    "item_id",
    "thread_id",
    "post_id",
    "parent_id",
    "platform",
    "dataset",
    "assignment_id",
    "identity_hate_speech",
    "hate_severity",
    "hate_target_group",
    "abuse_level",
    "abuse_target",
    "counterspeech",
    "confidence",
    "flag_for_review",
    "notes",
]

MERGED_TARGET_GROUP = "Race / ethnicity / nationality"
OLD_TARGET_GROUPS = {"Race / ethnicity", "Nationality / immigration status"}
ITEM_CACHE = {}
ASSIGNMENT_LOCK = threading.Lock()


def normalize_target_group(value):
    value = value or ""
    return MERGED_TARGET_GROUP if value in OLD_TARGET_GROUPS else value


def normalize_counterspeech(value):
    value = str(value or "").strip()
    if value in {"1", "Counterspeech to original post"}:
        return "1"
    if value in {"0", "Aligned with original post", "Neutral"}:
        return "0"
    if value == "Not applicable":
        return ""
    return value

def is_valid_hex_identifier(value: str) -> bool:
    return bool(HEX_ID_RE.fullmatch(value or ""))

def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_trial_items():
    return load_json(ITEMS_FILE)


def combine_text_and_ocr(text, ocr_text):
    text = (text or "").strip()
    ocr_text = (ocr_text or "").strip()

    if text and ocr_text:
        return f"{text}\n\nOCR text:\n{ocr_text}"
    if ocr_text:
        return f"OCR text:\n{ocr_text}"
    return text


def normalized_image_urls(image_urls):
    urls = []
    seen = set()
    for url in image_urls or []:
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def username_from_post(post):
    if not isinstance(post, dict):
        return ""

    for key in ("username", "author_username", "author_id", "author", "user", "name"):
        value = post.get(key)
        if isinstance(value, dict):
            value = value.get("username") or value.get("name") or value.get("id")
        if value:
            return str(value).lstrip("@")

    url = post.get("url") or post.get("post_url") or ""
    match = re.search(r"https?://(?:www\.)?(?:x\.com|twitter\.com)/([^/?#]+)", str(url), re.IGNORECASE)
    if match and match.group(1).lower() not in {"i", "intent", "share", "home"}:
        return match.group(1).lstrip("@")
    return ""


def make_context_post(post_id, speaker, text, image_urls=None):
    return {
        "post_id": post_id or "",
        "speaker": speaker or "",
        "text": text or "",
        "image_urls": normalized_image_urls(image_urls),
    }


def find_quoted_parent_id(text, valid_ids):
    for quoted_id in re.findall(r">>(\d+)", text or ""):
        if quoted_id in valid_ids:
            return quoted_id
    return ""


def thread_key(platform, thread_id):
    return f"{platform}:{thread_id}"


def base_item(platform, source, thread_id, post_id, parent_id, is_reply, target_text, context, image_urls, url="", date="", username=""):
    key = thread_key(platform, thread_id)
    return {
        "item_id": f"{key}:{post_id}",
        "thread_key": key,
        "thread_id": str(thread_id or ""),
        "post_id": str(post_id or ""),
        "parent_id": str(parent_id or "") if parent_id else None,
        "platform": platform,
        "dataset": source,
        "source": source,
        "is_reply": is_reply,
        "has_prior_hate_in_context": bool(context),
        "context": context,
        "target_text": target_text or "",
        "image_urls": normalized_image_urls(image_urls),
        "url": url or "",
        "date": date or "",
        "username": username or "",
        "lang": "",
    }


def convert_simple_reply_thread(thread, platform, source):
    thread_id = str(thread.get("post_id") or thread.get("id") or thread.get("url") or "")
    original_text = combine_text_and_ocr(thread.get("text", ""), thread.get("ocr_text", ""))
    original_context = make_context_post(
        thread_id,
        username_from_post(thread),
        original_text,
        thread.get("image_urls", []),
    )
    replies = thread.get("replies") or []
    if platform == "gab":
        replies = [
            reply
            for reply in replies
            if str(reply.get("reply_id") or "") != "gab-ad-comment"
        ]
    reply_by_id = {
        str(reply.get("reply_id")): reply
        for reply in replies
        if reply.get("reply_id")
    }
    valid_parent_ids = {thread_id, *reply_by_id.keys()}

    items = [
        base_item(
            platform=platform,
            source=source,
            thread_id=thread_id,
            post_id=thread_id,
            parent_id=None,
            is_reply=False,
            target_text=original_text,
            context=[],
            image_urls=thread.get("image_urls", []),
            url=thread.get("post_url", ""),
            date=thread.get("timestamp_iso") or thread.get("timestamp_raw", ""),
            username=username_from_post(thread),
        )
    ]

    for reply in replies:
        reply_id = str(reply.get("reply_id") or "")
        if not reply_id:
            continue

        reply_text = combine_text_and_ocr(reply.get("reply_text", ""), reply.get("ocr_text", ""))
        parent_id = find_quoted_parent_id(reply.get("reply_text", ""), valid_parent_ids) or thread_id
        context = [original_context]

        if parent_id != thread_id and parent_id in reply_by_id:
            parent = reply_by_id[parent_id]
            context.append(
                make_context_post(
                    parent_id,
                    username_from_post(parent),
                    combine_text_and_ocr(parent.get("reply_text", ""), parent.get("ocr_text", "")),
                    parent.get("image_urls", []),
                )
            )

        items.append(
            base_item(
                platform=platform,
                source=source,
                thread_id=thread_id,
                post_id=reply_id,
                parent_id=parent_id,
                is_reply=True,
                target_text=reply_text,
                context=context,
                image_urls=reply.get("image_urls", []),
                url=thread.get("post_url", ""),
                date=reply.get("reply_timestamp_iso") or reply.get("reply_timestamp_raw", ""),
                username=username_from_post(reply),
            )
        )

    return items


def twitter_media_urls(post):
    media = post.get("media") or {}
    urls = []
    for key in ("photos", "videos", "animated"):
        for entry in media.get(key) or []:
            if isinstance(entry, str):
                urls.append(entry)
            elif isinstance(entry, dict):
                urls.extend([entry.get("url"), entry.get("media_url_https"), entry.get("thumbnailUrl")])
    return normalized_image_urls(urls)


def tweet_text(post):
    return post.get("raw_content") or post.get("rawContent") or post.get("text") or ""


def convert_twitter_thread(thread, source):
    original = thread.get("post") or {}
    thread_id = str(original.get("tweet_id") or original.get("id") or original.get("url") or "")
    original_text = tweet_text(original)
    original_context = make_context_post(thread_id, username_from_post(original), original_text, twitter_media_urls(original))
    items = [
        base_item(
            platform="twitter",
            source=source,
            thread_id=thread_id,
            post_id=thread_id,
            parent_id=None,
            is_reply=False,
            target_text=original_text,
            context=[],
            image_urls=twitter_media_urls(original),
            url=original.get("url", ""),
            date=original.get("date", ""),
            username=username_from_post(original),
        )
    ]

    def walk_replies(replies, context_path, parent_id):
        for reply in replies or []:
            reply_id = str(reply.get("tweet_id") or reply.get("id") or reply.get("url") or "")
            if not reply_id:
                continue
            text = tweet_text(reply)
            items.append(
                base_item(
                    platform="twitter",
                    source=source,
                    thread_id=thread_id,
                    post_id=reply_id,
                    parent_id=parent_id,
                    is_reply=True,
                    target_text=text,
                    context=context_path,
                    image_urls=twitter_media_urls(reply),
                    url=reply.get("url", ""),
                    date=reply.get("date", ""),
                    username=username_from_post(reply),
                )
            )
            child_context = context_path + [make_context_post(reply_id, username_from_post(reply), text, twitter_media_urls(reply))]
            walk_replies(reply.get("nested_replies") or [], child_context, reply_id)

    walk_replies(thread.get("replies") or [], [original_context], thread_id)
    return items


def convert_forum_thread(key, thread, platform, source):
    thread_id = str(thread.get("original_id") or key)
    main_post = thread.get("main_post") or {}
    original_text = combine_text_and_ocr(main_post.get("text", ""), main_post.get("ocr_text", ""))
    if thread.get("title") and thread.get("title") not in original_text:
        original_text = f"{thread.get('title')}\n\n{original_text}".strip()
    original_context = make_context_post(thread_id, username_from_post(main_post), original_text, main_post.get("image_urls", []))
    items = [
        base_item(
            platform=platform,
            source=source,
            thread_id=thread_id,
            post_id=thread_id,
            parent_id=None,
            is_reply=False,
            target_text=original_text,
            context=[],
            image_urls=main_post.get("image_urls", []),
            url=thread.get("url", ""),
            date=thread.get("post_date", ""),
            username=username_from_post(main_post),
        )
    ]

    for index, reply in enumerate(thread.get("scraped_replies") or [], start=1):
        reply_id = str(reply.get("reply_id") or f"reply-{index}")
        reply_text = combine_text_and_ocr(reply.get("text", ""), reply.get("ocr_text", ""))
        items.append(
            base_item(
                platform=platform,
                source=source,
                thread_id=thread_id,
                post_id=reply_id,
                parent_id=thread_id,
                is_reply=True,
                target_text=reply_text,
                context=[original_context],
                image_urls=reply.get("image_urls", []),
                url=thread.get("url", ""),
                date=thread.get("post_date", ""),
                username=username_from_post(reply),
            )
        )

    return items


def platform_from_filename(path):
    name = os.path.basename(path)
    return name.split("_validation_original_threads.json")[0]


def convert_validation_file(path):
    source = os.path.basename(path)
    platform = platform_from_filename(path)
    data = load_json(path)
    threads = []

    if platform in {"4chan", "gab"}:
        for thread in data if isinstance(data, list) else []:
            threads.append(convert_simple_reply_thread(thread, platform, source))
    elif platform == "twitter":
        for thread in data if isinstance(data, list) else []:
            threads.append(convert_twitter_thread(thread, source))
    elif platform in {"stormfront", "vanguard"}:
        for key, thread in (data.items() if isinstance(data, dict) else []):
            threads.append(convert_forum_thread(key, thread, platform, source))

    return [thread_items for thread_items in threads if thread_items]


def load_full_threads():
    cache_key = (
        "full_threads",
        tuple(
            sorted(
                (name, os.path.getmtime(os.path.join(VALIDATION_DIR, name)))
                for name in os.listdir(VALIDATION_DIR)
                if name.endswith("_validation_original_threads.json")
            )
        ) if os.path.exists(VALIDATION_DIR) else (),
    )
    if cache_key in ITEM_CACHE:
        return ITEM_CACHE[cache_key]

    threads = []
    if os.path.exists(VALIDATION_DIR):
        for name in sorted(os.listdir(VALIDATION_DIR)):
            if not name.endswith("_validation_original_threads.json"):
                continue
            threads.extend(convert_validation_file(os.path.join(VALIDATION_DIR, name)))

    ITEM_CACHE.clear()
    ITEM_CACHE[cache_key] = threads
    return threads


def load_full_items():
    items = []
    for thread_items in load_full_threads():
        items.extend(thread_items)
    return items


def load_items(mode="trial"):
    if mode == "trial":
        return load_trial_items()
    if mode == "full":
        return load_full_items()
    return []


def load_assignment_state():
    if not os.path.exists(ASSIGNMENT_STATE_FILE):
        return {"by_user": {}, "by_thread": {}}
    try:
        with open(ASSIGNMENT_STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"by_user": {}, "by_thread": {}}
    if not isinstance(state, dict):
        return {"by_user": {}, "by_thread": {}}

    # Migrate the original {"assignments": {annotator_id: record}} layout on
    # read. The by-thread view is derived from by-user so the two indexes never
    # become independent sources of truth.
    by_user = state.get("by_user", state.get("assignments", {}))
    if not isinstance(by_user, dict):
        by_user = {}
    return assignment_state_with_thread_view(by_user)


def assignment_state_with_thread_view(by_user):
    by_thread = defaultdict(dict)
    for annotator_id, record in by_user.items():
        if not isinstance(record, dict):
            continue
        thread_key_value = record.get("thread_key")
        if not thread_key_value:
            continue
        by_thread[thread_key_value][annotator_id] = {
            "assigned_at": record.get("assigned_at", ""),
            "assignment_id": record.get("assignment_id", ""),
        }
    return {
        "by_user": by_user,
        "by_thread": dict(by_thread),
    }


def save_assignment_state(state):
    state = assignment_state_with_thread_view(state.get("by_user", {}))
    tmp_path = f"{ASSIGNMENT_STATE_FILE}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, ASSIGNMENT_STATE_FILE)


def is_assignment_active(record, now=None):
    now = now or datetime.now()
    try:
        assigned_at = datetime.fromisoformat(record.get("assigned_at", ""))
    except ValueError:
        return False
    return now - assigned_at < timedelta(hours=ASSIGNMENT_LEASE_HOURS)


def annotation_files(extension):
    if not os.path.exists(OUTPUT_DIR):
        return []
    return sorted(
        os.path.join(OUTPUT_DIR, name)
        for name in os.listdir(OUTPUT_DIR)
        if name.startswith("annotations_") and name.endswith(extension)
    )


def read_annotation_rows():
    rows = []
    for path in annotation_files(".csv"):
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                rows.extend(csv.DictReader(f))
        except OSError:
            continue
    return rows


def row_is_complete(row, item):
    if not row.get("identity_hate_speech"):
        return False
    if not row.get("abuse_level"):
        return False
    if item and item.get("is_reply") and not row.get("counterspeech"):
        return False
    if not row.get("confidence"):
        return False
    if row.get("identity_hate_speech") == "1" and (not row.get("hate_severity") or not row.get("hate_target_group")):
        return False
    if row.get("abuse_level") != "none" and not row.get("abuse_target"):
        return False
    return True


def annotation_progress_by_thread(thread_map):
    item_lookup = {item["item_id"]: item for items in thread_map.values() for item in items}
    latest = {}
    for row in read_annotation_rows():
        annotator_id = (row.get("annotator_id") or "").strip()
        item_id = row.get("item_id") or ""
        if not annotator_id or item_id not in item_lookup:
            continue
        key = (annotator_id, item_id)
        if row.get("timestamp", "") >= latest.get(key, {}).get("timestamp", ""):
            latest[key] = row

    by_thread = {
        key: {
            "complete_annotators": set(),
            "partial_annotators": set(),
            "annotator_items": defaultdict(set),
        }
        for key in thread_map
    }

    item_to_thread = {item["item_id"]: key for key, items in thread_map.items() for item in items}
    for (annotator_id, item_id), row in latest.items():
        item = item_lookup[item_id]
        if not row_is_complete(row, item):
            continue
        thread = item_to_thread[item_id]
        by_thread[thread]["annotator_items"][annotator_id].add(item_id)

    for key, info in by_thread.items():
        required_items = {item["item_id"] for item in thread_map[key]}
        for annotator_id, item_ids in info["annotator_items"].items():
            if required_items and required_items.issubset(item_ids):
                info["complete_annotators"].add(annotator_id)
            else:
                info["partial_annotators"].add(annotator_id)

    return by_thread


def annotator_progress_summary(annotator_id):
    """Return saved adaptive-assignment progress for one annotator."""
    threads = load_full_threads()
    thread_map = {items[0]["thread_key"]: items for items in threads if items}
    progress = annotation_progress_by_thread(thread_map)

    posts_annotated = sum(
        len(info["annotator_items"].get(annotator_id, set()))
        for info in progress.values()
    )
    threads_annotated = sum(
        annotator_id in info["complete_annotators"]
        for info in progress.values()
    )
    return {
        "threads_annotated": threads_annotated,
        "posts_annotated": posts_annotated,
    }


def saved_annotations_for_items(annotator_id, items):
    item_ids = {item["item_id"] for item in items}
    latest = {}
    for row in read_annotation_rows():
        if (row.get("annotator_id") or "").strip() != annotator_id:
            continue
        item_id = row.get("item_id") or ""
        if item_id not in item_ids:
            continue
        if row.get("timestamp", "") >= latest.get(item_id, {}).get("timestamp", ""):
            latest[item_id] = row

    saved = {}
    for item_id, row in latest.items():
        saved[item_id] = {
            "item_id": row.get("item_id", ""),
            "thread_id": row.get("thread_id", ""),
            "post_id": row.get("post_id", ""),
            "parent_id": row.get("parent_id", ""),
            "platform": row.get("platform", ""),
            "dataset": row.get("dataset", ""),
            "assignment_id": row.get("assignment_id", ""),
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
    return saved


def assignment_counts(thread_key_value, progress, assignments, current_annotator):
    completed = set(progress[thread_key_value]["complete_annotators"])
    active = set()
    now = datetime.now()
    for annotator_id, record in assignments.items():
        if annotator_id == current_annotator:
            continue
        if record.get("thread_key") == thread_key_value and is_assignment_active(record, now):
            active.add(annotator_id)
    return completed, active


def choose_thread_for_annotator(annotator_id, force_new=False):
    with ASSIGNMENT_LOCK:
        threads = load_full_threads()
        thread_map = {items[0]["thread_key"]: items for items in threads}
        progress = annotation_progress_by_thread(thread_map)
        state = load_assignment_state()
        assignments = state.setdefault("by_user", {})
        existing = assignments.get(annotator_id)

        if existing and not force_new:
            key = existing.get("thread_key")
            if key in thread_map:
                completed, active = assignment_counts(key, progress, assignments, annotator_id)
                if annotator_id not in progress[key]["complete_annotators"] and len(completed | active) < REQUIRED_ANNOTATORS_PER_THREAD:
                    existing["assigned_at"] = datetime.now().isoformat()
                    save_assignment_state(state)
                    return key, thread_map[key], "resumed"

        candidates = []
        for key, items in thread_map.items():
            info = progress[key]
            if annotator_id in info["complete_annotators"]:
                continue
            if annotator_id in info["partial_annotators"]:
                candidates.append((-1, key, items, "resumed_partial"))
                continue
            completed, active = assignment_counts(key, progress, assignments, annotator_id)
            occupied = completed | active
            if annotator_id in occupied or len(occupied) >= REQUIRED_ANNOTATORS_PER_THREAD:
                continue
            completed_count = len(completed)
            priority = {2: 0, 1: 1, 0: 2}.get(completed_count, 4)
            candidates.append((priority, key, items, "assigned"))

        if not candidates:
            assignments.pop(annotator_id, None)
            save_assignment_state(state)
            return None, [], "empty"

        best_priority = min(candidate[0] for candidate in candidates)
        priority_candidates = [candidate for candidate in candidates if candidate[0] == best_priority]

        # Always restore partial work first. For new assignments, keep platform
        # coverage even before randomly choosing a thread within that platform.
        if best_priority == -1:
            _, key, items, status = random.choice(priority_candidates)
        else:
            platform_load = defaultdict(int)
            for thread_key_value, info in progress.items():
                platform = thread_map[thread_key_value][0]["platform"]
                platform_load[platform] += len(info["complete_annotators"])
            now = datetime.now()
            for assigned_annotator, record in assignments.items():
                if assigned_annotator == annotator_id or not is_assignment_active(record, now):
                    continue
                assigned_key = record.get("thread_key")
                if assigned_key in thread_map:
                    platform_load[thread_map[assigned_key][0]["platform"]] += 1

            eligible_platforms = {
                candidate[2][0]["platform"]
                for candidate in priority_candidates
            }
            minimum_load = min(platform_load[platform] for platform in eligible_platforms)
            least_used_platforms = {
                platform for platform in eligible_platforms
                if platform_load[platform] == minimum_load
            }
            balanced_candidates = [
                candidate for candidate in priority_candidates
                if candidate[2][0]["platform"] in least_used_platforms
            ]
            _, key, items, status = random.choice(balanced_candidates)
        assignments[annotator_id] = {
            "thread_key": key,
            "assigned_at": datetime.now().isoformat(),
            "assignment_id": str(uuid.uuid4()),
        }
        save_assignment_state(state)
        return key, items, status


def current_assignment_id(annotator_id, thread_key_value):
    record = load_assignment_state().get("by_user", {}).get(annotator_id, {})
    if record.get("thread_key") == thread_key_value:
        return record.get("assignment_id", "")
    return ""


def csv_header_matches(path):
    if not os.path.exists(path):
        return True
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except OSError:
        return False
    return header == CSV_FIELDS


def append_csv_row(row):
    date_suffix = datetime.now().strftime('%Y%m%d')
    csv_file = os.path.join(OUTPUT_DIR, f"annotations_{date_suffix}.csv")
    if os.path.exists(csv_file) and not csv_header_matches(csv_file):
        csv_file = os.path.join(OUTPUT_DIR, f"annotations_{date_suffix}_dynamic.csv")

    file_exists = os.path.exists(csv_file)

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def append_jsonl(annotation_record):
    jsonl_file = os.path.join(
        OUTPUT_DIR,
        f"annotations_{datetime.now().strftime('%Y%m%d')}.jsonl"
    )
    with open(jsonl_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(annotation_record, ensure_ascii=False) + "\n")

def normalize_annotation(annotator_id, confirmation_code, timestamp, annotation):
    return {
        "timestamp": timestamp,
        "annotator_id": annotator_id,
        "confirmation_code": confirmation_code,
        "item_id": annotation.get("item_id", ""),
        "thread_id": annotation.get("thread_id", ""),
        "post_id": annotation.get("post_id", ""),
        "parent_id": annotation.get("parent_id", ""),
        "platform": annotation.get("platform", ""),
        "dataset": annotation.get("dataset", annotation.get("source", "")),
        "assignment_id": annotation.get("assignment_id", ""),
        "identity_hate_speech": annotation.get("identity_hate_speech", ""),
        "hate_severity": annotation.get("hate_severity", ""),
        "hate_target_group": normalize_target_group(annotation.get("hate_target_group", "")),
        "abuse_level": annotation.get("abuse_level", ""),
        "abuse_target": annotation.get("abuse_target", ""),
        "counterspeech": normalize_counterspeech(annotation.get("counterspeech", "")),
        "confidence": annotation.get("confidence", ""),
        "flag_for_review": annotation.get("flag_for_review", False),
        "notes": annotation.get("notes", ""),
    }

@app.route("/")
def index():
    return redirect("/instructions")


@app.route("/instructions")
def instructions():
    return render_template("instructions.html", show_back_to_annotation=False)


@app.route("/instructions/back")
def instructions_back():
    return render_template("instructions.html", show_back_to_annotation=True)


@app.route("/platform-info")
def platform_info():
    return render_template("platform_info.html", show_back_to_annotation=False)


@app.route("/platform-info/back")
def platform_info_back():
    return render_template("platform_info.html", show_back_to_annotation=True)


@app.route("/choose")
def choose_task():
    return render_template("choose.html")


@app.route("/annotate/<mode>")
def annotate(mode):
    if mode not in {"trial", "full"}:
        return redirect(url_for("choose_task"))

    task_config = {
        "trial": {
            "mode": "trial",
            "title": "Trial Annotation",
            "description": "Label the trial tweet conversations one post at a time.",
            "thread_label": "Tweet",
            "post_label": "Post in selected tweet",
            "context_label": "Tweet Context",
            "context_more": "Show full tweet context",
            "context_less": "Show less tweet context",
            "no_context": "No earlier tweet context provided. This item is the original post.",
            "thread_option": "Tweet",
            "completion_note": "",
            "submit_label": "Submit all annotations",
        },
        "full": {
            "mode": "full",
            "title": "Full Annotation",
            "description": "You will receive one assigned validation thread at a time. Complete every post in the thread, then request the next thread.",
            "thread_label": "Assigned thread",
            "post_label": "Post in selected thread",
            "context_label": "Thread Context",
            "context_more": "Show full thread context",
            "context_less": "Show less thread context",
            "no_context": "No earlier thread context provided. This item is the original post.",
            "thread_option": "Thread",
            "completion_note": "Complete this assigned thread before requesting the next one. You may pause and resume; saved progress will be restored, but you must complete every post in the assigned thread to be paid for it.",
            "submit_label": "Submit completed thread",
        },
    }[mode]

    return render_template("index.html", task=task_config)


@app.route("/items")
def items():
    response = jsonify({"items": load_items("trial")})
    response.cache_control.no_store = True
    response.cache_control.max_age = 0
    return response


@app.route("/items/<mode>")
def items_for_mode(mode):
    if mode not in {"trial", "full"}:
        return jsonify({"error": "Unknown annotation mode"}), 404

    if mode == "full":
        annotator_id = (request.args.get("annotator_id") or "").strip()
        if not is_valid_hex_identifier(annotator_id):
            return jsonify({"error": "annotator_id must be exactly 10 hex characters"}), 400
        thread_key_value, items, status = choose_thread_for_annotator(
            annotator_id,
            force_new=request.args.get("new") == "1",
        )
        assignment_id = current_assignment_id(annotator_id, thread_key_value) if thread_key_value else ""
        for item in items:
            item["assignment_id"] = assignment_id
        payload = {
            "items": items,
            "assignment": {
                "thread_key": thread_key_value,
                "status": status,
                "assignment_id": assignment_id,
                "required_annotators": REQUIRED_ANNOTATORS_PER_THREAD,
            },
            "saved_annotations": saved_annotations_for_items(annotator_id, items),
            "annotator_progress": annotator_progress_summary(annotator_id),
        }
    else:
        payload = {
            "items": load_items(mode),
            "assignment": None,
            "saved_annotations": {},
            "annotator_progress": None,
        }

    response = jsonify(payload)
    response.cache_control.no_store = True
    response.cache_control.max_age = 0
    return response

@app.route("/save_item", methods=["POST"])
def save_item():
    try:
        payload = request.get_json()
        annotator_id = (payload.get("annotator_id") or "").strip()
        annotation = payload.get("annotation")

        if not annotator_id or not annotation:
            return jsonify({"error": "Missing annotator_id or annotation"}), 400
        if not is_valid_hex_identifier(annotator_id):
            return jsonify({"error": "annotator_id must be exactly 10 hex characters"}), 400

        timestamp = datetime.now().isoformat()
        confirmation_code = str(uuid.uuid4())

        row = normalize_annotation(
            annotator_id=annotator_id,
            confirmation_code=confirmation_code,
            timestamp=timestamp,
            annotation=annotation,
        )

        append_csv_row(row)
        append_jsonl(row)

        return jsonify({
            "status": "success",
            "confirmation_code": confirmation_code,
            "annotator_progress": annotator_progress_summary(annotator_id),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/submit_all", methods=["POST"])
def submit_all():
    try:
        payload = request.get_json()
        annotator_id = (payload.get("annotator_id") or "").strip()
        annotations = payload.get("annotations", [])

        if not annotator_id or not annotations:
            return jsonify({"error": "Missing annotator_id or annotations"}), 400
        if not is_valid_hex_identifier(annotator_id):
            return jsonify({"error": "annotator_id must be exactly 10 hex characters"}), 400

        timestamp = datetime.now().isoformat()
        confirmation_code = str(uuid.uuid4())

        for annotation in annotations:
            row = normalize_annotation(
                annotator_id=annotator_id,
                confirmation_code=confirmation_code,
                timestamp=timestamp,
                annotation=annotation,
            )
            append_csv_row(row)
            append_jsonl(row)

        return jsonify({
            "status": "success",
            "confirmation_code": confirmation_code,
            "items_saved": len(annotations),
            "annotator_progress": annotator_progress_summary(annotator_id),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
