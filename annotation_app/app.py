from flask import Flask, request, jsonify, render_template, redirect
import csv
import json
import os
import re
import uuid
from datetime import datetime

#uv run flask --app annotation_app.app run

app = Flask(__name__)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(APP_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "annotation_storage")
ITEMS_FILE = os.path.join(DATA_DIR, "items.json")

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

def load_items():
    if not os.path.exists(ITEMS_FILE):
        return []
    with open(ITEMS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def append_csv_row(row):
    csv_file = os.path.join(
        OUTPUT_DIR,
        f"annotations_{datetime.now().strftime('%Y%m%d')}.csv"
    )
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
    if request.args.get("start") != "1":
        return redirect("/instructions")
    return render_template("index.html")

@app.route("/instructions")
def instructions():
    return render_template("instructions.html", show_back_to_annotation=False)

@app.route("/instructions/back")
def instructions_back():
    return render_template("instructions.html", show_back_to_annotation=True)


@app.route("/items")
def items():
    response = jsonify({"items": load_items()})
    response.cache_control.public = True
    response.cache_control.max_age = 3600
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
            "confirmation_code": confirmation_code
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
            "items_saved": len(annotations)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)