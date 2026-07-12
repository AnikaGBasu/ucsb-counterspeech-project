import json
import sys

input_path, output_path = sys.argv[1:3]

with open(input_path, encoding="utf-8") as source:
    rows = [json.loads(line) for line in source if line.strip()]

readable = []
for row in rows:
    annotation = row.get("annotation") or {}
    readable.append({
        "post_id": row.get("post_id"),
        "text": row.get("target_text"),
        "labels": {key: value for key, value in annotation.items() if key != "notes"},
        "reasoning": annotation.get("notes", ""),
    })

with open(output_path, "w", encoding="utf-8") as destination:
    json.dump(readable, destination, ensure_ascii=False, indent=2)
    destination.write("\n")
