#!/usr/bin/env python3
# Check inter-annotator agreement for annotation export CSV files.

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable


DEFAULT_FIELDS = [
    "identity_hate_speech",
    "hate_severity",
    "hate_target_group",
    "abuse_level",
    "abuse_target",
    "counterspeech",
    "flag_for_review",
]

METADATA_FIELDS = {
    "timestamp",
    "annotator_id",
    "confirmation_code",
    "item_id",
    "thread_id",
    "post_id",
    "parent_id",
    "notes",
}


@dataclass(frozen=True)
class FieldAgreement:
    field: str
    pair_count: int
    compared_labels: int
    item_count: int
    percent_agreement: float | None
    pairwise_cohen_kappa: float | None
    fleiss_kappa: float | None
    unanimous_items: int
    comparable_items: int


def parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min


def normalize_value(value: str | None, missing_as_category: bool) -> str | None:
    cleaned = (value or "").strip()
    if cleaned == "" and not missing_as_category:
        return None
    return cleaned


def load_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows.extend(reader)
    return rows


def latest_rows_by_annotator_item(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = ((row.get("annotator_id") or "").strip(), (row.get("item_id") or "").strip())
        if not all(key):
            continue
        previous = latest.get(key)
        if previous is None or parse_timestamp(row.get("timestamp", "")) >= parse_timestamp(
            previous.get("timestamp", "")
        ):
            latest[key] = row
    return list(latest.values())


def group_by_item(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    by_item: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        item_id = (row.get("item_id") or "").strip()
        annotator_id = (row.get("annotator_id") or "").strip()
        if item_id and annotator_id:
            by_item[item_id][annotator_id] = row
    return dict(by_item)


def complete_annotator_item_matrix(
    by_item: dict[str, dict[str, dict[str, str]]],
    expected_items: int | None = None,
) -> tuple[dict[str, dict[str, dict[str, str]]], list[str], list[str], dict[str, int]]:
    overlap_item_ids = sorted(item_id for item_id, rows in by_item.items() if len(rows) >= 2)
    item_set = set(overlap_item_ids)
    annotator_items: dict[str, set[str]] = defaultdict(set)

    for item_id in overlap_item_ids:
        for annotator_id in by_item[item_id]:
            annotator_items[annotator_id].add(item_id)

    required_count = expected_items or len(overlap_item_ids)
    coverage = {annotator_id: len(annotated_items & item_set) for annotator_id, annotated_items in annotator_items.items()}
    complete_annotators = sorted(
        annotator_id
        for annotator_id, item_count in coverage.items()
        if item_count >= required_count
    )

    if expected_items is not None and complete_annotators:
        common_items = set.intersection(*(annotator_items[annotator_id] for annotator_id in complete_annotators))
        overlap_item_ids = sorted(common_items & item_set)[:expected_items]
        item_set = set(overlap_item_ids)

    filtered = {
        item_id: {
            annotator_id: by_item[item_id][annotator_id]
            for annotator_id in complete_annotators
        }
        for item_id in overlap_item_ids
        if all(annotator_id in by_item[item_id] for annotator_id in complete_annotators)
    }
    return filtered, complete_annotators, overlap_item_ids, coverage


def cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if not left or len(left) != len(right):
        return None

    total = len(left)
    observed = sum(1 for a, b in zip(left, right) if a == b) / total
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = set(left_counts) | set(right_counts)
    expected = sum((left_counts[label] / total) * (right_counts[label] / total) for label in labels)

    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def fleiss_kappa_for_field(
    field: str,
    by_item: dict[str, dict[str, dict[str, str]]],
    missing_as_category: bool,
) -> float | None:
    item_label_counts: list[Counter[str]] = []
    expected_rater_count: int | None = None

    for annotator_rows in by_item.values():
        values = [
            value
            for value in (
                normalize_value(row.get(field), missing_as_category) for row in annotator_rows.values()
            )
            if value is not None
        ]
        if len(values) < 2:
            continue
        if expected_rater_count is None:
            expected_rater_count = len(values)
        elif len(values) != expected_rater_count:
            return None
        item_label_counts.append(Counter(values))

    if not item_label_counts or expected_rater_count is None or expected_rater_count < 2:
        return None

    item_count = len(item_label_counts)
    rater_count = expected_rater_count
    label_totals: Counter[str] = Counter()
    item_agreements: list[float] = []

    for label_counts in item_label_counts:
        label_totals.update(label_counts)
        squared_sum = sum(count * count for count in label_counts.values())
        item_agreements.append((squared_sum - rater_count) / (rater_count * (rater_count - 1)))

    observed = sum(item_agreements) / item_count
    total_ratings = item_count * rater_count
    expected = sum((count / total_ratings) ** 2 for count in label_totals.values())

    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def item_unanimity(
    field: str,
    by_item: dict[str, dict[str, dict[str, str]]],
    missing_as_category: bool,
) -> tuple[int, int]:
    unanimous = 0
    comparable = 0
    for annotator_rows in by_item.values():
        values = [
            value
            for value in (
                normalize_value(row.get(field), missing_as_category) for row in annotator_rows.values()
            )
            if value is not None
        ]
        if len(values) < 2:
            continue
        comparable += 1
        if len(set(values)) == 1:
            unanimous += 1
    return unanimous, comparable


def summarize_field(
    field: str,
    by_item: dict[str, dict[str, dict[str, str]]],
    missing_as_category: bool,
) -> FieldAgreement:
    agreements = 0
    compared = 0
    kappas: list[float] = []
    pair_count = 0
    compared_item_ids: set[str] = set()

    annotators = sorted({annotator for rows in by_item.values() for annotator in rows})
    for left_id, right_id in combinations(annotators, 2):
        left_values: list[str] = []
        right_values: list[str] = []

        for item_id, annotator_rows in by_item.items():
            if left_id not in annotator_rows or right_id not in annotator_rows:
                continue
            left_value = normalize_value(annotator_rows[left_id].get(field), missing_as_category)
            right_value = normalize_value(annotator_rows[right_id].get(field), missing_as_category)
            if left_value is None or right_value is None:
                continue
            left_values.append(left_value)
            right_values.append(right_value)
            compared_item_ids.add(item_id)

        if not left_values:
            continue

        pair_count += 1
        compared += len(left_values)
        agreements += sum(1 for a, b in zip(left_values, right_values) if a == b)
        kappa = cohen_kappa(left_values, right_values)
        if kappa is not None:
            kappas.append(kappa)

    percent = (agreements / compared) if compared else None
    mean_pairwise_cohen = (sum(kappas) / len(kappas)) if kappas else None
    fleiss = fleiss_kappa_for_field(field, by_item, missing_as_category)
    unanimous, comparable = item_unanimity(field, by_item, missing_as_category)
    return FieldAgreement(
        field,
        pair_count,
        compared,
        len(compared_item_ids),
        percent,
        mean_pairwise_cohen,
        fleiss,
        unanimous,
        comparable,
    )


def find_disagreements(
    fields: list[str],
    by_item: dict[str, dict[str, dict[str, str]]],
    missing_as_category: bool,
    limit: int,
) -> list[tuple[str, str, dict[str, str]]]:
    disagreements: list[tuple[str, str, dict[str, str]]] = []
    for item_id in sorted(by_item):
        annotator_rows = by_item[item_id]
        for field in fields:
            values = {
                annotator: value
                for annotator, row in sorted(annotator_rows.items())
                if (value := normalize_value(row.get(field), missing_as_category)) is not None
            }
            if len(values) >= 2 and len(set(values.values())) > 1:
                disagreements.append((item_id, field, values))
                if len(disagreements) >= limit:
                    return disagreements
    return disagreements


def infer_fields(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return DEFAULT_FIELDS
    return [
        field
        for field in rows[0]
        if field not in METADATA_FIELDS and field not in {"confidence"}
    ]


def format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def quality_label(kappa: float | None) -> str:
    if kappa is None:
        return "not enough data"
    if kappa < 0:
        return "worse than chance"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "strong"


def display_field_name(field: str) -> str:
    return field.replace("_", " ")


def print_readable_report(
    csv_files: list[Path],
    raw_row_count: int,
    compared_row_count: int,
    annotators: list[str],
    overlap_items: int,
    missing_as_category: bool,
    complete_mode: bool,
    coverage: dict[str, int],
    summaries: list[FieldAgreement],
    disagreements: list[tuple[str, str, dict[str, str]]],
) -> None:
    print("Inter-annotator agreement report")
    print("=" * 35)
    print(f"Files: {', '.join(str(path) for path in csv_files)}")
    print(f"Annotators: {len(annotators)}")
    for annotator in annotators:
        print(f"  - {annotator}")
    print(f"Raw rows: {raw_row_count}")
    print(f"Rows compared: {compared_row_count}")
    print(f"Items in agreement set: {overlap_items}")
    if complete_mode:
        print("Annotator filter: only annotators who completed every item in the agreement set")
    print(f"Blank cells: {'counted as labels' if missing_as_category else 'skipped'}")
    if coverage:
        print()
        print("Annotator coverage")
        for annotator_id, item_count in sorted(coverage.items(), key=lambda item: (-item[1], item[0])):
            marker = "included" if annotator_id in annotators else "excluded"
            print(f"  - {annotator_id}: {item_count}/{overlap_items} items ({marker})")
    print()

    print("Agreement by field")
    print("-" * 126)
    print(
        f"{'Field':<30} {'Agree':>8} {'Fleiss':>8} {'Pair Cohen':>11} {'Meaning':<18} "
        f"{'Items':>7} {'Pair labels':>11} {'Unanimous':>12}"
    )
    print("-" * 126)
    for summary in summaries:
        unanimous = (
            "n/a"
            if summary.comparable_items == 0
            else f"{summary.unanimous_items}/{summary.comparable_items}"
        )
        print(
            f"{display_field_name(summary.field):<30} "
            f"{format_percent(summary.percent_agreement):>8} "
            f"{format_rate(summary.fleiss_kappa):>8} "
            f"{format_rate(summary.pairwise_cohen_kappa):>11} "
            f"{quality_label(summary.fleiss_kappa):<18} "
            f"{summary.item_count:>7} "
            f"{summary.compared_labels:>11} "
            f"{unanimous:>12}"
        )
    print()
    print("How to read this")
    print("- Agreement is the raw percent of exact label matches.")
    print("- Fleiss is the multi-rater kappa across all included annotators.")
    print("- Pair Cohen is the average Cohen kappa across every annotator pair.")
    print("- Items is the number of items included for that field.")
    print("- Pair labels is the number of pairwise label comparisons.")
    print("- Unanimous means every included annotator agreed on that item.")

    if disagreements:
        print()
        print(f"First {len(disagreements)} disagreements")
        print("-" * 35)
        for index, (item_id, field, values) in enumerate(disagreements, start=1):
            print(f"{index}. {display_field_name(field)}")
            print(f"   item: {item_id}")
            for annotator, value in values.items():
                print(f"   {annotator}: {value}")


def print_csv_report(summaries: list[FieldAgreement]) -> None:
    print("field,pairs,item_count,compared_labels,percent_agreement,fleiss_kappa,mean_pairwise_cohen_kappa,unanimous_items")
    for summary in summaries:
        unanimous_text = (
            "n/a"
            if summary.comparable_items == 0
            else f"{summary.unanimous_items}/{summary.comparable_items}"
        )
        print(
            ",".join(
                [
                    summary.field,
                    str(summary.pair_count),
                    str(summary.item_count),
                    str(summary.compared_labels),
                    format_rate(summary.percent_agreement),
                    format_rate(summary.fleiss_kappa),
                    format_rate(summary.pairwise_cohen_kappa),
                    unanimous_text,
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute percent agreement, Fleiss kappa, and pairwise Cohen kappa for annotation CSV exports. "
            "By default, repeated annotator/item rows are deduplicated to the latest timestamp."
        )
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="One or more annotation CSV files, e.g. annotation_app/annotation_storage/annotations_20260709.csv",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        help="Specific annotation fields to compare. Defaults to the main label fields.",
    )
    parser.add_argument(
        "--all-label-fields",
        action="store_true",
        help="Compare every non-metadata field except confidence, including future label columns.",
    )
    parser.add_argument(
        "--missing-as-category",
        action="store_true",
        help="Treat blank cells as a real category instead of skipping that field comparison. This is forced on unless --include-partial-annotators is used.",
    )
    parser.add_argument(
        "--include-partial-annotators",
        action="store_true",
        help="Use all available annotator pairs instead of limiting to annotators who completed every shared item.",
    )
    parser.add_argument(
        "--expected-items",
        type=int,
        help="Expected number of completed items. Annotators with at least this many shared items are included.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep every row instead of using only the latest row for each annotator/item pair.",
    )
    parser.add_argument(
        "--show-disagreements",
        type=int,
        default=10,
        help="Number of item-level disagreements to print. Use 0 to hide them.",
    )
    parser.add_argument(
        "--csv-output",
        action="store_true",
        help="Print the compact CSV-style results table instead of the readable report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.csv_files)
    if not rows:
        print("No annotation rows found.")
        return 1

    fields = args.fields or (infer_fields(rows) if args.all_label_fields else DEFAULT_FIELDS)
    missing_fields = [field for field in fields if field not in rows[0]]
    if missing_fields:
        print(f"Missing requested fields: {', '.join(missing_fields)}")
        return 1

    analysis_rows = rows if args.keep_duplicates else latest_rows_by_annotator_item(rows)
    by_item = group_by_item(analysis_rows)
    raw_overlap_items = sum(1 for rows_for_item in by_item.values() if len(rows_for_item) >= 2)

    if args.include_partial_annotators:
        analysis_by_item = by_item
        annotators = sorted({annotator for rows_for_item in by_item.values() for annotator in rows_for_item})
        overlap_items = raw_overlap_items
        coverage = {
            annotator: sum(1 for rows_for_item in by_item.values() if annotator in rows_for_item)
            for annotator in annotators
        }
        missing_as_category = args.missing_as_category
    else:
        analysis_by_item, annotators, overlap_item_ids, coverage = complete_annotator_item_matrix(
            by_item,
            expected_items=args.expected_items,
        )
        overlap_items = len(overlap_item_ids)
        missing_as_category = True

    if len(annotators) < 2:
        print("Need at least two annotators who completed every item in the agreement set.")
        return 1

    summaries = [summarize_field(field, analysis_by_item, missing_as_category) for field in fields]
    disagreements = (
        find_disagreements(fields, analysis_by_item, missing_as_category, args.show_disagreements)
        if args.show_disagreements
        else []
    )

    if args.csv_output:
        print_csv_report(summaries)
    else:
        print_readable_report(
            csv_files=args.csv_files,
            raw_row_count=len(rows),
            compared_row_count=len(analysis_rows),
            annotators=annotators,
            overlap_items=overlap_items,
            missing_as_category=missing_as_category,
            complete_mode=not args.include_partial_annotators,
            coverage=coverage,
            summaries=summaries,
            disagreements=disagreements,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
