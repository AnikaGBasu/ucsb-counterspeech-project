"""Compare LLM annotations with valid human annotations on shared items."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
from pathlib import Path

from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from annotation_app.app import load_full_threads, load_trial_items, row_is_complete
from scripts.check_inter_annotator_agreement import cohen_kappa, fleiss_kappa_for_field


FIELDS = [
    "identity_hate_speech",
    "hate_severity",
    "hate_target_group",
    "abuse_level",
    "abuse_target",
    "counterspeech",
    "flag_for_review",
]
ORDINAL_LABELS = {
    "hate_severity": ["low", "medium", "high"],
    "abuse_level": ["none", "mild", "moderate", "severe"],
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMAN_DIR = PROJECT_ROOT / "annotation_app" / "annotation_storage"
DEFAULT_LLM_FILE = (
    PROJECT_ROOT
    / "llm_annotation_pipeline"
    / "results"
    / "rerun_2026-07-12"
    / "annotations_gpt-5.4-mini_medium_reasoning_seed20260710.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_HUMAN_DIR / "llm_human_agreement.json"


def normalized(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    cleaned = str(value).strip()
    if cleaned.lower() in {"true", "false"}:
        return cleaned.lower()
    return cleaned


def parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min


def item_lookup() -> dict[str, dict]:
    items = list(load_trial_items())
    items.extend(item for thread in load_full_threads() for item in thread)
    return {item["item_id"]: item for item in items}


def load_latest_valid_human_rows(human_dir: Path, items: dict[str, dict]) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    for path in sorted(human_dir.glob("annotations_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                annotator_id = (row.get("annotator_id") or "").strip()
                item_id = (row.get("item_id") or "").strip()
                if not annotator_id or item_id not in items:
                    continue
                key = (annotator_id, item_id)
                if parse_timestamp(row.get("timestamp", "")) >= parse_timestamp(
                    latest.get(key, {}).get("timestamp", "")
                ):
                    latest[key] = row
    return [row for row in latest.values() if row_is_complete(row, items[row["item_id"]])]


def load_llm_rows(path: Path) -> tuple[dict[str, dict], dict]:
    rows: dict[str, dict] = {}
    metadata: dict = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rows[record["item_id"]] = record["annotation"]
            if not metadata:
                metadata = {
                    "model": record.get("model"),
                    "reasoning_effort": record.get("reasoning_effort"),
                }
    return rows, metadata


def metric(left: list[str], right: list[str]) -> dict:
    agreements = sum(a == b for a, b in zip(left, right))
    return {
        "comparisons": len(left),
        "agreements": agreements,
        "percent_agreement": agreements / len(left) if left else None,
        "cohen_kappa": cohen_kappa(left, right),
    }


def majority_label(values: list[str]) -> str | None:
    counts = Counter(values)
    if not counts:
        return None
    most_common = counts.most_common()
    return most_common[0][0] if len(most_common) == 1 or most_common[0][1] > most_common[1][1] else None


def weighted_cohen(
    left: list[str],
    right: list[str],
    labels: list[str],
    weights: str,
) -> float | None:
    if not left or len(left) != len(right):
        return None
    score = cohen_kappa_score(left, right, labels=labels, weights=weights)
    return None if score != score else float(score)


def gwet_agreement(
    left: list[str],
    right: list[str],
    labels: list[str],
    weight_type: str | None = None,
) -> float | None:
    if not left or len(left) != len(right) or len(labels) < 2:
        return None
    positions = {label: index for index, label in enumerate(labels)}
    q = len(labels)
    if any(value not in positions for value in left + right):
        return None

    def weight(a: str, b: str) -> float:
        distance = abs(positions[a] - positions[b])
        if weight_type == "linear":
            return 1 - distance / (q - 1)
        if weight_type == "quadratic":
            return 1 - (distance / (q - 1)) ** 2
        return 1.0 if a == b else 0.0

    observed = sum(weight(a, b) for a, b in zip(left, right)) / len(left)
    pooled = Counter(left + right)
    total = 2 * len(left)
    marginal_term = sum(
        (pooled[label] / total) * (1 - pooled[label] / total)
        for label in labels
    )
    weight_sum = sum(weight(a, b) for a in labels for b in labels)
    expected = weight_sum * marginal_term / (q * (q - 1))
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def agreement_weight(left: str, right: str, labels: list[str], weight_type: str | None) -> float:
    if weight_type is None:
        return 1.0 if left == right else 0.0
    positions = {label: index for index, label in enumerate(labels)}
    distance = abs(positions[left] - positions[right])
    if weight_type == "linear":
        return 1 - distance / (len(labels) - 1)
    if weight_type == "quadratic":
        return 1 - (distance / (len(labels) - 1)) ** 2
    raise ValueError(f"Unsupported weight type: {weight_type}")


def gwet_multi_rater(
    rating_matrix: list[list[str]],
    labels: list[str],
    weight_type: str | None = None,
) -> float | None:
    if not rating_matrix or len(labels) < 2:
        return None
    rater_count = len(rating_matrix[0])
    if rater_count < 2 or any(len(row) != rater_count for row in rating_matrix):
        return None
    if any(value not in labels for row in rating_matrix for value in row):
        return None

    observed_by_item = []
    for row in rating_matrix:
        pair_weights = [
            agreement_weight(row[left], row[right], labels, weight_type)
            for left, right in combinations(range(rater_count), 2)
        ]
        observed_by_item.append(sum(pair_weights) / len(pair_weights))
    observed = sum(observed_by_item) / len(observed_by_item)

    pooled = Counter(value for row in rating_matrix for value in row)
    total = len(rating_matrix) * rater_count
    marginal_term = sum(
        (pooled[label] / total) * (1 - pooled[label] / total)
        for label in labels
    )
    weight_sum = sum(
        agreement_weight(left, right, labels, weight_type)
        for left in labels
        for right in labels
    )
    expected = weight_sum * marginal_term / (len(labels) * (len(labels) - 1))
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def classification_summary(gold: list[str], predicted: list[str]) -> dict:
    labels = sorted(set(gold) | set(predicted))
    if not gold:
        return {
            "comparisons": 0,
            "labels": labels,
            "percent_agreement": None,
            "cohen_kappa": None,
            "gwet_ac1": None,
            "confusion_matrix": [],
            "confusion_matrix_axes": "rows=human_gold, columns=llm_prediction",
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "per_class": {},
        }
    precision, recall, f1, support = precision_recall_fscore_support(
        gold,
        predicted,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        gold,
        predicted,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "comparisons": len(gold),
        "labels": labels,
        "percent_agreement": sum(a == b for a, b in zip(gold, predicted)) / len(gold),
        "cohen_kappa": cohen_kappa(gold, predicted),
        "gwet_ac1": gwet_agreement(gold, predicted, labels),
        "confusion_matrix": confusion_matrix(gold, predicted, labels=labels).tolist(),
        "confusion_matrix_axes": "rows=human_gold, columns=llm_prediction",
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }


def percentile_interval(values: list[float], confidence: float = 0.95) -> list[float] | None:
    clean = sorted(value for value in values if value is not None and value == value)
    if not clean:
        return None
    alpha = (1 - confidence) / 2
    low_index = round(alpha * (len(clean) - 1))
    high_index = round((1 - alpha) * (len(clean) - 1))
    return [clean[low_index], clean[high_index]]


def bootstrap_pair_metrics(
    gold: list[str],
    predicted: list[str],
    labels: list[str],
    iterations: int,
    seed: int,
    ordinal: bool = False,
) -> dict:
    if not gold:
        return {}
    rng = random.Random(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        indices = [rng.randrange(len(gold)) for _ in range(len(gold))]
        sampled_gold = [gold[index] for index in indices]
        sampled_predicted = [predicted[index] for index in indices]
        values["percent_agreement"].append(
            sum(a == b for a, b in zip(sampled_gold, sampled_predicted)) / len(indices)
        )
        values["cohen_kappa"].append(cohen_kappa(sampled_gold, sampled_predicted))
        values["gwet_ac1"].append(
            gwet_agreement(sampled_gold, sampled_predicted, labels)
        )
        macro_precision, macro_recall, macro_f1, _ = (
            precision_recall_fscore_support(
                sampled_gold,
                sampled_predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
        values["macro_precision"].append(float(macro_precision))
        values["macro_recall"].append(float(macro_recall))
        values["macro_f1"].append(float(macro_f1))
        if ordinal:
            values["linear_weighted_cohen_kappa"].append(
                weighted_cohen(sampled_gold, sampled_predicted, labels, "linear")
            )
            values["quadratic_weighted_cohen_kappa"].append(
                weighted_cohen(sampled_gold, sampled_predicted, labels, "quadratic")
            )
            values["gwet_ac2_linear"].append(
                gwet_agreement(sampled_gold, sampled_predicted, labels, "linear")
            )
            values["gwet_ac2_quadratic"].append(
                gwet_agreement(sampled_gold, sampled_predicted, labels, "quadratic")
            )
    return {
        metric_name: {
            "ci_95_percentile": percentile_interval(metric_values),
            "defined_replicates": sum(
                value is not None and value == value for value in metric_values
            ),
        }
        for metric_name, metric_values in values.items()
    }


def build_report(human_dir: Path, llm_file: Path) -> dict:
    items = item_lookup()
    humans = load_latest_valid_human_rows(human_dir, items)
    llm, llm_metadata = load_llm_rows(llm_file)
    shared_rows = [row for row in humans if row["item_id"] in llm]
    shared_item_ids = sorted({row["item_id"] for row in shared_rows})

    by_item: dict[str, list[dict]] = defaultdict(list)
    by_annotator: dict[str, list[dict]] = defaultdict(list)
    for row in shared_rows:
        by_item[row["item_id"]].append(row)
        by_annotator[row["annotator_id"]].append(row)

    shared_item_set = set(shared_item_ids)
    complete_annotators = sorted(
        annotator_id
        for annotator_id, rows in by_annotator.items()
        if {row["item_id"] for row in rows} == shared_item_set
    )
    human_row_lookup = {
        (row["annotator_id"], row["item_id"]): row for row in shared_rows
    }

    fields = {}
    for field in FIELDS:
        human_values = [normalized(row.get(field)) for row in shared_rows]
        llm_values = [normalized(llm[row["item_id"]].get(field)) for row in shared_rows]

        majority_human = []
        majority_llm = []
        tied_items = 0
        for item_id in shared_item_ids:
            vote = majority_label([normalized(row.get(field)) for row in by_item[item_id]])
            if vote is None:
                tied_items += 1
                continue
            majority_human.append(vote)
            majority_llm.append(normalized(llm[item_id].get(field)))

        pairwise_results = {
            annotator_id: metric(
                [normalized(llm[row["item_id"]].get(field)) for row in rows],
                [normalized(row.get(field)) for row in rows],
            )
            for annotator_id, rows in sorted(by_annotator.items())
        }
        defined_pairwise_kappas = [
            result["cohen_kappa"]
            for result in pairwise_results.values()
            if result["cohen_kappa"] is not None
        ]
        complete_pairwise_kappas = [
            pairwise_results[annotator_id]["cohen_kappa"]
            for annotator_id in complete_annotators
            if pairwise_results[annotator_id]["cohen_kappa"] is not None
        ]
        fleiss_rows = {
            item_id: {
                "llm": {field: normalized(llm[item_id].get(field))},
                **{
                    annotator_id: {
                        field: normalized(
                            human_row_lookup[(annotator_id, item_id)].get(field)
                        )
                    }
                    for annotator_id in complete_annotators
                },
            }
            for item_id in shared_item_ids
        }

        fields[field] = {
            "fleiss_kappa_llm_plus_complete_humans": {
                "items": len(shared_item_ids),
                "raters": 1 + len(complete_annotators),
                "human_annotators": complete_annotators,
                "fleiss_kappa": fleiss_kappa_for_field(field, fleiss_rows, True),
            },
            "pairwise_cohen_kappa_llm_vs_humans": {
                "human_pairs": len(pairwise_results),
                "pairs_with_defined_kappa": len(defined_pairwise_kappas),
                "mean_cohen_kappa_all_humans": (
                    sum(defined_pairwise_kappas) / len(defined_pairwise_kappas)
                    if defined_pairwise_kappas
                    else None
                ),
                "mean_cohen_kappa_complete_humans": (
                    sum(complete_pairwise_kappas) / len(complete_pairwise_kappas)
                    if complete_pairwise_kappas
                    else None
                ),
                "by_human_annotator": pairwise_results,
            },
            "llm_vs_each_human_rating": metric(llm_values, human_values),
            "llm_vs_human_majority": {
                **metric(majority_llm, majority_human),
                "tied_items_excluded": tied_items,
            },
        }

    majority_by_field = {
        field: {
            item_id: majority_label(
                [normalized(row.get(field)) for row in by_item[item_id]]
            )
            for item_id in shared_item_ids
        }
        for field in FIELDS
    }

    severity_labels = ORDINAL_LABELS["hate_severity"]
    severity_pairs = [
        (
            majority_by_field["hate_severity"][item_id],
            normalized(llm[item_id].get("hate_severity")),
        )
        for item_id in shared_item_ids
        if majority_by_field["hate_severity"][item_id] in severity_labels
        and normalized(llm[item_id].get("hate_severity")) in severity_labels
    ]
    conditional_item_ids = [
        item_id
        for item_id in shared_item_ids
        if majority_by_field["identity_hate_speech"][item_id] == "1"
    ]
    conditional_valid_pairs = [
        (
            majority_by_field["hate_severity"][item_id],
            normalized(llm[item_id].get("hate_severity")),
        )
        for item_id in conditional_item_ids
        if majority_by_field["hate_severity"][item_id] in severity_labels
        and normalized(llm[item_id].get("hate_severity")) in severity_labels
    ]
    conditional_invalid_items = [
        item_id
        for item_id in conditional_item_ids
        if majority_by_field["hate_severity"][item_id] not in severity_labels
        or normalized(llm[item_id].get("hate_severity")) not in severity_labels
    ]

    def weighted_summary(pairs: list[tuple[str, str]]) -> dict:
        gold = [left for left, _ in pairs]
        predicted = [right for _, right in pairs]
        return {
            "comparisons": len(pairs),
            "labels": severity_labels,
            "exact_agreements": sum(left == right for left, right in pairs),
            "percent_exact_agreement": (
                sum(left == right for left, right in pairs) / len(pairs)
                if pairs
                else None
            ),
            "human_label_counts": dict(sorted(Counter(gold).items())),
            "llm_label_counts": dict(sorted(Counter(predicted).items())),
            "linear_weighted_cohen_kappa": weighted_cohen(
                gold, predicted, severity_labels, "linear"
            ),
            "quadratic_weighted_cohen_kappa": weighted_cohen(
                gold, predicted, severity_labels, "quadratic"
            ),
            "gwet_ac2_linear": gwet_agreement(
                gold, predicted, severity_labels, "linear"
            ),
            "gwet_ac2_quadratic": gwet_agreement(
                gold, predicted, severity_labels, "quadratic"
            ),
        }

    weighted_and_conditional = {
        "hate_severity_on_valid_majority_pairs": weighted_summary(severity_pairs),
        "hate_severity_conditional_on_human_majority_identity_hate": {
            **weighted_summary(conditional_valid_pairs),
            "human_majority_identity_hate_items": len(conditional_item_ids),
            "items_excluded_for_tied_or_invalid_severity": len(conditional_invalid_items),
            "excluded_item_ids": conditional_invalid_items,
        },
    }

    conditional_human_rows = [
        row for row in shared_rows if normalized(row.get("identity_hate_speech")) == "1"
    ]
    conditional_human_valid_rows = [
        row
        for row in conditional_human_rows
        if normalized(row.get("hate_severity")) in severity_labels
        and normalized(llm[row["item_id"]].get("hate_severity")) in severity_labels
    ]
    conditional_human_pairs = [
        (
            normalized(row.get("hate_severity")),
            normalized(llm[row["item_id"]].get("hate_severity")),
        )
        for row in conditional_human_valid_rows
    ]
    conditional_by_annotator = {}
    for annotator_id in sorted(
        {row["annotator_id"] for row in conditional_human_rows}
    ):
        annotator_rows = [
            row
            for row in conditional_human_valid_rows
            if row["annotator_id"] == annotator_id
        ]
        annotator_pairs = [
            (
                normalized(row.get("hate_severity")),
                normalized(llm[row["item_id"]].get("hate_severity")),
            )
            for row in annotator_rows
        ]
        conditional_by_annotator[annotator_id] = weighted_summary(annotator_pairs)

    weighted_and_conditional[
        "hate_severity_conditional_on_each_human_identity_hate_rating"
    ] = {
        **weighted_summary(conditional_human_pairs),
        "human_identity_hate_rating_pairs": len(conditional_human_rows),
        "rating_pairs_excluded_for_invalid_severity": (
            len(conditional_human_rows) - len(conditional_human_valid_rows)
        ),
        "by_human_annotator": conditional_by_annotator,
    }

    gwet_by_field = {}
    for field in FIELDS:
        comparison_item_ids = [
            item_id
            for item_id in shared_item_ids
            if majority_by_field[field][item_id] is not None
        ]
        gold = [majority_by_field[field][item_id] for item_id in comparison_item_ids]
        predicted = [normalized(llm[item_id].get(field)) for item_id in comparison_item_ids]
        labels = sorted(set(gold) | set(predicted))
        gwet_by_field[field] = {
            "comparisons": len(comparison_item_ids),
            "labels": labels,
            "gwet_ac1": gwet_agreement(gold, predicted, labels),
            "human_majority_ties_excluded": len(shared_item_ids) - len(comparison_item_ids),
        }

    annotator_results = {}
    for annotator_id, rows in sorted(by_annotator.items()):
        annotator_results[annotator_id] = {
            "shared_items": len(rows),
            "fields": {
                field: metric(
                    [normalized(llm[row["item_id"]].get(field)) for row in rows],
                    [normalized(row.get(field)) for row in rows],
                )
                for field in FIELDS
            },
        }

    platform_counts = Counter(
        items[item_id].get("platform", "unknown") for item_id in shared_item_ids
    )
    bootstrap_iterations = 1000
    bootstrap_seed = 20260712

    sample_and_distributions = {
        "shared_items": len(shared_item_ids),
        "shared_human_rating_rows": len(shared_rows),
        "human_annotators": len(by_annotator),
        "complete_human_annotators": complete_annotators,
        "items_by_platform": dict(sorted(platform_counts.items())),
        "items_by_reply_status": dict(
            sorted(
                Counter(
                    "reply" if items[item_id].get("is_reply") else "original_post"
                    for item_id in shared_item_ids
                ).items()
            )
        ),
        "fields": {},
    }
    for field in FIELDS:
        majority_values = [
            majority_by_field[field][item_id]
            for item_id in shared_item_ids
            if majority_by_field[field][item_id] is not None
        ]
        sample_and_distributions["fields"][field] = {
            "human_rating_counts": dict(
                sorted(Counter(normalized(row.get(field)) for row in shared_rows).items())
            ),
            "human_majority_counts": dict(sorted(Counter(majority_values).items())),
            "human_majority_ties": sum(
                majority_by_field[field][item_id] is None
                for item_id in shared_item_ids
            ),
            "llm_counts": dict(
                sorted(
                    Counter(
                        normalized(llm[item_id].get(field))
                        for item_id in shared_item_ids
                    ).items()
                )
            ),
        }

    human_only = {}
    human_only_bootstrap = {}
    for field_index, field in enumerate(FIELDS):
        matrix = [
            [
                normalized(human_row_lookup[(annotator_id, item_id)].get(field))
                for annotator_id in complete_annotators
            ]
            for item_id in shared_item_ids
        ]
        labels = sorted({value for row in matrix for value in row})
        human_fleiss_rows = {
            item_id: {
                annotator_id: {
                    field: normalized(
                        human_row_lookup[(annotator_id, item_id)].get(field)
                    )
                }
                for annotator_id in complete_annotators
            }
            for item_id in shared_item_ids
        }
        pairwise_values = []
        pairwise_details = {}
        for left_id, right_id in combinations(complete_annotators, 2):
            left = [
                normalized(human_row_lookup[(left_id, item_id)].get(field))
                for item_id in shared_item_ids
            ]
            right = [
                normalized(human_row_lookup[(right_id, item_id)].get(field))
                for item_id in shared_item_ids
            ]
            result = metric(left, right)
            pairwise_details[f"{left_id}__{right_id}"] = result
            if result["cohen_kappa"] is not None:
                pairwise_values.append(result["cohen_kappa"])

        ordinal_labels = ORDINAL_LABELS.get(field)
        ordinal_matrix = (
            [
                row
                for row in matrix
                if all(value in ordinal_labels for value in row)
            ]
            if ordinal_labels
            else []
        )
        human_only[field] = {
            "items": len(shared_item_ids),
            "raters": len(complete_annotators),
            "annotators": complete_annotators,
            "labels": labels,
            "fleiss_kappa": fleiss_kappa_for_field(
                field, human_fleiss_rows, True
            ),
            "mean_pairwise_human_human_cohen_kappa": (
                sum(pairwise_values) / len(pairwise_values)
                if pairwise_values
                else None
            ),
            "human_pairs": len(pairwise_details),
            "pairwise_details": pairwise_details,
            "gwet_ac1": gwet_multi_rater(matrix, labels),
            "gwet_ac2_linear": (
                gwet_multi_rater(ordinal_matrix, ordinal_labels, "linear")
                if ordinal_matrix
                else None
            ),
            "gwet_ac2_quadratic": (
                gwet_multi_rater(ordinal_matrix, ordinal_labels, "quadratic")
                if ordinal_matrix
                else None
            ),
            "ordinal_items_with_all_human_ratings_applicable": len(ordinal_matrix),
        }

        rng = random.Random(bootstrap_seed + field_index)
        bootstrap_values = defaultdict(list)
        for _ in range(bootstrap_iterations):
            indices = [
                rng.randrange(len(shared_item_ids))
                for _ in range(len(shared_item_ids))
            ]
            sampled_matrix = [matrix[index] for index in indices]
            bootstrap_values["gwet_ac1"].append(
                gwet_multi_rater(sampled_matrix, labels)
            )
            if ordinal_labels:
                sampled_ordinal_matrix = [
                    row
                    for row in sampled_matrix
                    if all(value in ordinal_labels for value in row)
                ]
                bootstrap_values["gwet_ac2_linear"].append(
                    gwet_multi_rater(
                        sampled_ordinal_matrix, ordinal_labels, "linear"
                    )
                    if sampled_ordinal_matrix
                    else None
                )
                bootstrap_values["gwet_ac2_quadratic"].append(
                    gwet_multi_rater(
                        sampled_ordinal_matrix, ordinal_labels, "quadratic"
                    )
                    if sampled_ordinal_matrix
                    else None
                )
            sampled_pairwise = []
            for left_index, right_index in combinations(
                range(len(complete_annotators)), 2
            ):
                value = cohen_kappa(
                    [row[left_index] for row in sampled_matrix],
                    [row[right_index] for row in sampled_matrix],
                )
                if value is not None:
                    sampled_pairwise.append(value)
            bootstrap_values["mean_pairwise_human_human_cohen_kappa"].append(
                sum(sampled_pairwise) / len(sampled_pairwise)
                if sampled_pairwise
                else None
            )
            sampled_fleiss_rows = {
                str(index): {
                    str(rater_index): {field: value}
                    for rater_index, value in enumerate(row)
                }
                for index, row in enumerate(sampled_matrix)
            }
            bootstrap_values["fleiss_kappa"].append(
                fleiss_kappa_for_field(field, sampled_fleiss_rows, True)
            )
        human_only_bootstrap[field] = {
            metric_name: {
                "ci_95_percentile": percentile_interval(metric_values),
                "defined_replicates": sum(
                    value is not None and value == value
                    for value in metric_values
                ),
            }
            for metric_name, metric_values in bootstrap_values.items()
        }

    llm_vs_gold = {}
    llm_vs_gold_bootstrap = {}
    majority_pairs_by_field = {}
    for field_index, field in enumerate(FIELDS):
        item_ids = [
            item_id
            for item_id in shared_item_ids
            if majority_by_field[field][item_id] is not None
        ]
        gold = [majority_by_field[field][item_id] for item_id in item_ids]
        predicted = [normalized(llm[item_id].get(field)) for item_id in item_ids]
        majority_pairs_by_field[field] = (gold, predicted, item_ids)
        result = classification_summary(gold, predicted)
        result["human_majority_ties_excluded"] = len(shared_item_ids) - len(item_ids)
        llm_vs_gold[field] = result
        llm_vs_gold_bootstrap[field] = bootstrap_pair_metrics(
            gold,
            predicted,
            result["labels"],
            bootstrap_iterations,
            bootstrap_seed + 100 + field_index,
        )

    def conditional_result(
        field: str,
        eligible_item_ids: list[str],
        condition: str,
    ) -> tuple[dict, dict]:
        item_ids = [
            item_id
            for item_id in eligible_item_ids
            if majority_by_field[field][item_id] is not None
        ]
        gold = [majority_by_field[field][item_id] for item_id in item_ids]
        predicted = [normalized(llm[item_id].get(field)) for item_id in item_ids]
        result = classification_summary(gold, predicted)
        result.update(
            {
                "condition": condition,
                "eligible_items": len(eligible_item_ids),
                "human_majority_ties_excluded": len(eligible_item_ids) - len(item_ids),
            }
        )
        intervals = bootstrap_pair_metrics(
            gold,
            predicted,
            result["labels"],
            bootstrap_iterations,
            bootstrap_seed + 200 + len(eligible_item_ids),
        )
        return result, intervals

    human_hate_positive = [
        item_id
        for item_id in shared_item_ids
        if majority_by_field["identity_hate_speech"][item_id] == "1"
    ]
    human_abuse_positive = [
        item_id
        for item_id in shared_item_ids
        if majority_by_field["abuse_level"][item_id]
        not in {None, "", "none"}
    ]
    replies_only = [
        item_id for item_id in shared_item_ids if items[item_id].get("is_reply")
    ]
    conditional_metrics = {}
    conditional_bootstrap = {}
    for key, field, item_ids, condition in [
        (
            "hate_severity_on_human_hate_positive_items",
            "hate_severity",
            human_hate_positive,
            "human-majority identity_hate_speech == 1",
        ),
        (
            "hate_target_on_human_hate_positive_items",
            "hate_target_group",
            human_hate_positive,
            "human-majority identity_hate_speech == 1",
        ),
        (
            "abuse_target_on_human_abuse_positive_items",
            "abuse_target",
            human_abuse_positive,
            "human-majority abuse_level is not none",
        ),
        (
            "counterspeech_on_replies_only",
            "counterspeech",
            replies_only,
            "item is a reply",
        ),
    ]:
        result, intervals = conditional_result(field, item_ids, condition)
        conditional_metrics[key] = result
        conditional_bootstrap[key] = intervals

    ordinal_metrics = {}
    ordinal_bootstrap = {}
    ordinal_conditions = {
        "hate_severity": human_hate_positive,
        "abuse_level": [
            item_id
            for item_id in shared_item_ids
            if majority_by_field["abuse_level"][item_id] is not None
        ],
    }
    for field_index, field in enumerate(["hate_severity", "abuse_level"]):
        labels = ORDINAL_LABELS[field]
        eligible = ordinal_conditions[field]
        valid_item_ids = [
            item_id
            for item_id in eligible
            if majority_by_field[field][item_id] in labels
            and normalized(llm[item_id].get(field)) in labels
        ]
        gold = [majority_by_field[field][item_id] for item_id in valid_item_ids]
        predicted = [normalized(llm[item_id].get(field)) for item_id in valid_item_ids]
        ordinal_metrics[field] = {
            "eligible_items": len(eligible),
            "comparisons": len(valid_item_ids),
            "items_excluded_for_tied_or_non_ordinal_values": (
                len(eligible) - len(valid_item_ids)
            ),
            "labels": labels,
            "linear_weighted_cohen_kappa": weighted_cohen(
                gold, predicted, labels, "linear"
            ),
            "quadratic_weighted_cohen_kappa": weighted_cohen(
                gold, predicted, labels, "quadratic"
            ),
            "gwet_ac2_linear": gwet_agreement(
                gold, predicted, labels, "linear"
            ),
            "gwet_ac2_quadratic": gwet_agreement(
                gold, predicted, labels, "quadratic"
            ),
        }
        ordinal_bootstrap[field] = bootstrap_pair_metrics(
            gold,
            predicted,
            labels,
            bootstrap_iterations,
            bootstrap_seed + 300 + field_index,
            ordinal=True,
        )

    llm_vs_individual_humans = {}
    for annotator_id, rows in sorted(by_annotator.items()):
        if len(rows) < 20:
            continue
        llm_vs_individual_humans[annotator_id] = {
            "shared_items": len(rows),
            "fields": {
                field: classification_summary(
                    [normalized(row.get(field)) for row in rows],
                    [
                        normalized(llm[row["item_id"]].get(field))
                        for row in rows
                    ],
                )
                for field in FIELDS
            },
        }

    return {
        "summary": {
            "generated_at": datetime.now().astimezone().isoformat(),
            **llm_metadata,
            "llm_file": str(llm_file.relative_to(PROJECT_ROOT)),
            "valid_human_annotations": len(humans),
            "shared_human_llm_rating_pairs": len(shared_rows),
            "shared_items": len(shared_item_ids),
            "human_annotators_in_overlap": len(by_annotator),
            "complete_human_annotators": len(complete_annotators),
            "minimum_shared_items_for_individual_comparison": 20,
            "blank_handling": (
                "Blank conditional values are explicit not-applicable categories "
                "for nominal metrics and are excluded from ordinal metrics."
            ),
        },
        "1_sample_and_class_distributions": sample_and_distributions,
        "2_human_only_reliability": human_only,
        "3_llm_versus_human_majority_gold": llm_vs_gold,
        "4_conditional_metrics": conditional_metrics,
        "5_ordinal_metrics": ordinal_metrics,
        "6_llm_versus_individual_humans_minimum_20_items": (
            llm_vs_individual_humans
        ),
        "7_bootstrap_confidence_intervals": {
            "method": {
                "iterations": bootstrap_iterations,
                "confidence_level": 0.95,
                "interval": "percentile",
                "resampling_unit": "shared item",
                "seed": bootstrap_seed,
                "note": (
                    "Undefined kappa replicates are omitted and counted through "
                    "defined_replicates."
                ),
            },
            "human_only_reliability": human_only_bootstrap,
            "llm_versus_human_majority_gold": llm_vs_gold_bootstrap,
            "conditional_metrics": conditional_bootstrap,
            "ordinal_metrics": ordinal_bootstrap,
        },
    }



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-dir", type=Path, default=DEFAULT_HUMAN_DIR)
    parser.add_argument("--llm-file", type=Path, default=DEFAULT_LLM_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report(args.human_dir, args.llm_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(json.dumps(report["summary"], indent=2))
    print("\nLLM versus human-majority gold")
    for field, result in report["3_llm_versus_human_majority_gold"].items():
        print(
            f"{field:24} agreement={result['percent_agreement']}  "
            f"kappa={result['cohen_kappa']}  AC1={result['gwet_ac1']}  "
            f"macro-F1={result['macro_f1']}"
        )

    print("\nOrdinal metrics")
    print(json.dumps(report["5_ordinal_metrics"], indent=2))



if __name__ == "__main__":
    main()
