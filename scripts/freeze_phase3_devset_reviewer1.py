#!/usr/bin/env python3
"""Freeze a phase-3 development set using the user's reviewer-1 override.

This is intentionally separate from the normal two-reviewer freeze path.  The
output status records the single-review override and never modifies candidate_v6.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    from prepare_phase3_devset import _candidate_rows
except ModuleNotFoundError:  # imported as scripts.freeze_phase3_devset_reviewer1
    from scripts.prepare_phase3_devset import _candidate_rows


CATEGORIES = ["straight_walk", "turning_walk", "speed_walk", "arms_while_walking", "stop_and_walk"]
KEEP = "KEEP"

# The first automatic attempt is deliberately not used: it selected one of
# the dropped IDs again and admitted two semantically impure straight-walk
# descriptions.  These replacements were manually checked against the full
# source pool and are fixed here for reproducibility.
REVIEWER1_REPLACEMENT_IDS = {
    47457: 1933,   # clean long forward walk
    57652: 14053,  # four steps in a straight line
    43493: 48788,  # quarter-circle turn, distinct from the retained 180° pairs
    14567: 56467,  # short distance at a quick pace; no backward/stride confound
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_reviewer1(review_path: Path) -> dict[int, dict[str, str]]:
    decisions: dict[int, dict[str, str]] = {}
    for raw in review_path.read_text(encoding="utf-8").splitlines():
        if not raw.startswith("|") or raw.startswith("|---"):
            continue
        cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
        if len(cells) < 8 or not cells[0].isdigit() or not cells[1].isdigit():
            continue
        index, motion_id = int(cells[0]), int(cells[1])
        reviewer_a = cells[4]
        if not (reviewer_a.startswith("KEEP") or reviewer_a.startswith("DROP")):
            raise ValueError(f"reviewer1 decision for id={motion_id} is not KEEP/DROP: {reviewer_a}")
        decisions[motion_id] = {
            "index": str(index),
            "reviewer_a": reviewer_a,
            "note": cells[7] if len(cells) > 7 else "",
        }
    if len(decisions) != 20:
        raise ValueError(f"expected 20 reviewer1 decisions, found {len(decisions)}")
    return decisions


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    return len(left & right) / max(1, len(left | right))


def _replacement(
    pool: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    used_ids: set[int],
    replacement_for_id: int,
) -> dict[str, Any]:
    replacement_id = REVIEWER1_REPLACEMENT_IDS[replacement_for_id]
    matches = [row for row in pool if int(row["id"]) == replacement_id]
    if len(matches) != 1:
        raise ValueError(
            f"replacement id={replacement_id} for dropped id={replacement_for_id} "
            f"is missing or ambiguous in the same-category source pool"
        )
    row = matches[0]
    if int(row["id"]) in used_ids:
        raise ValueError(f"replacement id={replacement_id} is already used")
    similarity = max(
        (_jaccard(row["normalised_text"], item["normalised_text"]) for item in kept),
        default=0.0,
    )
    if similarity >= 0.58:
        raise ValueError(
            f"replacement id={replacement_id} is a near-duplicate of a retained item "
            f"(Jaccard={similarity:.3f})"
        )
    row = dict(row)
    row["max_similarity_to_kept"] = similarity
    return row


def freeze(candidate_path: Path, review_path: Path, source_path: Path, output_dir: Path) -> dict[str, Any]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate.get("status") != "CANDIDATE_NOT_FROZEN":
        raise ValueError("candidate must remain CANDIDATE_NOT_FROZEN")
    decisions = parse_reviewer1(review_path)
    source_items = json.loads(source_path.read_text(encoding="utf-8"))
    source_rows = _candidate_rows(source_items)
    by_category = {category: [row for row in source_rows if row["category"] == category] for category in CATEGORIES}
    selected_by_category: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
    candidate_ids = {int(item["id"]) for item in candidate["items"]}
    used_ids: set[int] = set()
    dropped: list[dict[str, Any]] = []
    for item in candidate["items"]:
        decision = decisions.get(int(item["id"]))
        if decision is None:
            raise ValueError(f"candidate item id={item['id']} missing reviewer1 decision")
        if decision["reviewer_a"].startswith(KEEP):
            kept = dict(item)
            kept["manual_review"] = {
                "reviewer_a": decision["reviewer_a"],
                "reviewer_b": "IGNORED_BY_USER",
                "adjudication": "USER_SINGLE_REVIEW_OVERRIDE",
            }
            kept["selection_status"] = "FROZEN_SINGLE_REVIEW_OVERRIDE"
            selected_by_category[item["category"]].append(kept)
            used_ids.add(int(item["id"]))
        else:
            dropped.append({
                "id": int(item["id"]),
                "category": item["category"],
                "text": item["motion_text_annot"],
                "reviewer1_decision": decision["reviewer_a"],
                "reviewer1_note": decision["note"],
            })
    replacements: list[dict[str, Any]] = []
    for dropped_item in dropped:
        category = dropped_item["category"]
        replacement_id = REVIEWER1_REPLACEMENT_IDS.get(dropped_item["id"])
        if replacement_id is None:
            raise ValueError(f"no fixed replacement for dropped id={dropped_item['id']}")
        if replacement_id in candidate_ids:
            raise ValueError(
                f"replacement id={replacement_id} is one of the candidate-v6 items; "
                "a dropped candidate must not be selected again"
            )
        replacement = _replacement(
            by_category[category], selected_by_category[category], used_ids, dropped_item["id"]
        )
        replacement["replacement_for_id"] = dropped_item["id"]
        replacement["replacement_reason"] = dropped_item["reviewer1_decision"]
        replacement["manual_review"] = {
            "reviewer_a": "REVIEWER1_SINGLE_REVIEW_REPLACEMENT",
            "reviewer_b": "IGNORED_BY_USER",
            "adjudication": "USER_SINGLE_REVIEW_OVERRIDE",
        }
        replacement["selection_status"] = "FROZEN_SINGLE_REVIEW_OVERRIDE"
        selected_by_category[category].append(replacement)
        used_ids.add(int(replacement["id"]))
        replacements.append(replacement)
    if any(len(selected_by_category[category]) != 4 for category in CATEGORIES):
        raise RuntimeError("freeze did not produce four items per category")
    items: list[dict[str, Any]] = []
    for category in CATEGORIES:
        for item in selected_by_category[category]:
            item = dict(item)
            item["dev_index"] = len(items)
            items.append(item)
    manifest = {
        "schema_version": "phase3_devset_frozen_v1",
        "status": "FROZEN_SINGLE_REVIEW_OVERRIDE",
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "base_candidate": str(candidate_path),
        "base_candidate_sha256": _sha256(candidate_path),
        "review_evidence": str(review_path),
        "review_evidence_sha256": _sha256(review_path),
        "review_protocol": "reviewer1 only by explicit user instruction; reviewer2 ignored",
        "categories": CATEGORIES,
        "items_per_category": 4,
        "target_size": 20,
        "formal_protocol": {"seeds": [0, 1, 2], "commands_degrees": [0.0, 5.0, 10.0], "units": 180},
        "dropped_by_reviewer1": dropped,
        "replacement_count": len(replacements),
        "replacement_ids": [int(item["id"]) for item in replacements],
        "replacement_selection": REVIEWER1_REPLACEMENT_IDS,
        "items": items,
        "warning": "This is not a two-independent-reviewer freeze; it is a user-authorized single-review override.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in {
        "frozen_manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        "README.md": "# Frozen development set v1 (reviewer1 override)\n\nStatus: `FROZEN_SINGLE_REVIEW_OVERRIDE`. Reviewer2 was explicitly ignored by the user; see `frozen_manifest.json`.\n",
    }.items():
        target = output_dir / filename
        if target.exists():
            raise FileExistsError(target)
        target.write_text(content, encoding="utf-8")
    with (output_dir / "frozen_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["dev_index", "id", "category", "motion_path", "motion_text_annot", "replacement_for_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: item.get(field, "") for field in fields} for item in items)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze(args.candidate, args.review, args.source, args.output_dir)
    print(json.dumps({"status": manifest["status"], "target_size": manifest["target_size"], "replacement_ids": manifest["replacement_ids"]}, indent=2))


if __name__ == "__main__":
    main()
