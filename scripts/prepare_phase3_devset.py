#!/usr/bin/env python3
"""Prepare a reviewable development-set candidate manifest for ViMoGen.

This script deliberately produces a *candidate* manifest only.  It never marks
the split as frozen, and it never runs the model.  The final 20-item split must
be reviewed by two independent reviewers and then frozen in a separate step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


CATEGORIES: dict[str, tuple[str, ...]] = {
    "straight_walk": (
        r"\bwalk(?:s|ed|ing)?\b",
        r"\bstroll(?:s|ed|ing)?\b",
        r"\bmarch(?:es|ed|ing)?\b",
        r"\bstride(?:s|d|ing)?\b",
        r"\bsteps? (?:forward|back(?:ward)?)\b",
    ),
    "turning_walk": (
        r"\bturn(?:s|ed|ing)?\b",
        r"\bturns? (?:left|right)\b",
        r"\bpivot(?:s|ed|ing)?\b",
        r"\bcircl(?:e|es|ed|ing)\b",
        r"\bzigzag\b",
    ),
    "speed_walk": (
        r"\bslow(?:ly|er)?\b",
        r"\bfast(?:er)?\b",
        r"\bquick(?:ly|er)?\b",
        r"\brapid(?:ly)?\b",
        r"\bhurr(?:y|ies|ied|ying)\b",
        r"\bleisurely\b",
        r"\bpace\b",
    ),
    "arms_while_walking": (
        r"\bwave(?:s|d|ing)?\b",
        r"\brais(?:e|es|ed|ing)\b",
        r"\bswing(?:s|ing)?\b",
        r"\bclap(?:s|ped|ping)?\b",
        r"\bgestur(?:e|es|ed|ing)\b",
        r"\barm(?:s)?\b",
        r"\bhand(?:s)?\b",
        r"\bthrow(?:s|n|ing)?\b",
    ),
    "stop_and_walk": (
        r"\bstop(?:s|ped|ping)?\b",
        r"\bpaus(?:e|es|ed|ing)\b",
        r"\bhalt(?:s|ed|ing)?\b",
        r"\bstart(?:s|ed|ing)?\b",
        r"\bbegin(?:s|ning)?\b",
    ),
}

WALK_PATTERNS = (
    r"\bwalk(?:s|ed|ing)?\b",
    r"\bstroll(?:s|ed|ing)?\b",
    r"\bmarch(?:es|ed|ing)?\b",
    r"\bstride(?:s|d|ing)?\b",
    r"\bsteps? (?:forward|back(?:ward)?)\b",
    r"\bpace(?:s|d|ing)?\b",
)

EXCLUDE_PATTERNS = {
    "running": r"\b(?:run|runs|ran|running|jog|jogs|jogging|sprint(?:s|ed|ing)?)\b",
    "stairs": r"\b(?:stairs?|steps?\s+(?:up|down)|staircase)\b",
    "sitting": r"\b(?:sit|sits|sat|sitting|chair|seated)\b",
    "crawling": r"\bcrawl(?:s|ed|ing)?\b",
    "falling_or_pushed": r"\b(?:fall|falls|fell|falling|push(?:es|ed|ing)?)\b",
    "obstacle": r"\b(?:obstacle| hurdle|jump(?:s|ed|ing)?\s+over)\b",
    "bending": r"\b(?:bend(?:s|ed|ing)?|stoop(?:s|ed|ing)?)\b",
    "dance": r"\bdanc(?:e|es|ed|ing)\b",
    "complex_object_or_jump": r"\b(?:jump(?:s|ed|ing)?|throw(?:s|n|ing)?|pick(?:s|ed|ing)?|carry(?:s|ied|ing)?|object|ball|king|queen|regal|aggressiv(?:e|ely))\b",
    "non_dominant_walk": r"\b(?:crouch(?:es|ed|ing)?|shov(?:e|es|ed|ing)|clockwise|counterclockwise|sideways|lean(?:s|ed|ing)?|look(?:s|ed|ing)?|standing up|sway(?:s|ed|ing)?|spin(?:s|ning)?|toppl(?:e|es|ed|ing)|bow(?:s|ed|ing)?|touch(?:es|ed|ing)?|lunge(?:s|d|ing)?|stance|balance|regain(?:s|ed|ing)?|open(?:s|ed|ing)?\s+(?:something|the)|hop(?:s|ped|ping)?|intimidat(?:e|es|ed|ing))\b",
    "explanatory_or_roleplay": r"\b(?:this is when|maybe when|trying to|playing a|would involve)\b",
}

CATEGORY_REJECT_PATTERNS = {
    "straight_walk": (
        CATEGORIES["turning_walk"],
        CATEGORIES["speed_walk"],
        CATEGORIES["arms_while_walking"],
        CATEGORIES["stop_and_walk"],
    ),
    "turning_walk": (
        CATEGORIES["speed_walk"],
        CATEGORIES["arms_while_walking"],
        CATEGORIES["stop_and_walk"],
    ),
    "speed_walk": (
        CATEGORIES["turning_walk"],
        CATEGORIES["arms_while_walking"],
        CATEGORIES["stop_and_walk"],
    ),
    "arms_while_walking": (
        CATEGORIES["turning_walk"],
        CATEGORIES["speed_walk"],
        CATEGORIES["stop_and_walk"],
    ),
    "stop_and_walk": (
        CATEGORIES["turning_walk"],
        CATEGORIES["speed_walk"],
        CATEGORIES["arms_while_walking"],
    ),
}

CATEGORY_EXTRA_REJECT = {
    "straight_walk": (r"\b(?:shift(?:s|ed|ing)?|face(?:s|d|ing)?|around|sideways|clockwise|counterclockwise)\b",),
    "turning_walk": (),
    "speed_walk": (r"\b(?:hurry|trying to|maybe when)\b",),
    "arms_while_walking": (r"\b(?:rotate|rotates|twist(?:s|ed|ing)?|above head|above chest|left to right|right to left)\b",),
    "stop_and_walk": (r"\b(?:turn(?:s|ed|ing)?|pivot(?:s|ed|ing)?|around|circle|shift(?:s|ed|ing)?|rotate)\b",),
}


def _matches(text: str, patterns: Iterable[str]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text)]


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for item in items:
        raw_text = str(item.get("motion_text_annot", ""))
        text = _normalise_text(raw_text)
        if not text or text in seen_text:
            continue
        seen_text.add(text)
        walk_matches = _matches(text, WALK_PATTERNS)
        exclusion_matches = {
            name: pattern
            for name, pattern in EXCLUDE_PATTERNS.items()
            if re.search(pattern, text)
        }
        if not walk_matches or exclusion_matches:
            continue
        matched_categories = {
            name: _matches(text, patterns)
            for name, patterns in CATEGORIES.items()
        }
        for category, matches in matched_categories.items():
            if not matches:
                continue
            rejected_by_category = any(
                _matches(text, reject_patterns)
                for reject_group in CATEGORY_REJECT_PATTERNS[category]
                for reject_patterns in (reject_group,)
            )
            if rejected_by_category or any(
                re.search(pattern, text) for pattern in CATEGORY_EXTRA_REJECT[category]
            ):
                continue
            score = len(matches) + 2 * len(walk_matches)
            rows.append(
                {
                    "id": item.get("id"),
                    "motion_path": item.get("motion_path"),
                    "motion_text_annot": raw_text,
                    "normalised_text": text,
                    "category": category,
                    "matched_walk_patterns": walk_matches,
                    "matched_category_patterns": matches,
                    "selection_score": score,
                }
            )
    return rows


def _select(rows: list[dict[str, Any]], per_category: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    used_ids: set[Any] = set()
    for category in CATEGORIES:
        pool = [row for row in rows if row["category"] == category]
        pool.sort(key=lambda row: (-int(row["selection_score"]), rng.random(), int(row["id"])))
        category_rows: list[dict[str, Any]] = []
        for row in pool:
            if row["id"] in used_ids:
                continue
            category_rows.append(row)
            used_ids.add(row["id"])
            if len(category_rows) == per_category:
                break
        if len(category_rows) < per_category:
            raise RuntimeError(
                f"category {category!r} has only {len(category_rows)} unique candidates; "
                f"need {per_category}"
            )
        selected.extend(category_rows)
    return selected


def build_manifest(source: Path, out_dir: Path, per_category: int, seed: int) -> dict[str, Any]:
    with source.open("r", encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise TypeError("ViMoGen-228K JSON must contain a list")
    rows = _candidate_rows(items)
    selected = _select(rows, per_category=per_category, seed=seed)
    for index, row in enumerate(selected):
        row["dev_index"] = index
        row["manual_review"] = {
            "reviewer_a": "PENDING",
            "reviewer_b": "PENDING",
            "adjudication": "PENDING",
        }
        row["selection_status"] = "CANDIDATE_NOT_FROZEN"
    pool_counts = {
        category: sum(row["category"] == category for row in rows)
        for category in CATEGORIES
    }
    manifest = {
        "schema_version": "phase3_devset_candidate_v1",
        "status": "CANDIDATE_NOT_FROZEN",
        "source": str(source),
        "source_sha256": _source_sha256(source),
        "selection_seed": seed,
        "items_per_category": per_category,
        "target_size": per_category * len(CATEGORIES),
        "categories": list(CATEGORIES),
        "candidate_pool_counts": pool_counts,
        "selection_rule": "walking-word required; excluded action patterns removed; exact normalized text deduplicated; deterministic score and seed tie-break",
        "required_review": [
            "Two independent reviewers must verify that every item is walking-dominant.",
            "Reviewers must remove near-duplicate texts and record exclusion reasons.",
            "A third reviewer resolves disagreements before status can become FROZEN.",
        ],
        "formal_protocol": {
            "seeds": [0, 1, 2],
            "commands_degrees": [0.0, 5.0, 10.0],
            "units": per_category * len(CATEGORIES) * 3 * 3,
        },
        "items": selected,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "candidate_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["dev_index", "id", "category", "motion_path", "motion_text_annot", "selection_score"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in selected)
    summary = [
        "# Phase 3 development-set candidate",
        "",
        "Status: **CANDIDATE_NOT_FROZEN**",
        "",
        f"Source SHA256: `{manifest['source_sha256']}`",
        f"Selection seed: `{seed}`",
        "",
        "This manifest is not a formal split until two independent reviewers and an adjudicator complete the review fields in `candidate_manifest.json`.",
        "",
        "## Candidate pool counts",
        "",
    ]
    for category, count in pool_counts.items():
        summary.append(f"- `{category}`: {count} candidates; selected {per_category}")
    summary.extend(
        [
            "",
            "## Formal evaluation units after freeze",
            "",
            f"{manifest['target_size']} texts × 3 seeds × 3 commands (0/+5/+10) = {manifest['formal_protocol']['units']} units per method.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    manifest = build_manifest(args.source, args.out_dir, args.per_category, args.seed)
    print(json.dumps({
        "status": manifest["status"],
        "target_size": manifest["target_size"],
        "candidate_pool_counts": manifest["candidate_pool_counts"],
        "source_sha256": manifest["source_sha256"],
        "out_dir": str(args.out_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
