"""Build a non-turning M1 holdout candidate without running the model.

The candidate is deliberately not frozen by this script.  It excludes every
item in the reviewer1 development split, excludes the turning category, and
uses a deterministic category-balanced selection.  A human review/freeze is
required before model outputs may be generated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_phase3_devset import _candidate_rows


CATEGORIES = ("straight_walk", "speed_walk", "arms_while_walking", "stop_and_walk")
EXPLICIT_STOP = re.compile(
    r"\b(?:stop|stops|stopped|stopping|pause|pauses|paused|pausing|"
    r"halt|halts|halted|halting)\b",
    flags=re.IGNORECASE,
)

# The source labels are noisy.  These lexical exclusions make the holdout
# conservative: a text mentioning a change of heading, backward motion, side
# stepping, or an object/trajectory interaction is not treated as non-turning.
NON_TURNING_REJECT = re.compile(
    r"\b(?:back(?:ward|wards)?|side\s*step(?:s|ped|ping)?|sideways|"
    r"side\s+to\s+side|zig[- ]?zag|circular|circle|curv(?:e|ed|ing)|veer(?:s|ed|ing)?|arc|"
    r"angle(?:s|d|ing)?|turn(?:s|ed|ing)?|"
    r"pivot(?:s|ed|ing)?|face(?:s|d|ing)?|kick(?:s|ed|ing)?|"
    r"slide(?:s|d|ing)?|clean(?:s|ed|ing)?|aggressiv(?:e|ely)|"
    r"knee(?:s|d)?|kneel(?:s|ed|ing)?|ground|object|something|diploma)\b|"
    r"\b(?:left|right)\s+side\b|\b(?:to|toward|towards|over\s+to)\s+"
    r"(?:his|her|the|hiis)?\s*(?:left|right|side)\b|"
    r"\bstep(?:s|ped|ping)?\s+to\s+the\s+side\b",
    flags=re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / max(1, len(a | b))


def build_candidate(
    *, source: Path, development_manifest: Path, output_dir: Path,
    per_category: int = 5, seed: int = 20260822, motion_root: Path | None = None,
) -> dict[str, object]:
    source_items = json.loads(source.read_text(encoding="utf-8"))
    development = json.loads(development_manifest.read_text(encoding="utf-8"))
    rows = _candidate_rows(source_items)
    motion_root = motion_root or (ROOT / "data/ViMoGen-228K")
    excluded_ids = {int(item["id"]) for item in development["items"]}
    excluded_texts = [item["normalised_text"] for item in development["items"]]
    rng = random.Random(seed)
    selected: list[dict[str, object]] = []
    selected_ids: set[int] = set(excluded_ids)
    for category in CATEGORIES:
        pool = [
            row for row in rows
            if row["category"] == category
            and int(row["id"]) not in excluded_ids
            and (motion_root / str(row["motion_path"])).exists()
        ]
        pool.sort(key=lambda row: (-int(row["selection_score"]), rng.random(), int(row["id"])))
        chosen: list[dict[str, object]] = []
        for row in pool:
            if int(row["id"]) in selected_ids:
                continue
            text = str(row["normalised_text"])
            if category == "stop_and_walk" and not EXPLICIT_STOP.search(text):
                continue
            if NON_TURNING_REJECT.search(text):
                continue
            if max((_jaccard(text, other) for other in excluded_texts), default=0.0) >= 0.58:
                continue
            if max((_jaccard(text, other["normalised_text"]) for other in chosen), default=0.0) >= 0.58:
                continue
            chosen.append(dict(row))
            selected_ids.add(int(row["id"]))
            if len(chosen) == per_category:
                break
        if len(chosen) != per_category:
            raise RuntimeError(f"category {category} has only {len(chosen)} usable holdout candidates")
        selected.extend(chosen)
    for index, row in enumerate(selected):
        row["holdout_index"] = index
        row["selection_status"] = "HOLDOUT_CANDIDATE_NOT_FROZEN"
        row["review_status"] = "PENDING_HUMAN_REVIEW"
    manifest = {
        "schema_version": "m1_nonturning_holdout_candidate_v1",
        "status": "HOLDOUT_CANDIDATE_NOT_FROZEN",
        "source": str(source),
        "source_sha256": _sha256(source),
        "development_manifest": str(development_manifest),
        "development_manifest_sha256": _sha256(development_manifest),
        "selection_seed": seed,
        "categories": list(CATEGORIES),
        "excluded_categories": ["turning_walk"],
        "items_per_category": per_category,
        "target_size": len(selected),
        "formal_protocol": {"seeds": [0, 1, 2], "commands_degrees": [0.0, 5.0, 10.0], "units": len(selected) * 3 * 3},
        "selection_rule": "exclude reviewer1 development IDs and turning_walk; deterministic category-balanced lexical candidates; reject near-duplicates at Jaccard >= 0.58",
        "review_required": "Freeze only after human review confirms non-turning semantics, valid motion file, and no near-duplicate leakage.",
        "items": selected,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "holdout_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "holdout_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["holdout_index", "id", "category", "motion_path", "motion_text_annot", "selection_score"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in selected)
    (output_dir / "README.md").write_text(
        "# M1 non-turning holdout candidate\n\n"
        "Status: `HOLDOUT_CANDIDATE_NOT_FROZEN`. Turning samples are excluded.\n\n"
        "This directory is a candidate only; do not run formal M1 evaluation until human review freezes it.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/ViMoGen-228K/ViMoGen-228K.json")
    parser.add_argument("--development-manifest", type=Path, default=ROOT / "diagnostics/phase3/devset_frozen/frozen_v2_reviewer1/frozen_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "diagnostics/phase3/devset_holdout_candidates/nonturning_v10")
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    manifest = build_candidate(
        source=args.source,
        development_manifest=args.development_manifest,
        output_dir=args.output_dir,
        per_category=args.per_category,
        seed=args.seed,
    )
    print(json.dumps({"status": manifest["status"], "target_size": manifest["target_size"], "categories": manifest["categories"]}, indent=2))


if __name__ == "__main__":
    main()
