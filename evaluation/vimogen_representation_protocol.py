"""Deterministic, source-disjoint ViMoGen representation split protocol.

The released ViMoGen annotation uses ``split`` for the original source
dataset, not for train/validation/test.  This module therefore defines the
project-local protocol used to evaluate the 276-D representation.  It keeps
the split policy independent from model training and records enough metadata
to audit leakage later.

The protocol has three optical-motion subsets:

* ``representation_dev_v1``: a deterministic, source-stratified sample from
  the large optical-motion sources; it may be used for tuning;
* ``representation_val_v1``: complete source groups KIT-ML, LAFAN1 and
  BEHAVE; it may be used for model-version selection;
* ``representation_test_v1``: complete source groups FIT3D, Mixamo,
  HumanSC3D, ARCTIC, RICH and EMDB; it is frozen before final evaluation.

The code deliberately does not call a model.  It only reads annotations and
motion files and writes manifests/validation reports.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEV_SOURCES = (
    "AMASS",
    "IDEA400",
    "100style",
    "OMOMO",
    "InterX",
    "CIRCLE",
    "TRU-MANS",
    "Chairs",
)
VAL_SOURCES = ("KIT-ML", "LAFAN1", "BEHAVE")
TEST_SOURCES = ("FIT3D", "Mixamo", "HumanSC3D", "ARCTIC", "RICH", "EMDB")
ALL_OPTICAL_SOURCES = DEV_SOURCES + VAL_SOURCES + TEST_SOURCES
OPTICAL_SUBSET = "Optical MoCap"
PROTOCOL_VERSION = "representation_source_holdout_v1"
LENGTH_BUCKETS = ((0, 74, "short"), (75, 124, "medium"), (125, 10**9, "long"))

# These counts are a version guard for the released 228K annotation.  A
# changed upstream annotation must be reviewed rather than silently accepted.
EXPECTED_OPTICAL_COUNTS = {
    "AMASS": 88967,
    "IDEA400": 13929,
    "100style": 11939,
    "OMOMO": 9736,
    "InterX": 9622,
    "CIRCLE": 8989,
    "TRU-MANS": 6611,
    "Chairs": 6376,
    "KIT-ML": 5404,
    "LAFAN1": 2375,
    "BEHAVE": 2354,
    "FIT3D": 1591,
    "Mixamo": 1133,
    "HumanSC3D": 1077,
    "ARCTIC": 1071,
    "RICH": 234,
    "EMDB": 134,
}


@dataclass(frozen=True)
class MotionFingerprint:
    """Small, JSON-friendly identity record for a motion tensor."""

    frame_count: int
    dimension: int
    dtype: str
    tensor_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "dimension": self.dimension,
            "dtype": self.dtype,
            "tensor_sha256": self.tensor_sha256,
        }


def _normalise_source(item: Mapping[str, Any]) -> str:
    value = item.get("split")
    if not isinstance(value, str) or not value:
        raise ValueError(f"annotation item has no source split: {item!r}")
    return value


def _normalise_subset(item: Mapping[str, Any]) -> str:
    value = item.get("subset")
    return value if isinstance(value, str) else ""


def _item_key(item: Mapping[str, Any], seed: int) -> str:
    raw = f"{seed}\0{item.get('id')}\0{item.get('motion_path')}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def length_bucket(frame_count: int) -> str:
    for lower, upper, label in LENGTH_BUCKETS:
        if lower <= frame_count <= upper:
            return label
    raise ValueError(f"invalid frame count: {frame_count}")


def _largest_remainder(total: int, capacities: Mapping[str, int]) -> dict[str, int]:
    if total < 0 or total > sum(capacities.values()):
        raise ValueError(f"cannot allocate {total} from capacities {capacities}")
    if not capacities:
        return {}
    capacity_total = sum(capacities.values())
    raw = {key: total * value / capacity_total for key, value in capacities.items()}
    result = {key: min(capacities[key], int(raw[key])) for key in capacities}
    remaining = total - sum(result.values())
    order = sorted(
        capacities,
        key=lambda key: (raw[key] - int(raw[key]), capacities[key], key),
        reverse=True,
    )
    for key in order:
        if remaining <= 0:
            break
        if result[key] < capacities[key]:
            result[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("largest-remainder allocation could not fill requested size")
    return result


def _optical_items(items: Sequence[Mapping[str, Any]], sources: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(sources)
    result = []
    for index, raw in enumerate(items):
        item = dict(raw)
        if _normalise_subset(item) != OPTICAL_SUBSET:
            continue
        if _normalise_source(item) not in wanted:
            continue
        if not item.get("motion_path"):
            raise ValueError(f"selected item has no motion_path at annotation index {index}")
        item["annotation_index"] = index
        item["source"] = _normalise_source(item)
        result.append(item)
    return result


def inventory(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(
        _normalise_source(item)
        for item in items
        if _normalise_subset(item) == OPTICAL_SUBSET
    )
    return {
        "total_items": len(items),
        "optical_total": sum(counts.values()),
        "optical_source_counts": dict(sorted(counts.items())),
        "subset_counts": dict(
            sorted(Counter(_normalise_subset(item) for item in items).items())
        ),
    }


def validate_annotation_inventory(
    items: Sequence[Mapping[str, Any]], *, require_release_counts: bool = True
) -> dict[str, Any]:
    """Validate the annotation shape and, by default, the released counts."""

    ids: set[str] = set()
    paths: set[str] = set()
    for index, item in enumerate(items):
        sample_id = str(item.get("id", ""))
        path = str(item.get("motion_path", ""))
        if not sample_id or not path:
            raise ValueError(f"annotation item {index} is missing id or motion_path")
        if sample_id in ids:
            raise ValueError(f"duplicate annotation id: {sample_id}")
        if path in paths:
            raise ValueError(f"duplicate annotation motion_path: {path}")
        ids.add(sample_id)
        paths.add(path)
    report = inventory(items)
    if require_release_counts:
        actual = report["optical_source_counts"]
        expected = {key: EXPECTED_OPTICAL_COUNTS[key] for key in EXPECTED_OPTICAL_COUNTS}
        if actual != expected:
            raise ValueError(
                "ViMoGen optical source counts differ from the frozen release: "
                f"expected {expected}, got {actual}"
            )
    return report


def _select_source_quota(items: Sequence[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[item["source"]].append(item)
    quotas = _largest_remainder(size, {source: len(rows) for source, rows in grouped.items()})
    selected: list[dict[str, Any]] = []
    for source in sorted(grouped):
        rows = sorted(grouped[source], key=lambda row: _item_key(row, seed))
        selected.extend(rows[: quotas[source]])
    return selected


def _select_source_and_length_stratified(
    items: Sequence[dict[str, Any]], frame_counts: Mapping[str, int], size: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if str(item["id"]) not in frame_counts:
            raise ValueError(f"missing frame count for {item['id']}")
        enriched = dict(item)
        enriched["frame_count"] = int(frame_counts[str(item["id"])])
        enriched["length_bucket"] = length_bucket(enriched["frame_count"])
        grouped[enriched["source"]].append(enriched)
    source_quotas = _largest_remainder(size, {source: len(rows) for source, rows in grouped.items()})
    selected: list[dict[str, Any]] = []
    for source in sorted(grouped):
        rows = grouped[source]
        by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_bucket[row["length_bucket"]].append(row)
        bucket_quotas = _largest_remainder(
            source_quotas[source], {bucket: len(bucket_rows) for bucket, bucket_rows in by_bucket.items()}
        )
        for bucket in sorted(by_bucket):
            bucket_rows = sorted(by_bucket[bucket], key=lambda row: _item_key(row, seed))
            selected.extend(bucket_rows[: bucket_quotas[bucket]])
    if len(selected) != size:
        raise RuntimeError(f"length-stratified selection returned {len(selected)} instead of {size}")
    return selected


def _annotate_selection(rows: Sequence[Mapping[str, Any]], split_name: str, role: str) -> list[dict[str, Any]]:
    output = []
    for split_index, row in enumerate(sorted(rows, key=lambda value: (value["source"], str(value["id"])) )):
        item = dict(row)
        item["split_name"] = split_name
        item["role"] = role
        item["split_index"] = split_index
        output.append(item)
    return output


def build_split_manifests(
    items: Sequence[Mapping[str, Any]],
    *,
    dev_size: int = 20_000,
    seed: int = 20260824,
    frame_counts: Mapping[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the three manifests without reading motion tensors.

    ``frame_counts`` is optional for a fast candidate.  Supplying it enables
    source-plus-length stratification for the development set.  Validation and
    test sets are complete source groups and never sampled.
    """

    validate_annotation_inventory(items)
    dev_pool = _optical_items(items, DEV_SOURCES)
    if frame_counts is None:
        dev_rows = _select_source_quota(dev_pool, dev_size, seed)
        dev_selection = "source_proportional_stable_hash"
    else:
        dev_rows = _select_source_and_length_stratified(dev_pool, frame_counts, dev_size, seed)
        dev_selection = "source_and_length_stratified_stable_hash"
    val_rows = _optical_items(items, VAL_SOURCES)
    test_rows = _optical_items(items, TEST_SOURCES)
    manifests = {
        "representation_dev_v1": {
            "role": "development",
            "sources": list(DEV_SOURCES),
            "selection": dev_selection,
            "requested_size": dev_size,
            "items": _annotate_selection(dev_rows, "representation_dev_v1", "tuning_allowed"),
        },
        "representation_val_v1": {
            "role": "validation",
            "sources": list(VAL_SOURCES),
            "selection": "complete_source_groups",
            "requested_size": len(val_rows),
            "items": _annotate_selection(val_rows, "representation_val_v1", "model_selection_allowed"),
        },
        "representation_test_v1": {
            "role": "final_blind_test",
            "sources": list(TEST_SOURCES),
            "selection": "complete_source_groups",
            "requested_size": len(test_rows),
            "items": _annotate_selection(test_rows, "representation_test_v1", "frozen_no_tuning"),
        },
    }
    validate_split_manifests(manifests, require_frozen=False)
    return manifests


def _load_motion_tensor(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only
        return torch.load(path, map_location="cpu")


def fingerprint_motion(path: Path) -> MotionFingerprint:
    """Load one motion file and return shape, dtype and tensor-content hash."""

    import torch

    value = _load_motion_tensor(path)
    if isinstance(value, dict):
        value = value.get("motion")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"motion file does not contain a tensor: {path}")
    if value.ndim != 2 or value.shape[-1] != 276:
        raise ValueError(f"expected [T,276] at {path}, got {tuple(value.shape)}")
    if value.shape[0] < 1 or not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"motion tensor is empty, non-floating, or non-finite: {path}")
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return MotionFingerprint(
        frame_count=int(value.shape[0]),
        dimension=int(value.shape[1]),
        dtype=str(value.dtype),
        tensor_sha256=hashlib.sha256(raw).hexdigest(),
    )


def build_unique_dev_manifest(
    items: Sequence[Mapping[str, Any]],
    *,
    dev_size: int,
    seed: int,
    motion_root: Path,
    mbench_root: Path | None = None,
) -> dict[str, Any]:
    """Build a materialized development split with stable duplicate replacement.

    ViMoGen annotations can contain two IDs pointing to identical tensors.
    The development split is sampled, so a duplicate is skipped and replaced
    by the next stable-hash item from the same source.  Validation and test
    remain complete source groups and use strict duplicate rejection.
    """

    dev_pool = _optical_items(items, DEV_SOURCES)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in dev_pool:
        grouped[item["source"]].append(item)
    quotas = _largest_remainder(dev_size, {source: len(rows) for source, rows in grouped.items()})
    mbench_hashes = _mbench_hashes(mbench_root) if mbench_root is not None else set()
    seen: dict[str, str] = {}
    duplicate_replacements: list[dict[str, str]] = []
    selected: list[dict[str, Any]] = []
    for source in sorted(grouped):
        count = 0
        for raw in sorted(grouped[source], key=lambda row: _item_key(row, seed)):
            if count >= quotas[source]:
                break
            path = motion_root / str(raw["motion_path"])
            fingerprint = fingerprint_motion(path)
            digest = fingerprint.tensor_sha256
            if digest in seen:
                duplicate_replacements.append(
                    {"source": source, "duplicate_id": str(raw["id"]), "kept_id": seen[digest]}
                )
                continue
            if digest in mbench_hashes:
                raise ValueError(f"development motion overlaps MBench: {raw['id']}")
            item = dict(raw)
            item["fingerprint"] = fingerprint.as_dict()
            item["length_bucket"] = length_bucket(fingerprint.frame_count)
            selected.append(item)
            seen[digest] = str(raw["id"])
            count += 1
        if count != quotas[source]:
            raise ValueError(f"source {source} has only {count} unique motions; need {quotas[source]}")
    manifest = {
        "role": "development",
        "sources": list(DEV_SOURCES),
        "selection": "source_proportional_stable_hash_with_content_dedup_replacement",
        "requested_size": dev_size,
        "items": _annotate_selection(selected, "representation_dev_v1", "tuning_allowed"),
        "materialized": True,
        "motion_root": str(motion_root),
        "duplicate_replacements": duplicate_replacements,
    }
    if len(manifest["items"]) != dev_size:
        raise RuntimeError("unique development selection did not reach requested size")
    return manifest


def materialize_manifest(
    manifest: Mapping[str, Any],
    *,
    motion_root: Path,
    mbench_root: Path | None = None,
    drop_duplicates: bool = False,
) -> dict[str, Any]:
    """Attach fingerprints and reject missing/invalid/duplicate motions."""

    result = json.loads(json.dumps(manifest))
    seen_hashes: dict[str, str] = {}
    unique_items: list[dict[str, Any]] = []
    excluded_duplicates: list[dict[str, Any]] = []
    for item in result["items"]:
        path = motion_root / str(item["motion_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        fingerprint = fingerprint_motion(path)
        item["fingerprint"] = fingerprint.as_dict()
        previous = seen_hashes.get(fingerprint.tensor_sha256)
        if previous is not None:
            duplicate = {
                "id": str(item["id"]),
                "motion_path": str(item["motion_path"]),
                "source": str(item.get("source", item.get("split", ""))),
                "duplicate_of_id": previous,
                "tensor_sha256": fingerprint.tensor_sha256,
                "reason": "exact_tensor_duplicate_excluded_from_effective_evaluation",
            }
            if not drop_duplicates:
                raise ValueError(f"duplicate motion tensor in one split: {previous} and {item['id']}")
            excluded_duplicates.append(duplicate)
            continue
        seen_hashes[fingerprint.tensor_sha256] = str(item["id"])
        unique_items.append(item)
    result["items"] = unique_items
    result["excluded_duplicate_rows"] = excluded_duplicates
    if mbench_root is not None:
        mbench_hashes = _mbench_hashes(mbench_root)
        overlap = sorted(set(seen_hashes) & mbench_hashes)
        if overlap:
            raise ValueError(f"representation split overlaps MBench motion tensors: {overlap[:5]}")
    result["materialized"] = True
    result["motion_root"] = str(motion_root)
    return result


def validate_materialized_overlap(manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Check tensor-content hashes across all three materialized splits."""

    owners: dict[str, tuple[str, str]] = {}
    counts: dict[str, int] = {}
    for split_name, manifest in manifests.items():
        rows = manifest.get("items", [])
        counts[split_name] = 0
        for item in rows:
            fingerprint = item.get("fingerprint")
            if not isinstance(fingerprint, Mapping) or not fingerprint.get("tensor_sha256"):
                raise ValueError(f"{split_name} item {item.get('id')} has no tensor fingerprint")
            digest = str(fingerprint["tensor_sha256"])
            owner = (split_name, str(item.get("id")))
            previous = owners.get(digest)
            if previous is not None and previous != owner:
                raise ValueError(
                    "duplicate motion tensor across representation splits: "
                    f"{previous[0]}/{previous[1]} and {split_name}/{item.get('id')}"
                )
            owners[digest] = owner
            counts[split_name] += 1
    return {"unique_tensor_hashes": len(owners), "counts": counts}


def _mbench_hashes(root: Path) -> set[str]:
    hashes: set[str] = set()
    for path in sorted(root.rglob("*.pt")):
        try:
            fingerprint = fingerprint_motion(path)
        except ValueError:
            continue
        hashes.add(fingerprint.tensor_sha256)
    return hashes


def validate_split_manifests(
    manifests: Mapping[str, Mapping[str, Any]], *, require_frozen: bool = True
) -> dict[str, Any]:
    expected_names = {"representation_dev_v1", "representation_val_v1", "representation_test_v1"}
    if set(manifests) != expected_names:
        raise ValueError(f"expected split names {sorted(expected_names)}, got {sorted(manifests)}")
    ids: dict[str, str] = {}
    paths: dict[str, str] = {}
    sources_by_split = {
        "representation_dev_v1": set(DEV_SOURCES),
        "representation_val_v1": set(VAL_SOURCES),
        "representation_test_v1": set(TEST_SOURCES),
    }
    counts: dict[str, int] = {}
    for name, manifest in manifests.items():
        if require_frozen and manifest.get("status") != "FROZEN":
            raise ValueError(f"{name} is not FROZEN")
        rows = manifest.get("items")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{name} has no items")
        allowed_sources = sources_by_split[name]
        for item in rows:
            source = str(item.get("source", item.get("split", "")))
            sample_id = str(item.get("id", ""))
            path = str(item.get("motion_path", ""))
            if source not in allowed_sources:
                raise ValueError(f"{name} contains source {source!r}")
            if sample_id in ids:
                raise ValueError(f"sample id overlaps {ids[sample_id]} and {name}: {sample_id}")
            if path in paths:
                raise ValueError(f"motion path overlaps {paths[path]} and {name}: {path}")
            ids[sample_id] = name
            paths[path] = name
        counts[name] = len(rows)
    expected_counts = {
        "representation_val_v1": sum(EXPECTED_OPTICAL_COUNTS[source] for source in VAL_SOURCES),
        "representation_test_v1": sum(EXPECTED_OPTICAL_COUNTS[source] for source in TEST_SOURCES),
    }
    for name, expected in expected_counts.items():
        excluded_count = len(manifests[name].get("excluded_duplicate_rows", []))
        if counts[name] + excluded_count != expected:
            raise ValueError(
                f"{name} expected {expected} source rows including excluded duplicates, "
                f"got {counts[name]} active + {excluded_count} excluded"
            )
    if counts["representation_dev_v1"] <= 0:
        raise ValueError("development split must not be empty")
    return {"status": "VALID", "counts": counts, "unique_ids": len(ids), "unique_paths": len(paths)}


def freeze_manifests(
    manifests: Mapping[str, Mapping[str, Any]], *, source_json: Path, seed: int
) -> dict[str, dict[str, Any]]:
    """Freeze manifests after validation; no data-dependent tuning occurs here."""

    validate_split_manifests(manifests, require_frozen=False)
    source_hash = hashlib.sha256(source_json.read_bytes()).hexdigest()
    frozen: dict[str, dict[str, Any]] = {}
    for name, raw in manifests.items():
        manifest = json.loads(json.dumps(raw))
        manifest.update(
            {
                "status": "FROZEN",
                "protocol_version": PROTOCOL_VERSION,
                "source_annotation": str(source_json),
                "source_annotation_sha256": source_hash,
                "selection_seed": seed,
                "frozen_rule": "source-disjoint; no post-hoc sample removal; final test is never used for tuning",
                "effective_item_count": len(manifest.get("items", [])),
                "excluded_duplicate_count": len(manifest.get("excluded_duplicate_rows", [])),
                "source_coverage_count": len(manifest.get("items", [])) + len(manifest.get("excluded_duplicate_rows", [])),
                "deduplication_policy": "keep_first_tensor_hash_by_stable_manifest_order; record later exact duplicates; never silently drop",
            }
        )
        frozen[name] = manifest
    validate_split_manifests(frozen, require_frozen=True)
    return frozen


def write_manifests(manifests: Mapping[str, Mapping[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, manifest in manifests.items():
        path = output_dir / f"{name}.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "ALL_OPTICAL_SOURCES",
    "DEV_SOURCES",
    "EXPECTED_OPTICAL_COUNTS",
    "MotionFingerprint",
    "PROTOCOL_VERSION",
    "TEST_SOURCES",
    "VAL_SOURCES",
    "build_split_manifests",
    "build_unique_dev_manifest",
    "fingerprint_motion",
    "freeze_manifests",
    "inventory",
    "length_bucket",
    "materialize_manifest",
    "validate_materialized_overlap",
    "validate_annotation_inventory",
    "validate_split_manifests",
    "write_manifests",
]
