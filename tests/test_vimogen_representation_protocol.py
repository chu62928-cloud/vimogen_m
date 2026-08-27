import json
from pathlib import Path

import pytest
import torch

from evaluation.vimogen_representation_protocol import (
    DEV_SOURCES,
    EXPECTED_OPTICAL_COUNTS,
    build_split_manifests,
    fingerprint_motion,
    freeze_manifests,
    materialize_manifest,
    validate_materialized_overlap,
    validate_split_manifests,
)


def _item(index: int, source: str) -> dict:
    return {
        "id": index,
        "motion_path": f"motions/{index}.pt",
        "motion_text_annot": f"motion {index}",
        "split": source,
        "subset": "Optical MoCap",
    }


def _small_items() -> list[dict]:
    # Patch the release-count guard for the small synthetic fixture by using
    # the public selection function with a monkeypatched validator below.
    rows = []
    index = 0
    for source in (*DEV_SOURCES, "KIT-ML", "LAFAN1", "BEHAVE", "FIT3D", "Mixamo", "HumanSC3D", "ARCTIC", "RICH", "EMDB"):
        for _ in range(4):
            rows.append(_item(index, source))
            index += 1
    return rows


def test_fingerprint_rejects_wrong_dimension(tmp_path: Path):
    path = tmp_path / "bad.pt"
    torch.save(torch.zeros(3, 275), path)
    with pytest.raises(ValueError, match="276"):
        fingerprint_motion(path)


def test_source_disjoint_validation_and_test_sets(monkeypatch):
    items = _small_items()
    monkeypatch.setattr(
        "evaluation.vimogen_representation_protocol.EXPECTED_OPTICAL_COUNTS",
        {source: 4 for source in (*DEV_SOURCES, "KIT-ML", "LAFAN1", "BEHAVE", "FIT3D", "Mixamo", "HumanSC3D", "ARCTIC", "RICH", "EMDB")},
    )
    monkeypatch.setattr(
        "evaluation.vimogen_representation_protocol.validate_annotation_inventory",
        lambda value: {"optical_total": len(value)},
    )
    manifests = build_split_manifests(items, dev_size=4, seed=7)
    summary = validate_split_manifests(manifests, require_frozen=False)
    assert summary["counts"]["representation_dev_v1"] == 4
    assert {row["source"] for row in manifests["representation_val_v1"]["items"]} == {"KIT-ML", "LAFAN1", "BEHAVE"}
    assert {row["source"] for row in manifests["representation_test_v1"]["items"]} == {"FIT3D", "Mixamo", "HumanSC3D", "ARCTIC", "RICH", "EMDB"}
    ids = [row["id"] for manifest in manifests.values() for row in manifest["items"]]
    assert len(ids) == len(set(ids))


def test_freeze_records_source_hash_and_requires_frozen_status(tmp_path: Path, monkeypatch):
    items = _small_items()
    monkeypatch.setattr(
        "evaluation.vimogen_representation_protocol.EXPECTED_OPTICAL_COUNTS",
        {source: 4 for source in (*DEV_SOURCES, "KIT-ML", "LAFAN1", "BEHAVE", "FIT3D", "Mixamo", "HumanSC3D", "ARCTIC", "RICH", "EMDB")},
    )
    monkeypatch.setattr(
        "evaluation.vimogen_representation_protocol.validate_annotation_inventory",
        lambda value: {"optical_total": len(value)},
    )
    manifests = build_split_manifests(items, dev_size=4, seed=11)
    source = tmp_path / "annotation.json"
    source.write_text(json.dumps(items), encoding="utf-8")
    frozen = freeze_manifests(manifests, source_json=source, seed=11)
    assert all(manifest["status"] == "FROZEN" for manifest in frozen.values())
    assert all(manifest["source_annotation_sha256"] for manifest in frozen.values())
    validate_split_manifests(frozen, require_frozen=True)


def test_materialize_rejects_duplicate_tensor_across_one_manifest(tmp_path: Path):
    value = torch.zeros(3, 276)
    for index in (1, 2):
        torch.save(value, tmp_path / f"{index}.pt")
    manifest = {
        "items": [
            {"id": "1", "motion_path": "1.pt", "source": "AMASS"},
            {"id": "2", "motion_path": "2.pt", "source": "AMASS"},
        ]
    }
    with pytest.raises(ValueError, match="duplicate motion tensor"):
        materialize_manifest(manifest, motion_root=tmp_path)


def test_materialize_can_record_and_exclude_duplicate_tensor(tmp_path: Path):
    value = torch.zeros(3, 276)
    torch.save(value, tmp_path / "1.pt")
    torch.save(value, tmp_path / "2.pt")
    manifest = {
        "items": [
            {"id": "1", "motion_path": "1.pt", "source": "AMASS"},
            {"id": "2", "motion_path": "2.pt", "source": "AMASS"},
        ]
    }
    result = materialize_manifest(manifest, motion_root=tmp_path, drop_duplicates=True)
    assert [item["id"] for item in result["items"]] == ["1"]
    assert result["excluded_duplicate_rows"][0]["id"] == "2"


def test_materialized_overlap_is_checked_across_splits():
    fingerprint = {"frame_count": 3, "dimension": 276, "dtype": "torch.float32", "tensor_sha256": "same"}
    manifests = {
        "representation_dev_v1": {"items": [{"id": "a", "fingerprint": fingerprint}]},
        "representation_val_v1": {"items": [{"id": "b", "fingerprint": fingerprint}]},
        "representation_test_v1": {"items": []},
    }
    with pytest.raises(ValueError, match="across representation splits"):
        validate_materialized_overlap(manifests)
