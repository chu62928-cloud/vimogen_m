from __future__ import annotations

import json
from argparse import Namespace

import pytest

from scripts.run_official_mbench_repair_parallel import resource_blockers
from scripts.run_official_mbench_repair_worker import merge_per_motion, validate_repaired_motions


def write_payload(path, motions):
    path.write_text(json.dumps({"motions": motions}), encoding="utf-8")
    return path


def repaired_motion(motion_id, **extra_dimensions):
    return {
        "id": motion_id,
        "dimensions": {
            "Body_Penetration": {"value": 0.25},
            "Pose_Quality": {"value": 0.5},
            **extra_dimensions,
        },
    }


def test_merge_preserves_official_disjoint_motion_subsets(tmp_path):
    old_path = write_payload(
        tmp_path / "old.json",
        [{"id": 0, "dimensions": {"Jitter_Degree": {"value": 1.0}}}],
    )
    repaired_path = write_payload(tmp_path / "repaired.json", [repaired_motion(150)])
    output_path = tmp_path / "merged.json"

    assert merge_per_motion(old_path, repaired_path, output_path) == 2
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert [motion["id"] for motion in output["motions"]] == [0, 150]
    assert output["motions"][0]["dimensions"] == {"Jitter_Degree": {"value": 1.0}}
    assert set(output["motions"][1]["dimensions"]) == {"Body_Penetration", "Pose_Quality"}
    assert output["repair"]["source_motion_count"] == 1
    assert output["repair"]["repaired_motion_count"] == 1


def test_merge_adds_repaired_dimensions_to_overlapping_motion(tmp_path):
    old_path = write_payload(
        tmp_path / "old.json",
        [{"id": 150, "prompt": "walk", "dimensions": {"Jitter_Degree": {"value": 1.0}}}],
    )
    repaired_path = write_payload(tmp_path / "repaired.json", [repaired_motion(150)])
    output_path = tmp_path / "merged.json"

    assert merge_per_motion(old_path, repaired_path, output_path) == 1
    motion = json.loads(output_path.read_text(encoding="utf-8"))["motions"][0]
    assert motion["prompt"] == "walk"
    assert set(motion["dimensions"]) == {
        "Jitter_Degree", "Body_Penetration", "Pose_Quality",
    }


@pytest.mark.parametrize("invalid_value", [None, float("nan"), float("inf"), True])
def test_merge_rejects_missing_or_nonfinite_repair_values(tmp_path, invalid_value):
    old_path = write_payload(tmp_path / "old.json", [])
    motion = repaired_motion(150)
    motion["dimensions"]["Body_Penetration"]["value"] = invalid_value
    repaired_path = write_payload(tmp_path / "repaired.json", [motion])

    with pytest.raises(RuntimeError, match="missing numeric Body_Penetration"):
        merge_per_motion(old_path, repaired_path, tmp_path / "merged.json")


def test_merge_rejects_duplicate_repaired_ids(tmp_path):
    old_path = write_payload(tmp_path / "old.json", [])
    repaired_path = write_payload(
        tmp_path / "repaired.json",
        [repaired_motion(150), repaired_motion(150)],
    )

    with pytest.raises(RuntimeError, match="duplicate repaired motion id 150"):
        merge_per_motion(old_path, repaired_path, tmp_path / "merged.json")


def test_validate_repaired_motions_rejects_silently_skipped_samples(tmp_path):
    repaired_path = write_payload(tmp_path / "repaired.json", [repaired_motion(150)])

    with pytest.raises(RuntimeError, match="expected 100 repaired motions, found 1"):
        validate_repaired_motions(repaired_path, expected_count=100)


def test_resource_guard_allows_launch_below_all_limits():
    snapshot = {
        "cpu_percent": 50.0,
        "gpu_utilization_percent": 65.0,
        "gpu_memory_percent": 40.0,
        "memory_working_set_percent": 35.0,
        "disk_free_gib": 120.0,
        "gpu_temperature_c": 60.0,
    }
    thresholds = Namespace(
        max_cpu_percent=85.0,
        max_gpu_utilization=90.0,
        max_gpu_memory_percent=70.0,
        max_memory_working_set_percent=75.0,
        min_free_disk_gib=64.0,
        max_gpu_temperature=78.0,
    )

    assert resource_blockers(snapshot, thresholds) == []


def test_resource_guard_blocks_processor_memory_gpu_and_disk_pressure():
    snapshot = {
        "cpu_percent": 90.0,
        "gpu_utilization_percent": 95.0,
        "gpu_memory_percent": 75.0,
        "memory_working_set_percent": 80.0,
        "disk_free_gib": 20.0,
        "gpu_temperature_c": 82.0,
    }
    thresholds = Namespace(
        max_cpu_percent=85.0,
        max_gpu_utilization=90.0,
        max_gpu_memory_percent=70.0,
        max_memory_working_set_percent=75.0,
        min_free_disk_gib=64.0,
        max_gpu_temperature=78.0,
    )

    blockers = resource_blockers(snapshot, thresholds)
    assert len(blockers) == 6
    assert any(blocker.startswith("cpu=") for blocker in blockers)
    assert any(blocker.startswith("gpu_memory=") for blocker in blockers)
    assert any(blocker.startswith("memory=") for blocker in blockers)
