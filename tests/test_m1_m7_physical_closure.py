from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from geometry.contacts import freeze_marker_contact_evidence
from evaluation.physical_metrics import (
    EVALUATED_PASS,
    NOT_EVALUATED,
    evaluate_physical_metrics_v3,
)
from evaluation.physical_reference import (
    MARKER_JOINTS,
    materialize_reference,
)
from motion_rep.phase1 import MOTION_LAYOUT, encode_rot6d
from experiments.build_s0_physical_table import build_rows
from scripts.calibrate_physical_thresholds import freeze_physical_thresholds
from scripts.freeze_s0_v1 import collect_sequence_records, freeze_s0_manifest


def _motion(batch: int = 1, frames: int = 5) -> torch.Tensor:
    motion = torch.zeros((batch, frames, MOTION_LAYOUT.total_dim), dtype=torch.float32)
    identity = torch.eye(3).expand(batch, frames, 21, 3, 3)
    motion[..., MOTION_LAYOUT.body_pose] = encode_rot6d(identity).reshape(
        batch, frames, 126
    )
    motion[..., MOTION_LAYOUT.root_rotation] = encode_rot6d(
        torch.eye(3).expand(batch, frames, 3, 3)
    )
    return motion


def _markers(frames: int = 6) -> dict[str, dict[str, torch.Tensor]]:
    zero = torch.zeros((1, frames, 3), dtype=torch.float32)
    return {
        "left": {"heel": zero.clone(), "toe": zero.clone()},
        "right": {"heel": zero.clone(), "toe": zero.clone()},
    }


def test_marker_contact_does_not_promote_an_airborne_toe_from_heel_contact() -> None:
    frames = 6
    markers = _markers(frames)
    markers["left"]["toe"][..., 2] = 0.10
    valid = torch.ones((1, frames), dtype=torch.bool)

    evidence = freeze_marker_contact_evidence(markers, valid, torch.zeros(1))

    assert evidence.contact["left"]["heel"][0, 1:].all()
    assert not evidence.contact["left"]["toe"].any()


def test_reference_uses_authoritative_fk_joint_markers_and_freezes_metadata() -> None:
    motion = _motion()
    valid = torch.ones((1, motion.shape[1]), dtype=torch.bool)
    reference = materialize_reference(
        motion, valid, sample_ids=["94"], seeds=[0]
    )

    assert reference["marker_joints"] == MARKER_JOINTS
    assert reference["marker_names"] == [
        "left_heel",
        "left_toe",
        "right_heel",
        "right_toe",
    ]
    assert reference["unit"] == "m"
    assert reference["world_up_axis"] == 2
    assert reference["m0_marker_world"].shape == (1, 4, motion.shape[1], 3)
    assert reference["valid_frame_mask"].dtype is torch.bool
    assert reference["contact_mask"].shape == (1, 4, motion.shape[1])
    assert reference["ground_plane"].shape == (1, 4)


def test_candidate_cannot_escape_by_redefining_m0_contact_mask() -> None:
    frames = 6
    baseline = _markers(frames)
    candidate = _markers(frames)
    # The candidate is lifted and moved while the M0 contact labels remain
    # frozen.  These frames must still contribute to floating/support/sliding.
    candidate["left"]["heel"][..., 0] = torch.arange(frames) * 0.02
    candidate["left"]["toe"][..., 0] = torch.arange(frames) * 0.02
    candidate["left"]["heel"][..., 2] = 0.05
    candidate["left"]["toe"][..., 2] = 0.05
    valid = torch.ones((1, frames), dtype=torch.bool)
    left = torch.tensor([[False, True, True, True, True, True]])
    empty = torch.zeros((1, frames), dtype=torch.bool)
    contacts = {
        "left": {"heel": left, "toe": left.clone()},
        "right": {"heel": empty, "toe": empty.clone()},
    }
    pairs = {
        side: {
            marker: mask[:, 1:] & mask[:, :-1]
            for marker, mask in side_masks.items()
        }
        for side, side_masks in contacts.items()
    }

    result = evaluate_physical_metrics_v3(
        candidate,
        baseline,
        valid,
        contacts,
        torch.zeros(1),
        contact_pair_masks=pairs,
    )
    row = result["per_sequence"][0]
    assert row["contact_evaluable"] is True
    assert row["floating_frame_rate"] is None
    assert row["contact_height_p95_mm"] > 0.0
    assert row["total_slide_distance_mm"] > 0.0
    assert row["support_height_error_p95_mm"] > 0.0


def test_raw_physical_metrics_are_not_a_gate_pass_without_frozen_thresholds() -> None:
    frames = 6
    markers = _markers(frames)
    valid = torch.ones((1, frames), dtype=torch.bool)
    left = torch.tensor([[False, True, True, True, True, True]])
    empty = torch.zeros((1, frames), dtype=torch.bool)
    contacts = {
        "left": {"heel": left, "toe": left.clone()},
        "right": {"heel": empty, "toe": empty.clone()},
    }
    pairs = {
        side: {
            marker: mask[:, 1:] & mask[:, :-1]
            for marker, mask in side_masks.items()
        }
        for side, side_masks in contacts.items()
    }
    raw = evaluate_physical_metrics_v3(
        markers,
        markers,
        valid,
        contacts,
        torch.zeros(1),
        contact_pair_masks=pairs,
    )
    assert raw["status"] == NOT_EVALUATED
    assert raw["reason"] == "THRESHOLDS_NOT_FROZEN"
    assert raw["per_sequence"][0]["penetration_frame_rate"] is None
    assert raw["per_sequence"][0]["floating_frame_rate"] is None

    thresholds = {
        "penetration_tolerance_mm": 1.0,
        "floating_height_threshold_mm": 25.0,
        "penetration_p95_mm": 1.0,
        "penetration_max_mm": 1.0,
        "penetration_frame_rate": 0.0,
        "contact_tangent_speed_p95_mm_per_frame": 1.0,
        "contact_tangent_speed_max_mm_per_frame": 1.0,
        "total_slide_distance_mm": 1.0,
        "max_segment_slide_distance_mm": 1.0,
        "contact_height_p95_mm": 1.0,
        "contact_height_max_mm": 1.0,
        "floating_frame_rate": 0.0,
        "support_height_error_p95_mm": 1.0,
    }
    judged = evaluate_physical_metrics_v3(
        markers,
        markers,
        valid,
        contacts,
        torch.zeros(1),
        contact_pair_masks=pairs,
        thresholds=thresholds,
    )
    assert judged["status"] == EVALUATED_PASS
    assert judged["physical_pass"] is True


def test_threshold_freeze_uses_only_m0_and_rejects_synthetic_failures(
    tmp_path: Path,
) -> None:
    motion = _motion(frames=8)
    valid = torch.ones((1, motion.shape[1]), dtype=torch.bool)
    reference = materialize_reference(
        motion, valid, sample_ids=["94"], seeds=[0], code_commit="abc123"
    )
    reference_path = tmp_path / "physical_reference.pt"
    torch.save(reference, reference_path)
    output = tmp_path / "thresholds"

    protocol = freeze_physical_thresholds(
        reference_path=reference_path,
        output=output,
        code_commit="abc123",
    )

    assert protocol["status"] == "FROZEN_PHYSICAL_THRESHOLDS"
    assert protocol["calibration_sources"] == ["PAIRED_M0_SELF_EVALUATION"]
    assert protocol["candidate_results_read"] is False
    assert protocol["synthetic_sanity"]["all_expected_failures_observed"] is True
    assert set(protocol["synthetic_sanity"]["cases"]) == {
        "penetration_20mm",
        "floating_50mm",
        "sliding_50mm_per_frame",
    }
    with pytest.raises(FileExistsError):
        freeze_physical_thresholds(
            reference_path=reference_path,
            output=output,
            code_commit="abc123",
        )


def test_physical_table_requires_and_summarizes_all_84_records() -> None:
    preliminary = {
        "methods": [
            {
                "method": f"M{method}",
                "label": f"M{method}",
                "sequence_count": 12,
            }
            for method in range(1, 8)
        ]
    }
    records = []
    for method in range(1, 8):
        for index in range(12):
            passed = index == 0
            records.append(
                {
                    "method": f"M{method}",
                    "physical": {
                        "status": "EVALUATED_PASS" if passed else "EVALUATED_FAIL",
                        "per_sequence": [
                            {
                                "penetration_p95_mm": 0.0,
                                "contact_tangent_speed_p95_mm_per_frame": 0.0,
                                "support_height_error_p95_mm": 0.0,
                                "physical_fail_reasons": []
                                if passed
                                else ["support_height_error_p95_mm_fail"],
                            }
                        ],
                    },
                }
            )
    rows = build_rows(
        preliminary,
        {"status": "S0_PHYSICAL_EVALUATED", "records": records},
    )

    assert len(rows) == 7
    assert all(row["physical"]["pass_count"] == 1 for row in rows)
    assert all(row["physical"]["fail_count"] == 11 for row in rows)


def test_freeze_s0_manifest_requires_complete_unique_84_records_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    records = []
    for method_index in range(1, 8):
        for seed in (0, 42):
            for dose in (-2.0, 0.0, 2.0):
                for prompt_id in ("94", "34122"):
                    records.append(
                        {
                            "run_id": f"M{method_index}_s{seed}_p{prompt_id}_d{dose:+g}",
                            "method_name": f"M{method_index}",
                            "prompt_id": prompt_id,
                            "seed": seed,
                            "target_dose_deg": dose,
                            "baseline_motion_id": f"m0-{seed}-{prompt_id}",
                            "noise_cache_id": f"noise-{seed}-{prompt_id}",
                            "code_commit": "3968ace",
                            "evaluator_version": "m1_m7_evaluator_v1",
                            "all_metrics": {},
                        }
                    )
    source = tmp_path / "records.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    output = tmp_path / "frozen"
    manifest = freeze_s0_manifest(
        source_paths=[source], output=output, code_commit="3968ace"
    )
    assert manifest["status"] == "S0_V1_FROZEN"
    assert manifest["sequence_count"] == 84
    with pytest.raises(FileExistsError):
        freeze_s0_manifest(source_paths=[source], output=output, code_commit="3968ace")


def test_progress_record_paths_are_resolved_from_repository_root(tmp_path: Path) -> None:
    sampling = tmp_path / "results" / "phase9" / "pelvis_m1_m7" / "s0_sampling"
    record_path = sampling / "M1" / "seed_000" / "dose_+0" / "attempt_01" / "evaluation" / "run" / "run_record.json"
    record_path.parent.mkdir(parents=True)
    record = {
        "run_id": "M1_s0_p94_d+0",
        "method_name": "M1",
        "prompt_id": "94",
        "seed": 0,
        "target_dose_deg": 0.0,
        "baseline_motion_id": "m0",
        "evaluator_version": "v1",
        "all_metrics": {},
    }
    record_path.write_text(json.dumps(record), encoding="utf-8")
    relative = record_path.relative_to(tmp_path).as_posix()
    progress = {
        "jobs": [
            {
                "method": "M1",
                "seed": 0,
                "dose": 0.0,
                "evaluation": {"status": "COMPLETED", "records": [relative, relative]},
            }
        ]
    }
    (sampling / "s0_matrix_progress.json").write_text(
        json.dumps(progress), encoding="utf-8"
    )
    records = collect_sequence_records([sampling])
    assert list(records) == [("M1", 0, 0.0, "94")]
