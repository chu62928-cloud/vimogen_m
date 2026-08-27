from __future__ import annotations

import json

import numpy as np
import torch

import evaluation.mbench_threeway as threeway
from evaluation.mbench_threeway_stats import holm_adjust, summarize_drift, summarize_official_mbench


def test_raw_channel_diagnostics_detects_known_mismatch():
    motion = torch.zeros(5, 276)
    motion[:, 126] = torch.arange(5, dtype=torch.float32)
    motion[:, 192] = 0.5
    metrics = threeway.raw_channel_diagnostics(motion)
    assert np.isclose(
        metrics["raw_position_velocity_residual_median_m_per_frame"],
        0.5 / 22.0,
    )
    assert metrics["raw_integrated_separation_endpoint_m"] > 0.0


def test_holm_adjust_is_monotone_and_bounded():
    values = holm_adjust([0.001, 0.02, 0.5])
    assert values[0] <= values[1] <= values[2]
    assert all(value is not None and 0.0 <= value <= 1.0 for value in values)


def test_organize_manifest_uses_one_source_for_all_methods(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    sample_dir = input_dir / "7"
    sample_dir.mkdir(parents=True)
    source = torch.zeros(4, 276)
    source[:, 126] = torch.arange(4, dtype=torch.float32)
    source_path = sample_dir / "motion_gen_condition_on_text.pt"
    torch.save(source, source_path)

    monkeypatch.setattr(
        threeway,
        "recover_motion_variants",
        lambda motion: {method: motion.clone() for method in threeway.METHODS},
    )
    monkeypatch.setattr(
        threeway,
        "motion_to_joints",
        lambda motion: np.zeros((motion.shape[0], 22, 3), dtype=np.float32),
    )

    output = tmp_path / "output"
    payload = threeway.organize_directory(
        input_dir,
        output,
        condition="m0",
        seed=0,
        expected_count=1,
    )
    assert payload["status"] == "VALID"
    record = payload["records"][0]
    assert len({record["source_file_sha256"]}) == 1
    assert set(record["methods"]) == set(threeway.METHODS)
    assert all((output / method / "7.npy").exists() for method in threeway.METHODS)
    saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert saved["record_count"] == 1


def test_drift_summary_is_reproducible(tmp_path):
    records = []
    for seed in (0, 1, 2):
        records.append({
            "sample_id": "0",
            "condition": "m0",
            "seed": seed,
            "methods": {
                method: {
                    "anchor_deviation_mean_m": float(index),
                    "anchor_deviation_endpoint_m": float(index),
                    "root_endpoint_deviation_m": float(index),
                    "anchor_drift_auc_m": float(index),
                    "anchor_drift_slope_m_per_frame": float(index),
                    "raw_position_velocity_residual_mean_m_per_frame": 0.1,
                    "raw_integrated_separation_endpoint_m": 0.2,
                    "raw_integrated_separation_auc_m": 0.3,
                }
                for index, method in enumerate(threeway.METHODS)
            },
        })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"condition": "m0", "records": records}), encoding="utf-8")
    first = summarize_drift([manifest], bootstrap_repetitions=100, bootstrap_seed=1)
    second = summarize_drift([manifest], bootstrap_repetitions=100, bootstrap_seed=1)
    assert first == second


def test_official_summary_collapses_seeds_and_is_reproducible(tmp_path):
    records = []
    dimensions = [
        "Jitter_Degree", "Ground_Penetration", "Foot_Floating",
        "Foot_Sliding", "Dynamic_Degree",
    ]
    for method_index, method in enumerate(threeway.METHODS):
        for seed in (0, 1, 2):
            per_motion = tmp_path / f"{method}_{seed}_per_motion_results.json"
            per_motion.write_text(json.dumps({
                "motions": [{
                    "id": 0,
                    "dimensions": {
                        dimension: {"value": float(method_index + seed / 10.0)}
                        for dimension in dimensions
                    },
                }]
            }), encoding="utf-8")
            record = tmp_path / f"{method}_{seed}_run_record.json"
            record.write_text(json.dumps({
                "status": "COMPLETED",
                "condition": "m0",
                "method": method,
                "seed": seed,
                "per_motion_path": str(per_motion),
            }), encoding="utf-8")
            records.append(record)
    first = summarize_official_mbench(records, bootstrap_repetitions=100, bootstrap_seed=3)
    second = summarize_official_mbench(records, bootstrap_repetitions=100, bootstrap_seed=3)
    assert first == second
    metric = first["conditions"]["m0"]["metrics"]["Jitter_Degree"]
    assert metric["sample_count"] == 1
    assert metric["method_summary"]["reconciled"]["median"] == 2.1
