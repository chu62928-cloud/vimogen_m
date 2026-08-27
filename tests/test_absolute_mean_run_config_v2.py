"""Configuration and safety gates for v2 generation units."""

import json
from pathlib import Path

import pytest

from scripts import run_absolute_mean_pelvis_v2 as runner


def test_v2_build_config_selects_full_fk_sagittal_protocol(tmp_path: Path):
    config = runner.build_config(
        split="development",
        target_mean_deg=5.0,
        seed=0,
        guidance_strength=1.0,
        shape_weight=0.1,
        run_root=tmp_path / "run",
        noise_cache=tmp_path / "noise",
        batch_size=2,
    )
    assert config.absolute_mean_pelvis.protocol == runner.PROTOCOL_NAME
    assert config.absolute_mean_pelvis.fusion_window == 9
    assert config.absolute_mean_pelvis.anchor_weight == 1.0
    assert config.absolute_mean_pelvis.sigma_min == 0.25
    assert config.absolute_mean_pelvis.sigma_max == 0.65
    assert config.m1.enabled is False
    assert config.representation.reconciliation.enabled is False


def test_v2_build_config_rejects_values_outside_frozen_grid(tmp_path: Path):
    with pytest.raises(ValueError, match="frozen development grid"):
        runner.build_config(
            split="development",
            target_mean_deg=5.0,
            seed=0,
            guidance_strength=3.0,
            shape_weight=0.1,
            run_root=tmp_path / "run",
            noise_cache=tmp_path / "noise",
            batch_size=1,
        )


def test_v2_runner_refuses_while_official_v3_is_running(tmp_path: Path, monkeypatch):
    state = tmp_path / "scheduler_state.json"
    state.write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
    monkeypatch.setattr(runner, "V3_STATE", state)
    with pytest.raises(RuntimeError, match="MBench v3 is RUNNING"):
        runner.assert_v3_not_running()
    state.write_text(json.dumps({"status": "USER_STOPPED_ARCHIVED"}), encoding="utf-8")
    runner.assert_v3_not_running()
