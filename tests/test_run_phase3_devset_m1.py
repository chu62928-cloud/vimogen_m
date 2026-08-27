from pathlib import Path

from scripts.run_phase3_devset_m1 import build_config


def test_window_mid_m1_config_is_explicit(tmp_path: Path):
    cfg = build_config(tmp_path / "inputs.json", 1, 10.0, tmp_path / "run", tmp_path / "cache")
    assert cfg.m1.enabled is True
    assert cfg.m1.target_delta_deg == 10.0
    assert cfg.m1.sigma_min == 0.25
    assert cfg.m1.sigma_max == 0.65
    assert cfg.m1.max_correction_rms == 0.05
    assert cfg.m1.heading_mode == "canonical_y"
    assert cfg.m0.batch_invariant is True
