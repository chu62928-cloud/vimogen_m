from pathlib import Path

from scripts.run_phase3_devset_m0 import build_config


def test_m0_config_is_text_only_sample_v1_and_disabled_m1(tmp_path: Path):
    cfg = build_config(tmp_path / "inputs.json", 2, tmp_path / "run", tmp_path / "cache", batch_size=4)
    assert cfg.experiment.global_seed == 2
    assert cfg.dataloader.test_local_batch == 4
    assert cfg.dataset.text_key == "prompt"
    assert cfg.m0.noise_protocol == "sample_v1"
    assert cfg.m0.batch_invariant is True
    assert cfg.m1.enabled is False
