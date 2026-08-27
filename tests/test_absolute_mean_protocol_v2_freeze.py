"""Protocol-freeze checks for full-FK sagittal absolute-mean v2."""

import json
from pathlib import Path

import pytest

from scripts import freeze_absolute_mean_pelvis_v2_protocol as freeze_v2


def test_v2_freeze_reuses_v1_data_byte_for_byte_and_refuses_overwrite(tmp_path: Path):
    if not (freeze_v2.V1_ROOT / "protocol.json").is_file():
        pytest.skip("frozen v1 protocol is not available")
    output = tmp_path / "absolute_mean_pelvis_v2"
    protocol = freeze_v2.freeze(output)
    assert protocol["protocol"] == "vimogen_absolute_mean_pelvis_v2_full_fk_sagittal"
    assert protocol["status"] == "FROZEN_BEFORE_V2_MODEL_RUNS"
    assert protocol["data"]["selection_reused_byte_for_byte_from_v1"] is True
    assert protocol["authority_pipeline"][-1] == "pack one internally consistent 276D tensor"
    assert protocol["angle_definition"]["turn_invariance"].startswith("the same local pelvis tilt")
    for filename, expected_hash in freeze_v2.DATA_HASHES.items():
        assert freeze_v2.sha256(output / "data" / filename) == expected_hash
    loaded = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert loaded == protocol
    with pytest.raises(FileExistsError):
        freeze_v2.freeze(output)
