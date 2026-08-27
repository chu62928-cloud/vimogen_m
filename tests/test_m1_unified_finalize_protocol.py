import json
from pathlib import Path

import pytest

from scripts.build_m1_unified_finalize_protocol import build_protocol


def test_protocol_is_reviewer1_only_and_declares_common_boundary(tmp_path: Path):
    manifest = tmp_path / "frozen_manifest.json"
    manifest.write_text(
        json.dumps({
            "status": "FROZEN_SINGLE_REVIEW_OVERRIDE",
            "review_protocol": "reviewer1_only_user_override",
            "source_sha256": "source",
        }),
        encoding="utf-8",
    )
    input_json = tmp_path / "input.json"
    input_json.write_text("[]", encoding="utf-8")
    output = tmp_path / "protocol.json"
    protocol = build_protocol(manifest, input_json, output)
    assert protocol["status"] == "FROZEN_PROTOCOL"
    assert protocol["review_protocol"] == "reviewer1_only_user_override"
    assert protocol["common_finalizer"]["boundary"] == "physical_Tplus1_to_Tx276"
    assert protocol["methods"]["B1"]["status"] == "INTENTIONAL_DIAGNOSTIC_BYPASS"
    assert protocol["noise"]["sample_protocol"] == "vimogen-sample-noise-v1"
    with pytest.raises(FileExistsError):
        build_protocol(manifest, input_json, output)
