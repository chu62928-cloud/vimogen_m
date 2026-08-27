import json
from pathlib import Path

from scripts.build_phase3_devset_inputs import build_input


def test_builds_text_only_input_without_reference_motion(tmp_path: Path):
    source = tmp_path / "source.json"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "inputs.json"
    source.write_text(json.dumps([{"id": i, "motion_path": f"motions/{i}.pt"} for i in range(20)]), encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "status": "FROZEN_SINGLE_REVIEW_OVERRIDE",
                    "items": [
                    {"id": i, "dev_index": i, "category": "a", "motion_text_annot": f"walk forward {i}"}
                    for i in range(20)
                ],
            }
        ),
        encoding="utf-8",
    )
    metadata = build_input(manifest, source, output)
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert metadata["condition"] == "text_only"
    assert len(rows) == 20
    assert all(row["use_ref_motion"] is False for row in rows)
    assert all(row["sample_id"] == str(row["global_id"]) for row in rows)
    assert output.with_suffix(output.suffix + ".meta.json").exists()
