import json
from pathlib import Path

from scripts.render_phase3_devset_review import render


def test_render_review_form(tmp_path: Path):
    manifest = {
        "status": "CANDIDATE_NOT_FROZEN",
        "items": [{"dev_index": 0, "id": 10, "category": "straight_walk", "motion_text_annot": "walk | forward"}],
    }
    source = tmp_path / "manifest.json"
    output = tmp_path / "review.md"
    source.write_text(json.dumps(manifest), encoding="utf-8")
    content = render(source, output)
    assert "Reviewer A" in content
    assert "walk \\| forward" in content
    assert output.exists()
