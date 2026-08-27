import json
from pathlib import Path

from scripts.prepare_phase3_devset import build_manifest


def test_candidate_manifest_is_not_frozen_and_has_protocol(tmp_path: Path):
    source = tmp_path / "data.json"
    items = []
    templates = {
        "straight_walk": "A person walks forward steadily.",
        "turning_walk": "A person walks forward and turns left.",
        "speed_walk": "A person walks quickly forward.",
        "arms_while_walking": "A person walks forward while waving both arms.",
        "stop_and_walk": "A person starts walking, pauses, and then walks forward.",
    }
    item_id = 0
    for text in templates.values():
        for suffix in range(2):
            items.append({"id": item_id, "motion_path": f"motions/{item_id}.pt", "motion_text_annot": f"{text} {suffix}"})
            item_id += 1
    items.append({"id": item_id, "motion_path": "motions/bad.pt", "motion_text_annot": "A person runs and sits on a chair."})
    source.write_text(json.dumps(items), encoding="utf-8")

    manifest = build_manifest(source, tmp_path / "out", per_category=1, seed=20260813)

    assert manifest["status"] == "CANDIDATE_NOT_FROZEN"
    assert manifest["target_size"] == 5
    assert manifest["formal_protocol"]["units"] == 45
    assert all(item["selection_status"] == "CANDIDATE_NOT_FROZEN" for item in manifest["items"])
    assert (tmp_path / "out" / "candidate_manifest.json").exists()
    assert (tmp_path / "out" / "candidate_manifest.csv").exists()


def test_excluded_actions_do_not_enter_pool(tmp_path: Path):
    source = tmp_path / "data.json"
    source.write_text(
        json.dumps([
            {"id": 1, "motion_path": "bad.pt", "motion_text_annot": "A person walks and then sits on a chair."},
            {"id": 2, "motion_path": "good.pt", "motion_text_annot": "A person walks slowly forward."},
        ]),
        encoding="utf-8",
    )
    try:
        build_manifest(source, tmp_path / "out", per_category=1, seed=1)
    except RuntimeError as exc:
        assert "category" in str(exc)
    else:
        raise AssertionError("insufficient category candidates should fail loudly")
