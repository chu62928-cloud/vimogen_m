import json

from scripts.build_m1_nonturning_holdout import EXPLICIT_STOP, build_candidate


def test_holdout_candidate_excludes_turning_and_development_ids(tmp_path):
    texts = [
        (1, "a person walks five steps forward"),
        (2, "the athlete strides forward repeatedly"),
        (3, "a person slowly walks forward"),
        (4, "a person quickly walks forward"),
        (5, "a person walks forward and waves arms"),
        (6, "a person walks forward and raises hands"),
        (7, "a person starts walking and then stops"),
        (8, "a person begins walking and pauses"),
        (9, "a person turns left while walking"),
    ]
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps([
            {"id": item_id, "motion_path": f"motions/{item_id}.pt", "motion_text_annot": text}
            for item_id, text in texts
        ]),
        encoding="utf-8",
    )
    for item_id, _ in texts:
        motion_path = tmp_path / "motions" / f"{item_id}.pt"
        motion_path.parent.mkdir(parents=True, exist_ok=True)
        motion_path.write_bytes(b"test")
    development = tmp_path / "development.json"
    development.write_text(json.dumps({"items": [{"id": 1, "normalised_text": texts[0][1]}]}), encoding="utf-8")
    output = tmp_path / "holdout"
    manifest = build_candidate(
        source=source,
        development_manifest=development,
        output_dir=output,
        per_category=1,
        seed=1,
        motion_root=tmp_path,
    )
    assert manifest["status"] == "HOLDOUT_CANDIDATE_NOT_FROZEN"
    assert len(manifest["items"]) == 4
    assert set(item["category"] for item in manifest["items"]) == {
        "straight_walk", "speed_walk", "arms_while_walking", "stop_and_walk"
    }
    assert all(item["id"] != 1 for item in manifest["items"])
    assert all(item["category"] != "turning_walk" for item in manifest["items"])
    assert all(
        EXPLICIT_STOP.search(item["normalised_text"])
        for item in manifest["items"]
        if item["category"] == "stop_and_walk"
    )
