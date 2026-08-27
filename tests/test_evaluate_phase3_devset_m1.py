from scripts.evaluate_phase3_devset_m1 import _sample_ids


def test_sample_id_reader(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('[{"sample_id": "a"}, {"sample_id": 2}]', encoding="utf-8")
    assert _sample_ids(path) == ["a", "2"]
