from scripts.validate_phase3_devset import build_matrix, validate_manifest


def _manifest(status="CANDIDATE_NOT_FROZEN"):
    items = []
    for category in ("a", "b"):
        for index in range(2):
            items.append(
                {
                    "id": len(items),
                    "dev_index": len(items),
                    "category": category,
                    "normalised_text": f"text {category} {index}",
                    "manual_review": {"reviewer_a": "PENDING", "reviewer_b": "PENDING", "adjudication": "PENDING"},
                }
            )
    return {"status": status, "categories": ["a", "b"], "items_per_category": 2, "items": items}


def test_candidate_matrix_is_preview_only():
    matrix = build_matrix(_manifest())
    assert matrix["status"] == "PREVIEW_NOT_FORMAL"
    assert matrix["target_units"] == 36
    assert matrix["do_not_run_until_manifest_frozen"] is True


def test_frozen_requires_all_reviews():
    manifest = _manifest(status="FROZEN")
    assert any("incomplete review" in error for error in validate_manifest(manifest, require_frozen=True))
    for item in manifest["items"]:
        item["manual_review"] = {"reviewer_a": "A", "reviewer_b": "B", "adjudication": "A"}
    assert validate_manifest(manifest, require_frozen=True) == []
    assert build_matrix(manifest)["status"] == "FROZEN_MATRIX_READY"


def test_user_authorized_single_review_override_is_formal_but_distinct():
    manifest = _manifest(status="FROZEN_SINGLE_REVIEW_OVERRIDE")
    for item in manifest["items"]:
        item["manual_review"] = {
            "reviewer_a": "USER_REVIEWER1",
            "reviewer_b": "IGNORED_BY_USER",
            "adjudication": "USER_SINGLE_REVIEW_OVERRIDE",
        }
    assert validate_manifest(manifest, require_frozen=True) == []
    matrix = build_matrix(manifest)
    assert matrix["status"] == "FROZEN_MATRIX_READY"
    assert matrix["do_not_run_until_manifest_frozen"] is False
