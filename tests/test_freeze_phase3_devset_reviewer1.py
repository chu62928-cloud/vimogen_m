from pathlib import Path

from scripts.freeze_phase3_devset_reviewer1 import _jaccard, parse_reviewer1


def test_reviewer1_parser_and_similarity(tmp_path: Path):
    review = tmp_path / "review.md"
    rows = ["| 序号 | ID | 类别 | 文本 | Reviewer A | Reviewer B | 裁决 | 备注 |", "|---:|---:|---|---|---|---|---|---|"]
    rows.extend(f"| {i} | {100+i} | straight_walk | text {i} | {'KEEP' if i == 0 else 'DROP: reason'} | PENDING | PENDING | |" for i in range(20))
    review.write_text("\n".join(rows), encoding="utf-8")
    decisions = parse_reviewer1(review)
    assert len(decisions) == 20
    assert decisions[100]["reviewer_a"] == "KEEP"
    assert decisions[101]["reviewer_a"].startswith("DROP")
    assert _jaccard("walk forward slowly", "walk forward quickly") >= 0.5
