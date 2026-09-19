"""Regression checks for publication-facing result tables."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ResultLedgerTest(unittest.TestCase):
    def test_s0_physical_method_counts_sum_to_totals(self) -> None:
        text = (ROOT / "result.md").read_text(encoding="utf-8")
        section = text.split("## 4. S1 有界配置筛选", 1)[0]
        rows = {}
        pattern = re.compile(
            r"^\| (M[1-7]†?) \| (\d+)/12 \| (\d+)/12 \|$", re.MULTILINE
        )
        for method, v1, v2 in pattern.findall(section):
            rows[method.rstrip("†")] = (int(v1), int(v2))

        self.assertEqual(set(rows), {f"M{i}" for i in range(1, 8)})
        self.assertEqual(sum(v1 for v1, _ in rows.values()), 7)
        self.assertEqual(sum(v2 for _, v2 in rows.values()), 22)
        self.assertIn("| **合计** | **7/84** | **22/84** |", section)


if __name__ == "__main__":
    unittest.main()
