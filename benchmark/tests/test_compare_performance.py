import sys
import unittest
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from compare_performance import (  # noqa: E402
    parse_measurements,
    report_datafusion_summary,
    report_summary,
    summarize_timings,
)


class ParseMeasurementsTest(unittest.TestCase):
    def test_parses_bloom_only_rows(self):
        output = "\n".join(
            [
                "query\tbloom_plan_ms\tbloom_exec_ms\tbloom_total_ms\tfull_rows\trow_locations\tdirect\trows",
                "1a\t1.000\t2.000\t3.000\t2\t0\t0\t1",
                "TOTAL\tbloom_median_sum_ms=3.000",
            ]
        )
        self.assertEqual(parse_measurements(output, "bloom"), {"1a": (0.003, 1)})

    def test_parses_datafusion_only_rows(self):
        output = "\n".join(
            [
                "query\tbaseline_plan_ms\tbaseline_exec_ms\tbaseline_total_ms\trows",
                "1a\t1.000\t2.000\t3.000\t1",
            ]
        )
        self.assertEqual(
            parse_measurements(output, "datafusion"), {"1a": (0.003, 1)}
        )

    def test_rejects_duplicate_rows(self):
        row = "1a\t1.000\t2.000\t3.000\t2\t0\t0\t1"
        with self.assertRaisesRegex(RuntimeError, "duplicate timing row"):
            parse_measurements(f"{row}\n{row}", "bloom")

    def test_rejects_missing_rows(self):
        with self.assertRaisesRegex(RuntimeError, "no bloom timing rows"):
            parse_measurements("unrelated output", "bloom")


class SummarizeTimingsTest(unittest.TestCase):
    def make_timings(self, workload, value):
        count = 113 if workload == "job" else 22
        timings = defaultdict(
            list,
            {f"q{index:03}": [value] * 3 for index in range(count)},
        )
        rows = {query: 1 for query in timings}
        return timings, rows

    def test_reports_query_regressions(self):
        base, base_rows = self.make_timings("tpch_sf1", 1.0)
        candidate, candidate_rows = self.make_timings("tpch_sf1", 1.0)
        candidate["q000"] = [1.2] * 3

        summary = summarize_timings(
            "tpch_sf1",
            base,
            candidate,
            base_rows,
            candidate_rows,
            3,
            1.10,
        )

        self.assertEqual(len(summary["query_regressions"]), 1)
        self.assertGreater(summary["geomean_ratio"], 1.0)

    def test_rejects_missing_queries(self):
        base, base_rows = self.make_timings("job", 1.0)
        candidate, candidate_rows = self.make_timings("job", 1.0)
        candidate.pop("q000")
        candidate_rows.pop("q000")

        with self.assertRaisesRegex(RuntimeError, "query-set mismatch"):
            summarize_timings(
                "job",
                base,
                candidate,
                base_rows,
                candidate_rows,
                3,
                1.10,
            )

    def test_rejects_row_count_changes(self):
        base, base_rows = self.make_timings("tpch_sf1", 1.0)
        candidate, candidate_rows = self.make_timings("tpch_sf1", 1.0)
        candidate_rows["q000"] = 2

        with self.assertRaisesRegex(RuntimeError, "row-count mismatch"):
            summarize_timings(
                "tpch_sf1",
                base,
                candidate,
                base_rows,
                candidate_rows,
                3,
                1.10,
            )

    def test_geomean_gate_uses_ratio_or_absolute_delta(self):
        cases = [
            (1.0, 1.11, True),
            (1.0, 1.06, True),
            (1.0, 1.04, False),
        ]
        for base, candidate, expected in cases:
            with self.subTest(candidate=candidate), redirect_stdout(StringIO()):
                summary = {
                    "workload": "tpch_sf1",
                    "base_geomean": base,
                    "candidate_geomean": candidate,
                    "geomean_ratio": candidate / base,
                    "base_total": base * 22,
                    "candidate_total": candidate * 22,
                    "query_regressions": [],
                }
                self.assertEqual(
                    report_summary(summary, 1.10, 0.050, False), expected
                )

    def test_datafusion_gate_requires_relative_and_absolute_slowdown(self):
        cases = [
            (1.0, 1.11, True),
            (1.0, 1.04, False),
            (0.010, 0.012, False),
        ]
        for datafusion, bloom, expected in cases:
            with self.subTest(bloom=bloom), redirect_stdout(StringIO()):
                summary = {
                    "workload": "tpch_sf1",
                    "base_geomean": datafusion,
                    "candidate_geomean": bloom,
                    "geomean_ratio": bloom / datafusion,
                    "base_total": datafusion * 22,
                    "candidate_total": bloom * 22,
                    "query_regressions": [],
                }
                self.assertEqual(
                    report_datafusion_summary(summary, 1.10, 0.050, False),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
