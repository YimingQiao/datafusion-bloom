#!/usr/bin/env python3
"""Compare Bloom query performance between a base and candidate runner.

Each round executes the complete workload with base Bloom, candidate Bloom,
and candidate stock DataFusion. Their order rotates between rounds. Per-query
medians are calculated across rounds. The CI gates follow aggregate workload
results; individual slow queries are reported to make a regression easy to
locate.
"""

import argparse
import math
import os
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
WORKLOADS = {
    "job": {
        "queries": 113,
        "query_dir": ROOT / "benchmark" / "job" / "queries",
    },
    "tpch_sf1": {
        "queries": 22,
        "query_dir": ROOT / "benchmark" / "tpch" / "queries",
    },
}


def positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_integer(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-runner", type=Path, required=True)
    parser.add_argument("--candidate-runner", type=Path, required=True)
    parser.add_argument("--job-data-dir", type=Path, required=True)
    parser.add_argument("--tpch-data-dir", type=Path, required=True)
    parser.add_argument(
        "--workload",
        action="append",
        choices=sorted(WORKLOADS),
        dest="workloads",
        help="Workload to compare; repeat to select both (default: both)",
    )
    parser.add_argument(
        "--sampling-mode",
        action="append",
        choices=("prepared", "instant"),
        dest="sampling_modes",
        help="Bloom sampling mode; repeat to select both (default: prepared)",
    )
    parser.add_argument(
        "--instant-parquet-row-groups",
        type=positive_integer,
        default=4,
    )
    parser.add_argument("--threads", type=positive_integer, default=1)
    parser.add_argument("--warmups", type=nonnegative_integer, default=0)
    parser.add_argument("--runs", type=positive_integer, default=1)
    parser.add_argument(
        "--rounds",
        type=positive_integer,
        default=3,
        help="Rotating three-way full-workload rounds (default: 3)",
    )
    parser.add_argument("--regression-ratio", type=float, default=1.10)
    parser.add_argument("--geomean-regression-seconds", type=float, default=0.050)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "performance_regression",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Report regressions without returning a failure status",
    )
    return parser.parse_args()


def parse_measurements(output, mode):
    """Parse single-engine TSV rows from the workload runner."""
    if mode not in {"bloom", "datafusion"}:
        raise ValueError(f"unknown runner mode: {mode}")
    expected_fields = 8 if mode == "bloom" else 5
    measurements = {}
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != expected_fields:
            continue
        try:
            total_seconds = float(fields[3]) / 1000.0
            rows = int(fields[-1])
        except ValueError:
            continue
        query = fields[0]
        if query in measurements:
            raise RuntimeError(f"duplicate timing row for {query}")
        if not math.isfinite(total_seconds) or total_seconds <= 0:
            raise RuntimeError(f"{query} produced a non-positive timing")
        measurements[query] = (total_seconds, rows)
    if not measurements:
        raise RuntimeError(f"runner produced no {mode} timing rows")
    return measurements


def warm_page_cache(data_dir):
    """Read workload Parquet files before starting a measured process."""
    for parquet_file in sorted(data_dir.rglob("*.parquet")):
        with parquet_file.open("rb", buffering=0) as source:
            while source.read(16 * 1024 * 1024):
                pass


def runner_command(args, workload, runner, side, sampling_mode):
    if workload == "job":
        data_dir = args.job_data_dir
        runner_workload = "job"
        extra = []
    else:
        data_dir = args.tpch_data_dir
        runner_workload = "tpch"
        extra = ["--scale-factor", "1"]
    command = [
        str(runner),
        "--workload",
        runner_workload,
        "--data-dir",
        str(data_dir),
        "--query-dir",
        str(WORKLOADS[workload]["query_dir"]),
        "--threads",
        str(args.threads),
        "--warmups",
        str(args.warmups),
        "--runs",
        str(args.runs),
        "--parquet-pushdown",
        *extra,
    ]
    if sampling_mode == "instant" and side != "datafusion":
        command += [
            "--instant-sampling",
            "--instant-parquet-row-groups",
            str(args.instant_parquet_row_groups),
        ]
    command.append("--baseline-only" if side == "datafusion" else "--bloom-only")
    return command


def run_once(args, workload, sampling_mode, side, runner, round_index):
    data_dir = args.job_data_dir if workload == "job" else args.tpch_data_dir
    warm_page_cache(data_dir)
    command = runner_command(args, workload, runner, side, sampling_mode)
    log_path = (
        args.out_dir
        / f"{workload}.{sampling_mode}.{side}.round{round_index + 1}.log"
    )
    print(
        f">>> {workload} {side} round {round_index + 1}: {shlex.join(command)}",
        flush=True,
    )
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.write_text(
        f"$ {shlex.join(command)}\n\n[stdout]\n{completed.stdout}\n"
        f"[stderr]\n{completed.stderr}",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{workload} {side} round {round_index + 1} failed with exit code "
            f"{completed.returncode}; see {log_path}"
        )
    mode = "datafusion" if side == "datafusion" else "bloom"
    return parse_measurements(completed.stdout, mode)


def merge_measurements(timings, row_counts, current):
    if timings and timings.keys() != current.keys():
        missing = sorted(timings.keys() - current.keys())
        added = sorted(current.keys() - timings.keys())
        raise RuntimeError(
            f"query set changed between rounds: missing={missing}, added={added}"
        )
    for query, (timing, rows) in current.items():
        if query in row_counts and row_counts[query] != rows:
            raise RuntimeError(
                f"{query} row count changed between rounds: "
                f"{row_counts[query]} != {rows}"
            )
        row_counts[query] = rows
        timings[query].append(timing)


def geometric_mean(values):
    if not values or any(value <= 0 for value in values):
        raise RuntimeError("geometric mean requires positive timings")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def summarize_timings(
    workload,
    base_timings,
    candidate_timings,
    base_rows,
    candidate_rows,
    expected_samples,
    regression_ratio,
):
    if base_timings.keys() != candidate_timings.keys():
        base_only = sorted(base_timings.keys() - candidate_timings.keys())
        candidate_only = sorted(candidate_timings.keys() - base_timings.keys())
        raise RuntimeError(
            f"query-set mismatch: base-only={base_only}, candidate-only={candidate_only}"
        )
    expected_queries = WORKLOADS[workload]["queries"]
    if len(base_timings) != expected_queries:
        raise RuntimeError(
            f"{workload} produced {len(base_timings)} queries; expected {expected_queries}"
        )

    rows = []
    for query in sorted(base_timings):
        base_samples = base_timings[query]
        candidate_samples = candidate_timings[query]
        if len(base_samples) != expected_samples or len(candidate_samples) != expected_samples:
            raise RuntimeError(
                f"{query} has {len(base_samples)} base and "
                f"{len(candidate_samples)} candidate samples; expected {expected_samples}"
            )
        if base_rows[query] != candidate_rows[query]:
            raise RuntimeError(
                f"{query} row-count mismatch: base={base_rows[query]}, "
                f"candidate={candidate_rows[query]}"
            )
        base_median = statistics.median(base_samples)
        candidate_median = statistics.median(candidate_samples)
        rows.append(
            {
                "query": query,
                "base": base_median,
                "candidate": candidate_median,
                "ratio": candidate_median / base_median,
            }
        )

    base_geomean = geometric_mean([row["base"] for row in rows])
    candidate_geomean = geometric_mean([row["candidate"] for row in rows])
    return {
        "workload": workload,
        "rows": rows,
        "query_regressions": [
            row for row in rows if row["ratio"] >= regression_ratio
        ],
        "base_geomean": base_geomean,
        "candidate_geomean": candidate_geomean,
        "geomean_ratio": candidate_geomean / base_geomean,
        "base_total": math.fsum(row["base"] for row in rows),
        "candidate_total": math.fsum(row["candidate"] for row in rows),
    }


def format_seconds(value):
    if abs(value) >= 1:
        return f"{value:.3f} s"
    return f"{value * 1000:.1f} ms"


def append_step_summary(lines):
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write("\n".join(lines) + "\n")


def emit_annotation(level, title, message):
    if os.getenv("GITHUB_ACTIONS") == "true":
        print(f"::{level} title={title}::{message}")


def report_summary(summary, regression_ratio, geomean_seconds, no_fail):
    base = summary["base_geomean"]
    candidate = summary["candidate_geomean"]
    ratio = summary["geomean_ratio"]
    delta = candidate - base
    gate_failed = ratio >= regression_ratio or delta >= geomean_seconds
    status = "REGRESSION" if gate_failed else "PASS"
    lines = [
        f"## Performance regression: `{summary['workload']}` "
        f"(`{summary.get('sampling_mode', 'prepared')}`) — {status}",
        "",
        "| Metric | Base | Candidate | Change |",
        "| --- | ---: | ---: | ---: |",
        f"| Query geomean | {format_seconds(base)} | {format_seconds(candidate)} "
        f"| {(ratio - 1) * 100:+.1f}% |",
        f"| Complete workload | {format_seconds(summary['base_total'])} "
        f"| {format_seconds(summary['candidate_total'])} "
        f"| {(summary['candidate_total'] / summary['base_total'] - 1) * 100:+.1f}% |",
        "",
    ]

    regressions = summary["query_regressions"]
    if regressions:
        lines += [
            f"Queries with median slowdown of at least "
            f"{(regression_ratio - 1) * 100:.0f}%:",
            "",
            "| Query | Base | Candidate | Change |",
            "| --- | ---: | ---: | ---: |",
        ]
        for row in sorted(regressions, key=lambda item: item["ratio"], reverse=True):
            lines.append(
                f"| `{row['query']}` | {format_seconds(row['base'])} "
                f"| {format_seconds(row['candidate'])} "
                f"| {(row['ratio'] - 1) * 100:+.1f}% |"
            )
            emit_annotation(
                "warning",
                "Performance query regression",
                f"{summary['workload']} {row['query']} slowed by "
                f"{(row['ratio'] - 1) * 100:.1f}%",
            )
        lines.append("")

    if gate_failed:
        message = (
            f"{summary['workload']} geomean changed from {format_seconds(base)} to "
            f"{format_seconds(candidate)} ({(ratio - 1) * 100:+.1f}%, "
            f"{format_seconds(delta)})"
        )
        emit_annotation(
            "warning" if no_fail else "error",
            "Performance geomean regression",
            message,
        )
    print("\n".join(lines))
    append_step_summary(lines)
    return gate_failed


def report_datafusion_summary(summary, slowdown_ratio, slowdown_seconds, no_fail):
    """Report candidate Bloom against candidate stock DataFusion."""
    datafusion = summary["base_geomean"]
    bloom = summary["candidate_geomean"]
    ratio = summary["geomean_ratio"]
    delta = bloom - datafusion
    # Requiring both a relative and absolute slowdown prevents tiny queries
    # from making a shared runner flaky while still rejecting a real loss.
    gate_failed = ratio >= slowdown_ratio and delta >= slowdown_seconds
    status = "REGRESSION" if gate_failed else "PASS"
    lines = [
        f"## Bloom against DataFusion: `{summary['workload']}` "
        f"(`{summary.get('sampling_mode', 'prepared')}`) — {status}",
        "",
        "| Metric | DataFusion | Bloom | Bloom speedup |",
        "| --- | ---: | ---: | ---: |",
        f"| Query geomean | {format_seconds(datafusion)} | {format_seconds(bloom)} "
        f"| {datafusion / bloom:.3f}× |",
        f"| Complete workload | {format_seconds(summary['base_total'])} "
        f"| {format_seconds(summary['candidate_total'])} "
        f"| {summary['base_total'] / summary['candidate_total']:.3f}× |",
        "",
    ]

    slower_queries = summary["query_regressions"]
    if slower_queries:
        lines += [
            f"Queries where Bloom is at least "
            f"{(slowdown_ratio - 1) * 100:.0f}% slower:",
            "",
            "| Query | DataFusion | Bloom | Change |",
            "| --- | ---: | ---: | ---: |",
        ]
        for row in sorted(slower_queries, key=lambda item: item["ratio"], reverse=True):
            lines.append(
                f"| `{row['query']}` | {format_seconds(row['base'])} "
                f"| {format_seconds(row['candidate'])} "
                f"| {(row['ratio'] - 1) * 100:+.1f}% |"
            )
        lines.append("")

    if gate_failed:
        message = (
            f"{summary['workload']} Bloom geomean is {format_seconds(bloom)} versus "
            f"DataFusion {format_seconds(datafusion)} "
            f"({(ratio - 1) * 100:+.1f}%, {format_seconds(delta)})"
        )
        emit_annotation(
            "warning" if no_fail else "error",
            "Bloom slower than DataFusion",
            message,
        )
    print("\n".join(lines))
    append_step_summary(lines)
    return gate_failed


def compare_workload(args, workload, sampling_mode):
    timings = {
        "base": defaultdict(list),
        "candidate": defaultdict(list),
        "datafusion": defaultdict(list),
    }
    row_counts = {"base": {}, "candidate": {}, "datafusion": {}}
    runners = {
        "base": args.base_runner,
        "candidate": args.candidate_runner,
        "datafusion": args.candidate_runner,
    }
    orders = (
        ("base", "candidate", "datafusion"),
        ("candidate", "datafusion", "base"),
        ("datafusion", "base", "candidate"),
    )
    for round_index in range(args.rounds):
        order = orders[round_index % len(orders)]
        for side in order:
            current = run_once(
                args,
                workload,
                sampling_mode,
                side,
                runners[side],
                round_index,
            )
            merge_measurements(timings[side], row_counts[side], current)
    version_summary = summarize_timings(
        workload,
        timings["base"],
        timings["candidate"],
        row_counts["base"],
        row_counts["candidate"],
        args.rounds,
        args.regression_ratio,
    )
    datafusion_summary = summarize_timings(
        workload,
        timings["datafusion"],
        timings["candidate"],
        row_counts["datafusion"],
        row_counts["candidate"],
        args.rounds,
        args.regression_ratio,
    )
    version_summary["sampling_mode"] = sampling_mode
    datafusion_summary["sampling_mode"] = sampling_mode
    return version_summary, datafusion_summary


def main():
    args = parse_args()
    args.base_runner = args.base_runner.resolve()
    args.candidate_runner = args.candidate_runner.resolve()
    args.job_data_dir = args.job_data_dir.resolve()
    args.tpch_data_dir = args.tpch_data_dir.resolve()
    for label, runner in (
        ("base", args.base_runner),
        ("candidate", args.candidate_runner),
    ):
        if not runner.is_file():
            raise SystemExit(f"{label} benchmark runner not found: {runner}")
    for label, data_dir in (
        ("JOB", args.job_data_dir),
        ("TPC-H SF1", args.tpch_data_dir),
    ):
        if not data_dir.is_dir() or not any(data_dir.rglob("*.parquet")):
            raise SystemExit(f"{label} Parquet data not found: {data_dir}")
    if not math.isfinite(args.regression_ratio) or args.regression_ratio <= 1:
        raise SystemExit("--regression-ratio must be greater than 1")
    if (
        not math.isfinite(args.geomean_regression_seconds)
        or args.geomean_regression_seconds <= 0
    ):
        raise SystemExit("--geomean-regression-seconds must be greater than zero")

    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    workloads = args.workloads or list(WORKLOADS)
    sampling_modes = args.sampling_modes or ["prepared"]
    failed = False
    try:
        for sampling_mode in sampling_modes:
            for workload in workloads:
                version_summary, datafusion_summary = compare_workload(
                    args, workload, sampling_mode
                )
                failed |= report_summary(
                    version_summary,
                    args.regression_ratio,
                    args.geomean_regression_seconds,
                    args.no_fail,
                )
                failed |= report_datafusion_summary(
                    datafusion_summary,
                    args.regression_ratio,
                    args.geomean_regression_seconds,
                    args.no_fail,
                )
    except RuntimeError as error:
        emit_annotation("error", "Performance benchmark failure", str(error))
        print(f"performance comparison failed: {error}", file=sys.stderr)
        return 1
    return 1 if failed and not args.no_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
