#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/../.." && pwd)
threads=${BLOOM_BENCH_THREADS:-1}
warmups=${BLOOM_BENCH_WARMUPS:-0}
runs=${BLOOM_BENCH_RUNS:-1}
scale_factor=${BLOOM_BENCH_TPCH_SCALE_FACTOR:-10}
sampling=${BLOOM_BENCH_SAMPLING:-prepared}
instant_row_groups=${BLOOM_BENCH_INSTANT_ROW_GROUPS:-4}
run_tag=${BLOOM_BENCH_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
output_dir=${BLOOM_BENCH_OUTPUT_DIR:-"$project_dir/benchmark_results/$run_tag"}

usage() {
    echo "usage: benchmark/scripts/run-benchmarks.sh [WORKLOAD ...]"
    echo "workloads: ceb-imdb job-compressed job-uncompressed stats-ceb tpch"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "$threads" -le 0 || "$runs" -le 0 || "$warmups" -lt 0 || "$instant_row_groups" -le 0 ]]; then
    echo "threads, runs, and instant row groups must be positive; warmups must be non-negative" >&2
    exit 1
fi

if [[ "$sampling" != "prepared" && "$sampling" != "instant" ]]; then
    echo "BLOOM_BENCH_SAMPLING must be prepared or instant" >&2
    exit 1
fi

sampling_args=()
if [[ "$sampling" == "instant" ]]; then
    sampling_args+=(--instant-sampling --instant-parquet-row-groups "$instant_row_groups")
fi

if [[ "$#" -eq 0 ]]; then
    set -- ceb-imdb job-compressed job-uncompressed stats-ceb tpch
fi

mkdir -p "$output_dir"

{
    echo "commit=$(git -C "$project_dir" rev-parse HEAD)"
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "threads=$threads"
    echo "warmups=$warmups"
    echo "runs=$runs"
    echo "sampling=$sampling"
    echo "instant_parquet_row_groups=$instant_row_groups"
    echo "rustc=$(rustc --version)"
    echo "kernel=$(uname -srmo)"
    lscpu | grep -E '^(Model name|Socket|Core|Thread|CPU\(s\)):' || true
} >"$output_dir/environment.txt"

run_workload() {
    local label=$1
    local data_root=$2
    shift 2
    local log="$output_dir/$label.log"
    if [[ ! -d "$data_root" ]]; then
        echo "benchmark data does not exist: $data_root" >&2
        echo "run the corresponding prepare script first" >&2
        exit 1
    fi
    echo "Prewarming Parquet pages under $data_root"
    while IFS= read -r -d '' parquet_file; do
        dd if="$parquet_file" of=/dev/null bs=8M status=none
    done < <(find "$data_root" -type f -name '*.parquet' -print0 | sort -z)
    echo "Running $label; full output: $log"
    cargo bench --manifest-path "$project_dir/Cargo.toml" --bench workload -- \
        "$@" \
        --threads "$threads" \
        --warmups "$warmups" \
        --runs "$runs" \
        "${sampling_args[@]}" \
        --parquet-pushdown 2>&1 | tee "$log"
}

for workload in "$@"; do
    case "$workload" in
        ceb-imdb)
            run_workload ceb-imdb "$project_dir/benchmark_data/job/parquet-largeutf8" \
                --workload ceb-imdb \
                --data-dir "$project_dir/benchmark_data/job/parquet-largeutf8" \
                --large-utf8
            ;;
        job-compressed)
            run_workload job-compressed "$project_dir/benchmark_data/job/parquet" \
                --workload job
            ;;
        job-uncompressed)
            run_workload job-uncompressed \
                "$project_dir/benchmark_data/job/parquet-uncompressed" \
                --workload job \
                --data-dir "$project_dir/benchmark_data/job/parquet-uncompressed"
            ;;
        stats-ceb)
            run_workload stats-ceb "$project_dir/benchmark_data/stats-ceb/parquet" \
                --workload stats-ceb
            ;;
        tpch)
            run_workload "tpch-sf$scale_factor" \
                "$project_dir/benchmark_data/tpch/sf$scale_factor/parquet" \
                --workload tpch \
                --scale-factor "$scale_factor"
            ;;
        *)
            echo "unknown workload: $workload" >&2
            usage >&2
            exit 1
            ;;
    esac
done

echo "Benchmark logs are under $output_dir"
