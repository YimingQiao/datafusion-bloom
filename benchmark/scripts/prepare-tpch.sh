#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/../.." && pwd)
scale_factor=${1:-10}
threads=${BLOOM_PREPARE_THREADS:-$(getconf _NPROCESSORS_ONLN)}
parts=${BLOOM_TPCH_PARTS:-$threads}
generator_version="3.0.0"
tool_root="$project_dir/benchmark_data/tools/tpchgen-$generator_version"
generator="$tool_root/bin/tpchgen-cli"
sf_dir="$project_dir/benchmark_data/tpch/sf$scale_factor"
parquet_dir="$sf_dir/parquet"
complete="$parquet_dir/_SUCCESS"

if ! [[ "$scale_factor" =~ ^[1-9][0-9]*$ ]]; then
    echo "scale factor must be a positive integer" >&2
    exit 1
fi
if ! [[ "$threads" =~ ^[1-9][0-9]*$ ]]; then
    echo "BLOOM_PREPARE_THREADS must be a positive integer" >&2
    exit 1
fi
if ! [[ "$parts" =~ ^[1-9][0-9]*$ ]]; then
    echo "BLOOM_TPCH_PARTS must be a positive integer" >&2
    exit 1
fi

if [[ ! -x "$generator" ]]; then
    mkdir -p "$tool_root"
    cargo install \
        --locked \
        --version "$generator_version" \
        --root "$tool_root" \
        tpchgen-cli
fi

query_count=$(find "$project_dir/benchmark/tpch/queries" -maxdepth 1 -type f -name 'q*.sql' | wc -l)
if [[ "$query_count" -ne 22 ]]; then
    echo "Expected 22 pinned TPC-H queries, found $query_count" >&2
    exit 1
fi

if [[ -f "$complete" ]]; then
    echo "TPC-H SF$scale_factor is already ready under $parquet_dir"
    exit 0
fi
if [[ -e "$parquet_dir" ]]; then
    echo "Incomplete destination exists: $parquet_dir; remove it explicitly and retry" >&2
    exit 1
fi

mkdir -p "$sf_dir"
"$generator" parquet \
    --scale-factor "$scale_factor" \
    --output-dir "$parquet_dir" \
    --parts "$parts" \
    --num-threads "$threads" \
    --compression 'ZSTD(3)' \
    --row-group-bytes 67108864 \
    --no-progress
touch "$complete"

echo "TPC-H SF$scale_factor is ready under $parquet_dir"
