#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/../.." && pwd)
data_dir="$project_dir/benchmark_data/stats-ceb"
commit="670cb8d4bf4cbfa32f94fdf17f33973d3fd67d1b"
archive="$data_dir/downloads/stats-ceb-$commit.tar.gz"
csv_dir="$data_dir/csv"
queries_dir="$data_dir/queries"
parquet_dir=${BLOOM_STATS_CEB_PARQUET_DIR:-"$data_dir/parquet"}
source_url="https://codeload.github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/tar.gz/$commit"
expected_sha256="ecdc919ddaeabea8cd2437f0faa90b1ebf973d4398955f877f325e79c2235b24"
expected_queries=146
threads=${BLOOM_PREPARE_THREADS:-$(getconf _NPROCESSORS_ONLN)}
compression=${BLOOM_STATS_CEB_COMPRESSION:-"zstd(3)"}
row_group_rows=${BLOOM_STATS_CEB_ROW_GROUP_ROWS:-262144}

mkdir -p "$data_dir/downloads"

if [[ ! -f "$archive" ]]; then
    echo "Downloading the pinned STATS-CEB source archive..."
    curl --fail --location --output "$archive" "$source_url"
fi

actual_sha256=$(sha256sum "$archive" | cut -d' ' -f1)
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "Unexpected STATS-CEB archive SHA-256: got $actual_sha256" >&2
    exit 1
fi

if [[ ! -f "$data_dir/.assets-complete" ]]; then
    if [[ -e "$csv_dir" || -e "$queries_dir" ]]; then
        echo "Incomplete STATS-CEB assets exist under $data_dir" >&2
        echo "Move csv/ and queries/ aside, then rerun this script." >&2
        exit 1
    fi

    extract_dir=$(mktemp -d "$data_dir/.extract.XXXXXX")
    staging_dir=$(mktemp -d "$data_dir/.assets.XXXXXX")
    cleanup() {
        rm -rf -- "$extract_dir" "$staging_dir"
    }
    trap cleanup EXIT

    tar -xzf "$archive" -C "$extract_dir"
    source_csv_dir=$(find "$extract_dir" -type d -path '*/datasets/stats_simplified' -print -quit)
    workload_file=$(find "$extract_dir" -type f -path '*/workloads/stats_CEB/stats_CEB.sql' -print -quit)
    if [[ -z "$source_csv_dir" || -z "$workload_file" ]]; then
        echo "The pinned archive does not contain the STATS-CEB data and workload" >&2
        exit 1
    fi

    mkdir -p "$staging_dir/csv" "$staging_dir/queries"
    for csv in users.csv posts.csv postLinks.csv postHistory.csv comments.csv votes.csv badges.csv tags.csv; do
        if [[ ! -f "$source_csv_dir/$csv" ]]; then
            echo "Missing STATS-CEB source file: $csv" >&2
            exit 1
        fi
        cp -- "$source_csv_dir/$csv" "$staging_dir/csv/$csv"
    done

    query_count=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ "$line" != *'||'* ]]; then
            echo "Malformed STATS-CEB workload line $((query_count + 1))" >&2
            exit 1
        fi
        cardinality=${line%%'||'*}
        sql=${line#*'||'}
        if [[ ! "$cardinality" =~ ^[0-9]+$ || -z "$sql" ]]; then
            echo "Malformed STATS-CEB workload line $((query_count + 1))" >&2
            exit 1
        fi
        query_count=$((query_count + 1))
        printf '%s\n' "$sql" >"$staging_dir/queries/$query_count.sql"
    done <"$workload_file"

    if [[ "$query_count" -ne "$expected_queries" ]]; then
        echo "Expected $expected_queries STATS-CEB queries, found $query_count" >&2
        exit 1
    fi

    mv -- "$staging_dir/csv" "$csv_dir"
    mv -- "$staging_dir/queries" "$queries_dir"
    touch "$data_dir/.assets-complete"
fi

query_count=$(find "$queries_dir" -maxdepth 1 -type f -name '*.sql' | wc -l)
if [[ "$query_count" -ne "$expected_queries" ]]; then
    echo "Expected $expected_queries STATS-CEB queries, found $query_count" >&2
    exit 1
fi

cd "$project_dir"
cargo run --release --example prepare_stats_ceb -- \
    "$csv_dir" "$parquet_dir" "$threads" "$compression" "$row_group_rows"

echo "STATS-CEB data is ready under $parquet_dir (compression: $compression, row-group rows: $row_group_rows)"
