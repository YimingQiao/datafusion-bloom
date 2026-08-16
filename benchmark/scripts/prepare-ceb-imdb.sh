#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/../.." && pwd)
data_dir="$project_dir/benchmark_data/ceb-imdb"
archive="$data_dir/downloads/imdb_pg_dataset.tar.gz"
queries_dir="$data_dir/queries"
parquet_dir=${BLOOM_CEB_IMDB_PARQUET_DIR:-"$project_dir/benchmark_data/job/parquet-largeutf8"}
source_url="https://codeload.github.com/RyanMarcus/imdb_pg_dataset/tar.gz/1f39e9aa85ee64249f60bfa59543e8707b228644"
expected_sha256="43f4b5984db5b281968a3f548a93cb00cbd8bad7850ce366641592117958754c"
expected_queries=3133

mkdir -p "$data_dir/downloads"

if [[ ! -f "$archive" ]]; then
    echo "Downloading the pinned CEB-IMDB query corpus..."
    curl --fail --location --output "$archive" "$source_url"
fi

actual_sha256=$(sha256sum "$archive" | cut -d' ' -f1)
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "Unexpected CEB-IMDB archive SHA-256: got $actual_sha256" >&2
    exit 1
fi

if [[ ! -f "$queries_dir/.complete" ]]; then
    if [[ -e "$queries_dir" ]]; then
        echo "Incomplete query directory exists: $queries_dir" >&2
        echo "Move it aside and rerun this script." >&2
        exit 1
    fi

    extract_dir=$(mktemp -d "$data_dir/.extract.XXXXXX")
    cleanup() {
        rm -rf -- "$extract_dir"
    }
    trap cleanup EXIT
    tar -xzf "$archive" -C "$extract_dir"
    source_queries=$(find "$extract_dir" -type d -name ceb-imdb-3k -print -quit)
    if [[ -z "$source_queries" ]]; then
        echo "The pinned archive does not contain ceb-imdb-3k" >&2
        exit 1
    fi
    mv -- "$source_queries" "$queries_dir"
    touch "$queries_dir/.complete"
fi

query_count=$(find "$queries_dir" -type f -name '*.sql' | wc -l)
if [[ "$query_count" -ne "$expected_queries" ]]; then
    echo "Expected $expected_queries CEB-IMDB queries, found $query_count" >&2
    exit 1
fi

if [[ ! -d "$parquet_dir" ]]; then
    echo "CEB-IMDB data does not exist: $parquet_dir" >&2
    echo "Prepare the LargeUtf8 JOB Parquet directory documented in benchmark/README.md first." >&2
    exit 1
fi

echo "CEB-IMDB queries are ready under $queries_dir (data: $parquet_dir)"
