#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/../.." && pwd)
data_dir="$project_dir/benchmark_data/job"
archive="$data_dir/downloads/imdb.tgz"
csv_dir="$data_dir/csv"
parquet_dir=${BLOOM_JOB_PARQUET_DIR:-"$data_dir/parquet"}
source_url="https://event.cwi.nl/da/job/imdb.tgz"
expected_bytes=1263193115
expected_sha256="25f9d893c54f903366e0c263f88db0d429dbc2b159d4987ebc1e203242a7e988"
threads=${BLOOM_PREPARE_THREADS:-$(getconf _NPROCESSORS_ONLN)}
compression=${BLOOM_JOB_COMPRESSION:-"zstd(3)"}
row_group_rows=${BLOOM_JOB_ROW_GROUP_ROWS:-262144}
dictionary_enabled=${BLOOM_JOB_DICTIONARY_ENABLED:-true}
integer_encoding=${BLOOM_JOB_INTEGER_ENCODING:-default}
string_type=${BLOOM_JOB_STRING_TYPE:-utf8}

mkdir -p "$data_dir/downloads" "$csv_dir" "$parquet_dir"

actual_bytes=0
if [[ -f "$archive" ]]; then
    actual_bytes=$(stat -c '%s' "$archive")
fi
if [[ "$actual_bytes" -ne "$expected_bytes" ]]; then
    echo "Downloading the standard JOB IMDB snapshot into this repository cache..."
    curl --fail --location --continue-at - --output "$archive" "$source_url"
fi

actual_sha256=$(sha256sum "$archive" | cut -d' ' -f1)
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "Unexpected IMDB archive SHA-256: got $actual_sha256" >&2
    exit 1
fi

actual_bytes=$(stat -c '%s' "$archive")
if [[ "$actual_bytes" -ne "$expected_bytes" ]]; then
    echo "Unexpected IMDB archive size: got $actual_bytes, expected $expected_bytes" >&2
    exit 1
fi

if [[ ! -f "$csv_dir/.complete" ]]; then
    echo "Extracting IMDB CSV files..."
    tar -xzf "$archive" -C "$csv_dir"
    touch "$csv_dir/.complete"
fi

query_count=$(find "$project_dir/benchmark/job/queries" -maxdepth 1 -type f -name '*.sql' | wc -l)
if [[ "$query_count" -ne 113 ]]; then
    echo "Expected 113 pinned JOB queries, found $query_count" >&2
    exit 1
fi

cd "$project_dir"
cargo run --release --example prepare_job -- \
    "$csv_dir" "$parquet_dir" "$threads" "$compression" "$row_group_rows" \
    "$dictionary_enabled" "$integer_encoding" "$string_type"

echo "JOB data is ready under $parquet_dir (compression: $compression, row-group rows: $row_group_rows, dictionary: $dictionary_enabled, integer encoding: $integer_encoding, strings: $string_type)"
