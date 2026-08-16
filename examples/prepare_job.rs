use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::config::{ParquetColumnOptions, TableParquetOptions};
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::prelude::{CsvReadOptions, SessionConfig, SessionContext};

type AnyResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Debug)]
struct Options {
    csv_dir: PathBuf,
    parquet_dir: PathBuf,
    threads: usize,
    compression: String,
    row_group_size: usize,
    dictionary_enabled: bool,
    integer_encoding: Option<String>,
}

#[tokio::main]
async fn main() -> AnyResult<()> {
    let options = parse_options()?;
    fs::create_dir_all(&options.parquet_dir)?;

    let config = SessionConfig::new()
        .with_target_partitions(options.threads)
        .with_batch_size(65_536);
    let context = SessionContext::new_with_config(config);

    for table in job_tables() {
        let source = options.csv_dir.join(format!("{}.csv", table.name));
        if !source.is_file() {
            return Err(format!("missing JOB CSV file: {}", source.display()).into());
        }

        let destination = options.parquet_dir.join(table.name);
        let complete = destination.join("_SUCCESS");
        if complete.is_file() {
            println!("skip {:<18} {}", table.name, destination.display());
            continue;
        }
        if destination.exists() {
            return Err(format!(
                "incomplete destination exists: {}; remove that table directory and retry",
                destination.display()
            )
            .into());
        }

        let building = options
            .parquet_dir
            .join(format!(".{}.building", table.name));
        if building.exists() {
            return Err(format!(
                "interrupted build directory exists: {}; remove it and retry",
                building.display()
            )
            .into());
        }

        println!("convert {:<18} {}", table.name, source.display());
        let schema = table.schema();
        let frame = context
            .read_csv(
                path_string(&source),
                CsvReadOptions::new()
                    .has_header(false)
                    .escape(b'\\')
                    .schema(schema.as_ref())
                    .newlines_in_values(true),
            )
            .await?;

        let mut parquet_options = TableParquetOptions::default();
        parquet_options.global.compression = Some(options.compression.clone());
        parquet_options.global.max_row_group_size = options.row_group_size;
        parquet_options.global.dictionary_enabled = Some(options.dictionary_enabled);
        if let Some(encoding) = &options.integer_encoding {
            for (column, kind) in table.columns {
                if matches!(kind, Kind::Int) {
                    parquet_options.column_specific_options.insert(
                        (*column).to_string(),
                        ParquetColumnOptions {
                            encoding: Some(encoding.clone()),
                            dictionary_enabled: Some(false),
                            ..ParquetColumnOptions::default()
                        },
                    );
                }
            }
        }
        let sink_output = frame
            .write_parquet(
                path_string(&building),
                DataFrameWriteOptions::new().with_single_file_output(false),
                Some(parquet_options),
            )
            .await?;

        fs::rename(&building, &destination)?;
        fs::write(&complete, b"complete\n")?;
        let sink_rows = sink_output
            .iter()
            .map(|batch| batch.num_rows())
            .sum::<usize>();
        println!(
            "ready   {:<18} {} (sink batches: {})",
            table.name,
            destination.display(),
            sink_rows
        );
    }

    Ok(())
}

fn parse_options() -> AnyResult<Options> {
    let mut arguments = env::args().skip(1);
    let csv_dir = arguments.next().ok_or(
            "usage: prepare_job <csv-dir> <parquet-dir> [threads] [compression] [row-group-rows] [dictionary-enabled] [integer-encoding]",
    )?;
    let parquet_dir = arguments.next().ok_or(
            "usage: prepare_job <csv-dir> <parquet-dir> [threads] [compression] [row-group-rows] [dictionary-enabled] [integer-encoding]",
    )?;
    let threads = arguments
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    let compression = arguments.next().unwrap_or_else(|| "zstd(3)".to_string());
    let row_group_size = arguments
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(262_144);
    let dictionary_enabled = arguments
        .next()
        .map(|value| value.parse::<bool>())
        .transpose()?
        .unwrap_or(true);
    let integer_encoding = arguments
        .next()
        .filter(|value| !value.eq_ignore_ascii_case("default"));
    if threads == 0 {
        return Err("threads must be greater than zero".into());
    }
    if row_group_size == 0 {
        return Err("row-group rows must be greater than zero".into());
    }
    if arguments.next().is_some() {
        return Err(
            "usage: prepare_job <csv-dir> <parquet-dir> [threads] [compression] [row-group-rows] [dictionary-enabled] [integer-encoding]".into(),
        );
    }
    Ok(Options {
        csv_dir: PathBuf::from(csv_dir),
        parquet_dir: PathBuf::from(parquet_dir),
        threads,
        compression,
        row_group_size,
        dictionary_enabled,
        integer_encoding,
    })
}

fn path_string(path: &Path) -> &str {
    path.to_str().expect("benchmark paths must be valid UTF-8")
}

#[derive(Debug, Clone, Copy)]
enum Kind {
    Int,
    Text,
}

#[derive(Debug)]
struct JobTable {
    name: &'static str,
    columns: &'static [(&'static str, Kind)],
}

impl JobTable {
    fn schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(
            self.columns
                .iter()
                .map(|(name, kind)| {
                    let data_type = match kind {
                        Kind::Int => DataType::Int32,
                        Kind::Text => DataType::Utf8,
                    };
                    Field::new(*name, data_type, true)
                })
                .collect::<Vec<_>>(),
        ))
    }
}

const I: Kind = Kind::Int;
const S: Kind = Kind::Text;

fn job_tables() -> Vec<JobTable> {
    vec![
        JobTable {
            name: "aka_name",
            columns: &[
                ("id", I),
                ("person_id", I),
                ("name", S),
                ("imdb_index", S),
                ("name_pcode_cf", S),
                ("name_pcode_nf", S),
                ("surname_pcode", S),
                ("md5sum", S),
            ],
        },
        JobTable {
            name: "aka_title",
            columns: &[
                ("id", I),
                ("movie_id", I),
                ("title", S),
                ("imdb_index", S),
                ("kind_id", I),
                ("production_year", I),
                ("phonetic_code", S),
                ("episode_of_id", I),
                ("season_nr", I),
                ("episode_nr", I),
                ("note", S),
                ("md5sum", S),
            ],
        },
        JobTable {
            name: "cast_info",
            columns: &[
                ("id", I),
                ("person_id", I),
                ("movie_id", I),
                ("person_role_id", I),
                ("note", S),
                ("nr_order", I),
                ("role_id", I),
            ],
        },
        JobTable {
            name: "char_name",
            columns: &[
                ("id", I),
                ("name", S),
                ("imdb_index", S),
                ("imdb_id", I),
                ("name_pcode_nf", S),
                ("surname_pcode", S),
                ("md5sum", S),
            ],
        },
        JobTable {
            name: "comp_cast_type",
            columns: &[("id", I), ("kind", S)],
        },
        JobTable {
            name: "company_name",
            columns: &[
                ("id", I),
                ("name", S),
                ("country_code", S),
                ("imdb_id", I),
                ("name_pcode_nf", S),
                ("name_pcode_sf", S),
                ("md5sum", S),
            ],
        },
        JobTable {
            name: "company_type",
            columns: &[("id", I), ("kind", S)],
        },
        JobTable {
            name: "complete_cast",
            columns: &[
                ("id", I),
                ("movie_id", I),
                ("subject_id", I),
                ("status_id", I),
            ],
        },
        JobTable {
            name: "info_type",
            columns: &[("id", I), ("info", S)],
        },
        JobTable {
            name: "keyword",
            columns: &[("id", I), ("keyword", S), ("phonetic_code", S)],
        },
        JobTable {
            name: "kind_type",
            columns: &[("id", I), ("kind", S)],
        },
        JobTable {
            name: "link_type",
            columns: &[("id", I), ("link", S)],
        },
        JobTable {
            name: "movie_companies",
            columns: &[
                ("id", I),
                ("movie_id", I),
                ("company_id", I),
                ("company_type_id", I),
                ("note", S),
            ],
        },
        JobTable {
            name: "movie_info",
            columns: &[
                ("id", I),
                ("movie_id", I),
                ("info_type_id", I),
                ("info", S),
                ("note", S),
            ],
        },
        JobTable {
            name: "movie_info_idx",
            columns: &[
                ("id", I),
                ("movie_id", I),
                ("info_type_id", I),
                ("info", S),
                ("note", S),
            ],
        },
        JobTable {
            name: "movie_keyword",
            columns: &[("id", I), ("movie_id", I), ("keyword_id", I)],
        },
        JobTable {
            name: "movie_link",
            columns: &[
                ("id", I),
                ("movie_id", I),
                ("linked_movie_id", I),
                ("link_type_id", I),
            ],
        },
        JobTable {
            name: "name",
            columns: &[
                ("id", I),
                ("name", S),
                ("imdb_index", S),
                ("imdb_id", I),
                ("gender", S),
                ("name_pcode_cf", S),
                ("name_pcode_nf", S),
                ("surname_pcode", S),
                ("md5sum", S),
            ],
        },
        JobTable {
            name: "person_info",
            columns: &[
                ("id", I),
                ("person_id", I),
                ("info_type_id", I),
                ("info", S),
                ("note", S),
            ],
        },
        JobTable {
            name: "role_type",
            columns: &[("id", I), ("role", S)],
        },
        JobTable {
            name: "title",
            columns: &[
                ("id", I),
                ("title", S),
                ("imdb_index", S),
                ("kind_id", I),
                ("production_year", I),
                ("imdb_id", I),
                ("phonetic_code", S),
                ("episode_of_id", I),
                ("season_nr", I),
                ("episode_nr", I),
                ("series_years", S),
                ("md5sum", S),
            ],
        },
    ]
}
