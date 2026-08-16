use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use datafusion::common::config::TableParquetOptions;
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
}

#[tokio::main]
async fn main() -> AnyResult<()> {
    let options = parse_options()?;
    fs::create_dir_all(&options.parquet_dir)?;

    let config = SessionConfig::new()
        .with_target_partitions(options.threads)
        .with_batch_size(65_536);
    let context = SessionContext::new_with_config(config);

    for table in stats_ceb_tables() {
        let source = options.csv_dir.join(table.csv_name);
        if !source.is_file() {
            return Err(format!("missing STATS-CEB CSV file: {}", source.display()).into());
        }

        let destination = options.parquet_dir.join(table.name);
        let complete = destination.join("_SUCCESS");
        if complete.is_file() {
            println!("skip {:<12} {}", table.name, destination.display());
            continue;
        }
        if destination.exists() {
            return Err(format!(
                "incomplete destination exists: {}; move it aside and retry",
                destination.display()
            )
            .into());
        }

        let building = options
            .parquet_dir
            .join(format!(".{}.building", table.name));
        if building.exists() {
            return Err(format!(
                "interrupted build directory exists: {}; move it aside and retry",
                building.display()
            )
            .into());
        }

        println!("convert {:<12} {}", table.name, source.display());
        let schema = table.schema();
        let frame = context
            .read_csv(
                path_string(&source),
                CsvReadOptions::new()
                    .has_header(true)
                    .schema(schema.as_ref()),
            )
            .await?;

        let mut parquet_options = TableParquetOptions::default();
        parquet_options.global.compression = Some(options.compression.clone());
        parquet_options.global.max_row_group_size = options.row_group_size;
        parquet_options.global.dictionary_enabled = Some(true);
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
            "ready   {:<12} {} (sink batches: {})",
            table.name,
            destination.display(),
            sink_rows
        );
    }

    Ok(())
}

fn parse_options() -> AnyResult<Options> {
    let mut arguments = env::args().skip(1);
    let usage =
        "usage: prepare_stats_ceb <csv-dir> <parquet-dir> [threads] [compression] [row-group-rows]";
    let csv_dir = arguments.next().ok_or(usage)?;
    let parquet_dir = arguments.next().ok_or(usage)?;
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
    if threads == 0 {
        return Err("threads must be greater than zero".into());
    }
    if row_group_size == 0 {
        return Err("row-group rows must be greater than zero".into());
    }
    if arguments.next().is_some() {
        return Err(usage.into());
    }
    Ok(Options {
        csv_dir: PathBuf::from(csv_dir),
        parquet_dir: PathBuf::from(parquet_dir),
        threads,
        compression,
        row_group_size,
    })
}

fn path_string(path: &Path) -> &str {
    path.to_str().expect("benchmark paths must be valid UTF-8")
}

#[derive(Debug, Clone, Copy)]
enum Kind {
    Int,
    Timestamp,
}

#[derive(Debug)]
struct StatsCebTable {
    name: &'static str,
    csv_name: &'static str,
    columns: &'static [(&'static str, Kind)],
}

impl StatsCebTable {
    fn schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(
            self.columns
                .iter()
                .map(|(name, kind)| {
                    let data_type = match kind {
                        Kind::Int => DataType::Int32,
                        Kind::Timestamp => DataType::Timestamp(TimeUnit::Microsecond, None),
                    };
                    Field::new(*name, data_type, true)
                })
                .collect::<Vec<_>>(),
        ))
    }
}

const I: Kind = Kind::Int;
const T: Kind = Kind::Timestamp;

fn stats_ceb_tables() -> Vec<StatsCebTable> {
    vec![
        StatsCebTable {
            name: "users",
            csv_name: "users.csv",
            columns: &[
                ("id", I),
                ("reputation", I),
                ("creationdate", T),
                ("views", I),
                ("upvotes", I),
                ("downvotes", I),
            ],
        },
        StatsCebTable {
            name: "posts",
            csv_name: "posts.csv",
            columns: &[
                ("id", I),
                ("posttypeid", I),
                ("creationdate", T),
                ("score", I),
                ("viewcount", I),
                ("owneruserid", I),
                ("answercount", I),
                ("commentcount", I),
                ("favoritecount", I),
                ("lasteditoruserid", I),
            ],
        },
        StatsCebTable {
            name: "postlinks",
            csv_name: "postLinks.csv",
            columns: &[
                ("id", I),
                ("creationdate", T),
                ("postid", I),
                ("relatedpostid", I),
                ("linktypeid", I),
            ],
        },
        StatsCebTable {
            name: "posthistory",
            csv_name: "postHistory.csv",
            columns: &[
                ("id", I),
                ("posthistorytypeid", I),
                ("postid", I),
                ("creationdate", T),
                ("userid", I),
            ],
        },
        StatsCebTable {
            name: "comments",
            csv_name: "comments.csv",
            columns: &[
                ("id", I),
                ("postid", I),
                ("score", I),
                ("creationdate", T),
                ("userid", I),
            ],
        },
        StatsCebTable {
            name: "votes",
            csv_name: "votes.csv",
            columns: &[
                ("id", I),
                ("postid", I),
                ("votetypeid", I),
                ("creationdate", T),
                ("userid", I),
                ("bountyamount", I),
            ],
        },
        StatsCebTable {
            name: "badges",
            csv_name: "badges.csv",
            columns: &[("id", I), ("userid", I), ("date", T)],
        },
        StatsCebTable {
            name: "tags",
            csv_name: "tags.csv",
            columns: &[("id", I), ("count", I), ("excerptpostid", I)],
        },
    ]
}
