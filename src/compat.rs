//! DataFusion-version and failure-policy boundary.
//!
//! Bloom uses DataFusion's public extension and physical-plan APIs. Keeping
//! version-sensitive policy here makes upgrades auditable without scattering
//! version checks through the transfer algorithm.

use datafusion::arrow::error::ArrowError;
use datafusion::common::DataFusionError;

/// Whether a failure came from an exhausted DataFusion resource pool.
pub(crate) fn is_resource_exhausted(error: &DataFusionError) -> bool {
    match error {
        DataFusionError::ResourcesExhausted(_) => true,
        DataFusionError::Context(_, source) | DataFusionError::Diagnostic(_, source) => {
            is_resource_exhausted(source)
        }
        DataFusionError::Shared(source) => is_resource_exhausted(source),
        DataFusionError::Collection(errors) => errors.iter().any(is_resource_exhausted),
        _ => false,
    }
}

/// Failures caused by Bloom planning or in-memory transfer can safely fall
/// back to the already-built native plan. Data-source and task failures are
/// preserved: retrying a missing/corrupt source is wasteful, and cancellation
/// must never start a second execution path.
pub(crate) fn is_recoverable_transfer_error(error: &DataFusionError) -> bool {
    match error {
        DataFusionError::ResourcesExhausted(_)
        | DataFusionError::NotImplemented(_)
        | DataFusionError::Internal(_)
        | DataFusionError::Plan(_)
        | DataFusionError::SchemaError(_, _) => true,
        DataFusionError::ArrowError(source, _) => is_recoverable_arrow_error(source),
        DataFusionError::Context(_, source) | DataFusionError::Diagnostic(_, source) => {
            is_recoverable_transfer_error(source)
        }
        DataFusionError::Shared(source) => is_recoverable_transfer_error(source),
        DataFusionError::Collection(errors) => {
            !errors.is_empty() && errors.iter().all(is_recoverable_transfer_error)
        }
        DataFusionError::ParquetError(_)
        | DataFusionError::ObjectStore(_)
        | DataFusionError::IoError(_)
        | DataFusionError::SQL(_, _)
        | DataFusionError::Configuration(_)
        | DataFusionError::Execution(_)
        | DataFusionError::ExecutionJoin(_)
        | DataFusionError::External(_)
        | DataFusionError::Substrait(_)
        | DataFusionError::Ffi(_) => false,
    }
}

fn is_recoverable_arrow_error(error: &ArrowError) -> bool {
    match error {
        // These variants can be produced by Bloom's casts, array kernels, or
        // ownership compaction. The untouched native plan is a safe fallback.
        ArrowError::NotYetImplemented(_)
        | ArrowError::CastError(_)
        | ArrowError::MemoryError(_)
        | ArrowError::SchemaError(_)
        | ArrowError::ComputeError(_)
        | ArrowError::InvalidArgumentError(_)
        | ArrowError::DictionaryKeyOverflowError
        | ArrowError::RunEndIndexOverflowError
        | ArrowError::OffsetOverflowError(_) => true,
        // These represent input, external-system, or data-evaluation failures.
        // Re-running the same source through the native plan would hide the
        // original failure and may repeat externally visible work.
        ArrowError::ExternalError(_)
        | ArrowError::ParseError(_)
        | ArrowError::DivideByZero
        | ArrowError::ArithmeticOverflow(_)
        | ArrowError::CsvError(_)
        | ArrowError::JsonError(_)
        | ArrowError::AvroError(_)
        | ArrowError::IoError(_, _)
        | ArrowError::IpcError(_)
        | ArrowError::ParquetError(_)
        | ArrowError::CDataInterface(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_nested_resource_exhaustion_as_recoverable() {
        let error = DataFusionError::Context(
            "Bloom materialization".to_string(),
            Box::new(DataFusionError::ResourcesExhausted("limit".to_string())),
        );
        assert!(is_resource_exhausted(&error));
        assert!(is_recoverable_transfer_error(&error));
    }

    #[test]
    fn does_not_retry_source_or_task_failures() {
        assert!(!is_recoverable_transfer_error(&DataFusionError::IoError(
            std::io::Error::new(std::io::ErrorKind::NotFound, "gone"),
        )));
        assert!(!is_recoverable_transfer_error(
            &DataFusionError::ArrowError(
                Box::new(ArrowError::IoError(
                    "gone".to_string(),
                    std::io::Error::new(std::io::ErrorKind::NotFound, "gone"),
                )),
                None,
            ),
        ));
        assert!(!is_recoverable_transfer_error(&DataFusionError::Execution(
            "bad input".to_string()
        ),));
    }

    #[test]
    fn retries_bloom_array_kernel_failures() {
        assert!(is_recoverable_transfer_error(&DataFusionError::ArrowError(
            Box::new(ArrowError::ComputeError("unsupported layout".to_string())),
            None,
        ),));
    }

    #[tokio::test]
    async fn does_not_retry_cancelled_tasks() {
        let task = tokio::spawn(async {
            std::future::pending::<()>().await;
        });
        task.abort();
        let error = DataFusionError::ExecutionJoin(Box::new(
            task.await
                .expect_err("aborted task must return a join error"),
        ));
        assert!(!is_recoverable_transfer_error(&error));
    }
}
