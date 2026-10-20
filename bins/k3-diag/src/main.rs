//! `k3-diag`: Operator diagnostic CLI binary.
//!
//! Provides a standalone zero-orchestration tool for verifying model correctness,
//! core affinity, and tokenization behavior on `SpacemiT` K3 hardware.

use std::env;
use std::process::ExitCode;

use k3_core::K3Error;

/// Command line arguments parsed for a diagnostic session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagConfig {
    /// Path to the local model file (e.g., GGUF format).
    pub model_path: Option<String>,
    /// Input prompt to evaluate.
    pub prompt: Option<String>,
    /// Target processor cluster ("x100" or "a100").
    pub cluster: Option<String>,
    /// Worker thread count.
    pub threads: Option<usize>,
    /// Bitmask for pinned core affinity.
    pub core_mask: Option<u64>,
    /// Configured context window limit in tokens.
    pub n_ctx: Option<usize>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<usize>,
}

impl Default for DiagConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            prompt: None,
            cluster: None,
            threads: Some(1),
            core_mask: None,
            n_ctx: Some(2048),
            max_tokens: Some(128),
        }
    }
}

impl DiagConfig {
    /// Parses CLI arguments from an iterator of string arguments.
    ///
    /// # Errors
    ///
    /// Returns a descriptive error message if arguments are malformed or missing values.
    pub fn parse_args<I>(mut args: I) -> Result<Option<Self>, String>
    where
        I: Iterator<Item = String>,
    {
        let mut config = Self::default();

        // Skip binary name if present
        let _ = args.next();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print_usage();
                    return Ok(None);
                }
                "--version" | "-v" => {
                    println!("k3-diag {}", env!("CARGO_PKG_VERSION"));
                    return Ok(None);
                }
                "--model" | "-m" => {
                    config.model_path = Some(
                        args.next()
                            .ok_or_else(|| "--model requires a file path".to_owned())?,
                    );
                }
                "--prompt" | "-p" => {
                    config.prompt = Some(
                        args.next()
                            .ok_or_else(|| "--prompt requires an input string".to_owned())?,
                    );
                }
                "--cluster" => {
                    let cluster = args
                        .next()
                        .ok_or_else(|| "--cluster requires 'x100' or 'a100'".to_owned())?;
                    if cluster != "x100" && cluster != "a100" {
                        return Err(format!(
                            "invalid cluster '{cluster}': expected 'x100' or 'a100'"
                        ));
                    }
                    config.cluster = Some(cluster);
                }
                "--threads" | "-t" => {
                    let val = args
                        .next()
                        .ok_or_else(|| "--threads requires an integer count".to_owned())?;
                    let n = val
                        .parse::<usize>()
                        .map_err(|_| format!("invalid thread count '{val}'"))?;
                    if n == 0 {
                        return Err("thread count must be at least 1".to_owned());
                    }
                    config.threads = Some(n);
                }
                "--cores" => {
                    let val = args
                        .next()
                        .ok_or_else(|| "--cores requires a hex or decimal mask".to_owned())?;
                    let mask = if let Some(hex) =
                        val.strip_prefix("0x").or_else(|| val.strip_prefix("0X"))
                    {
                        u64::from_str_radix(hex, 16)
                    } else {
                        val.parse::<u64>()
                    }
                    .map_err(|_| format!("invalid core mask '{val}'"))?;
                    config.core_mask = Some(mask);
                }
                "--n-ctx" => {
                    let val = args
                        .next()
                        .ok_or_else(|| "--n-ctx requires a token limit integer".to_owned())?;
                    let n = val
                        .parse::<usize>()
                        .map_err(|_| format!("invalid n-ctx '{val}'"))?;
                    config.n_ctx = Some(n);
                }
                "--max-tokens" => {
                    let val = args
                        .next()
                        .ok_or_else(|| "--max-tokens requires an integer".to_owned())?;
                    let n = val
                        .parse::<usize>()
                        .map_err(|_| format!("invalid max-tokens '{val}'"))?;
                    config.max_tokens = Some(n);
                }
                unknown => {
                    return Err(format!("unknown option '{unknown}'"));
                }
            }
        }

        Ok(Some(config))
    }
}

/// Maps domain [`K3Error`] variants to distinct process exit codes per US-01.7.1.
#[must_use]
pub const fn exit_code_for_error(err: &K3Error) -> u8 {
    match err {
        K3Error::CapabilityMismatch { .. } => 10,
        K3Error::InsufficientLockableMemory { .. } => 11,
        K3Error::ContextLengthExceeded { .. } => 12,
        K3Error::AffinityError { .. } => 13,
        K3Error::TensorBoundsError { .. } => 14,
    }
}

fn print_usage() {
    println!(
        "Usage: k3-diag --model <path> [options]\n\n\
         Standalone diagnostic model evaluation tool for SpacemiT K3.\n\n\
         Options:\n  \
           -m, --model <path>     Path to local model file (GGUF)\n  \
           -p, --prompt <text>    Input prompt to tokenize and evaluate\n  \
               --cluster <id>     Target processor cluster ('x100' or 'a100')\n  \
           -t, --threads <n>      Number of execution worker threads\n  \
               --cores <mask>     Pinned core affinity mask (e.g. 0x00FF)\n  \
               --n-ctx <tokens>   Configured context sequence length (default: 2048)\n  \
               --max-tokens <n>   Maximum generation tokens (default: 128)\n  \
           -h, --help             Show this help message\n  \
           -v, --version          Show version information"
    );
}

fn main() -> ExitCode {
    let args = env::args();
    let config = match DiagConfig::parse_args(args) {
        Ok(Some(cfg)) => cfg,
        Ok(None) => return ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[error] {err}");
            eprintln!("Run 'k3-diag --help' for usage instructions.");
            return ExitCode::from(2);
        }
    };

    if config.model_path.is_none() {
        eprintln!("[error] missing required argument: --model <path>");
        eprintln!("Run 'k3-diag --help' for usage instructions.");
        return ExitCode::from(2);
    }

    // Diagnostic run execution stub: wires k3-engine and k3-tokenizer
    println!(
        "k3-diag: initialized session (model: {}, threads: {})",
        config.model_path.as_deref().unwrap_or(""),
        config.threads.unwrap_or(1)
    );

    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_valid_options() {
        let args = vec![
            "k3-diag".to_string(),
            "--model".to_string(),
            "/path/to/model.gguf".to_string(),
            "--prompt".to_string(),
            "Explain RVV 1.0".to_string(),
            "--cluster".to_string(),
            "x100".to_string(),
            "--threads".to_string(),
            "8".to_string(),
            "--cores".to_string(),
            "0x00FF".to_string(),
            "--n-ctx".to_string(),
            "4096".to_string(),
            "--max-tokens".to_string(),
            "256".to_string(),
        ];

        let config = DiagConfig::parse_args(args.into_iter())
            .expect("parse succeeds")
            .expect("config present");

        assert_eq!(config.model_path.as_deref(), Some("/path/to/model.gguf"));
        assert_eq!(config.prompt.as_deref(), Some("Explain RVV 1.0"));
        assert_eq!(config.cluster.as_deref(), Some("x100"));
        assert_eq!(config.threads, Some(8));
        assert_eq!(config.core_mask, Some(0x00FF));
        assert_eq!(config.n_ctx, Some(4096));
        assert_eq!(config.max_tokens, Some(256));
    }

    #[test]
    fn test_parse_args_invalid_options() {
        let args = vec![
            "k3-diag".to_string(),
            "--cluster".to_string(),
            "invalid".to_string(),
        ];
        let err = DiagConfig::parse_args(args.into_iter()).unwrap_err();
        assert!(err.contains("invalid cluster"));

        let args2 = vec![
            "k3-diag".to_string(),
            "--threads".to_string(),
            "0".to_string(),
        ];
        let err2 = DiagConfig::parse_args(args2.into_iter()).unwrap_err();
        assert!(err2.contains("must be at least 1"));
    }

    #[test]
    fn test_exit_codes_mapping() {
        let cap_err = K3Error::CapabilityMismatch {
            architecture: "foo".to_owned(),
            reason: "bar".to_owned(),
        };
        assert_eq!(exit_code_for_error(&cap_err), 10);

        let mem_err = K3Error::InsufficientLockableMemory {
            required_bytes: 100,
            weight_bytes: 80,
            kv_bytes: 20,
            available_bytes: 50,
            rlimit_memlock_bytes: 50,
        };
        assert_eq!(exit_code_for_error(&mem_err), 11);

        let ctx_err = K3Error::ContextLengthExceeded {
            requested_tokens: 10,
            max_context_tokens: 8,
        };
        assert_eq!(exit_code_for_error(&ctx_err), 12);

        let aff_err = K3Error::AffinityError {
            requested_mask: 0x1,
            requested_threads: 2,
            reason: "mismatch".to_owned(),
        };
        assert_eq!(exit_code_for_error(&aff_err), 13);

        let bounds_err = K3Error::TensorBoundsError {
            tensor_name: "t".to_owned(),
            offset: 0,
            byte_len: 10,
            buffer_len: 5,
            reason: "oob".to_owned(),
        };
        assert_eq!(exit_code_for_error(&bounds_err), 14);
    }
}
