//! `k3-diag`: Operator diagnostic CLI binary.
//!
//! Provides a standalone zero-orchestration tool for verifying model correctness,
//! capability filtering, pre-flight memory admission, locked weight residency (`mlock` + `flock`),
//! core affinity, in-process tokenization, and real-time unbuffered token streaming on
//! `SpacemiT` K3 hardware (US-01, ADR-005, ADR-023, ADR-025, ADR-027, ADR-037, ADR-040).

use std::env;
use std::io::{self, Write};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use k3_core::{GgufFile, K3Error};
use k3_engine::{
    LockedModelWeights, SinglePromptEngine, SingleSlotKvArena, StreamAction, SystemMemoryInspector,
    WorkerPool, resolve_affinity,
};
use k3_tokenizer::{DetokenizeStep, GgufTokenizer};

/// Global atomic flag set when `SIGINT` or `SIGTERM` is received so the decode loop
/// exits cleanly and triggers RAII `Drop` guards (`munlock`, `munmap`, `flock(LOCK_UN)`).
static ABORT_REQUESTED: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_termination_signal(_sig: libc::c_int) {
    ABORT_REQUESTED.store(true, Ordering::SeqCst);
}

fn install_signal_handlers() {
    // SAFETY: handle_termination_signal is an async-signal-safe extern "C" function
    // that only performs an atomic relaxed/seqcst store on a static AtomicBool.
    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_termination_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            handle_termination_signal as *const () as libc::sighandler_t,
        );
    }
}

/// Returns `true` if an operator `SIGINT` or `SIGTERM` abort was requested.
#[must_use]
pub fn is_abort_requested() -> bool {
    ABORT_REQUESTED.load(Ordering::SeqCst)
}

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
            threads: None,
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

/// Executes the end-to-end standalone diagnostic evaluation pipeline (US-01.7.2):
/// 1. Resolve cluster/core affinity & bind pinned worker pool (`US-01.5`).
/// 2. Open GGUF model, verify capabilities, verify pre-flight RAM admission, and lock weights (`flock(LOCK_SH)` + `mmap` + `mlock`) (`US-01.2`, `US-01.3`).
/// 3. Allocate and `mlock` single-slot KV cache arena (`US-01.3.3`).
/// 4. Extract GGUF vocabulary & chat template, format prompt, and tokenize (`US-01.4`).
/// 5. Execute isolated single-prompt forward pass and stream detokenized UTF-8 pieces to `stdout` (`US-01.6`, `US-01.7.2`).
///
/// # Errors
///
/// Returns [`K3Error`] if any pre-flight admission check, affinity binding, tokenization,
/// or forward-pass step fails.
pub fn run_diagnostic_session<W: Write>(config: &DiagConfig, mut out: W) -> Result<(), K3Error> {
    let model_path = config
        .model_path
        .as_deref()
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "cli".to_owned(),
            reason: "missing required --model <path>".to_owned(),
        })?;
    let prompt_text = config.prompt.as_deref().unwrap_or("hello world");
    let n_ctx = config.n_ctx.unwrap_or(2048);
    let max_tokens = config.max_tokens.unwrap_or(128);

    // 1. Resolve cluster & core affinity and spawn pinned worker pool (US-01.5, Scenario 2)
    let resolved_affinity =
        resolve_affinity(config.cluster.as_deref(), config.core_mask, config.threads)?;
    let worker_pool = WorkerPool::new(resolved_affinity)?;
    worker_pool.verify_live_affinities()?;

    // 2. Open model with flock(LOCK_SH), GGUF parse, capability filter, pre-flight RAM admission, and mlock (US-01.2, US-01.3, Scenario 1 & 3)
    let inspector = SystemMemoryInspector::new();
    let (locked_weights, recipe, _footprint) =
        LockedModelWeights::open_with_admission(model_path, n_ctx, &inspector)?;

    // 3. Allocate and mlock single-slot 16-token paged KV cache arena (US-01.3.3)
    let mut kv_arena = SingleSlotKvArena::new(locked_weights.metadata(), n_ctx)?;

    // 4. Extract GGUF vocabulary/template and tokenize prompt in-process (US-01.4)
    let gguf_view = GgufFile::parse(locked_weights.as_bytes())?;
    let tokenizer = GgufTokenizer::from_gguf(&gguf_view)?;
    let prompt_tokens = tokenizer.tokenize_prompt(prompt_text, true, n_ctx)?;

    // 5. Execute isolated single-prompt forward pass and stream tokens unbuffered (US-01.6, US-01.7.2)
    let mut engine = SinglePromptEngine::new();
    let mut detokenizer = tokenizer.streaming_detokenizer();

    engine.evaluate_prompt_with_pool(
        recipe,
        &locked_weights,
        &mut kv_arena,
        Some(&worker_pool),
        &prompt_tokens,
        tokenizer.vocab_size(),
        max_tokens,
        |token_id| {
            if is_abort_requested() {
                return Ok(StreamAction::Stop);
            }

            match detokenizer.push_token(token_id)? {
                DetokenizeStep::Emit(piece) => {
                    let _ = out.write_all(piece.as_bytes());
                    let _ = out.flush();
                    Ok(StreamAction::Continue)
                }
                DetokenizeStep::Hold => Ok(StreamAction::Continue),
                DetokenizeStep::Stop(prefix) => {
                    if !prefix.is_empty() {
                        let _ = out.write_all(prefix.as_bytes());
                        let _ = out.flush();
                    }
                    Ok(StreamAction::Stop)
                }
            }
        },
    )?;

    if !detokenizer.is_stopped() && !is_abort_requested() {
        let trailing = detokenizer.finish();
        if !trailing.is_empty() {
            let _ = out.write_all(trailing.as_bytes());
            let _ = out.flush();
        }
    }

    let _ = out.write_all(b"\n");
    let _ = out.flush();

    // Re-verify worker affinities before teardown to guarantee zero cross-cluster drift during evaluation
    worker_pool.verify_live_affinities()?;

    // RAII Drop guards for `kv_arena`, `locked_weights`, and `worker_pool` execute deterministically here,
    // releasing mlock, munmap, and flock(LOCK_UN).
    Ok(())
}

fn main() -> ExitCode {
    install_signal_handlers();

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

    let stdout = io::stdout();
    if let Err(err) = run_diagnostic_session(&config, stdout) {
        eprintln!("{err}");
        return ExitCode::from(exit_code_for_error(&err));
    }

    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use k3_core::QuantType;
    use k3_core::test_helpers::create_synthetic_gguf_v3;
    use std::fs::File;

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

    #[test]
    fn test_run_diagnostic_session_streams_tokens_end_to_end() {
        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let temp_path = env::temp_dir().join(format!("k3_diag_unit_{}.gguf", std::process::id()));
        {
            let mut f = File::create(&temp_path).expect("create temp gguf");
            f.write_all(&bytes).expect("write gguf");
            f.sync_all().expect("sync gguf");
        }

        let cfg = DiagConfig {
            model_path: Some(temp_path.to_string_lossy().into_owned()),
            prompt: Some("hello world".to_owned()),
            cluster: None,
            threads: Some(1),
            core_mask: Some(0x0001),
            n_ctx: Some(64),
            max_tokens: Some(4),
        };

        let mut output_buf = Vec::new();
        run_diagnostic_session(&cfg, &mut output_buf).expect("run diagnostic session");
        let output_str = String::from_utf8(output_buf).expect("valid utf8 output");
        assert_eq!(output_str.trim(), "helloworldExplainRVV");

        let _ = std::fs::remove_file(&temp_path);
    }
}
