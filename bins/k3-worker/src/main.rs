//! Standalone binary for `k3-worker`: Isolated compute worker for the `SpacemiT` K3 platform.
//!
//! Executes forward passes with zero-copy shared memory tensor dispatch,
//! enforces read-only model weight residency with advisory locking (`flock(LOCK_SH)`),
//! and maintains strict core affinity with conditional `/proc/set_ai_thread` registration.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

use std::env;
use std::os::fd::FromRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::process::ExitCode;
use std::time::Duration;

use k3_worker::affinity::{parse_affinity_spec, pin_worker_core, verify_cluster_capabilities};
use k3_worker::cli::WorkerConfig;
use k3_worker::error::{
    EXIT_INVALID_ARGS, EXIT_PROTOCOL_ERROR,
};
use k3_worker::weights::LockedWeights;
use k3_worker::worker::WorkerEngine;

#[allow(clippy::too_many_lines)]
#[tokio::main]
async fn main() -> ExitCode {
    let args = env::args();
    let config = match WorkerConfig::parse_args(args) {
        Ok(Some(cfg)) => cfg,
        Ok(None) => return ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[k3-worker error] {err}");
            eprintln!("Run 'k3-worker --help' for usage information.");
            return ExitCode::from(EXIT_INVALID_ARGS as u8);
        }
    };

    if config.verbose {
        eprintln!(
            "[k3-worker] starting worker: model={}, cluster={}, affinity={:?}",
            config.model_path.display(),
            config.cluster,
            config.affinity
        );
    }

    // 1. Validate cluster and hardware capabilities
    if let Err(e) = verify_cluster_capabilities(&config.cluster, config.affinity.as_deref()) {
        eprintln!("[k3-worker error] capability verification failed: {e}");
        return ExitCode::from(e.exit_code() as u8);
    }

    // 2. Open and lock model weights read-only
    let weights = match LockedWeights::open(&config.model_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("[k3-worker error] failed to load model weights: {e}");
            return ExitCode::from(e.exit_code() as u8);
        }
    };

    if config.verbose {
        eprintln!(
            "[k3-worker] model weights mapped successfully: {} bytes (mlocked: {})",
            weights.len(),
            weights.is_locked()
        );
    }

    // 3. Pin thread affinity if requested
    if let Some(spec) = &config.affinity {
        match parse_affinity_spec(spec) {
            Ok((core, _)) => {
                // SAFETY: Scalar-only thread pinning routine avoids scalable vector register spills.
                let pin_res = unsafe { pin_worker_core(core) };
                if let Err(e) = pin_res {
                    eprintln!("[k3-worker error] failed to pin thread to core {core}: {e}");
                    return ExitCode::from(e.exit_code() as u8);
                }
                if config.verbose {
                    eprintln!("[k3-worker] pinned to core {core}");
                }
            }
            Err(e) => {
                eprintln!("[k3-worker error] invalid affinity spec: {e}");
                return ExitCode::from(EXIT_INVALID_ARGS as u8);
            }
        }
    }

    // 4. Connect to manager socket or bind listener
    let stream = if let Some(fd) = config.socket_fd {
        // SAFETY: fd was passed from supervisor via command line.
        unsafe { UnixStream::from_raw_fd(fd) }
    } else if let Some(path) = &config.socket_path {
        if config.listen {
            if path.exists() {
                let _ = std::fs::remove_file(path);
            }
            let listener = match UnixListener::bind(path) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("[k3-worker error] failed to bind socket {}: {e}", path.display());
                    return ExitCode::from(EXIT_PROTOCOL_ERROR as u8);
                }
            };
            if config.verbose {
                eprintln!("[k3-worker] listening on {}", path.display());
            }
            match listener.accept() {
                Ok((s, _)) => s,
                Err(e) => {
                    eprintln!("[k3-worker error] accept failed on {}: {e}", path.display());
                    return ExitCode::from(EXIT_PROTOCOL_ERROR as u8);
                }
            }
        } else {
            // Attempt connecting to supervisor with retry
            let mut connected = None;
            for _ in 0..20 {
                if let Ok(s) = UnixStream::connect(path) {
                    connected = Some(s);
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }

            match connected {
                Some(s) => s,
                None => {
                    // If target path does not exist, bind listener and wait for incoming connection
                    if !path.exists() {
                        let listener = match UnixListener::bind(path) {
                            Ok(l) => l,
                            Err(e) => {
                                eprintln!(
                                    "[k3-worker error] failed to connect or bind socket {}: {e}",
                                    path.display()
                                );
                                return ExitCode::from(EXIT_PROTOCOL_ERROR as u8);
                            }
                        };
                        match listener.accept() {
                            Ok((s, _)) => s,
                            Err(e) => {
                                eprintln!("[k3-worker error] accept failed on {}: {e}", path.display());
                                return ExitCode::from(EXIT_PROTOCOL_ERROR as u8);
                            }
                        }
                    } else {
                        eprintln!("[k3-worker error] failed to connect to socket {}", path.display());
                        return ExitCode::from(EXIT_PROTOCOL_ERROR as u8);
                    }
                }
            }
        }
    } else {
        return ExitCode::from(EXIT_INVALID_ARGS as u8);
    };

    // 5. Register OS signal handlers for graceful shutdown (exit code 0)
    let mut sigint = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[k3-worker error] failed to register SIGINT handler: {e}");
            return ExitCode::from(EXIT_PROTOCOL_ERROR as u8);
        }
    };

    let mut sigterm = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[k3-worker error] failed to register SIGTERM handler: {e}");
            return ExitCode::from(EXIT_PROTOCOL_ERROR as u8);
        }
    };

    let mut engine = WorkerEngine::new(config, weights);

    // 6. Spawn event loop in background task to allow reactive signal interruption
    let worker_task = tokio::task::spawn_blocking(move || engine.run_event_loop(stream));

    tokio::select! {
        _ = sigint.recv() => {
            eprintln!("[k3-worker] received SIGINT (Ctrl+C), clean exit");
            std::process::exit(0);
        }
        _ = sigterm.recv() => {
            eprintln!("[k3-worker] received SIGTERM, clean exit");
            std::process::exit(0);
        }
        res = worker_task => {
            match res {
                Ok(Ok(())) => ExitCode::SUCCESS,
                Ok(Err(err)) => {
                    eprintln!("[k3-worker error] event loop failed: {err}");
                    ExitCode::from(err.exit_code() as u8)
                }
                Err(e) => {
                    eprintln!("[k3-worker error] worker task panicked: {e}");
                    ExitCode::from(EXIT_PROTOCOL_ERROR as u8)
                }
            }
        }
    }
}
