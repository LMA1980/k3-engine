#![allow(clippy::pedantic, clippy::undocumented_unsafe_blocks)]

use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

const BIN_PATH: &str = env!("CARGO_BIN_EXE_k3-worker");

#[test]
fn test_cli_help_and_version_exit_0() {
    // --help
    let output = Command::new(BIN_PATH)
        .arg("--help")
        .output()
        .expect("run k3-worker --help");
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("USAGE:"));
    assert!(stdout.contains("--model"));

    // -h
    let output_h = Command::new(BIN_PATH)
        .arg("-h")
        .output()
        .expect("run k3-worker -h");
    assert_eq!(output_h.status.code(), Some(0));

    // --version
    let output_v = Command::new(BIN_PATH)
        .arg("--version")
        .output()
        .expect("run k3-worker --version");
    assert_eq!(output_v.status.code(), Some(0));
    let stdout_v = String::from_utf8_lossy(&output_v.stdout);
    assert!(stdout_v.contains("k3-worker"));

    // -V
    let output_short_v = Command::new(BIN_PATH)
        .arg("-V")
        .output()
        .expect("run k3-worker -V");
    assert_eq!(output_short_v.status.code(), Some(0));
}

#[test]
fn test_cli_invalid_args_exit_2() {
    // Missing all arguments
    let output = Command::new(BIN_PATH).output().expect("run k3-worker");
    assert_eq!(output.status.code(), Some(2));

    // Missing --socket/--socket-fd
    let output = Command::new(BIN_PATH)
        .args(["--model", "/tmp/fake.gguf"])
        .output()
        .expect("run k3-worker missing socket");
    assert_eq!(output.status.code(), Some(2));

    // Missing --model
    let output = Command::new(BIN_PATH)
        .args(["--socket", "/tmp/test.sock"])
        .output()
        .expect("run k3-worker missing model");
    assert_eq!(output.status.code(), Some(2));

    // Invalid cluster target
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            "/tmp/fake.gguf",
            "--socket",
            "/tmp/test.sock",
            "--cluster",
            "cuda",
        ])
        .output()
        .expect("run k3-worker invalid cluster");
    assert_eq!(output.status.code(), Some(2));

    // Unknown option
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            "/tmp/fake.gguf",
            "--socket",
            "/tmp/test.sock",
            "--bogus-flag",
        ])
        .output()
        .expect("run k3-worker unknown flag");
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn test_model_not_found_or_empty_exit_12() {
    // Non-existent model file -> exit code 12
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            "/path/to/nonexistent/weights.gguf",
            "--socket",
            "/tmp/fake.sock",
        ])
        .output()
        .expect("run k3-worker nonexistent model");
    assert_eq!(output.status.code(), Some(12));

    // Empty model file -> exit code 12
    let empty_path = std::env::temp_dir().join("k3_worker_test_empty.gguf");
    File::create(&empty_path).unwrap();
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            empty_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
        ])
        .output()
        .expect("run k3-worker empty model");
    let _ = std::fs::remove_file(&empty_path);
    assert_eq!(output.status.code(), Some(12));
}

#[test]
fn test_capability_mismatch_exit_10() {
    // Create a valid dummy model file
    let model_path = std::env::temp_dir().join("k3_worker_cap_model.gguf");
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(b"VALID_GGUF_V3_MODEL_HEADER").unwrap();
    }

    // 1. Requesting A100 cluster on non-riscv64 host without K3_SIMULATE_A100 -> exit code 10
    #[cfg(not(target_arch = "riscv64"))]
    {
        let output = Command::new(BIN_PATH)
            .env_remove("K3_SIMULATE_A100")
            .args([
                "--model",
                model_path.to_str().unwrap(),
                "--socket",
                "/tmp/fake.sock",
                "--cluster",
                "a100",
            ])
            .output()
            .expect("run k3-worker a100 without sim");
        assert_eq!(output.status.code(), Some(10));

        // 2. Requesting core 8 (A100 cluster) on non-riscv64 host -> exit code 10
        let output = Command::new(BIN_PATH)
            .env_remove("K3_SIMULATE_A100")
            .args([
                "--model",
                model_path.to_str().unwrap(),
                "--socket",
                "/tmp/fake.sock",
                "--affinity",
                "8",
            ])
            .output()
            .expect("run k3-worker core 8 without sim");
        assert_eq!(output.status.code(), Some(10));
    }

    // 3. Exclusive lock held on model file -> exit code 10
    let lock_file = File::open(&model_path).unwrap();
    let flock_ret = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(flock_ret, 0, "Exclusive lock acquired by test process");

    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "x100",
        ])
        .output()
        .expect("run k3-worker with exclusively locked model");
    assert_eq!(output.status.code(), Some(10));

    unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_UN) };
    drop(lock_file);
    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_cli_invalid_affinity_out_of_bounds_exit_2() {
    let model_path = std::env::temp_dir().join("k3_worker_aff64_model.gguf");
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(b"VALID_GGUF_HEADER").unwrap();
    }

    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--affinity",
            "64",
        ])
        .output()
        .expect("run k3-worker --affinity 64");

    let _ = std::fs::remove_file(&model_path);
    assert_eq!(
        output.status.code(),
        Some(2),
        "Invalid core index 64 must return exit code 2 (InvalidArguments)"
    );
}

#[test]
fn test_signal_termination_sigint_and_sigterm_exit_0() {
    let model_path = std::env::temp_dir().join("k3_worker_sig_test_model.gguf");
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(b"VALID_GGUF_HEADER").unwrap();
    }

    for &sig in &[libc::SIGINT, libc::SIGTERM] {
        let sig_name = if sig == libc::SIGINT {
            "SIGINT"
        } else {
            "SIGTERM"
        };
        let sock_path = std::env::temp_dir().join(format!("k3_worker_sig_{sig_name}.sock"));
        if sock_path.exists() {
            let _ = std::fs::remove_file(&sock_path);
        }

        let mut child = Command::new(BIN_PATH)
            .args([
                "--model",
                model_path.to_str().unwrap(),
                "--socket",
                sock_path.to_str().unwrap(),
                "--listen",
                "--cluster",
                "x100",
            ])
            .spawn()
            .expect("spawn worker");

        // Wait for socket to become available and connect to unblock listener.accept()
        let mut connected = None;
        let connect_deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < connect_deadline {
            if sock_path.exists() {
                if let Ok(stream) = UnixStream::connect(&sock_path) {
                    connected = Some(stream);
                    break;
                }
            }
            thread::sleep(Duration::from_millis(20));
        }
        let _stream = connected.expect("failed to connect to worker socket");

        // Short sleep to allow worker to register signal handlers and enter select!
        thread::sleep(Duration::from_millis(50));

        // Deliver OS signal
        unsafe {
            libc::kill(child.id() as i32, sig);
        }

        // Wait for exit, asserting it exits with code 0 within 2 seconds
        let wait_start = Instant::now();
        let mut exited = false;
        while wait_start.elapsed() < Duration::from_secs(2) {
            if let Ok(Some(_)) = child.try_wait() {
                exited = true;
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }

        if !exited {
            let _ = child.kill();
        }
        let status = child.wait().expect("wait for child");
        assert_eq!(
            status.code(),
            Some(0),
            "{sig_name} must exit cleanly with code 0"
        );
        assert!(
            exited,
            "{sig_name} termination timed out (> 2 seconds), deadlock detected"
        );

        if sock_path.exists() {
            let _ = std::fs::remove_file(&sock_path);
        }
    }

    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_cli_threads_rejections() {
    let model_path = std::env::temp_dir().join("k3_worker_threads_test_model.gguf");
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(b"VALID_GGUF_HEADER").unwrap();
    }

    // --threads 0 -> exit code 2 (InvalidArguments)
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--threads",
            "0",
        ])
        .output()
        .expect("run k3-worker --threads 0");
    assert_eq!(output.status.code(), Some(2));

    // --threads 7 on x100 -> exit code 10 (CapabilityMismatch: exceeds X100 cap of 6)
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "x100",
            "--threads",
            "7",
        ])
        .output()
        .expect("run k3-worker --threads 7 on x100");
    assert_eq!(output.status.code(), Some(10));

    // --affinity 0x00C0 on x100 encroaches on control cores -> exit code 10
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "x100",
            "--affinity",
            "0x00C0",
        ])
        .output()
        .expect("run k3-worker --affinity 0x00C0 on x100");
    assert_eq!(output.status.code(), Some(10));

    // --affinity 0x00FF on x100 encroaches on control cores 6-7 -> exit code 10
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "x100",
            "--affinity",
            "0x00FF",
        ])
        .output()
        .expect("run k3-worker --affinity 0x00FF on x100");
    assert_eq!(output.status.code(), Some(10));

    // --affinity 0-6 on x100 encroaches on control core 6 -> exit code 10
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "x100",
            "--affinity",
            "0-6",
        ])
        .output()
        .expect("run k3-worker --affinity 0-6 on x100");
    assert_eq!(output.status.code(), Some(10));

    // --affinity 0-3 with --threads 5 -> exit code 10 (threads exceed available cores)
    let output = Command::new(BIN_PATH)
        .args([
            "--model",
            model_path.to_str().unwrap(),
            "--socket",
            "/tmp/fake.sock",
            "--cluster",
            "x100",
            "--affinity",
            "0-3",
            "--threads",
            "5",
        ])
        .output()
        .expect("run k3-worker --threads 5 on 4 cores");
    assert_eq!(output.status.code(), Some(10));

    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_worker_engine_multi_thread_affinity_and_no_control_encroachment() {
    let model_path = std::env::temp_dir().join("k3_worker_engine_threads_model.gguf");
    {
        let mut f = File::create(&model_path).unwrap();
        f.write_all(b"VALID_GGUF_HEADER").unwrap();
    }

    let config = k3_engine::worker::WorkerConfig {
        socket_path: Some(std::path::PathBuf::from("/tmp/fake_engine.sock")),
        socket_fd: None,
        model_path: model_path.clone(),
        cluster: "x100".to_owned(),
        threads: Some(4),
        affinity: Some("0-3".to_owned()),
        verbose: false,
        listen: false,
    };
    let weights = k3_engine::worker::LockedWeights::open(&model_path).unwrap();
    let engine = k3_engine::worker::WorkerEngine::new(config, weights);

    assert_eq!(engine.thread_count(), 4);
    assert_eq!(engine.compute_mask(), 0x000F);
    assert_eq!(engine.assigned_cores(), &[0, 1, 2, 3]);
    // Assert zero compute threads ever run on dedicated active control cores 6 or 7
    for &core in engine.assigned_cores() {
        assert!(
            core < 6,
            "core {core} must not encroach on active control cores 6-7"
        );
        assert_eq!((1_u64 << core) & 0x00C0, 0);
    }

    let _ = std::fs::remove_file(&model_path);
}
