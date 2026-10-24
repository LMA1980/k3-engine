#![allow(
    clippy::pedantic,
    clippy::undocumented_unsafe_blocks
)]

use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::process::Command;

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
        .args(["--model", "/tmp/fake.gguf", "--socket", "/tmp/test.sock", "--cluster", "cuda"])
        .output()
        .expect("run k3-worker invalid cluster");
    assert_eq!(output.status.code(), Some(2));

    // Unknown option
    let output = Command::new(BIN_PATH)
        .args(["--model", "/tmp/fake.gguf", "--socket", "/tmp/test.sock", "--bogus-flag"])
        .output()
        .expect("run k3-worker unknown flag");
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn test_model_not_found_or_empty_exit_12() {
    // Non-existent model file -> exit code 12
    let output = Command::new(BIN_PATH)
        .args(["--model", "/path/to/nonexistent/weights.gguf", "--socket", "/tmp/fake.sock"])
        .output()
        .expect("run k3-worker nonexistent model");
    assert_eq!(output.status.code(), Some(12));

    // Empty model file -> exit code 12
    let empty_path = std::env::temp_dir().join("k3_worker_test_empty.gguf");
    File::create(&empty_path).unwrap();
    let output = Command::new(BIN_PATH)
        .args(["--model", empty_path.to_str().unwrap(), "--socket", "/tmp/fake.sock"])
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
