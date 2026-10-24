//! Hardware cluster affinity, capability probing, and conditional `/proc/set_ai_thread` registration.

#[cfg(target_os = "linux")]
use core::mem::size_of;
#[cfg(target_arch = "riscv64")]
use std::path::Path;

use super::error::WorkerError;

/// Core bitmask for Cluster 0 (X100 Out-of-Order cores 0–7, VLEN = 256 bits).
pub const X100_CLUSTER_MASK: u64 = 0x00FF;

/// Core bitmask for Cluster 1 (A100 In-Order compute cores 8–15, VLEN = 1024 bits, XSMTVDotII).
pub const A100_CLUSTER_MASK: u64 = 0xFF00;

/// Parses an affinity core index or hex/decimal mask string into a single core ID and mask.
///
/// # Errors
/// Returns [`WorkerError::InvalidArguments`] if affinity string cannot be parsed.
pub fn parse_affinity_spec(spec: &str) -> Result<(usize, u64), WorkerError> {
    let s = spec.trim();
    if s.is_empty() {
        return Err(WorkerError::InvalidArguments(
            "affinity spec cannot be empty".to_owned(),
        ));
    }

    // Try parsing as single core number first (e.g. "8")
    if let Ok(core) = s.parse::<usize>() {
        if core >= 64 {
            return Err(WorkerError::InvalidArguments(format!(
                "core index {core} exceeds 63"
            )));
        }
        return Ok((core, 1_u64 << core));
    }

    // Try parsing as hex bitmask (e.g. "0xFF00" or "0x00FF")
    let mask = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).map_err(|e| {
            WorkerError::InvalidArguments(format!("invalid hex affinity mask '{s}': {e}"))
        })?
    } else {
        s.parse::<u64>().map_err(|e| {
            WorkerError::InvalidArguments(format!("invalid affinity mask '{s}': {e}"))
        })?
    };

    if mask == 0 {
        return Err(WorkerError::InvalidArguments(
            "affinity mask cannot be 0".to_owned(),
        ));
    }

    // Lowest set bit represents the primary core
    let primary_core = mask.trailing_zeros() as usize;
    Ok((primary_core, mask))
}

/// Verifies whether the host platform supports the requested cluster and affinity settings.
///
/// Returns `Ok(())` if supported.
/// Returns `Err(WorkerError::CapabilityMismatch)` (exit code 10) if requested cluster or core is unavailable.
///
/// # Errors
/// Returns [`WorkerError`] on capability mismatch.
pub fn verify_cluster_capabilities(
    cluster: &str,
    affinity_spec: Option<&str>,
) -> Result<(), WorkerError> {
    let sim = std::env::var("K3_SIMULATE_A100").ok();
    verify_cluster_capabilities_with_sim(cluster, affinity_spec, sim.as_deref())
}

/// Verifies whether the host platform supports the requested cluster and affinity settings,
/// with an explicit simulation flag parameter.
///
/// # Errors
/// Returns [`WorkerError`] on capability mismatch.
pub fn verify_cluster_capabilities_with_sim(
    cluster: &str,
    affinity_spec: Option<&str>,
    sim: Option<&str>,
) -> Result<(), WorkerError> {
    let mut requires_a100 = cluster.eq_ignore_ascii_case("a100");

    if let Some(spec) = affinity_spec {
        let (core, mask) = parse_affinity_spec(spec)?;
        if core >= 8 || (mask & A100_CLUSTER_MASK) != 0 {
            requires_a100 = true;
        }
    }

    if !requires_a100 {
        return Ok(());
    }

    // 1. Check test simulation environment override
    if sim == Some("1") {
        return Ok(());
    }

    // 2. Non-riscv64 platforms (macOS, x86_64, aarch64) cannot execute A100 kernels or matrix ops
    #[cfg(not(target_arch = "riscv64"))]
    {
        Err(WorkerError::CapabilityMismatch(format!(
            "host architecture is not riscv64; SpacemiT A100 compute cores (cpu8-15) are unavailable for cluster '{cluster}'"
        )))
    }

    // 3. On riscv64, probe device-tree / sysfs for physical A100 presence
    #[cfg(target_arch = "riscv64")]
    {
        let is_a100_present = Path::new("/sys/devices/system/cpu/cpu8/of_node/cpu-ai").exists()
            || std::fs::read("/sys/devices/system/cpu/cpu8/of_node/compatible")
                .map(|b| String::from_utf8_lossy(&b).contains("spacemit,a100"))
                .unwrap_or(false);

        if !is_a100_present {
            return Err(WorkerError::CapabilityMismatch(
                "host does not possess physical SpacemiT K3 A100 compute cores (cpu8-15)"
                    .to_owned(),
            ));
        }

        Ok(())
    }
}

/// Pins the current worker thread to the assigned physical core.
///
/// # Safety
/// Invokes low-level OS threading syscalls. Marked `#[inline(never)]` and keeps a scalar-only
/// stack frame to prevent scalable vector spill corruption during cross-cluster migrations.
#[inline(never)]
pub unsafe fn pin_worker_core(assigned_core: usize) -> Result<(), WorkerError> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: Direct Linux syscall without vector registers in stack frame.
        unsafe {
            let tid = libc::syscall(libc::SYS_gettid);
            if assigned_core >= 8 {
                let path = b"/proc/set_ai_thread\0";
                let fd = libc::open(path.as_ptr().cast::<libc::c_char>(), libc::O_WRONLY);
                if fd >= 0 {
                    let mut buf = [0u8; 32];
                    let mut val = tid;
                    let mut tmp = [0u8; 32];
                    let mut i = 0;
                    if val == 0 {
                        buf[0] = b'0';
                        i = 1;
                    } else {
                        while val > 0 {
                            tmp[i] = b'0' + (val % 10) as u8;
                            val /= 10;
                            i += 1;
                        }
                        for j in 0..i {
                            buf[j] = tmp[i - 1 - j];
                        }
                    }
                    buf[i] = b'\n';
                    let _ = libc::write(fd, buf.as_ptr().cast::<libc::c_void>(), i + 1);
                    let _ = libc::close(fd);
                }
            }

            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(assigned_core, &mut set);
            let ret = libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set);
            if ret != 0 {
                let err = std::io::Error::last_os_error();
                return Err(WorkerError::Internal(format!(
                    "sched_setaffinity failed for core {assigned_core}: {err}"
                )));
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        // Non-Linux hosts (e.g. macOS host development): no-op
        let _ = assigned_core;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_affinity_spec() {
        assert_eq!(parse_affinity_spec("0").unwrap(), (0, 0x0001));
        assert_eq!(parse_affinity_spec("8").unwrap(), (8, 0x0100));
        assert_eq!(parse_affinity_spec("0x00FF").unwrap(), (0, 0x00FF));
        assert_eq!(parse_affinity_spec("0xFF00").unwrap(), (8, 0xFF00));
    }

    #[test]
    fn test_capability_check_x100() {
        assert!(verify_cluster_capabilities("x100", Some("2")).is_ok());
    }

    #[test]
    fn test_capability_check_a100_non_riscv() {
        #[cfg(not(target_arch = "riscv64"))]
        {
            let res = verify_cluster_capabilities_with_sim("a100", None, None);
            assert!(matches!(res, Err(WorkerError::CapabilityMismatch(_))));
            if let Err(e) = res {
                assert_eq!(e.exit_code(), 10);
            }

            let res_core8 = verify_cluster_capabilities_with_sim("x100", Some("8"), None);
            assert!(matches!(res_core8, Err(WorkerError::CapabilityMismatch(_))));
        }
    }

    #[test]
    fn test_capability_check_a100_simulated() {
        assert!(verify_cluster_capabilities_with_sim("a100", None, Some("1")).is_ok());
        assert!(verify_cluster_capabilities_with_sim("x100", Some("8"), Some("1")).is_ok());
    }
}
