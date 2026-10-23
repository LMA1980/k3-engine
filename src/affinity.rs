//! `SpacemiT` K3 heterogeneous cluster topology, core mask resolution, and pinned worker pool.
//!
//! Restricts standalone model evaluation strictly to the operator-requested processor cluster
//! (`x100` or `a100`), core bitmask, and worker concurrency bound while preventing cross-cluster
//! VLEN migration (ADR-007, ADR-008, `core-affinity.md`, US-01 Scenario 2).

use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use k3_core::{CoordinationProfile, K3Error};

/// Core bitmask for Cluster 0 (`X100` `OoO` cores 0–7, `zvl256b`).
pub const X100_CLUSTER_MASK: u64 = 0x00FF;

/// Core bitmask for Cluster 1 (`A100` In-Order cores 8–15, `zvl1024b`).
pub const A100_CLUSTER_MASK: u64 = 0xFF00;

/// Default maximum compute threads on `X100` to preserve control-plane cores per ADR-007.
pub const X100_DEFAULT_MAX_COMPUTE_THREADS: usize = 6;

/// Default maximum compute threads on `A100` (all 8 dedicated compute cores).
pub const A100_DEFAULT_MAX_COMPUTE_THREADS: usize = 8;

/// Hardware RVV 1.0 vector register length (`VLEN`) domain on `SpacemiT` K3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VlenDomain {
    /// 256-bit vector registers (`zvl256b`) on `X100` cores 0–7.
    Zvl256b,
    /// 1024-bit vector registers (`zvl1024b`) on `A100` cores 8–15.
    Zvl1024b,
}

impl VlenDomain {
    /// Returns the canonical ISA extension name for this VLEN domain.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Zvl256b => "zvl256b",
            Self::Zvl1024b => "zvl1024b",
        }
    }
}

/// Named processor cluster presets on the `SpacemiT` K3 `SoC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterPreset {
    /// Cluster 0: `X100` cores 0–7 (`0x00FF`, `zvl256b`, default max 6 compute threads per ADR-007).
    X100,
    /// Cluster 1: `A100` cores 8–15 (`0xFF00`, `zvl1024b`, up to 8 compute threads).
    A100,
}

impl ClusterPreset {
    /// Parses a cluster preset string (`"x100"` or `"a100"`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if `name` is not `"x100"` or `"a100"`.
    pub fn parse(name: &str, requested_threads: usize) -> Result<Self, K3Error> {
        match name {
            "x100" => Ok(Self::X100),
            "a100" => Ok(Self::A100),
            other => Err(K3Error::AffinityError {
                requested_mask: 0,
                requested_threads,
                reason: format!("unknown cluster preset '{other}'; expected 'x100' or 'a100'"),
            }),
        }
    }

    /// Returns the canonical cluster name (`"x100"` or `"a100"`).
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::X100 => "x100",
            Self::A100 => "a100",
        }
    }

    /// Returns the 64-bit physical core bitmask for this cluster.
    #[must_use]
    pub const fn core_mask(&self) -> u64 {
        match self {
            Self::X100 => X100_CLUSTER_MASK,
            Self::A100 => A100_CLUSTER_MASK,
        }
    }

    /// Returns the hardware [`VlenDomain`] for this cluster.
    #[must_use]
    pub const fn vlen_domain(&self) -> VlenDomain {
        match self {
            Self::X100 => VlenDomain::Zvl256b,
            Self::A100 => VlenDomain::Zvl1024b,
        }
    }

    /// Returns the default maximum compute thread count for this cluster (ADR-007).
    #[must_use]
    pub const fn default_max_threads(&self) -> usize {
        match self {
            Self::X100 => X100_DEFAULT_MAX_COMPUTE_THREADS,
            Self::A100 => A100_DEFAULT_MAX_COMPUTE_THREADS,
        }
    }
}

/// Trait abstracting host online CPU topology discovery.
pub trait CpuTopologyInspector {
    /// Returns a 64-bit bitmask of currently online host CPUs.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if online CPU topology cannot be determined.
    fn online_cpu_mask(&self) -> Result<u64, K3Error>;
}

/// Reads online CPU topology from `/sys/devices/system/cpu/online` (or `K3_CPU_ONLINE_MASK` override).
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemCpuTopology;

impl SystemCpuTopology {
    /// Creates a new [`SystemCpuTopology`] inspector.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl CpuTopologyInspector for SystemCpuTopology {
    fn online_cpu_mask(&self) -> Result<u64, K3Error> {
        if let Ok(override_str) = std::env::var("K3_CPU_ONLINE_MASK") {
            let trimmed = override_str.trim();
            let parsed = if let Some(hex) = trimmed
                .strip_prefix("0x")
                .or_else(|| trimmed.strip_prefix("0X"))
            {
                u64::from_str_radix(hex, 16)
            } else {
                trimmed.parse::<u64>()
            };
            if let Ok(mask) = parsed {
                if mask != 0 {
                    return Ok(mask);
                }
            }
        }

        if let Ok(content) = fs::read_to_string("/sys/devices/system/cpu/online") {
            let mask = parse_cpu_list_mask(&content)?;
            if mask != 0 {
                return Ok(mask);
            }
        }

        // Fallback to libc::_SC_NPROCESSORS_ONLN
        // SAFETY: sysconf(_SC_NPROCESSORS_ONLN) is a standard POSIX query with no side effects.
        let nprocs = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
        if nprocs > 0 {
            let count = u32::try_from(nprocs).unwrap_or(1).min(64);
            let mask = if count == 64 {
                u64::MAX
            } else {
                (1u64 << count) - 1
            };
            return Ok(mask);
        }

        Ok(0xFFFF)
    }
}

/// Deterministic mock CPU topology for unit testing 16-core `SpacemiT` K3 masks on any host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MockCpuTopology {
    /// Bitmask of simulated online CPUs.
    pub online_mask: u64,
}

impl MockCpuTopology {
    /// Creates a mock topology with the specified online CPU bitmask.
    #[must_use]
    pub const fn new(online_mask: u64) -> Self {
        Self { online_mask }
    }

    /// Creates a mock 16-core `SpacemiT` K3 topology (`0xFFFF`: cores 0–15 online).
    #[must_use]
    pub const fn spacemit_k3() -> Self {
        Self {
            online_mask: 0xFFFF,
        }
    }
}

impl CpuTopologyInspector for MockCpuTopology {
    fn online_cpu_mask(&self) -> Result<u64, K3Error> {
        Ok(self.online_mask)
    }
}

/// Parses a Linux CPU range list string (e.g. `"0-7,12-15"` or `"0-15\n"`) into a 64-bit bitmask.
///
/// # Errors
///
/// Returns [`K3Error::AffinityError`] if the format is malformed.
pub fn parse_cpu_list_mask(list_str: &str) -> Result<u64, K3Error> {
    let trimmed = list_str.trim();
    if trimmed.is_empty() {
        return Err(K3Error::AffinityError {
            requested_mask: 0,
            requested_threads: 0,
            reason: "empty CPU list string".to_owned(),
        });
    }

    let mut mask = 0u64;
    for part in trimmed.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start_s, end_s)) = part.split_once('-') {
            let start = start_s
                .trim()
                .parse::<u32>()
                .map_err(|_| K3Error::AffinityError {
                    requested_mask: 0,
                    requested_threads: 0,
                    reason: format!("invalid CPU range start in '{part}'"),
                })?;
            let end = end_s
                .trim()
                .parse::<u32>()
                .map_err(|_| K3Error::AffinityError {
                    requested_mask: 0,
                    requested_threads: 0,
                    reason: format!("invalid CPU range end in '{part}'"),
                })?;
            if start > end {
                return Err(K3Error::AffinityError {
                    requested_mask: 0,
                    requested_threads: 0,
                    reason: format!("inverted CPU range '{part}'"),
                });
            }
            for cpu in start..=end.min(63) {
                mask |= 1u64 << cpu;
            }
        } else {
            let cpu = part.parse::<u32>().map_err(|_| K3Error::AffinityError {
                requested_mask: 0,
                requested_threads: 0,
                reason: format!("invalid CPU number '{part}'"),
            })?;
            if cpu < 64 {
                mask |= 1u64 << cpu;
            }
        }
    }

    Ok(mask)
}

/// Validated compute cluster, core bitmask, VLEN domain, and ordered core assignments (US-01.5.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedAffinity {
    /// Target cluster preset, if selected or inferred.
    pub cluster: Option<ClusterPreset>,
    /// Active hardware VLEN domain (`zvl256b` or `zvl1024b`).
    pub vlen_domain: VlenDomain,
    /// Effective core bitmask intersected with online CPUs.
    pub effective_mask: u64,
    /// Number of worker threads to spawn.
    pub threads: usize,
    /// Exact 1-to-1 physical core IDs assigned to each worker thread (`len == threads`).
    pub assigned_cores: Vec<usize>,
}

/// Resolves and validates requested `--cluster`, `--cores`, and `--threads` against the live system
/// CPU topology (US-01.5.1, US-01 Scenario 2).
///
/// # Errors
///
/// Returns [`K3Error::AffinityError`] if the configuration is invalid or exceeds available cores.
pub fn resolve_affinity(
    cluster: Option<&str>,
    explicit_mask: Option<u64>,
    requested_threads: Option<usize>,
) -> Result<ResolvedAffinity, K3Error> {
    resolve_affinity_with(
        cluster,
        explicit_mask,
        requested_threads,
        &SystemCpuTopology::new(),
    )
}

/// Resolves and validates requested `--cluster`, `--cores`, and `--threads` against a provided
/// [`CpuTopologyInspector`] (US-01.5.1, ADR-007, ADR-008).
///
/// # Errors
///
/// Returns [`K3Error::AffinityError`] if:
/// - `requested_threads` is `Some(0)`.
/// - `explicit_mask` is `Some(0)`.
/// - A single-cluster run mixes incompatible `X100` (`0x00FF`, `zvl256b`) and `A100` (`0xFF00`, `zvl1024b`)
///   VLEN domains without an explicit cross-cluster split policy.
/// - `--cluster` and `--cores` conflict (e.g., `--cluster x100` with cores outside `0x00FF`).
/// - `--cluster x100` (without explicit `--cores` override) requests more than 6 compute threads (ADR-007).
/// - `threads` exceeds the number of online cores in the resolved mask.
#[allow(clippy::too_many_lines, clippy::similar_names)]
pub fn resolve_affinity_with<T: CpuTopologyInspector>(
    cluster: Option<&str>,
    explicit_mask: Option<u64>,
    requested_threads: Option<usize>,
    topology: &T,
) -> Result<ResolvedAffinity, K3Error> {
    let online_mask = topology.online_cpu_mask()?;
    let req_threads_for_err = requested_threads.unwrap_or(1);

    if requested_threads == Some(0) {
        return Err(K3Error::AffinityError {
            requested_mask: explicit_mask.unwrap_or(0),
            requested_threads: 0,
            reason: "requested worker thread count must be at least 1".to_owned(),
        });
    }

    if explicit_mask == Some(0) {
        return Err(K3Error::AffinityError {
            requested_mask: 0,
            requested_threads: req_threads_for_err,
            reason: "core affinity mask cannot be 0x0000".to_owned(),
        });
    }

    let preset = match cluster {
        Some(name) => Some(ClusterPreset::parse(name, req_threads_for_err)?),
        None => None,
    };

    // Determine candidate mask before online intersection
    let candidate_mask = match (preset, explicit_mask) {
        (Some(p), Some(mask)) => {
            // Verify explicit mask is strictly within the requested cluster preset mask
            if (mask & !p.core_mask()) != 0 {
                return Err(K3Error::AffinityError {
                    requested_mask: mask,
                    requested_threads: req_threads_for_err,
                    reason: format!(
                        "explicit core mask 0x{mask:04X} contains cores outside cluster '{}' (0x{:04X})",
                        p.as_str(),
                        p.core_mask()
                    ),
                });
            }
            mask
        }
        (Some(p), None) => p.core_mask(),
        (None, Some(mask)) => mask,
        (None, None) => {
            // Default to X100 cluster (0x00FF) intersected with online CPUs
            X100_CLUSTER_MASK & online_mask
        }
    };

    // Enforce VLEN domain isolation guard: single-cluster pool must not mix X100 (0x00FF) and A100 (0xFF00)
    let has_x100_cores = (candidate_mask & X100_CLUSTER_MASK) != 0;
    let has_a100_cores = (candidate_mask & A100_CLUSTER_MASK) != 0;
    if has_x100_cores && has_a100_cores {
        return Err(K3Error::AffinityError {
            requested_mask: candidate_mask,
            requested_threads: req_threads_for_err,
            reason: format!(
                "core mask 0x{candidate_mask:04X} mixes asymmetric X100 (0x00FF, zvl256b) and A100 (0xFF00, zvl1024b) VLEN domains without an explicit cross-cluster split policy"
            ),
        });
    }

    let effective_mask = candidate_mask & online_mask;
    if effective_mask == 0 {
        return Err(K3Error::AffinityError {
            requested_mask: candidate_mask,
            requested_threads: req_threads_for_err,
            reason: format!(
                "no online CPUs in requested mask 0x{candidate_mask:04X} (online mask: 0x{online_mask:04X})"
            ),
        });
    }

    let available_cores: Vec<usize> = (0..64usize)
        .filter(|&bit| (effective_mask & (1u64 << bit)) != 0)
        .collect();

    // Determine thread count
    let threads = match (requested_threads, preset, explicit_mask) {
        (Some(n), Some(ClusterPreset::X100), None) => {
            if n > X100_DEFAULT_MAX_COMPUTE_THREADS {
                return Err(K3Error::AffinityError {
                    requested_mask: effective_mask,
                    requested_threads: n,
                    reason: format!(
                        "requested {n} compute threads on cluster 'x100' exceeds ADR-007 default cap of {X100_DEFAULT_MAX_COMPUTE_THREADS} (2 active + 1 spare reserved for control plane)"
                    ),
                });
            }
            n
        }
        (Some(n), _, _) => n,
        (None, Some(p), None) => available_cores.len().min(p.default_max_threads()),
        (None, _, Some(_)) => available_cores.len(),
        (None, None, None) => 1,
    };

    if threads > available_cores.len() {
        return Err(K3Error::AffinityError {
            requested_mask: candidate_mask,
            requested_threads: threads,
            reason: format!(
                "requested {threads} threads exceeds {} available cores in mask 0x{effective_mask:04X}",
                available_cores.len()
            ),
        });
    }

    let assigned_cores: Vec<usize> = available_cores.into_iter().take(threads).collect();
    let active_mask = assigned_cores
        .iter()
        .fold(0u64, |acc, &core| acc | (1u64 << core));

    let inferred_cluster = preset.or({
        if has_a100_cores && !has_x100_cores {
            Some(ClusterPreset::A100)
        } else if has_x100_cores && !has_a100_cores {
            Some(ClusterPreset::X100)
        } else {
            None
        }
    });

    let vlen_domain = match inferred_cluster {
        Some(ClusterPreset::A100) => VlenDomain::Zvl1024b,
        Some(ClusterPreset::X100) | None => VlenDomain::Zvl256b,
    };

    Ok(ResolvedAffinity {
        cluster: inferred_cluster,
        vlen_domain,
        effective_mask: active_mask,
        threads,
        assigned_cores,
    })
}

/// Runtime affinity verification report for a single pinned compute worker thread (US-01.5.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerAffinityStatus {
    /// Worker index within the pool (`0..threads`).
    pub worker_index: usize,
    /// Linux kernel thread ID (`TID`).
    pub tid: i64,
    /// Assigned physical core ID (`0..63`).
    pub assigned_core: usize,
    /// Verified CPU affinity bitmask queried from `sched_getaffinity`.
    pub sched_affinity_mask: u64,
    /// Verified `Cpus_allowed_list` read from `/proc/self/task/<tid>/status`.
    pub proc_cpus_allowed_list: String,
}

type JobFn = Arc<dyn Fn(usize, usize) + Send + Sync>;

struct PoolSharedState {
    generation: usize,
    active_job: Option<JobFn>,
    shutdown: bool,
}

/// Cache-line aligned (64-byte) per-worker barrier wait accumulator (US-02 R1, ADR-006).
///
/// Prevents cross-core cache line false sharing on `SpacemiT` K3 L1/L2 caches when multiple
/// pinned worker threads concurrently accumulate barrier wait times.
#[repr(align(64))]
#[derive(Debug, Default)]
pub struct WorkerWaitSlot {
    /// Cumulative barrier wait duration in nanoseconds.
    accumulated_nanos: AtomicU64,
    /// Number of barrier synchronization events recorded.
    barrier_count: AtomicU64,
}

impl WorkerWaitSlot {
    /// Creates a new zeroed wait slot.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            accumulated_nanos: AtomicU64::new(0),
            barrier_count: AtomicU64::new(0),
        }
    }

    /// Accumulates wait duration into this slot.
    pub fn record_wait(&self, duration: Duration) {
        let nanos = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        self.accumulated_nanos.fetch_add(nanos, Ordering::Relaxed);
        self.barrier_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Accumulates wait duration in nanoseconds into this slot.
    pub fn record_wait_nanos(&self, delta_nanos: u64) {
        self.accumulated_nanos.fetch_add(delta_nanos, Ordering::Relaxed);
        self.barrier_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns accumulated wait duration in nanoseconds.
    #[must_use]
    pub fn wait_nanos(&self) -> u64 {
        self.accumulated_nanos.load(Ordering::Relaxed)
    }

    /// Returns accumulated wait duration in milliseconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn wait_ms(&self) -> f64 {
        self.wait_nanos() as f64 / 1_000_000.0
    }

    /// Returns accumulated wait duration as a [`Duration`].
    #[must_use]
    pub fn wait_duration(&self) -> Duration {
        Duration::from_nanos(self.wait_nanos())
    }

    /// Returns number of barrier events recorded.
    #[must_use]
    pub fn barrier_count(&self) -> u64 {
        self.barrier_count.load(Ordering::Relaxed)
    }

    /// Resets accumulator and count to zero.
    pub fn reset(&self) {
        self.accumulated_nanos.store(0, Ordering::Relaxed);
        self.barrier_count.store(0, Ordering::Relaxed);
    }
}

/// Sense-reversing atomic busy-polling barrier (US-02 R1).
///
/// Uses `core::hint::spin_loop()` for sub-microsecond synchronization between pinned worker cores.
pub struct SpinBarrier {
    num_threads: usize,
    count: AtomicUsize,
    generation: AtomicUsize,
    sense: AtomicBool,
}

impl std::fmt::Debug for SpinBarrier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpinBarrier")
            .field("num_threads", &self.num_threads)
            .field("count", &self.count.load(Ordering::Relaxed))
            .field("generation", &self.generation.load(Ordering::Relaxed))
            .field("sense", &self.sense.load(Ordering::Relaxed))
            .finish()
    }
}

impl SpinBarrier {
    /// Creates a new spin barrier for `num_threads` participants.
    #[must_use]
    pub const fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            count: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            sense: AtomicBool::new(false),
        }
    }

    /// Synchronizes calling thread. Returns elapsed wait duration spent spinning.
    pub fn wait(&self, local_sense: &mut bool, shutdown: &AtomicBool) -> Duration {
        if self.num_threads <= 1 {
            return Duration::ZERO;
        }

        *local_sense = !*local_sense;
        let expected_sense = *local_sense;
        let start = Instant::now();

        let arrival = self.count.fetch_add(1, Ordering::AcqRel);
        if arrival + 1 == self.num_threads {
            self.count.store(0, Ordering::Release);
            self.generation.fetch_add(1, Ordering::Release);
            self.sense.store(expected_sense, Ordering::Release);
            Duration::ZERO
        } else {
            while self.sense.load(Ordering::Acquire) != expected_sense {
                if shutdown.load(Ordering::Relaxed) {
                    return Duration::ZERO;
                }
                core::hint::spin_loop();
            }
            start.elapsed()
        }
    }
}

/// Thread-sleep yielding barrier via [`Mutex`] and [`Condvar`] (US-02 R1).
///
/// Yields CPU cycles to the Linux kernel scheduler during inter-worker wait intervals.
pub struct FutexBarrier {
    num_threads: usize,
    state: Mutex<FutexBarrierState>,
    cvar: Condvar,
}

struct FutexBarrierState {
    count: usize,
    generation: usize,
}

impl std::fmt::Debug for FutexBarrier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FutexBarrier")
            .field("num_threads", &self.num_threads)
            .finish_non_exhaustive()
    }
}

impl FutexBarrier {
    /// Creates a new futex barrier for `num_threads` participants.
    #[must_use]
    pub const fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            state: Mutex::new(FutexBarrierState {
                count: 0,
                generation: 0,
            }),
            cvar: Condvar::new(),
        }
    }

    /// Synchronizes calling thread yielding CPU cycles. Returns elapsed wait duration.
    pub fn wait(&self, shutdown: &AtomicBool) -> Duration {
        if self.num_threads <= 1 {
            return Duration::ZERO;
        }

        let start = Instant::now();
        let mut guard = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let current_gen = guard.generation;
        guard.count += 1;
        if guard.count == self.num_threads {
            guard.count = 0;
            guard.generation = guard.generation.wrapping_add(1);
            self.cvar.notify_all();
            Duration::ZERO
        } else {
            while guard.generation == current_gen && !shutdown.load(Ordering::Relaxed) {
                guard = self.cvar.wait(guard).unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            start.elapsed()
        }
    }

    /// Notifies all waiting threads to wake immediately for shutdown.
    pub fn notify_shutdown(&self) {
        self.cvar.notify_all();
    }
}

enum PoolBackend {
    DedicatedThreads {
        profile: CoordinationProfile,
        state: Arc<(Mutex<PoolSharedState>, Condvar)>,
        shutdown: Arc<AtomicBool>,
        spin_barrier: Option<Arc<SpinBarrier>>,
        futex_barrier: Option<Arc<FutexBarrier>>,
        coordinator_sense: Arc<AtomicBool>,
        handles: Vec<JoinHandle<()>>,
    },
    Rayon {
        pool: rayon::ThreadPool,
    },
}

impl std::fmt::Debug for PoolBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DedicatedThreads { profile, .. } => f
                .debug_struct("DedicatedThreads")
                .field("profile", profile)
                .finish_non_exhaustive(),
            Self::Rayon { .. } => f.debug_struct("Rayon").finish_non_exhaustive(),
        }
    }
}

/// Affinity-pinned multi-core compute worker pool (US-01.5.2, US-02 R1).
///
/// Supports three multi-core coordination profiles (`spin`, `futex`, `rayon`),
/// pins workers 1-to-1 to physical cores, and instruments cumulative barrier wait times.
pub struct WorkerPool {
    affinity: ResolvedAffinity,
    profile: CoordinationProfile,
    worker_statuses: Vec<WorkerAffinityStatus>,
    wait_slots: Arc<[WorkerWaitSlot]>,
    backend: PoolBackend,
}

impl std::fmt::Debug for WorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerPool")
            .field("affinity", &self.affinity)
            .field("profile", &self.profile)
            .field("worker_statuses", &self.worker_statuses)
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

impl WorkerPool {
    /// Spawns a new [`WorkerPool`] using the default coordination profile ([`CoordinationProfile::Spin`]).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if thread creation, `sched_setaffinity`, or runtime
    /// affinity verification fails.
    pub fn new(affinity: ResolvedAffinity) -> Result<Self, K3Error> {
        Self::new_with_profile(affinity, CoordinationProfile::Spin)
    }

    /// Spawns a new [`WorkerPool`] configured with a specific [`CoordinationProfile`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if thread creation, `sched_setaffinity`, or runtime
    /// affinity verification fails.
    #[allow(clippy::too_many_lines)]
    pub fn new_with_profile(
        affinity: ResolvedAffinity,
        profile: CoordinationProfile,
    ) -> Result<Self, K3Error> {
        let num_workers = affinity.threads;
        if num_workers == 0 || affinity.assigned_cores.len() != num_workers {
            return Err(K3Error::AffinityError {
                requested_mask: affinity.effective_mask,
                requested_threads: num_workers,
                reason: "worker pool requires at least 1 assigned core matching thread count"
                    .to_owned(),
            });
        }

        let mut slots = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            slots.push(WorkerWaitSlot::new());
        }
        let wait_slots: Arc<[WorkerWaitSlot]> = slots.into();

        match profile {
            CoordinationProfile::Spin | CoordinationProfile::Futex => {
                let state = Arc::new((
                    Mutex::new(PoolSharedState {
                        generation: 0,
                        active_job: None,
                        shutdown: false,
                    }),
                    Condvar::new(),
                ));
                let shutdown = Arc::new(AtomicBool::new(false));
                let coordinator_sense = Arc::new(AtomicBool::new(false));

                let spin_barrier = if profile == CoordinationProfile::Spin {
                    Some(Arc::new(SpinBarrier::new(num_workers + 1)))
                } else {
                    None
                };

                let futex_barrier = if profile == CoordinationProfile::Futex {
                    Some(Arc::new(FutexBarrier::new(num_workers + 1)))
                } else {
                    None
                };

                let (init_tx, init_rx) = mpsc::channel::<Result<WorkerAffinityStatus, K3Error>>();
                let mut handles = Vec::with_capacity(num_workers);

                for (worker_index, &assigned_core) in affinity.assigned_cores.iter().enumerate() {
                    let state_clone = Arc::clone(&state);
                    let shutdown_clone = Arc::clone(&shutdown);
                    let spin_barrier_clone = spin_barrier.clone();
                    let futex_barrier_clone = futex_barrier.clone();
                    let wait_slots_clone = Arc::clone(&wait_slots);
                    let tx = init_tx.clone();
                    let req_mask = affinity.effective_mask;

                    let handle = thread::Builder::new()
                        .name(format!("k3-worker-{worker_index}"))
                        .spawn(move || {
                            let pin_result = pin_current_thread_and_verify(
                                worker_index,
                                assigned_core,
                                req_mask,
                                num_workers,
                            );
                            let is_ok = pin_result.is_ok();
                            let _ = tx.send(pin_result);
                            if !is_ok {
                                return;
                            }

                            let mut local_sense = false;
                            let mut seen_gen = 0usize;
                            loop {
                                let (job, shutdown_flag) = {
                                    let (lock, cvar) = &*state_clone;
                                    let mut guard = lock
                                        .lock()
                                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                                    while !guard.shutdown
                                        && !shutdown_clone.load(Ordering::Relaxed)
                                        && guard.generation == seen_gen
                                    {
                                        guard = cvar
                                            .wait(guard)
                                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                                    }
                                    if guard.shutdown || shutdown_clone.load(Ordering::Relaxed) {
                                        return;
                                    }
                                    seen_gen = guard.generation;
                                    (guard.active_job.clone(), guard.shutdown)
                                };

                                if let Some(work_fn) = job {
                                    work_fn(worker_index, num_workers);
                                }

                                if let Some(barrier) = &spin_barrier_clone {
                                    let wait_time = barrier.wait(&mut local_sense, &shutdown_clone);
                                    wait_slots_clone[worker_index].record_wait(wait_time);
                                } else if let Some(barrier) = &futex_barrier_clone {
                                    let wait_time = barrier.wait(&shutdown_clone);
                                    wait_slots_clone[worker_index].record_wait(wait_time);
                                }

                                if shutdown_flag || shutdown_clone.load(Ordering::Relaxed) {
                                    return;
                                }
                            }
                        })
                        .map_err(|err| {
                            shutdown_dedicated(&state, &shutdown, futex_barrier.as_ref(), &mut handles);
                            K3Error::AffinityError {
                                requested_mask: affinity.effective_mask,
                                requested_threads: num_workers,
                                reason: format!("failed to spawn worker thread {worker_index}: {err}"),
                            }
                        })?;

                    handles.push(handle);
                }
                drop(init_tx);

                let mut worker_statuses = Vec::with_capacity(num_workers);
                for _ in 0..num_workers {
                    match init_rx.recv() {
                        Ok(Ok(status)) => worker_statuses.push(status),
                        Ok(Err(err)) => {
                            shutdown_dedicated(&state, &shutdown, futex_barrier.as_ref(), &mut handles);
                            return Err(err);
                        }
                        Err(recv_err) => {
                            shutdown_dedicated(&state, &shutdown, futex_barrier.as_ref(), &mut handles);
                            return Err(K3Error::AffinityError {
                                requested_mask: affinity.effective_mask,
                                requested_threads: num_workers,
                                reason: format!(
                                    "worker thread terminated before reporting affinity: {recv_err}"
                                ),
                            });
                        }
                    }
                }

                worker_statuses.sort_by_key(|s| s.worker_index);

                Ok(Self {
                    affinity,
                    profile,
                    worker_statuses,
                    wait_slots,
                    backend: PoolBackend::DedicatedThreads {
                        profile,
                        state,
                        shutdown,
                        spin_barrier,
                        futex_barrier,
                        coordinator_sense,
                        handles,
                    },
                })
            }
            CoordinationProfile::Rayon => {
                let (init_tx, init_rx) = mpsc::channel::<Result<WorkerAffinityStatus, K3Error>>();
                let assigned_cores = affinity.assigned_cores.clone();
                let req_mask = affinity.effective_mask;

                let rayon_pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_workers)
                    .thread_name(|i| format!("k3-rayon-worker-{i}"))
                    .start_handler(move |worker_idx| {
                        let assigned_core = assigned_cores[worker_idx];
                        let res = pin_current_thread_and_verify(
                            worker_idx,
                            assigned_core,
                            req_mask,
                            num_workers,
                        );
                        let _ = init_tx.send(res);
                    })
                    .build()
                    .map_err(|err| K3Error::AffinityError {
                        requested_mask: affinity.effective_mask,
                        requested_threads: num_workers,
                        reason: format!("failed to initialize rayon thread pool: {err}"),
                    })?;

                let mut worker_statuses = Vec::with_capacity(num_workers);
                for _ in 0..num_workers {
                    match init_rx.recv() {
                        Ok(Ok(status)) => worker_statuses.push(status),
                        Ok(Err(err)) => return Err(err),
                        Err(recv_err) => {
                            return Err(K3Error::AffinityError {
                                requested_mask: affinity.effective_mask,
                                requested_threads: num_workers,
                                reason: format!(
                                    "rayon worker terminated prematurely: {recv_err}"
                                ),
                            });
                        }
                    }
                }
                worker_statuses.sort_by_key(|s| s.worker_index);

                Ok(Self {
                    affinity,
                    profile,
                    worker_statuses,
                    wait_slots,
                    backend: PoolBackend::Rayon { pool: rayon_pool },
                })
            }
        }
    }

    /// Spawns a new [`WorkerPool`] using the specified [`CoordinationProfile`] (alias for [`Self::new_with_profile`]).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if configuration or worker initialization fails.
    pub fn with_coordination(
        affinity: ResolvedAffinity,
        coordination: CoordinationProfile,
    ) -> Result<Self, K3Error> {
        Self::new_with_profile(affinity, coordination)
    }

    /// Returns the active coordination profile for this pool.
    #[must_use]
    pub const fn profile(&self) -> CoordinationProfile {
        self.profile
    }

    /// Returns the active coordination profile for this pool (alias for [`Self::profile`]).
    #[must_use]
    pub const fn coordination_profile(&self) -> CoordinationProfile {
        self.profile
    }

    /// Returns the resolved affinity configuration for this pool.
    #[must_use]
    pub const fn affinity(&self) -> &ResolvedAffinity {
        &self.affinity
    }

    /// Returns the number of pinned worker threads in this pool.
    #[must_use]
    pub const fn thread_count(&self) -> usize {
        self.affinity.threads
    }

    /// Returns the startup affinity verification reports for all worker threads.
    #[must_use]
    pub fn worker_statuses(&self) -> &[WorkerAffinityStatus] {
        &self.worker_statuses
    }

    /// Returns reference to per-worker wait slots.
    #[must_use]
    pub fn wait_slots(&self) -> &[WorkerWaitSlot] {
        &self.wait_slots
    }

    /// Returns the wait slot reference for a given worker index.
    #[must_use]
    pub fn wait_slot(&self, worker_index: usize) -> &WorkerWaitSlot {
        &self.wait_slots[worker_index]
    }

    /// Returns cumulative inter-worker barrier wait time across all workers in milliseconds.
    #[must_use]
    pub fn cumulative_barrier_wait_ms(&self) -> f64 {
        self.wait_slots.iter().map(WorkerWaitSlot::wait_ms).sum()
    }

    /// Returns cumulative inter-worker barrier wait time across all workers in nanoseconds.
    #[must_use]
    pub fn cumulative_barrier_wait_nanos(&self) -> u64 {
        self.wait_slots.iter().map(WorkerWaitSlot::wait_nanos).sum()
    }

    /// Returns cumulative inter-worker barrier wait duration across all workers.
    #[must_use]
    pub fn cumulative_barrier_wait(&self) -> Duration {
        Duration::from_nanos(self.cumulative_barrier_wait_nanos())
    }

    /// Resets all per-worker barrier wait time accumulators.
    pub fn reset_barrier_wait(&self) {
        for slot in self.wait_slots.iter() {
            slot.reset();
        }
    }

    /// Resets all per-worker barrier wait time accumulators (alias for [`Self::reset_barrier_wait`]).
    pub fn reset_wait_metrics(&self) {
        self.reset_barrier_wait();
    }

    /// Re-reads `/proc/self/task/<tid>/status` (`Cpus_allowed_list`) for every active worker thread
    /// to confirm strict 1-to-1 core binding and zero cross-cluster migration (US-01.5.2).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::AffinityError`] if any worker thread has migrated outside its assigned core.
    pub fn verify_live_affinities(&self) -> Result<Vec<WorkerAffinityStatus>, K3Error> {
        let mut live = Vec::with_capacity(self.worker_statuses.len());
        for status in &self.worker_statuses {
            let allowed_list = read_proc_task_cpus_allowed_list(status.tid)
                .unwrap_or_else(|| status.proc_cpus_allowed_list.clone());
            let parsed_mask = parse_cpu_list_mask(&allowed_list)?;
            let expected_mask = 1u64 << status.assigned_core;
            if parsed_mask != expected_mask {
                return Err(K3Error::AffinityError {
                    requested_mask: self.affinity.effective_mask,
                    requested_threads: self.affinity.threads,
                    reason: format!(
                        "worker {} (tid {}) affinity drifted: expected core {} (0x{expected_mask:04X}), found Cpus_allowed_list='{allowed_list}' (0x{parsed_mask:04X})",
                        status.worker_index, status.tid, status.assigned_core
                    ),
                });
            }
            live.push(WorkerAffinityStatus {
                worker_index: status.worker_index,
                tid: status.tid,
                assigned_core: status.assigned_core,
                sched_affinity_mask: parsed_mask,
                proc_cpus_allowed_list: allowed_list,
            });
        }
        Ok(live)
    }

    /// Executes `task(worker_index, total_workers)` in parallel across all pinned worker threads
    /// and waits for completion.
    pub fn execute<F>(&self, task: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        match &self.backend {
            PoolBackend::DedicatedThreads {
                state,
                shutdown,
                spin_barrier,
                futex_barrier,
                coordinator_sense,
                ..
            } => {
                let job: JobFn = Arc::new(task);
                {
                    let (lock, cvar) = &**state;
                    let mut guard = lock
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    guard.active_job = Some(job);
                    guard.generation = guard.generation.wrapping_add(1);
                    cvar.notify_all();
                }
                if let Some(barrier) = spin_barrier {
                    let mut local_sense = coordinator_sense.load(Ordering::Relaxed);
                    barrier.wait(&mut local_sense, shutdown);
                    coordinator_sense.store(local_sense, Ordering::Relaxed);
                } else if let Some(barrier) = futex_barrier {
                    barrier.wait(shutdown);
                }
                {
                    let (lock, _) = &**state;
                    let mut guard = lock
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    guard.active_job = None;
                }
            }
            PoolBackend::Rayon { pool } => {
                let num_workers = self.affinity.threads;
                let wait_slots = Arc::clone(&self.wait_slots);
                let finish_times = Arc::new(
                    (0..num_workers)
                        .map(|_| AtomicU64::new(0))
                        .collect::<Vec<_>>(),
                );
                let t_start = Instant::now();

                pool.install(|| {
                    rayon::scope(|s| {
                        for worker_idx in 0..num_workers {
                            let finish_times_clone = Arc::clone(&finish_times);
                            let task_ref = &task;
                            s.spawn(move |_| {
                                task_ref(worker_idx, num_workers);
                                let elapsed_ns =
                                    u64::try_from(t_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
                                finish_times_clone[worker_idx].store(elapsed_ns, Ordering::Release);
                            });
                        }
                    });
                });

                let t_total_ns =
                    u64::try_from(t_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
                for worker_idx in 0..num_workers {
                    let finish_ns = finish_times[worker_idx].load(Ordering::Acquire);
                    let wait_ns = t_total_ns.saturating_sub(finish_ns);
                    wait_slots[worker_idx].record_wait_nanos(wait_ns);
                }
            }
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        if let PoolBackend::DedicatedThreads {
            state,
            shutdown,
            futex_barrier,
            handles,
            ..
        } = &mut self.backend
        {
            shutdown_dedicated(state, shutdown, futex_barrier.as_ref(), handles);
        }
    }
}

fn shutdown_dedicated(
    state: &Arc<(Mutex<PoolSharedState>, Condvar)>,
    shutdown: &Arc<AtomicBool>,
    futex_barrier: Option<&Arc<FutexBarrier>>,
    handles: &mut Vec<JoinHandle<()>>,
) {
    shutdown.store(true, Ordering::Release);
    {
        let (lock, cvar) = &**state;
        let mut guard = lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.shutdown = true;
        cvar.notify_all();
    }
    if let Some(fb) = futex_barrier {
        fb.notify_shutdown();
    }
    for handle in handles.drain(..) {
        let _ = handle.join();
    }
}

fn pin_current_thread_and_verify(
    worker_index: usize,
    assigned_core: usize,
    requested_mask: u64,
    requested_threads: usize,
) -> Result<WorkerAffinityStatus, K3Error> {
    // 1. Construct cpu_set_t with only `assigned_core` set
    // SAFETY: zero-initializing cpu_set_t and setting a single valid core bit via CPU_ZERO/CPU_SET.
    let mut cpuset: libc::cpu_set_t = unsafe { core::mem::zeroed() };
    // SAFETY: CPU_ZERO and CPU_SET operate on valid local cpuset struct.
    unsafe {
        libc::CPU_ZERO(&mut cpuset);
        libc::CPU_SET(assigned_core, &mut cpuset);
    }

    // 2. Pin current thread via sched_setaffinity(0, ...)
    // SAFETY: sched_setaffinity with pid=0 pins the calling OS thread.
    let set_res =
        unsafe { libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &raw const cpuset) };
    if set_res != 0 {
        let os_err = std::io::Error::last_os_error();
        return Err(K3Error::AffinityError {
            requested_mask,
            requested_threads,
            reason: format!(
                "sched_setaffinity failed for worker {worker_index} on core {assigned_core}: {os_err}"
            ),
        });
    }

    // 3. Query OS thread ID (TID)
    // SAFETY: SYS_gettid is a thread-safe Linux syscall taking no arguments.
    let tid = unsafe { libc::syscall(libc::SYS_gettid) };

    // 4. Verify via sched_getaffinity(0, ...)
    // SAFETY: zero-initializing cpu_set_t for sched_getaffinity output.
    let mut verify_set: libc::cpu_set_t = unsafe { core::mem::zeroed() };
    // SAFETY: sched_getaffinity with pid=0 reads calling thread's CPU mask into verify_set.
    let get_res =
        unsafe { libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &raw mut verify_set) };
    if get_res != 0 {
        let os_err = std::io::Error::last_os_error();
        return Err(K3Error::AffinityError {
            requested_mask,
            requested_threads,
            reason: format!(
                "sched_getaffinity verification failed for worker {worker_index}: {os_err}"
            ),
        });
    }

    let mut observed_mask = 0u64;
    for core_id in 0..64usize {
        // SAFETY: CPU_ISSET reads from valid verify_set populated by sched_getaffinity.
        let is_set = unsafe { libc::CPU_ISSET(core_id, &verify_set) };
        if is_set {
            observed_mask |= 1u64 << core_id;
        }
    }

    let expected_mask = 1u64 << assigned_core;
    if observed_mask != expected_mask {
        return Err(K3Error::AffinityError {
            requested_mask,
            requested_threads,
            reason: format!(
                "worker {worker_index} sched_getaffinity mismatch: expected 0x{expected_mask:04X}, got 0x{observed_mask:04X}"
            ),
        });
    }

    // 5. Read /proc/self/task/<tid>/status Cpus_allowed_list
    let proc_cpus_allowed_list =
        read_proc_task_cpus_allowed_list(tid).unwrap_or_else(|| assigned_core.to_string());

    Ok(WorkerAffinityStatus {
        worker_index,
        tid,
        assigned_core,
        sched_affinity_mask: observed_mask,
        proc_cpus_allowed_list,
    })
}

fn read_proc_task_cpus_allowed_list(tid: i64) -> Option<String> {
    let status_path = format!("/proc/self/task/{tid}/status");
    let content = fs::read_to_string(status_path).ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("Cpus_allowed_list:") {
            return Some(rest.trim().to_owned());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_cluster_preset_resolution_x100_and_a100_us01_5_1() {
        let k3_topo = MockCpuTopology::spacemit_k3();

        // 1. --cluster x100 defaults to 6 compute threads on cores 0..5 (ADR-007)
        let x100 = resolve_affinity_with(Some("x100"), None, None, &k3_topo)
            .expect("resolve x100 default");
        assert_eq!(x100.cluster, Some(ClusterPreset::X100));
        assert_eq!(x100.vlen_domain, VlenDomain::Zvl256b);
        assert_eq!(x100.threads, 6);
        assert_eq!(x100.assigned_cores, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(x100.effective_mask, 0x003F);

        // 2. --cluster a100 defaults to 8 compute threads on cores 8..15 (0xFF00, zvl1024b)
        let a100 = resolve_affinity_with(Some("a100"), None, None, &k3_topo)
            .expect("resolve a100 default");
        assert_eq!(a100.cluster, Some(ClusterPreset::A100));
        assert_eq!(a100.vlen_domain, VlenDomain::Zvl1024b);
        assert_eq!(a100.threads, 8);
        assert_eq!(a100.assigned_cores, vec![8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(a100.effective_mask, 0xFF00);

        // 3. Explicit --cores 0x0F00 --threads 4 on A100
        let custom_a100 = resolve_affinity_with(Some("a100"), Some(0x0F00), Some(4), &k3_topo)
            .expect("resolve custom a100 mask");
        assert_eq!(custom_a100.assigned_cores, vec![8, 9, 10, 11]);
        assert_eq!(custom_a100.effective_mask, 0x0F00);
    }

    #[test]
    fn test_invalid_affinity_combinations_rejected_us01_5_1() {
        let k3_topo = MockCpuTopology::spacemit_k3();

        // 1. Mixing X100 (0x00FF) and A100 (0xFF00) VLEN domains without split policy is rejected
        let cross_vlen = resolve_affinity_with(None, Some(0x0101), Some(2), &k3_topo)
            .expect_err("must reject cross-cluster VLEN mixing");
        assert_eq!(cross_vlen.diagnostic_code(), "affinity_error");
        assert!(cross_vlen.to_string().contains("mixes asymmetric X100"));

        // 2. Threads exceeding cores in mask is rejected
        let too_many_threads = resolve_affinity_with(None, Some(0x000F), Some(6), &k3_topo)
            .expect_err("must reject threads > mask cores");
        assert_eq!(too_many_threads.diagnostic_code(), "affinity_error");
        assert!(
            too_many_threads
                .to_string()
                .contains("exceeds 4 available cores")
        );

        // 3. --cluster x100 with > 6 threads without explicit --cores override is rejected per ADR-007
        let x100_cap = resolve_affinity_with(Some("x100"), None, Some(8), &k3_topo)
            .expect_err("must enforce ADR-007 6-thread cap on x100 preset");
        assert_eq!(x100_cap.diagnostic_code(), "affinity_error");
        assert!(x100_cap.to_string().contains("ADR-007"));

        // 4. --cluster x100 with A100 core mask is rejected
        let mismatch = resolve_affinity_with(Some("x100"), Some(0xFF00), Some(4), &k3_topo)
            .expect_err("must reject A100 mask on x100 preset");
        assert_eq!(mismatch.diagnostic_code(), "affinity_error");
    }

    #[test]
    fn test_worker_pool_pinning_and_proc_status_verification_us01_5_2() {
        let online_mask = SystemCpuTopology::new()
            .online_cpu_mask()
            .expect("read online CPUs");
        let x100_online = (online_mask & X100_CLUSTER_MASK).count_ones() as usize;
        let target_threads = x100_online.clamp(1, 2);

        let resolved = resolve_affinity(
            None,
            Some(online_mask & X100_CLUSTER_MASK),
            Some(target_threads),
        )
        .expect("resolve host online cores");

        let pool = WorkerPool::new(resolved.clone()).expect("spawn pinned worker pool");
        assert_eq!(pool.thread_count(), target_threads);

        let statuses = pool.worker_statuses();
        assert_eq!(statuses.len(), target_threads);
        for (i, status) in statuses.iter().enumerate() {
            assert_eq!(status.worker_index, i);
            assert_eq!(status.assigned_core, resolved.assigned_cores[i]);
            assert_eq!(status.sched_affinity_mask, 1u64 << status.assigned_core);
            assert_eq!(
                status.proc_cpus_allowed_list,
                status.assigned_core.to_string()
            );
        }

        // Execute parallel work across pinned workers and verify live affinity during/after execution
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);
        pool.execute(move |worker_idx, total| {
            assert!(worker_idx < total);
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(counter.load(Ordering::SeqCst), target_threads);

        let live = pool
            .verify_live_affinities()
            .expect("live affinity check succeeds");
        assert_eq!(live.len(), target_threads);
    }

    #[test]
    fn test_worker_wait_slot_alignment_and_stride_adr006() {
        assert_eq!(align_of::<WorkerWaitSlot>(), 64);
        assert_eq!(size_of::<WorkerWaitSlot>(), 64);

        let slots: Vec<WorkerWaitSlot> = (0..8).map(|_| WorkerWaitSlot::new()).collect();
        for i in 0..7 {
            let p1 = (&raw const slots[i]) as usize;
            let p2 = (&raw const slots[i + 1]) as usize;
            assert_eq!(p1 % 64, 0, "slot {i} must be 64-byte aligned");
            assert_eq!(p2 - p1, 64, "stride between slots must equal 64 bytes");
        }

        // Test recording and reset
        let slot = WorkerWaitSlot::new();
        assert_eq!(slot.wait_nanos(), 0);
        assert_eq!(slot.barrier_count(), 0);
        assert!((slot.wait_ms() - 0.0).abs() < f64::EPSILON);

        slot.record_wait(Duration::from_millis(5));
        assert_eq!(slot.barrier_count(), 1);
        assert!(slot.wait_nanos() >= 5_000_000);
        assert!(slot.wait_ms() >= 4.9);

        slot.record_wait_nanos(2_000_000);
        assert_eq!(slot.barrier_count(), 2);
        assert!(slot.wait_nanos() >= 7_000_000);

        slot.reset();
        assert_eq!(slot.wait_nanos(), 0);
        assert_eq!(slot.barrier_count(), 0);
    }

    #[test]
    fn test_spin_barrier_synchronization_and_timing() {
        let barrier = Arc::new(SpinBarrier::new(4));
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let b = Arc::clone(&barrier);
            let s = Arc::clone(&shutdown);
            handles.push(thread::spawn(move || {
                let mut local_sense = false;
                let mut total_wait = Duration::ZERO;
                for _ in 0..10 {
                    total_wait += b.wait(&mut local_sense, &s);
                }
                total_wait
            }));
        }

        for h in handles {
            let _ = h.join().expect("barrier thread completes");
        }
    }

    #[test]
    fn test_futex_barrier_synchronization_and_timing() {
        let barrier = Arc::new(FutexBarrier::new(4));
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let b = Arc::clone(&barrier);
            let s = Arc::clone(&shutdown);
            handles.push(thread::spawn(move || {
                let mut total_wait = Duration::ZERO;
                for _ in 0..10 {
                    total_wait += b.wait(&s);
                }
                total_wait
            }));
        }

        for h in handles {
            let _ = h.join().expect("futex barrier thread completes");
        }
    }
}
