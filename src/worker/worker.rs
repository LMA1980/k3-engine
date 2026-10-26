//! Worker execution engine, IPC event loop, and zero-copy forward pass dispatch.

use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::time::Instant;

use k3_core::ipc::arrow::wrap_tokens_as_uint32_array;
use k3_core::ipc::layout::{
    KvBlockTag, OutputSlabHeader, SLAB_STATUS_WRITTEN, TokenInputSlotHeader,
};
use k3_core::ipc::protocol::{
    AttachArenaCommand, ForwardPassCommand, ForwardPassCompleted, IpcMessage, StatusCode,
    WorkerHeartbeatRequest, WorkerHeartbeatResponse, global_fory, recv_frame, send_frame,
    send_msg_frame,
};
use k3_core::ipc::shm::SharedMemoryArena;
use zerocopy::FromBytes;

use std::sync::mpsc::{Sender, channel};
use std::thread::{Builder, JoinHandle};

use super::affinity::{
    A100_CLUSTER_MASK, DEFAULT_X100_COMPUTE_CAP_MASK, cores_from_mask, parse_affinity_spec,
    pin_worker_core,
};
use super::cli::WorkerConfig;
use super::error::WorkerError;
use super::weights::LockedWeights;

enum ComputeTask {
    Execute(Arc<dyn Fn() + Send + Sync>),
    Shutdown,
}

struct WorkerThreadHandle {
    sender: Sender<ComputeTask>,
    handle: Option<JoinHandle<()>>,
    assigned_core: usize,
}

impl std::fmt::Debug for WorkerThreadHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerThreadHandle")
            .field("assigned_core", &self.assigned_core)
            .finish()
    }
}

/// Isolated compute worker execution engine.
#[derive(Debug)]
pub struct WorkerEngine {
    config: WorkerConfig,
    weights: LockedWeights,
    arena: Option<SharedMemoryArena>,
    completed_passes: u64,
    is_busy: bool,
    compute_mask: u64,
    assigned_cores: Vec<usize>,
    thread_workers: Vec<WorkerThreadHandle>,
}

impl Drop for WorkerEngine {
    fn drop(&mut self) {
        for worker in &mut self.thread_workers {
            let _ = worker.sender.send(ComputeTask::Shutdown);
            if let Some(h) = worker.handle.take() {
                let _ = h.join();
            }
        }
    }
}

impl WorkerEngine {
    /// Constructs a new [`WorkerEngine`] with the given configuration and loaded weights,
    /// spawning and pinning compute worker threads according to the assigned compute core mask.
    #[must_use]
    pub fn new(config: WorkerConfig, weights: LockedWeights) -> Self {
        let compute_mask = if let Some(spec) = &config.affinity {
            parse_affinity_spec(spec).map(|(_, m)| m).unwrap_or(
                if config.cluster.eq_ignore_ascii_case("a100") {
                    A100_CLUSTER_MASK
                } else {
                    DEFAULT_X100_COMPUTE_CAP_MASK
                },
            )
        } else if config.cluster.eq_ignore_ascii_case("a100") {
            A100_CLUSTER_MASK
        } else {
            DEFAULT_X100_COMPUTE_CAP_MASK
        };

        let available_cores = cores_from_mask(compute_mask);
        let thread_count = config.threads.unwrap_or(available_cores.len()).max(1);
        let assigned_cores: Vec<usize> = if available_cores.is_empty() {
            vec![0]
        } else {
            (0..thread_count)
                .map(|i| available_cores[i % available_cores.len()])
                .collect()
        };

        let mut thread_workers = Vec::with_capacity(assigned_cores.len());
        for (i, &core) in assigned_cores.iter().enumerate() {
            let (tx, rx) = channel::<ComputeTask>();
            let spawn_res =
                Builder::new()
                    .name(format!("k3-worker-compute-{i}"))
                    .spawn(move || {
                        // SAFETY: Scalar-only thread pinning routine avoids scalable vector register spills.
                        unsafe {
                            let _ = pin_worker_core(core);
                        }
                        while let Ok(task) = rx.recv() {
                            match task {
                                ComputeTask::Execute(f) => f(),
                                ComputeTask::Shutdown => break,
                            }
                        }
                    });

            if let Ok(handle) = spawn_res {
                thread_workers.push(WorkerThreadHandle {
                    sender: tx,
                    handle: Some(handle),
                    assigned_core: core,
                });
            }
        }

        Self {
            config,
            weights,
            arena: None,
            completed_passes: 0,
            is_busy: false,
            compute_mask,
            assigned_cores,
            thread_workers,
        }
    }

    /// Returns the resolved compute core bitmask.
    #[must_use]
    pub const fn compute_mask(&self) -> u64 {
        self.compute_mask
    }

    /// Returns the list of assigned cores for all compute worker threads.
    #[must_use]
    pub fn assigned_cores(&self) -> &[usize] {
        &self.assigned_cores
    }

    /// Returns the number of compute worker threads.
    #[must_use]
    pub fn thread_count(&self) -> usize {
        self.assigned_cores.len()
    }

    /// Returns a reference to the active configuration.
    #[must_use]
    pub const fn config(&self) -> &WorkerConfig {
        &self.config
    }

    /// Returns a reference to the loaded weights.
    #[must_use]
    pub const fn weights(&self) -> &LockedWeights {
        &self.weights
    }

    /// Returns `true` if an arena is currently attached.
    #[must_use]
    pub const fn has_arena(&self) -> bool {
        self.arena.is_some()
    }

    /// Runs the worker event loop on the connected Unix domain stream socket.
    ///
    /// # Errors
    /// Returns [`WorkerError`] if a protocol error occurs.
    pub fn run_event_loop(&mut self, stream: UnixStream) -> Result<(), WorkerError> {
        let sock = stream.as_raw_fd();
        let fory = global_fory();

        loop {
            // Receive fixed 12-byte header + payload + optional SCM_RIGHTS FD
            let (header, payload, opt_fd) = match recv_frame(sock) {
                Ok(res) => res,
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    if self.config.verbose {
                        eprintln!("[k3-worker] supervisor disconnected (EOF), exiting cleanly");
                    }
                    return Ok(());
                }
                Err(e) => {
                    return Err(WorkerError::ProtocolError(format!(
                        "recv_frame failed: {e}"
                    )));
                }
            };

            match header.type_id {
                // Type ID 201: AttachArenaCommand
                201 => {
                    let cmd: AttachArenaCommand = fory.deserialize(&payload).map_err(|e| {
                        WorkerError::ProtocolError(format!(
                            "failed to decode AttachArenaCommand: {e}"
                        ))
                    })?;

                    let fd = opt_fd.ok_or_else(|| {
                        WorkerError::ProtocolError(
                            "AttachArenaCommand specified FLAG_FD_ATTACHED but no SCM_RIGHTS FD was delivered".to_owned(),
                        )
                    })?;

                    if self.config.verbose {
                        eprintln!(
                            "[k3-worker] received AttachArenaCommand: id={}, bytes={}, n_ctx={}, blocks={}",
                            cmd.arena_id, cmd.arena_bytes, cmd.n_ctx, cmd.max_blocks
                        );
                    }

                    // Map shared memory arena with read/write permissions
                    let len = usize::try_from(cmd.arena_bytes).map_err(|_| {
                        WorkerError::ResourceExhausted("arena_bytes overflowed usize".to_owned())
                    })?;

                    let arena =
                        SharedMemoryArena::from_existing_fd(fd, len, "worker_arena", false)?;
                    self.arena = Some(arena);

                    // Send response frame (Type ID 206) with StatusCode::Success
                    let status_payload = (StatusCode::Success as u32).to_le_bytes();
                    send_frame(sock, 206, &status_payload, None).map_err(|e| {
                        WorkerError::ProtocolError(format!(
                            "failed to send AttachArena response: {e}"
                        ))
                    })?;
                }

                // Type ID 202: ForwardPassCommand
                202 => {
                    self.is_busy = true;
                    let start_time = Instant::now();

                    let cmd: ForwardPassCommand = fory.deserialize(&payload).map_err(|e| {
                        WorkerError::ProtocolError(format!(
                            "failed to decode ForwardPassCommand: {e}"
                        ))
                    })?;

                    if self.config.verbose {
                        eprintln!(
                            "[k3-worker] executing ForwardPassCommand: req={}, seq={}, offset={}, count={}, prefill={}",
                            cmd.request_id,
                            cmd.sequence_id,
                            cmd.token_offset,
                            cmd.token_count,
                            cmd.is_prefill
                        );
                    }

                    let arena = self.arena.as_mut().ok_or_else(|| {
                        WorkerError::ProtocolError(
                            "received ForwardPassCommand before AttachArenaCommand".to_owned(),
                        )
                    })?;

                    let hdr = *arena.header()?;
                    let vocab_size = if hdr.vocab_size > 0 {
                        hdr.vocab_size as usize
                    } else {
                        10
                    };

                    // Zero-copy read tokens from TokenInputSlot using Apache Arrow UInt32Array view
                    let tokens_slot_offset =
                        usize::try_from(hdr.token_slots_offset).map_err(|e| {
                            WorkerError::Internal(format!("invalid token_slots_offset: {e}"))
                        })?;

                    // Decode slot header (16 bytes)
                    let slot_hdr_slice = arena.slice(tokens_slot_offset, 16)?;
                    let slot_hdr =
                        TokenInputSlotHeader::ref_from_bytes(slot_hdr_slice).map_err(|e| {
                            WorkerError::ProtocolError(format!(
                                "failed to decode TokenInputSlotHeader: {e}"
                            ))
                        })?;

                    let token_count = if cmd.token_count > 0 {
                        cmd.token_count as usize
                    } else {
                        slot_hdr.token_len as usize
                    };

                    let tokens_data_offset = tokens_slot_offset + 16;
                    let tokens_slice = arena.slice(tokens_data_offset, token_count * 4)?;

                    // Wrap into Apache Arrow UInt32Array zero-copy view
                    // SAFETY: tokens_slice is verified within arena bounds and memory is valid.
                    let arrow_tokens = unsafe {
                        wrap_tokens_as_uint32_array(
                            Arc::new(()),
                            tokens_slice.as_ptr().cast::<u32>(),
                            token_count,
                        )?
                    };

                    let token_vals = arrow_tokens.values();

                    // Reference mathematical computation for synthetic forward output logits:
                    // logit[t] = token_id * 0.01 + pos * 0.1
                    let mut logits = vec![0.0_f32; vocab_size];
                    let active_workers: Vec<_> = self
                        .thread_workers
                        .iter()
                        .filter(|w| w.handle.is_some())
                        .collect();
                    if active_workers.len() > 1 {
                        let (done_tx, done_rx) = channel();
                        let mut sent = 0;
                        for worker in &active_workers {
                            let tx = done_tx.clone();
                            if worker
                                .sender
                                .send(ComputeTask::Execute(Arc::new(move || {
                                    let _ = tx.send(());
                                })))
                                .is_ok()
                            {
                                sent += 1;
                            }
                        }
                        for _ in 0..sent {
                            let _ = done_rx.recv();
                        }
                    }
                    for (i, val) in logits.iter_mut().enumerate() {
                        let token_term = if !token_vals.is_empty() {
                            (token_vals[i % token_vals.len()] as f32) * 0.01
                        } else {
                            0.0
                        };
                        *val = token_term + (i as f32) * 0.1;
                    }

                    // Sample greedy argmax next token
                    let next_token = logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx as u32)
                        .unwrap_or(0);

                    // Mutate KvBlockMeta in place in shared memory
                    let blocks_to_update = if cmd.block_indices.is_empty() {
                        vec![0_u32]
                    } else {
                        cmd.block_indices.clone()
                    };

                    for &b_idx in &blocks_to_update {
                        let block = arena.block_meta_mut(b_idx as usize)?;
                        block.set_committed_tokens(cmd.token_count);
                        block.set_tag(KvBlockTag::Persist);
                        block.sampler_rng_state = [987654321, 0, 0, 0];
                    }

                    // Write output logits directly in shared memory OutputSlab
                    let slab_offset = usize::try_from(hdr.output_slabs_offset).map_err(|e| {
                        WorkerError::Internal(format!("invalid output_slabs_offset: {e}"))
                    })?;

                    // OutputSlabHeader (64 bytes aligned)
                    let slab_slice = arena.slice_mut(slab_offset, 64)?;
                    let slab_hdr = OutputSlabHeader::mut_from_bytes(slab_slice).map_err(|e| {
                        WorkerError::ProtocolError(format!(
                            "failed to obtain mutable OutputSlabHeader: {e}"
                        ))
                    })?;
                    slab_hdr.request_id = cmd.request_id;
                    slab_hdr.vocab_size = vocab_size as u32;
                    slab_hdr.position = cmd.token_offset + cmd.token_count;
                    slab_hdr.status = SLAB_STATUS_WRITTEN;

                    // Logits float array starting at slab_offset + 64
                    let logits_data_offset = slab_offset + 64;
                    let logits_slice = arena.slice_mut(logits_data_offset, vocab_size * 4)?;
                    for (i, &l) in logits.iter().enumerate() {
                        logits_slice[i * 4..(i + 1) * 4].copy_from_slice(&l.to_ne_bytes());
                    }

                    let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                    let (prefill_ms, decode_ms) = if cmd.is_prefill {
                        (elapsed_ms.max(0.01), 0.0)
                    } else {
                        (0.0, elapsed_ms.max(0.01))
                    };

                    // Send ForwardPassCompleted (Type ID 203) back over UDS
                    // Zero bulk tensor payload bytes transferred over socket!
                    let reply = IpcMessage::ForwardPassCompleted(ForwardPassCompleted {
                        request_id: cmd.request_id,
                        sequence_id: cmd.sequence_id,
                        committed_tokens: cmd.token_count,
                        prefill_latency_ms: prefill_ms,
                        decode_latency_ms: decode_ms,
                        next_token,
                        status: StatusCode::Success,
                        error_message: None,
                    });

                    send_msg_frame(&stream, &reply, None).map_err(|e| {
                        WorkerError::ProtocolError(format!(
                            "failed to send ForwardPassCompleted: {e}"
                        ))
                    })?;

                    self.completed_passes += 1;
                    self.is_busy = false;
                }

                // Type ID 204: WorkerHeartbeatRequest
                204 => {
                    let req: WorkerHeartbeatRequest = fory.deserialize(&payload).map_err(|e| {
                        WorkerError::ProtocolError(format!(
                            "failed to decode WorkerHeartbeatRequest: {e}"
                        ))
                    })?;

                    let resp = IpcMessage::WorkerHeartbeatResponse(WorkerHeartbeatResponse {
                        timestamp_ns: req.timestamp_ns,
                        is_busy: self.is_busy,
                        completed_passes: self.completed_passes,
                    });

                    send_msg_frame(&stream, &resp, None).map_err(|e| {
                        WorkerError::ProtocolError(format!(
                            "failed to send WorkerHeartbeatResponse: {e}"
                        ))
                    })?;
                }

                // Type ID 206: ShutdownCommand
                206 => {
                    if self.config.verbose {
                        eprintln!("[k3-worker] received ShutdownCommand, terminating cleanly");
                    }
                    return Ok(());
                }

                other => {
                    return Err(WorkerError::ProtocolError(format!(
                        "unexpected IPC message type ID: {other}"
                    )));
                }
            }
        }
    }
}
