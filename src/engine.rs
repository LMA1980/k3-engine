//! Cycle-exclusive single-prompt forward-pass engine loop (US-01.6.1, ADR-005, ADR-027).
//!
//! Enforces strict single-prompt cycle exclusivity (zero cross-prompt batching),
//! transitions deterministically from prompt prefill to token-by-token autoregressive decode,
//! and invokes an unbuffered streaming callback on every generated token.

use std::sync::atomic::{AtomicBool, Ordering};

use k3_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights};

use crate::affinity::WorkerPool;
use crate::kv::SingleSlotKvArena;

/// Execution phase of the single-prompt forward pass state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnginePhase {
    /// Engine is idle and ready to accept a single prompt.
    Idle,
    /// Executing prompt prefill across all input tokens.
    Prefill,
    /// Executing autoregressive single-token decode steps.
    Decode,
    /// Forward pass completed cleanly.
    Completed,
}

/// Flow-control action returned by the streaming token callback after each generated token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamAction {
    /// Continue generating the next token.
    Continue,
    /// Stop generation immediately (e.g., EOS/stop sequence reached or signal abort).
    Stop,
}

/// Summary statistics and generated token sequence from a completed single-prompt evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationSummary {
    /// Number of prompt tokens processed during prefill.
    pub prompt_tokens: usize,
    /// Sequence of generated token IDs produced during prefill + decode.
    pub generated_tokens: Vec<u32>,
    /// Final KV cache sequence cursor position (`prompt_tokens + decode_steps`).
    pub final_position: usize,
    /// Final state machine phase (`EnginePhase::Completed`).
    pub final_phase: EnginePhase,
}

/// Single-slot forward-pass orchestrator enforcing strict cycle exclusivity (ADR-005, US-01.6.1).
#[derive(Debug)]
pub struct SinglePromptEngine {
    active_cycle: AtomicBool,
    phase: EnginePhase,
}

impl Default for SinglePromptEngine {
    fn default() -> Self {
        Self::new()
    }
}

struct CycleGuard<'a> {
    flag: &'a AtomicBool,
}

impl<'a> CycleGuard<'a> {
    fn acquire(flag: &'a AtomicBool, arch: &str) -> Result<Self, K3Error> {
        if flag
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: "cycle exclusivity violation (ADR-005): another prompt is already executing on this engine instance".to_owned(),
            });
        }
        Ok(Self { flag })
    }
}

impl Drop for CycleGuard<'_> {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::SeqCst);
    }
}

impl SinglePromptEngine {
    /// Creates a new idle [`SinglePromptEngine`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            active_cycle: AtomicBool::new(false),
            phase: EnginePhase::Idle,
        }
    }

    /// Returns the current phase of the engine state machine.
    #[must_use]
    pub const fn phase(&self) -> EnginePhase {
        self.phase
    }

    /// Returns `true` if a prompt is currently executing on this engine instance.
    #[must_use]
    pub fn is_busy(&self) -> bool {
        self.active_cycle.load(Ordering::SeqCst)
    }

    /// Executes an isolated single-prompt forward pass (prefill + autoregressive decode)
    /// without a multi-thread worker pool (single-thread baseline).
    ///
    /// # Errors
    ///
    /// - Returns [`K3Error::CapabilityMismatch`] if another prompt is already active (ADR-005)
    ///   or if `prompt_tokens` / `vocab_size` is empty.
    /// - Returns [`K3Error::ContextLengthExceeded`] if `prompt_tokens.len() > kv_arena.n_ctx()`.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_prompt<F>(
        &mut self,
        recipe: &dyn ArchRecipe,
        weights: &dyn ModelWeights,
        kv_arena: &mut SingleSlotKvArena,
        prompt_tokens: &[u32],
        vocab_size: usize,
        max_tokens: usize,
        on_token: F,
    ) -> Result<EvaluationSummary, K3Error>
    where
        F: FnMut(u32) -> Result<StreamAction, K3Error>,
    {
        self.evaluate_prompt_with_pool(
            recipe,
            weights,
            kv_arena,
            None,
            prompt_tokens,
            vocab_size,
            max_tokens,
            on_token,
        )
    }

    /// Executes an isolated single-prompt forward pass dispatching across an optional pinned
    /// [`WorkerPool`] (US-01.5.2, US-01.6.1, US-01 Scenario 1 & 2).
    ///
    /// # Errors
    ///
    /// - Returns [`K3Error::CapabilityMismatch`] if another prompt is already active (ADR-005)
    ///   or if `prompt_tokens` / `vocab_size` is empty.
    /// - Returns [`K3Error::ContextLengthExceeded`] if `prompt_tokens.len() > kv_arena.n_ctx()`.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_prompt_with_pool<F>(
        &mut self,
        recipe: &dyn ArchRecipe,
        weights: &dyn ModelWeights,
        kv_arena: &mut SingleSlotKvArena,
        worker_pool: Option<&WorkerPool>,
        prompt_tokens: &[u32],
        vocab_size: usize,
        max_tokens: usize,
        mut on_token: F,
    ) -> Result<EvaluationSummary, K3Error>
    where
        F: FnMut(u32) -> Result<StreamAction, K3Error>,
    {
        let _guard = CycleGuard::acquire(&self.active_cycle, recipe.architecture_id())?;

        if prompt_tokens.is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: recipe.architecture_id().to_owned(),
                reason: "prompt token sequence cannot be empty".to_owned(),
            });
        }

        if vocab_size == 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: recipe.architecture_id().to_owned(),
                reason: "vocabulary size must be greater than zero".to_owned(),
            });
        }

        let n_ctx = kv_arena.n_ctx();
        let mut logits = vec![0.0_f32; vocab_size];
        let mut ctx =
            ExecutionContext::new(prompt_tokens, n_ctx, &mut logits, kv_arena.as_mut_slice())?;

        // 1. Transition Idle -> Prefill
        self.phase = EnginePhase::Prefill;
        if let Some(pool) = worker_pool {
            pool.execute(|_worker_idx, _total_workers| {
                // Synchronize pinned worker threads on prefill barrier
            });
        }
        recipe.prefill_step(&mut ctx, weights)?;

        let mut generated_tokens = Vec::with_capacity(max_tokens);
        if max_tokens == 0 {
            self.phase = EnginePhase::Completed;
            return Ok(EvaluationSummary {
                prompt_tokens: prompt_tokens.len(),
                generated_tokens,
                final_position: ctx.position(),
                final_phase: self.phase,
            });
        }

        // Sample first token produced by prefill logits
        let mut current_token =
            sample_greedy_argmax(ctx.logits()).ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: recipe.architecture_id().to_owned(),
                reason: "empty logits buffer after prefill step".to_owned(),
            })?;

        generated_tokens.push(current_token);
        if on_token(current_token)? == StreamAction::Stop {
            self.phase = EnginePhase::Completed;
            return Ok(EvaluationSummary {
                prompt_tokens: prompt_tokens.len(),
                generated_tokens,
                final_position: ctx.position(),
                final_phase: self.phase,
            });
        }

        // 2. Transition Prefill -> Decode
        self.phase = EnginePhase::Decode;
        while generated_tokens.len() < max_tokens && ctx.position() < n_ctx {
            ctx.prepare_decode(current_token)?;
            if let Some(pool) = worker_pool {
                pool.execute(|_worker_idx, _total_workers| {
                    // Synchronize pinned worker threads on decode step
                });
            }
            recipe.decode_step(&mut ctx, weights)?;

            current_token =
                sample_greedy_argmax(ctx.logits()).ok_or_else(|| K3Error::CapabilityMismatch {
                    architecture: recipe.architecture_id().to_owned(),
                    reason: "empty logits buffer after decode step".to_owned(),
                })?;

            generated_tokens.push(current_token);
            if on_token(current_token)? == StreamAction::Stop {
                break;
            }
        }

        self.phase = EnginePhase::Completed;
        Ok(EvaluationSummary {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens,
            final_position: ctx.position(),
            final_phase: self.phase,
        })
    }
}

/// Deterministic greedy (temperature `0.0`) argmax token sampler with lowest-token-ID tie-breaking.
#[must_use]
pub fn sample_greedy_argmax(logits: &[f32]) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }

    let mut best_id = 0_usize;
    let mut best_score = logits[0];

    for (i, &score) in logits.iter().enumerate().skip(1) {
        if score > best_score {
            best_score = score;
            best_id = i;
        }
    }

    u32::try_from(best_id).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use k3_core::GgufFile;
    use k3_core::QuantType;
    use k3_core::test_helpers::create_synthetic_gguf_v3;

    #[cfg(feature = "arch-synthetic")]
    #[test]
    fn test_single_prompt_engine_prefill_and_decode_loop_us01_6_1() {
        use arch_synthetic::SyntheticRecipe;

        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let gguf = GgufFile::parse(&bytes).expect("parse gguf");
        let mut kv_arena = SingleSlotKvArena::new(gguf.metadata(), 32).expect("allocate kv arena");
        let recipe = SyntheticRecipe::new();

        let mut engine = SinglePromptEngine::new();
        assert_eq!(engine.phase(), EnginePhase::Idle);

        let mut streamed = Vec::new();
        let summary = engine
            .evaluate_prompt(&recipe, &gguf, &mut kv_arena, &[1, 9, 10], 16, 4, |tok| {
                streamed.push(tok);
                Ok(StreamAction::Continue)
            })
            .expect("evaluate prompt");

        assert_eq!(summary.prompt_tokens, 3);
        assert_eq!(summary.generated_tokens, vec![9, 10, 11, 12]);
        assert_eq!(streamed, vec![9, 10, 11, 12]);
        assert_eq!(summary.final_position, 3 + 3); // 3 prefill + 3 decode steps = 6
        assert_eq!(summary.final_phase, EnginePhase::Completed);
        assert!(!engine.is_busy());
    }

    #[test]
    fn test_cycle_exclusivity_guard_rejects_concurrent_prompt_adr005() {
        let flag = AtomicBool::new(false);
        let _guard1 = CycleGuard::acquire(&flag, "synthetic").expect("first guard acquires slot");
        let err = CycleGuard::acquire(&flag, "synthetic")
            .err()
            .expect("second concurrent prompt must be rejected");
        assert_eq!(err.diagnostic_code(), "capability_mismatch");
        assert!(err.to_string().contains("cycle exclusivity violation"));
    }
}
