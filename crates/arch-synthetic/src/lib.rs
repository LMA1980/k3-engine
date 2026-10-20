//! `arch-synthetic`: Synthetic architecture recipe and deterministic token sampler.
//!
//! Implements [`ArchRecipe`] for the reference synthetic model architecture,
//! providing execution step lifecycle methods and greedy argmax sampling
//! with lowest-token-ID tie-breaking per ADR-005.

use k3_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType};

/// Synthetic architecture recipe implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct SyntheticRecipe;

impl SyntheticRecipe {
    /// Creates a new instance of the synthetic architecture recipe.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl ArchRecipe for SyntheticRecipe {
    fn architecture_id(&self) -> &'static str {
        "synthetic"
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::F32 | QuantType::F16 | QuantType::BF16 | QuantType::Q4_K_M
        )
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let prompt_len = ctx.prompt_tokens().len();
        ctx.advance_position(prompt_len)?;

        // Produce initial logits for next token selection.
        let logits = ctx.logits_mut();
        if !logits.is_empty() {
            logits.fill(0.0);
            logits[0] = 1.0;
        }

        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let token = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: "decode_step called without a staged decode token".to_owned(),
            })?;

        ctx.advance_position(1)?;

        // Simulate deterministic autoregressive next-token logits.
        let logits = ctx.logits_mut();
        if !logits.is_empty() {
            logits.fill(0.0);
            let next_slot = (usize::try_from(token).unwrap_or(0) + 1) % logits.len();
            logits[next_slot] = 2.0;
        }

        Ok(())
    }
}

/// Deterministically selects the token ID with the highest logit score.
///
/// Ties are broken by selecting the lowest token ID to guarantee reproducible
/// execution per ADR-005. Returns `None` if `logits` is empty.
#[must_use]
pub fn sample_argmax(logits: &[f32]) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }

    let mut best_id = 0_usize;
    let mut best_score = logits[0];

    for (i, &score) in logits.iter().enumerate().skip(1) {
        // If score is strictly greater, update.
        // In case of a tie (score == best_score), keep the lowest index.
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

    struct DummyWeights;

    impl ModelWeights for DummyWeights {
        fn tensor(&self, name: &str) -> Result<k3_core::TensorView<'_>, K3Error> {
            Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "no tensors in dummy weights".to_owned(),
            })
        }

        fn tensor_count(&self) -> usize {
            0
        }

        fn total_weight_bytes(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_synthetic_recipe_lifecycle() {
        let recipe = SyntheticRecipe::new();
        assert_eq!(recipe.architecture_id(), "synthetic");
        assert!(recipe.supports_quant(QuantType::Q4_K_M));
        assert!(recipe.supports_quant(QuantType::F32));
        assert!(!recipe.supports_quant(QuantType::Q8_0));

        let prompt = [10_u32, 20];
        let mut logits = [0.0_f32; 8];
        let mut kv_cache = [0.0_f32; 16];
        let weights = DummyWeights;

        let mut ctx = ExecutionContext::new(&prompt, 4, &mut logits, &mut kv_cache)
            .expect("valid execution context");

        recipe
            .prefill_step(&mut ctx, &weights)
            .expect("prefill succeeds");
        assert_eq!(ctx.position(), 2);
        assert_eq!(sample_argmax(ctx.logits()), Some(0));

        ctx.prepare_decode(0).expect("prepare decode");
        recipe
            .decode_step(&mut ctx, &weights)
            .expect("decode succeeds");
        assert_eq!(ctx.position(), 3);
        assert_eq!(sample_argmax(ctx.logits()), Some(1));
    }

    #[test]
    fn test_sample_argmax_tie_breaking() {
        // Tie between index 2 and index 4: must select lowest index (2).
        let logits = [0.0_f32, 1.0, 5.0, 3.0, 5.0, 2.0];
        assert_eq!(sample_argmax(&logits), Some(2));

        // Empty logits
        assert_eq!(sample_argmax(&[]), None);

        // Single logit
        assert_eq!(sample_argmax(&[42.0]), Some(0));
    }
}
