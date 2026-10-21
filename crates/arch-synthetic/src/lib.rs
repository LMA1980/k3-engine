//! `arch-synthetic`: Synthetic architecture recipe and deterministic token sampler.
//!
//! Implements [`ArchRecipe`] for the reference synthetic model architecture,
//! reading `mlock`ed weight tensors, updating and reading the 16-token paged KV cache,
//! dispatching `kernels-rvv` primitives, and performing greedy argmax sampling
//! with lowest-token-ID tie-breaking per ADR-005, ADR-023, and ADR-024.

use k3_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType};

#[cfg(feature = "with-kernels")]
use kernels_rvv::{BaselineKernel, VectorKernel};

/// Synthetic architecture recipe implementation (US-01.6.3).
#[derive(Debug, Clone, Copy, Default)]
pub struct SyntheticRecipe;

impl SyntheticRecipe {
    /// Creates a new instance of the synthetic architecture recipe.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    fn execute_layer_kernels(
        token_seed: u32,
        position: usize,
        kv_cache: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<f32, K3Error> {
        #[expect(clippy::cast_precision_loss, reason = "token_seed fits in f32")]
        let base_val = ((token_seed % 17) as f32) * 0.1 + 0.5;
        let hidden = [base_val; 256];

        // 1. If `blk.0.attn_q.weight` (Q4_K_M) exists in locked weights, run quantized mat-vec
        let mut q_proj = [base_val; 2];
        if weights.contains_tensor("blk.0.attn_q.weight") {
            let q_tensor = weights.tensor("blk.0.attn_q.weight")?;
            if q_tensor.quant() == QuantType::Q4_K_M && q_tensor.num_elements() == 512 {
                #[cfg(feature = "with-kernels")]
                {
                    BaselineKernel::new().vec_dot_q4_k_m(&mut q_proj, &q_tensor, &hidden)?;
                }
                #[cfg(not(feature = "with-kernels"))]
                {
                    let _ = &hidden;
                    q_proj[0] = base_val;
                    q_proj[1] = base_val * 0.5;
                }
            }
        }

        // 2. If `output_norm.weight` (F32) exists in locked weights, run RMSNorm
        let mut norm_out = [q_proj[0]; 16];
        if weights.contains_tensor("output_norm.weight") {
            let norm_tensor = weights.tensor("output_norm.weight")?;
            if norm_tensor.quant() == QuantType::F32 && norm_tensor.num_elements() == 16 {
                let raw = norm_tensor.as_bytes();
                let mut w_f32 = [1.0_f32; 16];
                for (i, slot) in w_f32.iter_mut().enumerate() {
                    let b = &raw[i * 4..(i + 1) * 4];
                    let val = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    *slot = if val.is_finite() && val.abs() < 1e4 {
                        val
                    } else {
                        1.0
                    };
                }
                let norm_in = [q_proj[0].clamp(-10.0, 10.0) + 1.0; 16];
                #[cfg(feature = "with-kernels")]
                {
                    BaselineKernel::new().rmsnorm(&mut norm_out, &norm_in, &w_f32, 1e-5)?;
                }
                #[cfg(not(feature = "with-kernels"))]
                {
                    norm_out = norm_in;
                }
            }
        }

        // 3. Populate and read the single-slot 16-token paged KV cache slice
        let mut kv_attn = norm_out[0];
        if !kv_cache.is_empty() {
            let kv_len = kv_cache.len();
            let stride = 16.min(kv_len);
            let start = (position.saturating_mul(stride)) % kv_len;
            for i in 0..stride {
                let idx = (start + i) % kv_len;
                kv_cache[idx] = norm_out[i % norm_out.len()];
            }

            let window_end = (start + stride).min(kv_len);
            let kv_window = &kv_cache[start..window_end];
            let q_window = &norm_out[..kv_window.len()];
            #[cfg(feature = "with-kernels")]
            {
                kv_attn = BaselineKernel::new().dot_product_f32(kv_window, q_window);
            }
            #[cfg(not(feature = "with-kernels"))]
            {
                let _ = q_window;
                kv_attn = kv_window[0];
            }
        }

        let bounded_bias = if kv_attn.is_finite() {
            (kv_attn.abs() % 0.25) * 0.01
        } else {
            0.0
        };
        Ok(bounded_bias)
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
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let prompt_tokens = ctx.prompt_tokens();
        let prompt_len = prompt_tokens.len();

        // Process each prompt token sequentially to populate KV cache positions 0..prompt_len
        let mut bias = 0.0_f32;
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            bias = Self::execute_layer_kernels(tok, pos, ctx.kv_cache_mut(), weights)?;
        }

        ctx.advance_position(prompt_len)?;

        // Produce initial deterministic logits for next token selection.
        let logits = ctx.logits_mut();
        if !logits.is_empty() {
            logits.fill(0.0);
            let target_slot = if logits.len() > 9 { 9 } else { 0 };
            logits[target_slot] = 1.0 + bias;
        }

        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let token = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: "decode_step called without a staged decode token".to_owned(),
            })?;

        let pos = ctx.position();
        let bias = Self::execute_layer_kernels(token, pos, ctx.kv_cache_mut(), weights)?;
        ctx.advance_position(1)?;

        // Produce deterministic autoregressive next-token logits.
        let logits = ctx.logits_mut();
        if !logits.is_empty() {
            logits.fill(0.0);
            let next_slot = (usize::try_from(token).unwrap_or(0) + 1) % logits.len();
            logits[next_slot] = 2.0 + bias;
        }

        Ok(())
    }
}

/// Deterministically selects the token ID with the highest logit score (temperature `0.0`).
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
    use k3_core::GgufFile;
    use k3_core::test_helpers::create_synthetic_gguf_v3;

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
    fn test_synthetic_recipe_with_gguf_weights_and_kv_determinism_us01_6_3() {
        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic gguf");
        let recipe = SyntheticRecipe::new();

        let prompt = [1_u32, 9, 10];
        let mut seq1 = Vec::new();
        let mut seq2 = Vec::new();

        for out_seq in [&mut seq1, &mut seq2] {
            let mut logits = [0.0_f32; 16];
            let mut kv_cache = [0.0_f32; 128];
            let mut ctx = ExecutionContext::new(&prompt, 16, &mut logits, &mut kv_cache)
                .expect("create context");

            recipe.prefill_step(&mut ctx, &gguf).expect("prefill");
            let mut tok = sample_argmax(ctx.logits()).expect("sample prefill");
            out_seq.push(tok);

            for _ in 0..3 {
                ctx.prepare_decode(tok).expect("prepare decode");
                recipe.decode_step(&mut ctx, &gguf).expect("decode");
                tok = sample_argmax(ctx.logits()).expect("sample decode");
                out_seq.push(tok);
            }

            // Confirm KV cache was populated with non-zero state
            assert!(ctx.kv_cache().iter().any(|&v| v != 0.0));
        }

        // Bit-identical token sequences across repeated runs
        assert_eq!(seq1, seq2);
        assert_eq!(seq1, vec![9, 10, 11, 12]);
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
