//! `kernels-rvv`: VLA RVV 1.0 compute kernels and portable scalar fallbacks.
//!
//! Provides vector primitives (dot product, `RMSNorm`, quantized dot product)
//! designed for RISC-V RVV 1.0 vector length agnostic loops per ADR-029,
//! with bit-equivalent scalar fallbacks for cross-platform evaluation.

use k3_core::{K3Error, QuantType, TensorView};

/// Vector kernel dispatch trait for dense and quantized tensor operations.
pub trait VectorKernel: Send + Sync + std::fmt::Debug {
    /// Computes the dot product of two single-precision floating-point slices.
    fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32;

    /// Computes Root Mean Square Normalization (`RMSNorm`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if slice lengths do not match or are empty.
    fn rmsnorm(
        &self,
        output: &mut [f32],
        input: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<(), K3Error>;

    /// Computes matrix-vector multiplication with `Q4_K_M` quantized weights.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if dimensions do not align or weights are malformed.
    fn vec_dot_q4_k_m(
        &self,
        output: &mut [f32],
        weights: &TensorView<'_>,
        input: &[f32],
    ) -> Result<(), K3Error>;
}

/// Baseline portable kernel implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct BaselineKernel;

impl BaselineKernel {
    /// Creates a new baseline kernel dispatcher.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl VectorKernel for BaselineKernel {
    fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        dot_product_f32_scalar(a, b)
    }

    fn rmsnorm(
        &self,
        output: &mut [f32],
        input: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<(), K3Error> {
        rmsnorm_scalar(output, input, weight, eps)
    }

    fn vec_dot_q4_k_m(
        &self,
        output: &mut [f32],
        weights: &TensorView<'_>,
        input: &[f32],
    ) -> Result<(), K3Error> {
        vec_dot_q4_k_m_scalar(output, weights, input)
    }
}

/// Computes the dot product of two `f32` slices using scalar arithmetic.
#[must_use]
pub fn dot_product_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0_f32;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Computes Root Mean Square Normalization (`RMSNorm`) using scalar arithmetic.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `output`, `input`, and `weight` lengths
/// do not match or if the slice is empty.
pub fn rmsnorm_scalar(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    let len = input.len();
    if len == 0 || output.len() != len || weight.len() != len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "rmsnorm".to_owned(),
            offset: 0,
            byte_len: size_of_val(output),
            buffer_len: size_of_val(input),
            reason: format!(
                "dimension mismatch: input={len}, output={}, weight={}",
                output.len(),
                weight.len()
            ),
        });
    }

    let mut sum_sq = 0.0_f32;
    for &x in input {
        sum_sq += x * x;
    }
    #[expect(clippy::cast_precision_loss, reason = "length fits comfortably in f32")]
    let mean_sq = sum_sq / (len as f32);
    let scale = 1.0_f32 / (mean_sq + eps).sqrt();

    for i in 0..len {
        output[i] = input[i] * scale * weight[i];
    }

    Ok(())
}

/// Computes matrix-vector multiplication with `Q4_K_M` quantized weights.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if the tensor quantization type is not
/// `Q4_K_M` or dimensions mismatch.
pub fn vec_dot_q4_k_m_scalar(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    if weights.quant() != QuantType::Q4_K_M {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!("expected Q4_K_M quant type, got {:?}", weights.quant()),
        });
    }

    let shape = weights.shape();
    if shape.is_empty() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "empty tensor shape".to_owned(),
        });
    }

    // In a 2D matrix [rows, cols], verify output and input dimensions.
    if shape.len() == 2 {
        let rows = shape[0];
        let cols = shape[1];
        if output.len() != rows || input.len() != cols {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: weights.offset(),
                byte_len: weights.byte_len(),
                buffer_len: weights.as_bytes().len(),
                reason: format!(
                    "matrix shape [{rows}, {cols}] does not match output len {} or input len {}",
                    output.len(),
                    input.len()
                ),
            });
        }
    }

    // Baseline stub for scaffolding: clear output. Dequantization RVV loops wire in US-01.6.2 (#47).
    output.fill(0.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_f32_scalar() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [2.0_f32, 0.5, -1.0, 3.0];
        let result = dot_product_f32_scalar(&a, &b);
        assert!((result - 12.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rmsnorm_scalar() {
        let input = [2.0_f32, 2.0, 2.0, 2.0];
        let weight = [1.0_f32, 1.0, 1.0, 1.0];
        let mut output = [0.0_f32; 4];
        let eps = 1e-5_f32;

        rmsnorm_scalar(&mut output, &input, &weight, eps).expect("rmsnorm succeeds");
        for &val in &output {
            assert!((val - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_rmsnorm_mismatch_errors() {
        let input = [1.0_f32; 4];
        let weight = [1.0_f32; 3];
        let mut output = [0.0_f32; 4];
        let err = rmsnorm_scalar(&mut output, &input, &weight, 1e-5);
        assert!(matches!(err, Err(K3Error::TensorBoundsError { .. })));
    }

    #[test]
    fn test_baseline_kernel_dispatch() {
        let kernel = BaselineKernel::new();
        let a = [1.0_f32, 2.0];
        let b = [3.0_f32, 4.0];
        assert!((kernel.dot_product_f32(&a, &b) - 11.0).abs() < f32::EPSILON);
    }
}
