//! `k3-engine`: Low-level compute runtime for the `SpacemiT` K3 AI platform.
//!
//! Architecture recipes (`arch-*`) and compute kernels (`kernels-*`) are compiled
//! in-process via Cargo feature gates per ADR-024, targeting `riscv64gc-unknown-linux-gnu`
//! with RVV 1.0 (`+v`) per ADR-029.

pub use k3_core::{
    self, ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, TensorView,
};

/// Indicates whether the `arch-synthetic` architecture recipe feature gate is enabled.
pub const HAS_ARCH_SYNTHETIC: bool = cfg!(feature = "arch-synthetic");

/// Indicates whether the `kernels-rvv` RVV 1.0 kernel feature gate is enabled.
pub const HAS_KERNELS_RVV: bool = cfg!(feature = "kernels-rvv");

#[cfg(test)]
mod tests {
    use super::{HAS_ARCH_SYNTHETIC, HAS_KERNELS_RVV, QuantType};

    #[test]
    fn feature_gates_match_cfg() {
        assert_eq!(HAS_ARCH_SYNTHETIC, cfg!(feature = "arch-synthetic"));
        assert_eq!(HAS_KERNELS_RVV, cfg!(feature = "kernels-rvv"));
        assert_eq!(QuantType::Q4_K_M.as_str(), "Q4_K_M");
    }
}
