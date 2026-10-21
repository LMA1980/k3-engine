//! `k3-engine`: Low-level compute runtime for the `SpacemiT` K3 AI platform.
//!
//! Architecture recipes (`arch-*`) and compute kernels (`kernels-*`) are compiled
//! in-process via Cargo feature gates per ADR-024, targeting `riscv64gc-unknown-linux-gnu`
//! with RVV 1.0 (`+v`) per ADR-029.

pub mod affinity;
pub mod capability;
pub mod engine;
pub mod kv;
pub mod memory;
pub mod weights;

pub use affinity::{
    A100_CLUSTER_MASK, A100_DEFAULT_MAX_COMPUTE_THREADS, ClusterPreset, CpuTopologyInspector,
    MockCpuTopology, ResolvedAffinity, SystemCpuTopology, VlenDomain, WorkerAffinityStatus,
    WorkerPool, X100_CLUSTER_MASK, X100_DEFAULT_MAX_COMPUTE_THREADS, parse_cpu_list_mask,
    resolve_affinity, resolve_affinity_with,
};
pub use capability::{RecipeRegistry, verify_metadata_capabilities, verify_model_capabilities};
pub use engine::{
    EnginePhase, EvaluationSummary, SinglePromptEngine, StreamAction, sample_greedy_argmax,
};
pub use k3_core::{
    self, ArchRecipe, ExecutionContext, GgufFile, K3Error, ModelMetadata, ModelWeights, QuantType,
    TensorDescriptor, TensorView,
};
pub use kv::SingleSlotKvArena;
pub use memory::{
    DEFAULT_PAGE_SIZE, KV_BLOCK_SIZE, MemoryFootprint, MemoryInspector, MockMemoryInspector,
    SystemMemoryInspector, calculate_footprint, check_memory_admission,
    check_memory_admission_with, system_page_size,
};
pub use weights::LockedModelWeights;

#[cfg(feature = "arch-synthetic")]
pub use arch_synthetic;

#[cfg(feature = "kernels-rvv")]
pub use kernels_rvv;

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

    #[cfg(feature = "arch-synthetic")]
    #[test]
    fn arch_synthetic_recipe_wired() {
        use super::ArchRecipe;
        use super::arch_synthetic::SyntheticRecipe;

        let recipe = SyntheticRecipe::new();
        assert_eq!(recipe.architecture_id(), "synthetic");
    }

    #[cfg(feature = "kernels-rvv")]
    #[test]
    fn kernels_rvv_wired() {
        use super::kernels_rvv::{BaselineKernel, VectorKernel};

        let kernel = BaselineKernel::new();
        assert!((kernel.dot_product_f32(&[2.0], &[3.0]) - 6.0).abs() < f32::EPSILON);
    }
}
