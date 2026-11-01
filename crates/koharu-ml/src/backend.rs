use anyhow::{Result, bail};
use koharu_runtime::{Backend, Device, Hardware};
use koharu_torch::{Cuda, Kind, nn};

pub(crate) trait TryIntoDevice<T> {
    fn try_into_device(self) -> Result<T>;
}

impl TryIntoDevice<koharu_torch::Device> for Device {
    fn try_into_device(self) -> Result<koharu_torch::Device> {
        match &self.backend {
            Backend::Cpu => Ok(koharu_torch::Device::Cpu),
            Backend::Cuda | Backend::Rocm => {
                // PyTorch's cuDNN switch controls MIOpen on ROCm. MIOpen is not reliable
                // on Windows and causes runtime issues, so disable it there.
                Cuda::set_user_enabled_cudnn(if cfg!(windows) {
                    matches!(&self.backend, Backend::Cuda)
                } else {
                    true
                });
                Ok(koharu_torch::Device::Cuda(self.index))
            }
            Backend::Vulkan if self.index == 0 => Ok(if koharu_torch::utils::has_vulkan() {
                koharu_torch::Device::Vulkan
            } else {
                koharu_torch::Device::Cpu
            }),
            Backend::Metal
                if self.index == 0 && cfg!(all(target_os = "macos", target_arch = "aarch64")) =>
            {
                Ok(koharu_torch::Device::Mps)
            }
            Backend::Vulkan | Backend::Metal => {
                bail!(
                    "Torch cannot address {} device index {}",
                    self.backend,
                    self.index
                )
            }
            Backend::Other(_) => bail!(
                "the {} backend cannot be represented by a Torch device",
                self.backend
            ),
        }
    }
}

/// Precision for the default detector and inpainter, which keep their
/// results within comparison tolerance in bfloat16. Unlike [`set_precision`],
/// it also uses bfloat16 on Apple GPUs, where it roughly halves their time.
pub(crate) fn set_reduced_precision(var_store: &mut nn::VarStore) {
    if matches!(var_store.device(), koharu_torch::Device::Mps) && mps_supports_bfloat16() {
        apply_precision(var_store, true);
    } else {
        set_precision(var_store);
    }
}

/// MPS gained bfloat16 kernels in macOS 14; older systems reject the first op.
fn mps_supports_bfloat16() -> bool {
    static SUPPORTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        koharu_torch::Tensor::f_ones([2, 2], (Kind::BFloat16, koharu_torch::Device::Mps))
            .and_then(|tensor| tensor.f_matmul(&tensor))
            .is_ok()
    })
}

pub(crate) fn set_precision(var_store: &mut nn::VarStore) {
    let hardware = Hardware::discover();
    let device_supports_bfloat16 = hardware
        .device()
        .is_some_and(|device| match &device.backend {
            Backend::Cuda => device.compute_capability() >= 80,
            Backend::Rocm => device.target().is_some_and(|target| {
                matches!(target, "gfx908" | "gfx90a" | "gfx942" | "gfx950")
                    || target.starts_with("gfx11")
                    || target.starts_with("gfx12")
            }),
            Backend::Cpu | Backend::Vulkan | Backend::Metal | Backend::Other(_) => false,
        });
    let use_bfloat16 =
        matches!(var_store.device(), koharu_torch::Device::Cuda(_)) && device_supports_bfloat16;
    apply_precision(var_store, use_bfloat16);
}

fn apply_precision(var_store: &mut nn::VarStore, use_bfloat16: bool) {
    if var_store.is_empty() {
        var_store.set_kind(if use_bfloat16 {
            Kind::BFloat16
        } else {
            Kind::Float
        });
    } else if use_bfloat16 {
        var_store.bfloat16();
    } else {
        var_store.float();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires the dynamically loaded LibTorch runtime and an Apple GPU"]
    async fn reduced_precision_uses_bfloat16_on_metal() -> anyhow::Result<()> {
        crate::init().await?;
        let mut var_store = nn::VarStore::new(koharu_torch::Device::Mps);
        set_reduced_precision(&mut var_store);
        assert_eq!(var_store.kind(), Kind::BFloat16);
        Ok(())
    }
}
