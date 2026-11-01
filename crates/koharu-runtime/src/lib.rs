mod device;
mod hardware;
mod network;
mod runtime;
mod source;
mod store;

pub mod download;

pub use device::{Backend, Device, DeviceType};
pub use hardware::Hardware;
pub use runtime::{Feature, Package, Runtime, Torch};
pub use source::HuggingFaceFile;
pub use store::Store;

/// Builds the shared client used by remote APIs and long-running model requests.
///
/// Connection setup and response reads follow the shared HTTP policy: a read
/// that stalls for longer than the configured read timeout fails. There is no
/// total request deadline and no retry, so callers retain cancellation
/// ownership of long generations and non-idempotent generation requests are
/// never replayed automatically.
pub fn http_client() -> anyhow::Result<reqwest::Client> {
    network::http()
}
