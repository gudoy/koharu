use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use sysinfo::{ProcessesToUpdate, System, get_current_pid};

mod vram;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceResources {
    pub name: String,
    pub selected: bool,
    pub memory_budget_bytes: Option<u64>,
    pub memory_used_bytes: Option<u64>,
    pub utilization_percent: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResourceSnapshot {
    pub process_memory_bytes: u64,
    pub system_memory_bytes: u64,
    pub process_cpu_percent: f32,
    pub devices: Vec<DeviceResources>,
}

pub(crate) struct ResourceMonitor {
    started: AtomicBool,
    sampled: AtomicBool,
    sampled_notify: tokio::sync::Notify,
    changed: tokio::sync::watch::Sender<ResourceSnapshot>,
    device: koharu_ml::Device,
}

impl ResourceMonitor {
    pub(crate) fn new(device: &koharu_ml::Device) -> Arc<Self> {
        let (changed, _) = tokio::sync::watch::channel(ResourceSnapshot::default());
        Arc::new(Self {
            started: AtomicBool::new(false),
            sampled: AtomicBool::new(false),
            sampled_notify: tokio::sync::Notify::new(),
            changed,
            device: device.clone(),
        })
    }

    pub(crate) fn start(self: &Arc<Self>) {
        if self.started.swap(true, Ordering::AcqRel) {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            self.started.store(false, Ordering::Release);
            return;
        };
        let monitor = Arc::downgrade(self);
        let device = self.device.clone();
        runtime.spawn(async move {
            let mut system = System::new();
            let mut vram = vram::Monitor::new(device.clone());
            let mut vram_unavailable = false;
            let pid = get_current_pid().ok();
            let logical_processors =
                std::thread::available_parallelism().map_or(1, |count| count.get()) as f32;
            // Most page-local model calls finish in well under 500 ms. A shorter
            // interval keeps their VRAM use and utilization visible in the UI
            // without polling in a tight loop.
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            let mut host_refresh_tick = 0_u8;
            loop {
                interval.tick().await;
                let Some(monitor) = monitor.upgrade() else {
                    return;
                };
                if host_refresh_tick == 0 {
                    system.refresh_memory();
                    if let Some(pid) = pid {
                        system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
                    }
                }
                host_refresh_tick = (host_refresh_tick + 1) % 5;
                let process = pid.and_then(|pid| system.process(pid));
                let system_memory = vram::SystemMemory {
                    total_bytes: system.total_memory(),
                    used_bytes: system.used_memory(),
                };
                let devices = match vram.sample(system_memory) {
                    Ok(devices) => {
                        vram_unavailable = false;
                        devices
                    }
                    Err(error) => {
                        if !vram_unavailable {
                            tracing::debug!(%error, "accelerator memory telemetry is unavailable");
                            vram_unavailable = true;
                        }
                        vram::unavailable(&device)
                    }
                };
                let snapshot = ResourceSnapshot {
                    process_memory_bytes: process.map_or(0, sysinfo::Process::memory),
                    system_memory_bytes: system_memory.total_bytes,
                    process_cpu_percent: process.map_or(0.0, sysinfo::Process::cpu_usage)
                        / logical_processors,
                    devices,
                };
                // Host values refresh on every fifth tick; a repeated sample
                // would only wake every subscriber with the same values.
                monitor.changed.send_if_modified(|current| {
                    let modified = *current != snapshot;
                    if modified {
                        *current = snapshot;
                    }
                    modified
                });
                monitor.sampled.store(true, Ordering::Release);
                monitor.sampled_notify.notify_waiters();
            }
        });
    }

    pub(crate) async fn wait_for_sample(&self) {
        if self.sampled.load(Ordering::Acquire) {
            return;
        }
        let notified = self.sampled_notify.notified();
        if self.sampled.load(Ordering::Acquire) {
            return;
        }
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), notified).await;
    }

    pub(crate) fn subscribe(&self) -> tokio::sync::watch::Receiver<ResourceSnapshot> {
        self.changed.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::ResourceMonitor;

    #[tokio::test]
    async fn unchanged_samples_do_not_notify_subscribers() {
        let monitor = ResourceMonitor::new(&koharu_ml::Device::cpu());
        let mut changes = monitor.subscribe();
        monitor.start();

        // Host memory and CPU refresh on every fifth tick, so most samples
        // repeat the previous one.
        let mut notified = 0;
        for _ in 0..10 {
            let sampled = monitor.sampled_notify.notified();
            tokio::time::timeout(Duration::from_secs(2), sampled)
                .await
                .expect("the monitor samples every 100 ms");
            if changes.has_changed().unwrap() {
                changes.borrow_and_update();
                notified += 1;
            }
        }
        // The first sample always differs from the default snapshot.
        assert!(
            (1..=3).contains(&notified),
            "{notified} of 10 samples notified subscribers"
        );
    }
}
