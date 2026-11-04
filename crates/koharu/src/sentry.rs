use sentry::{ClientOptions, IntoDsn};
use tracing_subscriber::registry::LookupSpan;

/// Set to `1` to send no crash, error or session reports.
const DISABLE_TELEMETRY: &str = "KOHARU_DISABLE_TELEMETRY";

/// Starts Sentry. Without a DSN, which local builds lack, or with telemetry
/// disabled, the client stays inactive and sends nothing.
pub fn initialize() -> sentry::ClientInitGuard {
    let mut options = ClientOptions::new()
        .maybe_release(sentry::release_name!())
        .send_default_pii(true)
        .sample_rate(0.1)
        .auto_session_tracking(true);
    options.dsn = if std::env::var_os(DISABLE_TELEMETRY).is_some_and(|value| value == "1") {
        None
    } else {
        option_env!("SENTRY_DSN")
            .into_dsn()
            .expect("invalid SENTRY_DSN environment variable")
    };
    sentry::init(options)
}

pub fn tracing_layer<S>() -> impl tracing_subscriber::Layer<S>
where
    S: tracing::Subscriber + for<'span> LookupSpan<'span>,
{
    sentry::integrations::tracing::layer()
}
