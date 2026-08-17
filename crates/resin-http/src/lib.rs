mod anthropic;
mod openai;
mod protocol;
mod ui;

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use axum::{
    Router,
    extract::DefaultBodyLimit,
    routing::{get, post},
};
use tokio::sync::mpsc;

pub use protocol::{ConversationRequest, FinishReason, GenerationEvent, GenerationJob};

#[derive(Clone, Copy, Debug)]
pub struct RouteConfig {
    pub enable_openai: bool,
    pub enable_anthropic: bool,
    pub enable_demo: bool,
}

impl Default for RouteConfig {
    fn default() -> Self {
        Self {
            enable_openai: true,
            enable_anthropic: true,
            enable_demo: true,
        }
    }
}

#[derive(Clone)]
struct AppState {
    jobs: mpsc::UnboundedSender<GenerationJob>,
    model: Arc<str>,
    next_id: Arc<AtomicU64>,
}

impl AppState {
    fn request_id(&self, prefix: &str) -> String {
        let sequence = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}_{sequence:016x}")
    }
}

pub fn router(
    model: impl Into<Arc<str>>,
    jobs: mpsc::UnboundedSender<GenerationJob>,
    routes: RouteConfig,
) -> Router {
    let state = AppState {
        jobs,
        model: model.into(),
        next_id: Arc::new(AtomicU64::new(1)),
    };
    let mut router = Router::new().route("/health", get(openai::health));
    if routes.enable_demo {
        router = router.route("/", get(openai::index));
    }
    if routes.enable_openai {
        router = router
            .route("/v1/models", get(openai::models))
            .route("/v1/chat/completions", post(openai::chat_completions))
            .route("/v1/responses", post(openai::responses));
    }
    if routes.enable_anthropic {
        router = router.route("/v1/messages", post(anthropic::messages));
    }
    router
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state)
}

pub async fn serve(
    listener: tokio::net::TcpListener,
    model: impl Into<Arc<str>>,
    jobs: mpsc::UnboundedSender<GenerationJob>,
    routes: RouteConfig,
) -> std::io::Result<()> {
    axum::serve(listener, router(model, jobs, routes))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
}
