use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use http_body_util::BodyExt;
use resin_http::{FinishReason, GenerationEvent, GenerationJob, RouteConfig, router};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tower::ServiceExt;

fn test_router() -> axum::Router {
    let (jobs, mut receiver) = mpsc::unbounded_channel::<GenerationJob>();
    tokio::spawn(async move {
        while let Some(job) = receiver.recv().await {
            let _ = job
                .events
                .send(GenerationEvent::Started { input_tokens: 3 });
            let _ = job
                .events
                .send(GenerationEvent::ReasoningDelta("brief thought".to_owned()));
            let _ = job
                .events
                .send(GenerationEvent::TextDelta("hello".to_owned()));
            let _ = job.events.send(GenerationEvent::Finished {
                reason: FinishReason::Stop,
                input_tokens: 3,
                cached_tokens: 2,
                output_tokens: 2,
                elapsed_ms: 20.0,
                generation_elapsed_ms: 10.0,
                time_to_first_token_ms: 10.0,
            });
        }
    });
    router("test-model", jobs, RouteConfig::default())
}

fn tool_router() -> axum::Router {
    let (jobs, mut receiver) = mpsc::unbounded_channel::<GenerationJob>();
    tokio::spawn(async move {
        while let Some(job) = receiver.recv().await {
            let _ = job
                .events
                .send(GenerationEvent::Started { input_tokens: 7 });
            let _ = job
                .events
                .send(GenerationEvent::ReasoningDelta("check weather".to_owned()));
            let _ = job.events.send(GenerationEvent::ToolCallStart {
                index: 0,
                id: "call_weather".to_owned(),
                name: "weather".to_owned(),
            });
            let _ = job.events.send(GenerationEvent::ToolCallArgumentsDelta {
                index: 0,
                delta: r#"{"city":"Paris"}"#.to_owned(),
            });
            let _ = job.events.send(GenerationEvent::Finished {
                reason: FinishReason::Stop,
                input_tokens: 7,
                cached_tokens: 4,
                output_tokens: 4,
                elapsed_ms: 30.0,
                generation_elapsed_ms: 20.0,
                time_to_first_token_ms: 10.0,
            });
        }
    });
    router("test-model", jobs, RouteConfig::default())
}

async fn json_response(app: axum::Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn chat_completions_returns_reasoning_text_and_usage() {
    let request = Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            })
            .to_string(),
        ))
        .unwrap();

    let (status, body) = json_response(test_router(), request).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["choices"][0]["message"]["content"], "hello");
    assert_eq!(
        body["choices"][0]["message"]["reasoning_content"],
        "brief thought"
    );
    assert_eq!(body["usage"]["total_tokens"], 5);
    assert_eq!(body["usage"]["prompt_tokens_details"]["cached_tokens"], 2);
}

#[tokio::test]
async fn responses_api_returns_reasoning_and_message_items() {
    let request = Request::post("/v1/responses")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"model": "test-model", "input": "hi"}).to_string(),
        ))
        .unwrap();

    let (status, body) = json_response(test_router(), request).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "response");
    assert_eq!(body["output"][0]["type"], "reasoning");
    assert_eq!(body["output"][1]["type"], "message");
    assert_eq!(body["output"][1]["content"][0]["text"], "hello");
    assert_eq!(body["usage"]["input_tokens_details"]["cached_tokens"], 2);
}

#[tokio::test]
async fn anthropic_messages_returns_thinking_and_text_blocks() {
    let request = Request::post("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            })
            .to_string(),
        ))
        .unwrap();

    let (status, body) = json_response(test_router(), request).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["type"], "message");
    assert_eq!(body["content"][0]["type"], "thinking");
    assert_eq!(body["content"][1], json!({"type": "text", "text": "hello"}));
    assert_eq!(body["usage"]["output_tokens"], 2);
}

#[tokio::test]
async fn route_flags_remove_disabled_apis_and_demo() {
    let (jobs, _receiver) = mpsc::unbounded_channel();
    let app = router(
        "test-model",
        jobs,
        RouteConfig {
            enable_openai: false,
            enable_anthropic: false,
            enable_demo: false,
        },
    );
    for path in ["/", "/v1/models", "/v1/chat/completions", "/v1/messages"] {
        let response = app
            .clone()
            .oneshot(Request::get(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
    }
    let response = app
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn chat_completions_streams_tool_calls_and_resin_metrics() {
    let request = Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "weather?"}],
                "stream": true,
                "stream_options": {"include_usage": true},
            })
            .to_string(),
        ))
        .unwrap();

    let response = tool_router().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/event-stream"
    );
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let events = std::str::from_utf8(&body).unwrap();
    assert!(events.contains(r#""reasoning_content":"check weather""#));
    assert!(events.contains(r#""name":"weather""#));
    assert!(events.contains(r#""finish_reason":"tool_calls""#));
    assert!(events.contains(r#""generation_elapsed_ms":20.0"#));
    assert!(events.contains(r#""cached_tokens":4"#));
    assert!(events.contains(r#""tokens_per_second":200.0"#));
    assert!(events.contains("data: [DONE]"));
}

#[tokio::test]
async fn tool_calls_map_to_responses_and_anthropic_shapes() {
    let responses_request = Request::post("/v1/responses")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"model": "test-model", "input": "weather?"}).to_string(),
        ))
        .unwrap();
    let (status, response) = json_response(tool_router(), responses_request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["output"][1]["type"], "function_call");
    assert_eq!(response["output"][1]["name"], "weather");
    assert_eq!(response["output"][1]["arguments"], r#"{"city":"Paris"}"#);

    let messages_request = Request::post("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "weather?"}],
            })
            .to_string(),
        ))
        .unwrap();
    let (status, message) = json_response(tool_router(), messages_request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(message["content"][1]["type"], "tool_use");
    assert_eq!(message["content"][1]["input"], json!({"city": "Paris"}));
    assert_eq!(message["stop_reason"], "tool_use");
}

#[tokio::test]
async fn responses_and_anthropic_streams_emit_protocol_events() {
    let responses_request = Request::post("/v1/responses")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"model": "test-model", "input": "hi", "stream": true}).to_string(),
        ))
        .unwrap();
    let response = test_router().oneshot(responses_request).await.unwrap();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let events = std::str::from_utf8(&body).unwrap();
    assert!(events.contains(r#""type":"response.reasoning_summary_text.delta""#));
    assert!(events.contains(r#""type":"response.output_text.delta""#));
    assert!(events.contains(r#""type":"response.completed""#));

    let messages_request = Request::post("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
            })
            .to_string(),
        ))
        .unwrap();
    let response = test_router().oneshot(messages_request).await.unwrap();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let events = std::str::from_utf8(&body).unwrap();
    assert!(events.contains("event: message_start"));
    assert!(events.contains(r#""type":"thinking_delta""#));
    assert!(events.contains(r#""type":"text_delta""#));
    assert!(events.contains("event: message_stop"));
}
