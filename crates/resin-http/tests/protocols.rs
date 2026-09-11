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
    let (jobs, mut receiver) = mpsc::channel::<GenerationJob>(32);
    tokio::spawn(async move {
        while let Some(job) = receiver.recv().await {
            let _ = job
                .events
                .try_send(GenerationEvent::Started { input_tokens: 3 });
            let _ = job
                .events
                .try_send(GenerationEvent::ReasoningDelta("brief thought".to_owned()));
            let _ = job
                .events
                .try_send(GenerationEvent::TextDelta("hello".to_owned()));
            let _ = job.events.try_send(GenerationEvent::Finished {
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
    let (jobs, mut receiver) = mpsc::channel::<GenerationJob>(32);
    tokio::spawn(async move {
        while let Some(job) = receiver.recv().await {
            let _ = job
                .events
                .try_send(GenerationEvent::Started { input_tokens: 7 });
            let _ = job
                .events
                .try_send(GenerationEvent::ReasoningDelta("check weather".to_owned()));
            let _ = job.events.try_send(GenerationEvent::ToolCallStart {
                index: 0,
                id: "call_weather".to_owned(),
                name: "weather".to_owned(),
            });
            let _ = job
                .events
                .try_send(GenerationEvent::ToolCallArgumentsDelta {
                    index: 0,
                    delta: r#"{"city":"Paris"}"#.to_owned(),
                });
            let _ = job.events.try_send(GenerationEvent::Finished {
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
    let (jobs, _receiver) = mpsc::channel(32);
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

fn generation_request(path: &str, stream: bool) -> Request<Body> {
    let body = if path == "/v1/responses" {
        json!({"model": "test-model", "input": "hi", "stream": stream})
    } else {
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "stream": stream,
        })
    };
    Request::post(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test]
async fn full_admission_queue_rejects_each_api_without_waiting() {
    for path in ["/v1/chat/completions", "/v1/responses", "/v1/messages"] {
        let (jobs, receiver) = mpsc::channel(1);
        let app = router("test-model", jobs, RouteConfig::default());
        let first = app
            .clone()
            .oneshot(generation_request(path, true))
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let (status, body) = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            json_response(app, generation_request(path, false)),
        )
        .await
        .expect("overload must be rejected immediately");
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("queue is full")
        );
        assert_eq!(receiver.len(), 1);
    }
}

#[tokio::test]
async fn stopped_actor_fails_health_and_generation_admission() {
    let (jobs, receiver) = mpsc::channel(1);
    let app = router("test-model", jobs, RouteConfig::default());
    drop(receiver);
    let (status, _) = json_response(
        app.clone(),
        Request::get("/health").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let (status, _) = json_response(app, generation_request("/v1/responses", false)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn slow_stream_has_a_bounded_queue_and_does_not_block_another_response() {
    let (jobs, mut receiver) = mpsc::channel(2);
    let app = router("test-model", jobs, RouteConfig::default());
    let slow = app
        .clone()
        .oneshot(generation_request("/v1/chat/completions", true))
        .await
        .unwrap();
    let slow_job = receiver.recv().await.unwrap();
    for _ in 0..resin_http::EVENT_QUEUE_CAPACITY {
        slow_job
            .events
            .try_send(GenerationEvent::TextDelta("x".to_owned()))
            .unwrap();
    }
    assert!(matches!(
        slow_job
            .events
            .try_send(GenerationEvent::TextDelta("overflow".to_owned())),
        Err(mpsc::error::TrySendError::Full(_)),
    ));
    drop(slow_job);

    let fast = tokio::spawn(json_response(
        app,
        generation_request("/v1/responses", false),
    ));
    let fast_job = receiver.recv().await.unwrap();
    fast_job
        .events
        .try_send(GenerationEvent::TextDelta("finished".to_owned()))
        .unwrap();
    fast_job
        .events
        .try_send(GenerationEvent::Finished {
            reason: FinishReason::Stop,
            input_tokens: 1,
            cached_tokens: 0,
            output_tokens: 1,
            elapsed_ms: 1.0,
            generation_elapsed_ms: 1.0,
            time_to_first_token_ms: 0.0,
        })
        .unwrap();
    let (status, _) = tokio::time::timeout(std::time::Duration::from_secs(1), fast)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status, StatusCode::OK);

    let bytes = slow.into_body().collect().await.unwrap().to_bytes();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(text.contains("ended before completion"));
    assert!(!text.contains("overflow"));
}

#[tokio::test]
async fn interrupted_streams_report_failure_instead_of_silent_success() {
    for path in ["/v1/chat/completions", "/v1/responses", "/v1/messages"] {
        let (jobs, mut receiver) = mpsc::channel(1);
        let app = router("test-model", jobs, RouteConfig::default());
        let response = app.oneshot(generation_request(path, true)).await.unwrap();
        let job = receiver.recv().await.unwrap();
        job.events
            .try_send(GenerationEvent::TextDelta("partial".to_owned()))
            .unwrap();
        drop(job);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains("ended before completion"), "{path}");
    }
}

#[tokio::test]
async fn programmatic_shutdown_closes_the_listener() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (jobs, _receiver) = mpsc::channel(1);
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(resin_http::serve_with_shutdown(
        listener,
        "test-model",
        jobs,
        RouteConfig::default(),
        async {
            let _ = stopped.await;
        },
    ));
    stop.send(()).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(1), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(tokio::net::TcpStream::connect(address).await.is_err());
}
