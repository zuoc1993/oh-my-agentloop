//! Shared helpers for integration tests.

#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::stream::{self, StreamExt};
use oh_my_agentloop::{
    AgentLoopConfig, AgentMessage, AssistantMessage, ContentBlock, ConvertToLlmFn, Message, Model,
    ModelCost, OnPayloadFn, StopReason, StreamEvent, StreamFn, TextContent, Transport, Usage,
    UserContent, UserMessage,
};
use serde::Serialize;

pub fn json_value<T: Serialize>(v: &T) -> serde_json::Value {
    serde_json::to_value(v).expect("serialize")
}

/// Poll `cond` every 5ms up to `timeout`; panic if it never becomes true.
///
/// Prefer this over `tokio::time::sleep(X).await; assert!(cond);` which races
/// with the event being observed and is flaky on busy CI runners.
pub async fn wait_for<F>(timeout: Duration, label: &str, mut cond: F)
where
    F: FnMut() -> bool,
{
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if cond() {
            return;
        }
        if std::time::Instant::now() >= deadline {
            panic!("wait_for timed out after {timeout:?}: {label}");
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

pub fn test_model() -> Model {
    Model {
        id: "mock".into(),
        name: "mock".into(),
        api: "openai-responses".into(),
        provider: "openai".into(),
        base_url: "https://example.invalid".into(),
        reasoning: false,
        input: vec!["text".into()],
        cost: ModelCost::default(),
        context_window: 8192,
        max_tokens: 2048,
    }
}

pub fn user_message(text: &str) -> AgentMessage {
    AgentMessage::User(UserMessage {
        content: UserContent::Plain(text.into()),
        timestamp: 0,
    })
}

pub fn assistant_text(text: &str, model: &Model) -> AssistantMessage {
    AssistantMessage {
        content: vec![ContentBlock::Text(TextContent {
            text: text.into(),
            text_signature: None,
        })],
        model: model.id.clone(),
        provider: model.provider.clone(),
        api: model.api.clone(),
        response_id: None,
        stop_reason: StopReason::Stop,
        error_message: None,
        usage: Usage::default(),
        timestamp: 0,
    }
}

/// Identity convert: keep standard roles only.
pub fn identity_convert() -> ConvertToLlmFn {
    Arc::new(|messages| {
        Box::pin(async move {
            messages
                .into_iter()
                .filter_map(|m| m.into_message())
                .collect::<Vec<Message>>()
        })
    })
}

pub fn base_loop_config(model: Model, convert_to_llm: ConvertToLlmFn) -> AgentLoopConfig {
    AgentLoopConfig {
        model,
        reasoning: None,
        convert_to_llm,
        transform_context: None,
        get_api_key: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        tool_execution: Default::default(),
        before_tool_call: None,
        after_tool_call: None,
        api_key: None,
        session_id: None,
        transport: Transport::Sse,
        temperature: None,
        max_tokens: None,
        thinking_budgets: None,
        max_retry_delay_ms: None,
        on_payload: None,
    }
}

/// Stream that pushes only a terminal `done` (no incremental `start`), like the TS mock tests.
pub fn stream_done_only(final_msg: AssistantMessage) -> StreamFn {
    Arc::new(move |_model, _ctx, _req| {
        let msg = final_msg.clone();
        Box::pin(async move {
            let s = futures::stream::iter(vec![Ok(StreamEvent::Done { message: msg })]);
            Ok(Box::pin(s) as oh_my_agentloop::LlmEventStream)
        })
    })
}

/// Stream with TS-style `start` + partial-bearing `text_delta` + `done`.
pub fn stream_with_partial_deltas(
    final_msg: AssistantMessage,
    delta_partial: AssistantMessage,
) -> StreamFn {
    Arc::new(move |_model, _ctx, _req| {
        let empty = AssistantMessage {
            content: vec![ContentBlock::Text(TextContent {
                text: String::new(),
                text_signature: None,
            })],
            model: final_msg.model.clone(),
            provider: final_msg.provider.clone(),
            api: final_msg.api.clone(),
            response_id: None,
            stop_reason: StopReason::Stop,
            error_message: None,
            usage: Usage::default(),
            timestamp: 0,
        };
        let start = empty.clone();
        let delta_p = delta_partial.clone();
        let done_m = final_msg.clone();
        Box::pin(async move {
            let events = vec![
                Ok(StreamEvent::Start {
                    partial: start.clone(),
                }),
                Ok(StreamEvent::TextDelta {
                    content_index: 0,
                    delta: String::new(),
                    partial: delta_p,
                }),
                Ok(StreamEvent::Done { message: done_m }),
            ];
            let s = futures::stream::iter(events);
            Ok(Box::pin(s) as oh_my_agentloop::LlmEventStream)
        })
    })
}

pub async fn collect_loop_events(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<oh_my_agentloop::AgentEvent>,
) -> Vec<oh_my_agentloop::AgentEvent> {
    let mut out = Vec::new();
    while let Some(e) = rx.recv().await {
        out.push(e);
    }
    out
}

pub fn noop_on_payload() -> OnPayloadFn {
    Arc::new(|_payload, _model| Box::pin(async { None }))
}

/// Two LLM rounds: first `Done` uses `first`, second uses `second` (tool-use + follow-up parity).
pub fn stream_two_rounds(first: AssistantMessage, second: AssistantMessage) -> StreamFn {
    let call_index = Arc::new(AtomicUsize::new(0));
    Arc::new(move |_model, _ctx, _req| {
        let first = first.clone();
        let second = second.clone();
        let call_index = call_index.clone();
        Box::pin(async move {
            let idx = call_index.fetch_add(1, Ordering::SeqCst);
            let msg = if idx == 0 {
                first.clone()
            } else {
                second.clone()
            };
            let s = futures::stream::iter(vec![Ok(StreamEvent::Done { message: msg })]);
            Ok(Box::pin(s) as oh_my_agentloop::LlmEventStream)
        })
    })
}

/// Yields `Start` then slow `TextDelta` events so `abort()` can fire mid-stream (pi-mono e2e parity).
pub fn slow_stream_for_abort(model: Model) -> StreamFn {
    const WORDS: &[&str] = &[
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
        "twelve", "thirteen", "fourteen", "fifteen",
    ];
    let idx = Arc::new(AtomicUsize::new(0));
    Arc::new(move |_m, _ctx, _req| {
        let model = model.clone();
        let idx = idx.clone();
        Box::pin(async move {
            let s = stream::repeat(()).then(move |_| {
                let model = model.clone();
                let idx = idx.clone();
                async move {
                    tokio::time::sleep(Duration::from_millis(15)).await;
                    let i = idx.fetch_add(1, Ordering::SeqCst);
                    if i == 0 {
                        let empty = AssistantMessage {
                            content: vec![ContentBlock::Text(TextContent {
                                text: String::new(),
                                text_signature: None,
                            })],
                            model: model.id.clone(),
                            provider: model.provider.clone(),
                            api: model.api.clone(),
                            response_id: None,
                            stop_reason: StopReason::Stop,
                            error_message: None,
                            usage: Usage::default(),
                            timestamp: 0,
                        };
                        Ok(StreamEvent::Start { partial: empty })
                    } else if i <= WORDS.len() {
                        let text = WORDS[..i].join(" ");
                        let partial = assistant_text(&text, &model);
                        Ok(StreamEvent::TextDelta {
                            content_index: 0,
                            delta: format!("{} ", WORDS[i - 1]),
                            partial,
                        })
                    } else {
                        let text = WORDS.join(" ");
                        let partial = assistant_text(&text, &model);
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        Ok(StreamEvent::TextDelta {
                            content_index: 0,
                            delta: String::new(),
                            partial,
                        })
                    }
                }
            });
            Ok(Box::pin(s) as oh_my_agentloop::LlmEventStream)
        })
    })
}

pub fn stream_waits_for_cancel(model: Model) -> StreamFn {
    Arc::new(move |_model, _ctx, request| {
        let model = model.clone();
        Box::pin(async move {
            let cancel = request.cancel.clone();
            let s = stream::once(async move {
                cancel.cancelled().await;
                Ok(StreamEvent::Error {
                    message: AssistantMessage {
                        content: vec![ContentBlock::Text(TextContent {
                            text: String::new(),
                            text_signature: None,
                        })],
                        model: model.id.clone(),
                        provider: model.provider.clone(),
                        api: model.api.clone(),
                        response_id: None,
                        stop_reason: StopReason::Aborted,
                        error_message: Some("Aborted".into()),
                        usage: Usage::default(),
                        timestamp: 0,
                    },
                })
            });
            Ok(Box::pin(s) as oh_my_agentloop::LlmEventStream)
        })
    })
}
