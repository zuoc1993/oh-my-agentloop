mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::collect_loop_events;
use futures::stream::{self, StreamExt};
use oh_my_agentloop::{
    agent_loop, AgentContext, AgentError, AgentEvent, AgentLoopConfig, AgentMessage,
    AssistantMessage, ContentBlock, Message, Model, ModelCost, RunOutcome, StopReason, StreamEvent,
    StreamFn, StreamFnAdapter, StreamOptions, StreamProvider, StreamRequest, TextContent,
    Transport, Usage, UserContent, UserMessage,
};
use oh_my_agentloop::{Agent, AgentOptions, InitialAgentState, ThinkingLevel};
use tokio_util::sync::CancellationToken;

fn test_model() -> Model {
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

fn user_message(text: &str) -> AgentMessage {
    AgentMessage::User(UserMessage {
        content: UserContent::Plain(text.into()),
        timestamp: 0,
    })
}

fn assistant_text(text: &str, model: &Model) -> AssistantMessage {
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

fn base_loop_config(model: Model) -> AgentLoopConfig {
    AgentLoopConfig {
        model,
        reasoning: None,
        convert_to_llm: Arc::new(|messages| {
            Box::pin(async move {
                messages
                    .into_iter()
                    .filter_map(|message| message.into_message())
                    .collect::<Vec<Message>>()
            })
        }),
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

#[test]
fn run_contract_types_are_public_and_instantiable() {
    let request = StreamRequest {
        options: StreamOptions {
            api_key: None,
            reasoning: None,
            session_id: None,
            transport: Transport::Sse,
            temperature: None,
            max_tokens: None,
            thinking_budgets: None,
            max_retry_delay_ms: None,
            on_payload: None,
        },
        cancel: CancellationToken::new(),
    };

    let completed = RunOutcome::Completed {
        new_messages: Arc::new(vec![]),
    };
    let failed = RunOutcome::Failed {
        new_messages: Arc::new(vec![user_message("boom")]),
        error: AgentError::Aborted,
    };
    let aborted = RunOutcome::Aborted {
        new_messages: Arc::new(vec![AgentMessage::Assistant(assistant_text(
            "bye",
            &test_model(),
        ))]),
    };

    assert_eq!(request.options.transport, Transport::Sse);
    assert!(!request.cancel.is_cancelled());
    assert!(matches!(completed, RunOutcome::Completed { .. }));
    assert!(matches!(failed, RunOutcome::Failed { .. }));
    assert!(matches!(aborted, RunOutcome::Aborted { .. }));

    let ctx = AgentContext {
        system_prompt: "sys".into(),
        messages: Arc::new(vec![user_message("hello")]),
        tools: vec![],
    };
    assert_eq!(ctx.messages.len(), 1);
}

/// Ensures `stream_assistant_response` passes the same cancellation handle the loop owns:
/// cancelling the token given to `agent_loop` must cancel `StreamRequest.cancel` seen by `stream_fn`.
#[tokio::test]
async fn stream_request_cancel_is_same_channel_as_loop_cancellation_token() {
    let loop_cancel = CancellationToken::new();
    let (tx, rx) = tokio::sync::oneshot::channel::<CancellationToken>();
    let tx_slot = Arc::new(Mutex::new(Some(tx)));

    let model = test_model();
    let final_msg = assistant_text("ok", &model);
    let stream_fn: StreamFn = Arc::new(move |_m, _ctx, req| {
        if let Some(sender) = tx_slot.lock().unwrap().take() {
            let _ = sender.send(req.cancel.clone());
        }
        let msg = final_msg.clone();
        Box::pin(async move {
            let s = futures::stream::iter(vec![Ok(StreamEvent::Done { message: msg })]);
            Ok(Box::pin(s) as oh_my_agentloop::LlmEventStream)
        })
    });

    let ctx = AgentContext {
        system_prompt: "sys".into(),
        messages: Arc::new(vec![]),
        tools: vec![],
    };
    let config = base_loop_config(model);

    let events_rx = agent_loop(
        vec![user_message("hi")],
        ctx,
        config,
        loop_cancel.clone(),
        Arc::new(StreamFnAdapter(stream_fn)),
    );

    let drain = tokio::spawn(collect_loop_events(events_rx));

    let provider_cancel = rx
        .await
        .expect("stream_fn should deliver provider-facing cancel token");

    assert!(
        !provider_cancel.is_cancelled(),
        "provider StreamRequest.cancel must not be cancelled before loop token is cancelled"
    );

    loop_cancel.cancel();

    assert!(
        provider_cancel.is_cancelled(),
        "StreamRequest.cancel must share the loop cancellation channel (clone of the same token)"
    );

    drain.await.expect("drain task join");
}

#[tokio::test]
async fn low_level_stream_errors_emit_run_failed_before_receiver_closes() {
    let model = test_model();
    let config = base_loop_config(model.clone());
    let ctx = AgentContext {
        system_prompt: String::new(),
        messages: Arc::new(vec![]),
        tools: vec![],
    };
    let stream_fn: StreamFn = Arc::new(|_model, _ctx, _request| {
        Box::pin(async move { Err(AgentError::Stream("boom".into())) })
    });

    let events = collect_loop_events(agent_loop(
        vec![user_message("hi")],
        ctx,
        config,
        CancellationToken::new(),
        Arc::new(StreamFnAdapter(stream_fn)),
    ))
    .await;

    assert!(
        matches!(
            events.last(),
            Some(AgentEvent::RunFailed { error_message, .. })
                if error_message == "Stream error: boom"
        ),
        "terminal events were {:?}",
        events
    );
}

#[tokio::test]
async fn low_level_cancelled_runs_emit_run_aborted_before_receiver_closes() {
    let model = test_model();
    let config = base_loop_config(model.clone());
    let ctx = AgentContext {
        system_prompt: String::new(),
        messages: Arc::new(vec![]),
        tools: vec![],
    };
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();

    let stream_fn: StreamFn = Arc::new(move |_model, _ctx, _request| {
        let model = model.clone();
        Box::pin(async move {
            let partial = assistant_text("", &model);
            let tail = stream::repeat(()).then(move |_| {
                let model = model.clone();
                async move {
                    tokio::time::sleep(Duration::from_millis(15)).await;
                    Ok(StreamEvent::TextDelta {
                        content_index: 0,
                        delta: "tick ".into(),
                        partial: assistant_text("tick", &model),
                    })
                }
            });
            let stream =
                stream::once(async move { Ok(StreamEvent::Start { partial }) }).chain(tail);
            Ok(Box::pin(stream) as oh_my_agentloop::LlmEventStream)
        })
    });

    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(40)).await;
        cancel_clone.cancel();
    });

    let events = collect_loop_events(agent_loop(
        vec![user_message("cancel me")],
        ctx,
        config,
        cancel,
        Arc::new(StreamFnAdapter(stream_fn)),
    ))
    .await;

    assert!(
        matches!(events.last(), Some(AgentEvent::RunAborted { .. })),
        "terminal events were {:?}",
        events
    );
}

#[tokio::test]
async fn provider_stream_error_events_emit_run_failed() {
    let model = test_model();
    let config = base_loop_config(model.clone());
    let ctx = AgentContext {
        system_prompt: String::new(),
        messages: Arc::new(vec![]),
        tools: vec![],
    };

    let mut error_message = assistant_text("", &model);
    error_message.stop_reason = StopReason::Error;
    error_message.error_message = Some("provider boom".into());

    let stream_fn: StreamFn = Arc::new(move |_model, _ctx, _request| {
        let error_message = error_message.clone();
        Box::pin(async move {
            let stream = stream::once(async move {
                Ok(StreamEvent::Error {
                    message: error_message,
                })
            });
            Ok(Box::pin(stream) as oh_my_agentloop::LlmEventStream)
        })
    });

    let events = collect_loop_events(agent_loop(
        vec![user_message("hi")],
        ctx,
        config,
        CancellationToken::new(),
        Arc::new(StreamFnAdapter(stream_fn)),
    ))
    .await;

    assert!(
        matches!(events.last(), Some(AgentEvent::RunFailed { .. })),
        "terminal events were {:?}",
        events
    );
}

#[tokio::test]
async fn cancel_before_first_partial_emits_aborted_terminal_and_message() {
    let model = test_model();
    let config = base_loop_config(model.clone());
    let ctx = AgentContext {
        system_prompt: String::new(),
        messages: Arc::new(vec![]),
        tools: vec![],
    };
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();

    let stream_fn: StreamFn = Arc::new(move |_model, _ctx, _request| {
        let model = model.clone();
        Box::pin(async move {
            let late_message = assistant_text("too late", &model);
            let stream = stream::once(async move {
                tokio::time::sleep(Duration::from_millis(40)).await;
                Ok(StreamEvent::Done {
                    message: late_message,
                })
            });
            Ok(Box::pin(stream) as oh_my_agentloop::LlmEventStream)
        })
    });

    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        cancel_clone.cancel();
    });

    let events = collect_loop_events(agent_loop(
        vec![user_message("cancel before start")],
        ctx,
        config,
        cancel,
        Arc::new(StreamFnAdapter(stream_fn)),
    ))
    .await;

    let turn_end_message = events.iter().find_map(|event| match event {
        AgentEvent::TurnEnd { message, .. } => Some(message),
        _ => None,
    });
    let Some(AgentMessage::Assistant(message)) = turn_end_message else {
        panic!(
            "expected assistant turn_end message, got {:?}",
            turn_end_message
        );
    };

    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.error_message.as_deref(), Some("Aborted"));
    assert!(
        matches!(events.last(), Some(AgentEvent::RunAborted { .. })),
        "terminal events were {:?}",
        events
    );
}

#[tokio::test]
async fn streams_without_terminal_events_emit_run_failed() {
    let model = test_model();
    let config = base_loop_config(model.clone());
    let ctx = AgentContext {
        system_prompt: String::new(),
        messages: Arc::new(vec![]),
        tools: vec![],
    };
    let stream_fn: StreamFn = Arc::new(move |_model, _ctx, _request| {
        Box::pin(async move {
            let stream = stream::empty();
            Ok(Box::pin(stream) as oh_my_agentloop::LlmEventStream)
        })
    });

    let events = collect_loop_events(agent_loop(
        vec![user_message("hi")],
        ctx,
        config,
        CancellationToken::new(),
        Arc::new(StreamFnAdapter(stream_fn)),
    ))
    .await;

    assert!(
        matches!(events.last(), Some(AgentEvent::RunFailed { .. })),
        "terminal events were {:?}",
        events
    );
}

#[tokio::test]
async fn partial_stream_without_terminal_events_emits_run_failed() {
    let model = test_model();
    let config = base_loop_config(model.clone());
    let ctx = AgentContext {
        system_prompt: String::new(),
        messages: Arc::new(vec![]),
        tools: vec![],
    };
    let start = assistant_text("", &model);
    let updated = assistant_text("half", &model);
    let stream_fn: StreamFn = Arc::new(move |_model, _ctx, _request| {
        let start = start.clone();
        let updated = updated.clone();
        Box::pin(async move {
            let stream = stream::iter(vec![
                Ok(StreamEvent::Start { partial: start }),
                Ok(StreamEvent::TextDelta {
                    content_index: 0,
                    delta: "half".into(),
                    partial: updated,
                }),
            ]);
            Ok(Box::pin(stream) as oh_my_agentloop::LlmEventStream)
        })
    });

    let events = collect_loop_events(agent_loop(
        vec![user_message("hi")],
        ctx,
        config,
        CancellationToken::new(),
        Arc::new(StreamFnAdapter(stream_fn)),
    ))
    .await;

    let turn_end_message = events.iter().find_map(|event| match event {
        AgentEvent::TurnEnd { message, .. } => Some(message),
        _ => None,
    });
    let Some(AgentMessage::Assistant(message)) = turn_end_message else {
        panic!(
            "expected assistant turn_end message, got {:?}",
            turn_end_message
        );
    };

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Stream ended without terminal event")
    );
    assert!(matches!(
        events.last(),
        Some(AgentEvent::RunFailed { error_message, .. })
            if error_message == "Stream ended without terminal event"
    ));
}

fn agent_options(stream_provider: Arc<dyn StreamProvider>, model: Model) -> AgentOptions {
    AgentOptions {
        initial_state: Some(InitialAgentState {
            system_prompt: Some("You are helpful.".into()),
            model: Some(model),
            thinking_level: Some(ThinkingLevel::Off),
            tools: Some(vec![]),
            messages: None,
        }),
        convert_to_llm: None,
        transform_context: None,
        stream_provider,
        get_api_key: None,
        before_tool_call: None,
        after_tool_call: None,
        steering_mode: None,
        follow_up_mode: None,
        session_id: None,
        transport: None,
        tool_execution: None,
        api_key: None,
        temperature: None,
        max_tokens: None,
        thinking_budgets: None,
        max_retry_delay_ms: None,
        on_payload: None,
    }
}

#[tokio::test]
async fn agent_prompt_returns_runtime_error_and_keeps_failure_message() {
    let model = test_model();
    let stream_fn: StreamFn = Arc::new(|_model, _ctx, _request| {
        Box::pin(async move { Err(AgentError::Stream("boom".into())) })
    });
    let agent = Agent::new(agent_options(Arc::new(StreamFnAdapter(stream_fn)), model.clone()));

    let err = agent.prompt_text("hello", None).await.unwrap_err();
    assert!(matches!(err, AgentError::Stream(message) if message == "boom"));

    let state = agent.state();
    let last = state.messages.last().expect("failure assistant");
    let AgentMessage::Assistant(message) = last else {
        panic!("expected assistant failure");
    };
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(state.error_message.as_deref(), Some("Stream error: boom"));
}

#[tokio::test]
async fn agent_prompt_returns_provider_error_terminal_as_runtime_error() {
    let model = test_model();
    let mut provider_error = assistant_text("", &model);
    provider_error.stop_reason = StopReason::Error;
    provider_error.error_message = Some("provider boom".into());

    let stream_fn: StreamFn = Arc::new(move |_model, _ctx, _request| {
        let provider_error = provider_error.clone();
        Box::pin(async move {
            let stream = stream::once(async move {
                Ok(StreamEvent::Error {
                    message: provider_error,
                })
            });
            Ok(Box::pin(stream) as oh_my_agentloop::LlmEventStream)
        })
    });

    let agent = Agent::new(agent_options(Arc::new(StreamFnAdapter(stream_fn)), model.clone()));

    let err = agent.prompt_text("hello", None).await.unwrap_err();
    assert!(matches!(err, AgentError::Stream(message) if message == "provider boom"));

    let state = agent.state();
    let last = state.messages.last().expect("provider error assistant");
    let AgentMessage::Assistant(message) = last else {
        panic!("expected assistant provider error");
    };
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_message.as_deref(), Some("provider boom"));
    assert_eq!(state.error_message.as_deref(), Some("provider boom"));
}

#[tokio::test]
async fn agent_prompt_normalizes_missing_provider_error_message() {
    let model = test_model();
    let mut provider_error = assistant_text("", &model);
    provider_error.stop_reason = StopReason::Error;
    provider_error.error_message = None;

    let stream_fn: StreamFn = Arc::new(move |_model, _ctx, _request| {
        let provider_error = provider_error.clone();
        Box::pin(async move {
            let stream = stream::once(async move {
                Ok(StreamEvent::Error {
                    message: provider_error,
                })
            });
            Ok(Box::pin(stream) as oh_my_agentloop::LlmEventStream)
        })
    });

    let agent = Agent::new(agent_options(Arc::new(StreamFnAdapter(stream_fn)), model.clone()));
    let events = Arc::new(Mutex::new(Vec::<AgentEvent>::new()));
    let events_sub = events.clone();
    let _sub = agent.subscribe(move |event, _cancel| {
        let events_sub = events_sub.clone();
        async move {
            events_sub.lock().unwrap().push(event);
        }
    });

    let err = agent.prompt_text("hello", None).await.unwrap_err();
    assert!(matches!(err, AgentError::Stream(message) if message == "Unknown stream error"));

    let state = agent.state();
    let last = state.messages.last().expect("provider error assistant");
    let AgentMessage::Assistant(message) = last else {
        panic!("expected assistant provider error");
    };
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Unknown stream error")
    );
    assert_eq!(state.error_message.as_deref(), Some("Unknown stream error"));
    assert!(matches!(
        events.lock().unwrap().last(),
        Some(AgentEvent::RunFailed { error_message, .. })
            if error_message == "Unknown stream error"
    ));
}

#[tokio::test]
async fn completed_terminal_event_wins_over_late_cancel() {
    let model = test_model();
    let stream_fn: StreamFn = Arc::new(move |model_arg, _ctx, request| {
        Box::pin(async move {
            let message = assistant_text("done", &model_arg);
            let cancel = request.cancel.clone();
            let stream = stream::once(async move {
                cancel.cancel();
                Ok(StreamEvent::Done { message })
            });
            Ok(Box::pin(stream) as oh_my_agentloop::LlmEventStream)
        })
    });

    let agent = Agent::new(agent_options(Arc::new(StreamFnAdapter(stream_fn)), model.clone()));
    let events = Arc::new(Mutex::new(Vec::<AgentEvent>::new()));
    let events_sub = events.clone();
    let _sub = agent.subscribe(move |event, _cancel| {
        let events_sub = events_sub.clone();
        async move {
            events_sub.lock().unwrap().push(event);
        }
    });

    agent.prompt_text("hello", None).await.unwrap();

    let state = agent.state();
    let last = state.messages.last().expect("completed assistant");
    let AgentMessage::Assistant(message) = last else {
        panic!("expected assistant completion");
    };
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.error_message, None);
    assert_eq!(state.error_message, None);
    assert!(matches!(
        events.lock().unwrap().last(),
        Some(AgentEvent::RunCompleted { .. })
    ));
}

#[tokio::test]
async fn agent_prompt_returns_aborted_for_provider_supplied_aborted_terminal() {
    let model = test_model();
    let mut aborted_message = assistant_text("", &model);
    aborted_message.stop_reason = StopReason::Aborted;
    aborted_message.error_message = Some("Aborted".into());

    let stream_fn: StreamFn = Arc::new(move |_model, _ctx, _request| {
        let aborted_message = aborted_message.clone();
        Box::pin(async move {
            let stream = stream::once(async move {
                Ok(StreamEvent::Error {
                    message: aborted_message,
                })
            });
            Ok(Box::pin(stream) as oh_my_agentloop::LlmEventStream)
        })
    });

    let agent = Agent::new(agent_options(Arc::new(StreamFnAdapter(stream_fn)), model.clone()));
    let events = Arc::new(Mutex::new(Vec::<AgentEvent>::new()));
    let events_sub = events.clone();
    let _sub = agent.subscribe(move |event, _cancel| {
        let events_sub = events_sub.clone();
        async move {
            events_sub.lock().unwrap().push(event);
        }
    });

    let err = agent.prompt_text("hello", None).await.unwrap_err();
    assert!(matches!(err, AgentError::Aborted));

    let state = agent.state();
    let last = state.messages.last().expect("aborted assistant");
    let AgentMessage::Assistant(message) = last else {
        panic!("expected assistant aborted terminal");
    };
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.error_message.as_deref(), Some("Aborted"));
    assert!(matches!(
        events.lock().unwrap().last(),
        Some(AgentEvent::RunAborted { .. })
    ));
}

#[tokio::test]
async fn agent_prompt_returns_aborted_for_provider_construction_abort() {
    let model = test_model();
    let stream_fn: StreamFn =
        Arc::new(|_model, _ctx, _request| Box::pin(async move { Err(AgentError::Aborted) }));

    let agent = Agent::new(agent_options(Arc::new(StreamFnAdapter(stream_fn)), model.clone()));
    let events = Arc::new(Mutex::new(Vec::<AgentEvent>::new()));
    let events_sub = events.clone();
    let _sub = agent.subscribe(move |event, _cancel| {
        let events_sub = events_sub.clone();
        async move {
            events_sub.lock().unwrap().push(event);
        }
    });

    let err = agent.prompt_text("hello", None).await.unwrap_err();
    assert!(matches!(err, AgentError::Aborted));

    let state = agent.state();
    let last = state.messages.last().expect("aborted assistant");
    let AgentMessage::Assistant(message) = last else {
        panic!("expected assistant aborted terminal");
    };
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert_eq!(message.error_message.as_deref(), Some("Aborted"));
    assert_eq!(state.error_message.as_deref(), Some("Aborted"));
    assert!(matches!(
        events.lock().unwrap().last(),
        Some(AgentEvent::RunAborted { .. })
    ));
}

#[tokio::test]
async fn early_abort_does_not_reuse_historical_aborted_message() {
    let model = test_model();
    let stream_called = Arc::new(AtomicBool::new(false));
    let stream_called_for_fn = stream_called.clone();
    let stream_fn: StreamFn = Arc::new(move |model_arg, _ctx, _request| {
        let stream_called_for_fn = stream_called_for_fn.clone();
        Box::pin(async move {
            stream_called_for_fn.store(true, Ordering::SeqCst);
            let msg = assistant_text("too late", &model_arg);
            let stream = stream::once(async move { Ok(StreamEvent::Done { message: msg }) });
            Ok(Box::pin(stream) as oh_my_agentloop::LlmEventStream)
        })
    });

    let mut old_aborted = assistant_text("", &model);
    old_aborted.stop_reason = StopReason::Aborted;
    old_aborted.error_message = Some("old abort".into());

    let mut options = agent_options(Arc::new(StreamFnAdapter(stream_fn)), model.clone());
    options.initial_state = Some(InitialAgentState {
        system_prompt: Some("You are helpful.".into()),
        model: Some(model),
        thinking_level: Some(ThinkingLevel::Off),
        tools: Some(vec![]),
        messages: Some(vec![
            user_message("previous"),
            AgentMessage::Assistant(old_aborted),
        ]),
    });

    let agent = Agent::new(options);
    let agent_for_abort = agent.clone();
    let _sub = agent.subscribe(move |event, _cancel| {
        let agent_for_abort = agent_for_abort.clone();
        async move {
            if matches!(
                event,
                AgentEvent::MessageEnd {
                    message: AgentMessage::User(_)
                }
            ) {
                agent_for_abort.abort();
            }
        }
    });

    let err = agent.prompt_text("hello", None).await.unwrap_err();
    assert!(matches!(err, AgentError::Aborted));
    assert!(
        !stream_called.load(Ordering::SeqCst),
        "stream_fn should not run once the listener aborts before the stream starts"
    );
    assert_eq!(agent.state().error_message.as_deref(), Some("Run aborted"));
}

#[tokio::test]
async fn agent_continue_returns_runtime_error_from_low_level_failure() {
    let model = test_model();
    let stream_fn: StreamFn = Arc::new(|_model, _ctx, _request| {
        Box::pin(async move { Err(AgentError::Stream("continue boom".into())) })
    });
    let mut options = agent_options(Arc::new(StreamFnAdapter(stream_fn)), model.clone());
    options.initial_state = Some(InitialAgentState {
        system_prompt: Some("You are helpful.".into()),
        model: Some(model),
        thinking_level: Some(ThinkingLevel::Off),
        tools: Some(vec![]),
        messages: Some(vec![user_message("hello")]),
    });
    let agent = Agent::new(options);

    let err = agent.continue_run().await.unwrap_err();
    assert!(matches!(err, AgentError::Stream(message) if message == "continue boom"));
}
