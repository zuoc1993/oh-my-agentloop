//! Parity tests for the low-level agent loop (see pi-mono `agent-loop.ts`).

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use common::{
    assistant_text, base_loop_config, collect_loop_events, identity_convert, noop_on_payload,
    stream_done_only, stream_with_partial_deltas, test_model, user_message,
};
use oh_my_agentloop::{
    agent_loop, agent_loop_continue, run_agent_loop_continue, AgentContext, AgentError, AgentEvent,
    AgentMessage, AssistantMessage, ContentBlock, ConvertToLlmFn, GetApiKeyFn, OnPayloadFn,
    RunOutcome, StopReason, StreamEvent, StreamFn, StreamFnAdapter, StreamOptions, TextContent,
    Transport, Usage,
};
use tokio_util::sync::CancellationToken;

fn event_types(events: &[AgentEvent]) -> Vec<&'static str> {
    events
        .iter()
        .map(|e| match e {
            AgentEvent::AgentStart => "agent_start",
            AgentEvent::RunCompleted { .. } => "run_completed",
            AgentEvent::RunFailed { .. } => "run_failed",
            AgentEvent::RunAborted { .. } => "run_aborted",
            AgentEvent::TurnStart => "turn_start",
            AgentEvent::TurnEnd { .. } => "turn_end",
            AgentEvent::MessageStart { .. } => "message_start",
            AgentEvent::MessageUpdate { .. } => "message_update",
            AgentEvent::MessageEnd { .. } => "message_end",
            AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
            AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
            AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
            _ => "unknown",
        })
        .collect()
}

#[tokio::test]
async fn prompt_path_emits_expected_event_sequence_for_done_only_stream() {
    let model = test_model();
    let config = base_loop_config(model.clone(), identity_convert());
    let ctx = AgentContext {
        system_prompt: "You are helpful.".into(),
        messages: Arc::new(vec![]),
        tools: vec![],
    };
    let prompt = user_message("Hello");
    let assistant = assistant_text("Hi there!", &model);
    let cancel = CancellationToken::new();
    let rx = agent_loop(
        vec![prompt.clone()],
        ctx,
        config,
        cancel,
        stream_done_only(assistant.clone()),
    );
    let events = collect_loop_events(rx).await;
    let types = event_types(&events);
    assert!(
        types.iter().position(|t| *t == "agent_start")
            < types.iter().position(|t| *t == "turn_start"),
        "agent_start before turn_start: {:?}",
        types
    );
    assert!(types.contains(&"message_start"));
    assert!(types.contains(&"message_end"));
    assert!(types.contains(&"turn_end"));
    assert_eq!(types.last(), Some(&"run_completed"));
    let starts: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::MessageStart { message } => Some(message),
            _ => None,
        })
        .collect();
    assert_eq!(starts.len(), 2, "user + assistant message_start");
    assert_eq!(starts[0].role(), "user");
    assert_eq!(starts[1].role(), "assistant");
}

#[tokio::test]
async fn transform_context_runs_before_convert_to_llm() {
    let model = test_model();
    let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));
    let o_t = order.clone();
    let transform: oh_my_agentloop::TransformContextFn = Arc::new(move |msgs, _cancel| {
        let o_t = o_t.clone();
        Box::pin(async move {
            o_t.lock().unwrap().push("transform");
            msgs
        })
    });
    let o_c = order.clone();
    let convert: ConvertToLlmFn = Arc::new(move |msgs| {
        let o_c = o_c.clone();
        Box::pin(async move {
            o_c.lock().unwrap().push("convert");
            msgs.into_iter().filter_map(|m| m.into_message()).collect()
        })
    });
    let mut config = base_loop_config(model.clone(), convert);
    config.transform_context = Some(transform);
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![]),
        tools: vec![],
    };
    let cancel = CancellationToken::new();
    let rx = agent_loop(
        vec![user_message("x")],
        ctx,
        config,
        cancel,
        stream_done_only(assistant_text("ok", &model)),
    );
    collect_loop_events(rx).await;
    assert_eq!(*order.lock().unwrap(), vec!["transform", "convert"]);
}

#[tokio::test]
async fn follow_up_messages_run_only_after_agent_would_stop() {
    let model = test_model();
    let follow_up_calls = Arc::new(AtomicUsize::new(0));
    let fu = follow_up_calls.clone();
    let get_follow_up: oh_my_agentloop::GetMessagesFn = Arc::new(move || {
        let fu = fu.clone();
        Box::pin(async move {
            let n = fu.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                vec![user_message("follow-up")]
            } else {
                vec![]
            }
        })
    });
    let mut config = base_loop_config(model.clone(), identity_convert());
    config.get_follow_up_messages = Some(get_follow_up);
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![]),
        tools: vec![],
    };
    let cancel = CancellationToken::new();
    let rx = agent_loop(
        vec![user_message("start")],
        ctx,
        config,
        cancel,
        stream_done_only(assistant_text("first", &model)),
    );
    let events = collect_loop_events(rx).await;
    assert_eq!(
        follow_up_calls.load(Ordering::SeqCst),
        2,
        "called once when stopping, again on final stop"
    );
    let turn_starts = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::TurnStart))
        .count();
    assert!(
        turn_starts >= 2,
        "second turn after follow-up: {:?}",
        event_types(&events)
    );
}

#[tokio::test]
async fn stream_options_include_resolved_key_session_transport_temperature_tokens_budgets_retry_on_payload(
) {
    let model = test_model();
    let captured = Arc::new(Mutex::new(None::<StreamOptions>));
    let cap = captured.clone();
    let budgets = oh_my_agentloop::ThinkingBudgets {
        minimal: Some(1),
        low: Some(2),
        medium: Some(3),
        high: Some(4),
        xhigh: Some(5),
    };
    let hook: OnPayloadFn = noop_on_payload();
    let hook_ptr = Arc::as_ptr(&hook);
    let get_key: GetApiKeyFn = Arc::new(|_| Box::pin(async { Some("key-from-fn".into()) }));
    let mut config = base_loop_config(model.clone(), identity_convert());
    config.get_api_key = Some(get_key);
    config.api_key = Some("key-static".into());
    config.session_id = Some("sess-1".into());
    config.transport = Transport::Ws;
    config.temperature = Some(0.7);
    config.max_tokens = Some(99);
    config.thinking_budgets = Some(budgets.clone());
    config.max_retry_delay_ms = Some(12345);
    config.on_payload = Some(hook.clone());
    let assistant = assistant_text("ok", &model);
    let stream_fn: StreamFn = Arc::new(move |m, ctx, req| {
        let cap = cap.clone();
        let assistant = assistant.clone();
        let mut g = cap.lock().unwrap();
        *g = Some(req.options.clone());
        drop(g);
        assert_eq!(m.id, "mock");
        assert!(!ctx.messages.is_empty());
        Box::pin(async move {
            let s = futures::stream::iter(vec![Ok(StreamEvent::Done { message: assistant })]);
            Ok(Box::pin(s) as oh_my_agentloop::LlmEventStream)
        })
    });
    let ctx = AgentContext {
        system_prompt: "sys".into(),
        messages: Arc::new(vec![]),
        tools: vec![],
    };
    let cancel = CancellationToken::new();
    let rx = agent_loop(vec![user_message("hi")], ctx, config, cancel, Arc::new(StreamFnAdapter(stream_fn)));
    collect_loop_events(rx).await;
    let opts = captured.lock().unwrap().take().expect("options captured");
    assert_eq!(opts.api_key.as_deref(), Some("key-from-fn"));
    assert_eq!(opts.session_id.as_deref(), Some("sess-1"));
    assert_eq!(opts.transport, Transport::Ws);
    assert_eq!(opts.temperature, Some(0.7));
    assert_eq!(opts.max_tokens, Some(99));
    assert_eq!(opts.thinking_budgets.as_ref(), Some(&budgets));
    assert_eq!(opts.max_retry_delay_ms, Some(12345));
    let opt_hook = opts.on_payload.expect("on_payload forwarded");
    assert_eq!(Arc::as_ptr(&opt_hook), hook_ptr);
}

#[tokio::test]
async fn message_update_uses_provider_partial_from_stream_event() {
    let model = test_model();
    let mut partial_msg = assistant_text("__PROVIDER_PARTIAL__", &model);
    partial_msg.stop_reason = StopReason::Stop;
    let final_msg = assistant_text("final", &model);
    let config = base_loop_config(model.clone(), identity_convert());
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![]),
        tools: vec![],
    };
    let cancel = CancellationToken::new();
    let rx = agent_loop(
        vec![user_message("u")],
        ctx,
        config,
        cancel,
        stream_with_partial_deltas(final_msg.clone(), partial_msg.clone()),
    );
    let events = collect_loop_events(rx).await;
    let updates: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::MessageUpdate {
                message,
                stream_event,
            } => Some((message.clone(), stream_event.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(updates.len(), 1);
    let (msg, ev) = &updates[0];
    let AgentMessage::Assistant(am) = msg else {
        panic!("expected assistant");
    };
    assert_eq!(
        assistant_text_content(am),
        Some("__PROVIDER_PARTIAL__".into())
    );
    match ev {
        StreamEvent::TextDelta { partial, .. } => {
            assert_eq!(
                assistant_text_content(partial),
                Some("__PROVIDER_PARTIAL__".into())
            );
        }
        _ => panic!("expected TextDelta, got {:?}", ev),
    }
}

fn assistant_text_content(m: &AssistantMessage) -> Option<String> {
    m.content.iter().find_map(|b| match b {
        ContentBlock::Text(t) => Some(t.text.clone()),
        _ => None,
    })
}

#[tokio::test]
async fn agent_loop_continue_rejects_empty_context_before_spawn() {
    let model = test_model();
    let config = base_loop_config(model, identity_convert());
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![]),
        tools: vec![],
    };
    let err = agent_loop_continue(
        ctx,
        config,
        CancellationToken::new(),
        stream_done_only(AssistantMessage {
            content: vec![ContentBlock::Text(TextContent {
                text: String::new(),
                text_signature: None,
            })],
            model: "m".into(),
            provider: "p".into(),
            api: "a".into(),
            response_id: None,
            stop_reason: StopReason::Stop,
            error_message: None,
            usage: Usage::default(),
            timestamp: 0,
        }),
    )
    .unwrap_err();
    assert!(matches!(err, AgentError::NoMessages));
}

#[tokio::test]
async fn agent_loop_continue_rejects_assistant_tail() {
    let model = test_model();
    let config = base_loop_config(model.clone(), identity_convert());
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![
            user_message("u"),
            AgentMessage::Assistant(assistant_text("a", &model)),
        ]),
        tools: vec![],
    };
    let err = agent_loop_continue(
        ctx,
        config,
        CancellationToken::new(),
        stream_done_only(assistant_text("x", &model)),
    )
    .unwrap_err();
    assert!(matches!(err, AgentError::ContinueFromAssistant));
}

#[tokio::test]
async fn run_agent_loop_continue_emits_only_new_assistant_message_events() {
    let model = test_model();
    let config = base_loop_config(model.clone(), identity_convert());
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![user_message("Hello")]),
        tools: vec![],
    };
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let emitter = oh_my_agentloop::EventEmitter::new(move |event| {
        let tx = tx.clone();
        async move {
            let _ = tx.send(event);
        }
    });
    let cancel = CancellationToken::new();
    let stream_fn = stream_done_only(assistant_text("Response", &model));
        let outcome = run_agent_loop_continue(ctx, config, &emitter, cancel, &*stream_fn)
        .await
        .expect("continue ok");
    let RunOutcome::Completed {
        new_messages: new_msgs,
    } = outcome
    else {
        panic!("expected completed outcome");
    };
    assert_eq!(new_msgs.len(), 1);
    assert_eq!(new_msgs[0].role(), "assistant");
    let mut events = Vec::new();
    while let Ok(e) = rx.try_recv() {
        events.push(e);
    }
    let ends: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::MessageEnd { message } => Some(message.role()),
            _ => None,
        })
        .collect();
    assert_eq!(ends, vec!["assistant"]);
}
