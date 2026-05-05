//! Tool execution pipeline parity (see pi-mono `agent-loop.test.ts`).

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use std::sync::Arc as StdArc;

use common::{
    assistant_text, base_loop_config, collect_loop_events, identity_convert, test_model,
    user_message,
};
use oh_my_agentloop::{
    agent_loop, AfterToolCallContext, AfterToolCallResult, Agent, AgentContext, AgentError,
    AgentEvent, AgentMessage, AgentOptions, AgentTool, AgentToolResult, AssistantMessage,
    BeforeToolCallContext, BeforeToolCallResult, ContentBlock, InitialAgentState, Message, Model,
    StopReason, StreamEvent, StreamFn, StreamFnAdapter, StreamProvider, TextContent, ThinkingLevel,
    ToolCallContent, ToolExecutionMode, Usage, UserContent,
};
use serde_json::{json, Value};
use tokio::sync::Barrier;
use tokio_util::sync::CancellationToken;

fn echo_parameters_schema() -> Value {
    json!({
        "type": "object",
        "properties": { "value": { "type": "string" } },
        "required": ["value"]
    })
}

fn assistant_with_tool_calls(model: &Model, calls: Vec<ToolCallContent>) -> AssistantMessage {
    AssistantMessage {
        content: calls.into_iter().map(ContentBlock::ToolCall).collect(),
        model: model.id.clone(),
        provider: model.provider.clone(),
        api: model.api.clone(),
        response_id: None,
        stop_reason: StopReason::ToolUse,
        error_message: None,
        usage: Usage::default(),
        timestamp: 0,
    }
}

/// Two-round stream: first `tool_use`, then final text response.
fn stream_two_rounds(first: AssistantMessage, second: AssistantMessage) -> Arc<dyn StreamProvider> {
    let call_index = Arc::new(AtomicUsize::new(0));
    let f: StreamFn = Arc::new(move |_model, _ctx, _req| {
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
    });
    Arc::new(StreamFnAdapter(f))
}

fn agent_options(
    stream_provider: Arc<dyn StreamProvider>,
    model: Model,
    tools: Vec<Arc<dyn AgentTool>>,
    tool_execution: ToolExecutionMode,
) -> AgentOptions {
    AgentOptions {
        initial_state: Some(InitialAgentState {
            system_prompt: Some("You are helpful.".into()),
            model: Some(model),
            thinking_level: Some(ThinkingLevel::Off),
            tools: Some(tools),
            messages: None,
        }),
        convert_to_llm: Some(identity_convert()),
        transform_context: None,
        stream_provider,
        get_api_key: None,
        before_tool_call: None,
        after_tool_call: None,
        steering_mode: None,
        follow_up_mode: None,
        session_id: None,
        transport: None,
        tool_execution: Some(tool_execution),
        api_key: None,
        temperature: None,
        max_tokens: None,
        thinking_budgets: None,
        max_retry_delay_ms: None,
        on_payload: None,
    }
}

struct EchoTool {
    executed: Arc<Mutex<Vec<Value>>>,
}

#[async_trait]
impl AgentTool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn label(&self) -> &str {
        "Echo"
    }
    fn description(&self) -> &str {
        "Echo tool"
    }
    fn parameters(&self) -> Value {
        echo_parameters_schema()
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _cancel: tokio_util::sync::CancellationToken,
        on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>>,
    ) -> Result<AgentToolResult, oh_my_agentloop::AgentError> {
        self.executed.lock().unwrap().push(params.clone());
        if let Some(cb) = on_update {
            cb(AgentToolResult {
                content: vec![ContentBlock::Text(TextContent {
                    text: "partial".into(),
                    text_signature: None,
                })],
                details: None,
            });
        }
        let v = params
            .get("value")
            .map(|x| {
                x.as_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| x.to_string())
            })
            .unwrap_or_default();
        Ok(AgentToolResult {
            content: vec![ContentBlock::Text(TextContent {
                text: format!("echoed: {v}"),
                text_signature: None,
            })],
            details: Some(json!({ "value": v })),
        })
    }
}

struct AbortOnCancelTool {
    started: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl AgentTool for AbortOnCancelTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn label(&self) -> &str {
        "Echo"
    }
    fn description(&self) -> &str {
        "Abort on cancel tool"
    }
    fn parameters(&self) -> Value {
        echo_parameters_schema()
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        _params: Value,
        cancel: tokio_util::sync::CancellationToken,
        _on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>>,
    ) -> Result<AgentToolResult, oh_my_agentloop::AgentError> {
        self.started.lock().unwrap().push(tool_call_id.to_string());
        cancel.cancelled().await;
        Err(oh_my_agentloop::AgentError::Aborted)
    }
}

#[tokio::test]
async fn missing_tool_produces_error_tool_result() {
    let model = test_model();
    let tc = ToolCallContent {
        id: "tool-1".into(),
        name: "does_not_exist".into(),
        arguments: json!({}),
    };
    let assistant = assistant_with_tool_calls(&model, vec![tc]);
    let final_text = assistant_text("done", &model);
    let config = base_loop_config(model.clone(), identity_convert());
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![]),
        tools: vec![],
    };
    let cancel = CancellationToken::new();
    let rx = agent_loop(
        vec![user_message("hi")],
        ctx,
        config,
        cancel,
        stream_two_rounds(assistant, final_text),
    );
    let events = collect_loop_events(rx).await;
    let ends: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolExecutionEnd {
                is_error, result, ..
            } => Some((*is_error, result.content.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(ends.len(), 1);
    assert!(ends[0].0, "missing tool should be error result");
    let text = match &ends[0].1[0] {
        ContentBlock::Text(t) => t.text.as_str(),
        _ => panic!("expected text block"),
    };
    assert!(
        text.contains("does_not_exist") && text.to_lowercase().contains("not found"),
        "message={text:?}"
    );
}

#[tokio::test]
async fn missing_tool_stays_in_band_and_run_still_completes() {
    let model = test_model();
    let tc = ToolCallContent {
        id: "tool-1".into(),
        name: "does_not_exist".into(),
        arguments: json!({}),
    };
    let assistant = assistant_with_tool_calls(&model, vec![tc]);
    let final_text = assistant_text("done", &model);
    let config = base_loop_config(model.clone(), identity_convert());
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![]),
        tools: vec![],
    };

    let events = collect_loop_events(agent_loop(
        vec![user_message("hi")],
        ctx,
        config,
        CancellationToken::new(),
        stream_two_rounds(assistant, final_text),
    ))
    .await;

    let tool_end = events.iter().find_map(|event| match event {
        AgentEvent::ToolExecutionEnd {
            is_error, result, ..
        } if *is_error => Some(result.clone()),
        _ => None,
    });
    let tool_result_message = events.iter().find_map(|event| match event {
        AgentEvent::MessageEnd {
            message: AgentMessage::ToolResult(tool_result),
        } if tool_result.is_error => Some(tool_result.clone()),
        _ => None,
    });

    let tool_end = tool_end.expect("missing tool should emit error ToolExecutionEnd");
    let tool_result_message =
        tool_result_message.expect("missing tool should stay in-band as error ToolResult");
    let tool_end_text = match &tool_end.content[0] {
        ContentBlock::Text(text) => text.text.as_str(),
        _ => panic!("expected text block"),
    };
    let tool_result_text = match &tool_result_message.content[0] {
        ContentBlock::Text(text) => text.text.as_str(),
        _ => panic!("expected text block"),
    };

    assert!(
        tool_end_text.contains("does_not_exist")
            && tool_end_text.to_lowercase().contains("not found"),
        "tool execution error message={tool_end_text:?}"
    );
    assert!(
        tool_result_text.contains("does_not_exist")
            && tool_result_text.to_lowercase().contains("not found"),
        "tool result error message={tool_result_text:?}"
    );
    assert!(
        matches!(events.last(), Some(AgentEvent::RunCompleted { .. })),
        "terminal events were {:?}",
        events
    );
}

#[tokio::test]
async fn invalid_arguments_yield_validation_error_tool_result() {
    let model = test_model();
    let executed = Arc::new(Mutex::new(Vec::<Value>::new()));
    let tool = Arc::new(EchoTool {
        executed: executed.clone(),
    });
    let tc = ToolCallContent {
        id: "tool-1".into(),
        name: "echo".into(),
        arguments: json!({ "value": 42 }),
    };
    let assistant = assistant_with_tool_calls(&model, vec![tc]);
    let final_text = assistant_text("done", &model);
    let config = base_loop_config(model.clone(), identity_convert());
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![]),
        tools: vec![tool],
    };
    let cancel = CancellationToken::new();
    let rx = agent_loop(
        vec![user_message("hi")],
        ctx,
        config,
        cancel,
        stream_two_rounds(assistant, final_text),
    );
    let events = collect_loop_events(rx).await;
    assert!(
        executed.lock().unwrap().is_empty(),
        "tool must not run when validation fails"
    );
    let end = events.iter().find_map(|e| match e {
        AgentEvent::ToolExecutionEnd {
            is_error, result, ..
        } if *is_error => Some(result.content.clone()),
        _ => None,
    });
    let content = end.expect("error tool_execution_end");
    let text = match &content[0] {
        ContentBlock::Text(t) => t.text.as_str(),
        _ => panic!("expected text"),
    };
    assert!(
        text.contains("Validation failed") && text.contains("echo"),
        "{text}"
    );
}

#[tokio::test]
async fn prepare_arguments_runs_before_validation() {
    let model = test_model();
    let executed = Arc::new(Mutex::new(Vec::<Vec<Value>>::new()));

    struct EditTool {
        executed: Arc<Mutex<Vec<Vec<Value>>>>,
    }

    #[async_trait]
    impl AgentTool for EditTool {
        fn name(&self) -> &str {
            "edit"
        }
        fn label(&self) -> &str {
            "Edit"
        }
        fn description(&self) -> &str {
            "Edit tool"
        }
        fn parameters(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": { "type": "string" },
                                "newText": { "type": "string" }
                            },
                            "required": ["oldText", "newText"]
                        }
                    }
                },
                "required": ["edits"]
            })
        }

        fn prepare_arguments(&self, args: Value) -> Value {
            if !args.is_object() {
                return args;
            }
            let old = args.get("oldText").and_then(|x| x.as_str());
            let new = args.get("newText").and_then(|x| x.as_str());
            if let (Some(o), Some(n)) = (old, new) {
                let mut edits = args
                    .get("edits")
                    .and_then(|e| e.as_array())
                    .cloned()
                    .unwrap_or_default();
                edits.push(json!({ "oldText": o, "newText": n }));
                return json!({ "edits": edits });
            }
            args
        }

        async fn execute(
            &self,
            _tool_call_id: &str,
            params: Value,
            _cancel: tokio_util::sync::CancellationToken,
            _on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>>,
        ) -> Result<AgentToolResult, oh_my_agentloop::AgentError> {
            let edits = params
                .get("edits")
                .and_then(|e| e.as_array())
                .cloned()
                .unwrap_or_default();
            self.executed.lock().unwrap().push(edits);
            Ok(AgentToolResult {
                content: vec![ContentBlock::Text(TextContent {
                    text: format!(
                        "edited {}",
                        self.executed.lock().unwrap().last().unwrap().len()
                    ),
                    text_signature: None,
                })],
                details: None,
            })
        }
    }

    let tool = Arc::new(EditTool {
        executed: executed.clone(),
    });
    let tc = ToolCallContent {
        id: "tool-1".into(),
        name: "edit".into(),
        arguments: json!({ "oldText": "before", "newText": "after" }),
    };
    let assistant = assistant_with_tool_calls(&model, vec![tc]);
    let final_text = assistant_text("done", &model);
    let config = base_loop_config(model.clone(), identity_convert());
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![]),
        tools: vec![tool],
    };
    let cancel = CancellationToken::new();
    let rx = agent_loop(
        vec![user_message("edit something")],
        ctx,
        config,
        cancel,
        stream_two_rounds(assistant, final_text),
    );
    collect_loop_events(rx).await;
    assert_eq!(
        *executed.lock().unwrap(),
        vec![vec![json!({"oldText":"before","newText":"after"})]]
    );
}

#[tokio::test]
async fn before_tool_call_can_block() {
    let model = test_model();
    let executed = Arc::new(Mutex::new(Vec::<Value>::new()));
    let tool = Arc::new(EchoTool {
        executed: executed.clone(),
    });
    let hook_executed = Arc::new(AtomicUsize::new(0));
    let h = hook_executed.clone();
    let before: oh_my_agentloop::BeforeToolCallHookFn = StdArc::new(move |_ctx, _cancel| {
        let h = h.clone();
        Box::pin(async move {
            h.fetch_add(1, Ordering::SeqCst);
            Some(BeforeToolCallResult {
                block: true,
                reason: Some("blocked-by-test".into()),
            })
        })
    });
    let mut config = base_loop_config(model.clone(), identity_convert());
    config.before_tool_call = Some(before);

    let tc = ToolCallContent {
        id: "tool-1".into(),
        name: "echo".into(),
        arguments: json!({ "value": "hello" }),
    };
    let assistant = assistant_with_tool_calls(&model, vec![tc]);
    let final_text = assistant_text("done", &model);
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![]),
        tools: vec![tool],
    };
    let cancel = CancellationToken::new();
    let rx = agent_loop(
        vec![user_message("x")],
        ctx,
        config,
        cancel,
        stream_two_rounds(assistant, final_text),
    );
    let events = collect_loop_events(rx).await;
    assert_eq!(hook_executed.load(Ordering::SeqCst), 1);
    assert!(executed.lock().unwrap().is_empty(), "execute must not run");
    let end = events.iter().find_map(|e| match e {
        AgentEvent::ToolExecutionEnd {
            is_error, result, ..
        } if *is_error => Some(result.content[0].clone()),
        _ => None,
    });
    match end.expect("blocked end") {
        ContentBlock::Text(t) => assert!(t.text.contains("blocked-by-test")),
        _ => panic!("text"),
    }
}

#[tokio::test]
async fn before_tool_call_mutated_args_execute_without_revalidation() {
    let model = test_model();
    let executed = Arc::new(Mutex::new(Vec::<Value>::new()));
    let tool = Arc::new(EchoTool {
        executed: executed.clone(),
    });
    let before: oh_my_agentloop::BeforeToolCallHookFn =
        StdArc::new(move |ctx: BeforeToolCallContext, _cancel| {
            Box::pin(async move {
                let mut g = ctx.args.lock().unwrap();
                *g = json!({ "value": 123 });
                None
            })
        });
    let mut config = base_loop_config(model.clone(), identity_convert());
    config.before_tool_call = Some(before);

    let tc = ToolCallContent {
        id: "tool-1".into(),
        name: "echo".into(),
        arguments: json!({ "value": "hello" }),
    };
    let assistant = assistant_with_tool_calls(&model, vec![tc]);
    let final_text = assistant_text("done", &model);
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![]),
        tools: vec![tool],
    };
    let cancel = CancellationToken::new();
    let rx = agent_loop(
        vec![user_message("x")],
        ctx,
        config,
        cancel,
        stream_two_rounds(assistant, final_text),
    );
    collect_loop_events(rx).await;
    assert_eq!(*executed.lock().unwrap(), vec![json!({ "value": 123 })]);
}

#[tokio::test]
async fn after_tool_call_can_override_content_details_and_error_flag() {
    let model = test_model();
    let tool = Arc::new(EchoTool {
        executed: Arc::new(Mutex::new(vec![])),
    });
    let after: oh_my_agentloop::AfterToolCallHookFn =
        StdArc::new(move |ctx: AfterToolCallContext, _cancel| {
            Box::pin(async move {
                assert!(!ctx.is_error);
                Some(AfterToolCallResult {
                    content: Some(vec![ContentBlock::Text(TextContent {
                        text: "replaced".into(),
                        text_signature: None,
                    })]),
                    details: Some(json!({"patched": true})),
                    is_error: Some(true),
                })
            })
        });
    let mut config = base_loop_config(model.clone(), identity_convert());
    config.after_tool_call = Some(after);

    let tc = ToolCallContent {
        id: "tool-1".into(),
        name: "echo".into(),
        arguments: json!({ "value": "x" }),
    };
    let assistant = assistant_with_tool_calls(&model, vec![tc]);
    let final_text = assistant_text("done", &model);
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![]),
        tools: vec![tool],
    };
    let cancel = CancellationToken::new();
    let rx = agent_loop(
        vec![user_message("x")],
        ctx,
        config,
        cancel,
        stream_two_rounds(assistant, final_text),
    );
    let events = collect_loop_events(rx).await;
    let msg = events.iter().find_map(|e| match e {
        AgentEvent::MessageEnd {
            message: AgentMessage::ToolResult(tr),
        } => Some(tr.clone()),
        _ => None,
    });
    let tr = msg.expect("tool result message");
    assert!(tr.is_error);
    assert_eq!(tr.details, Some(json!({"patched": true})));
    match &tr.content[0] {
        ContentBlock::Text(t) => assert_eq!(t.text, "replaced"),
        _ => panic!("text"),
    }
}

#[tokio::test]
async fn tool_on_update_emits_tool_execution_update_events() {
    let model = test_model();
    let tool = Arc::new(EchoTool {
        executed: Arc::new(Mutex::new(vec![])),
    });
    let tc = ToolCallContent {
        id: "tool-1".into(),
        name: "echo".into(),
        arguments: json!({ "value": "a" }),
    };
    let assistant = assistant_with_tool_calls(&model, vec![tc]);
    let final_text = assistant_text("done", &model);
    let config = base_loop_config(model.clone(), identity_convert());
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![]),
        tools: vec![tool],
    };
    let cancel = CancellationToken::new();
    let rx = agent_loop(
        vec![user_message("x")],
        ctx,
        config,
        cancel,
        stream_two_rounds(assistant, final_text),
    );
    let events = collect_loop_events(rx).await;
    let updates: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolExecutionUpdate {
                tool_call_id,
                partial_result,
                ..
            } => Some((tool_call_id.clone(), partial_result.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0].0, "tool-1");
    match &updates[0].1.content[0] {
        ContentBlock::Text(t) => assert_eq!(t.text, "partial"),
        _ => panic!("partial"),
    }
}

#[tokio::test]
async fn parallel_mode_runs_concurrently_but_tool_results_in_source_order() {
    let model = test_model();
    let executed_order = Arc::new(Mutex::new(Vec::<String>::new()));
    let barrier = Arc::new(Barrier::new(2));

    struct ParallelEcho {
        barrier: Arc<Barrier>,
        executed_order: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl AgentTool for ParallelEcho {
        fn name(&self) -> &str {
            "echo"
        }
        fn label(&self) -> &str {
            "Echo"
        }
        fn description(&self) -> &str {
            "Echo"
        }
        fn parameters(&self) -> Value {
            echo_parameters_schema()
        }

        async fn execute(
            &self,
            _tool_call_id: &str,
            params: Value,
            _cancel: tokio_util::sync::CancellationToken,
            _on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>>,
        ) -> Result<AgentToolResult, oh_my_agentloop::AgentError> {
            let v = params.get("value").and_then(|x| x.as_str()).unwrap_or("");
            if v == "first" {
                self.barrier.wait().await;
                self.executed_order.lock().unwrap().push("first".into());
            } else if v == "second" {
                self.executed_order
                    .lock()
                    .unwrap()
                    .push("second_seen".into());
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                self.barrier.wait().await;
            }
            Ok(AgentToolResult {
                content: vec![ContentBlock::Text(TextContent {
                    text: format!("echoed:{v}"),
                    text_signature: None,
                })],
                details: None,
            })
        }
    }

    let tool: Arc<dyn AgentTool> = Arc::new(ParallelEcho {
        barrier: barrier.clone(),
        executed_order: executed_order.clone(),
    });

    let tc1 = ToolCallContent {
        id: "tool-1".into(),
        name: "echo".into(),
        arguments: json!({ "value": "first" }),
    };
    let tc2 = ToolCallContent {
        id: "tool-2".into(),
        name: "echo".into(),
        arguments: json!({ "value": "second" }),
    };
    let assistant = assistant_with_tool_calls(&model, vec![tc1, tc2]);
    let final_text = assistant_text("done", &model);
    let mut config = base_loop_config(model.clone(), identity_convert());
    config.tool_execution = ToolExecutionMode::Parallel;
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![]),
        tools: vec![tool.clone()],
    };
    let cancel = CancellationToken::new();
    let rx = agent_loop(
        vec![user_message("x")],
        ctx,
        config,
        cancel,
        stream_two_rounds(assistant, final_text),
    );
    let events = collect_loop_events(rx).await;
    let order = executed_order.lock().unwrap().clone();
    assert!(
        order.iter().any(|s| s == "second_seen"),
        "second should start before first finishes: {order:?}"
    );

    let tool_result_ids: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::MessageEnd {
                message: AgentMessage::ToolResult(tr),
            } => Some(tr.tool_call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_result_ids, vec!["tool-1", "tool-2"]);
}

#[tokio::test]
async fn steering_messages_injected_only_after_all_tool_calls_finish() {
    let model = test_model();
    let executed = Arc::new(Mutex::new(Vec::<Value>::new()));
    let tool = Arc::new(EchoTool {
        executed: executed.clone(),
    });
    let executed_for_steering = executed.clone();
    let queued_delivered = Arc::new(AtomicUsize::new(0));
    let qd = queued_delivered.clone();
    let steering: oh_my_agentloop::GetMessagesFn = StdArc::new(move || {
        let ex = executed_for_steering.lock().unwrap().clone();
        let qd = qd.clone();
        Box::pin(async move {
            if !ex.is_empty() && qd.load(Ordering::SeqCst) == 0 {
                qd.fetch_add(1, Ordering::SeqCst);
                return vec![user_message("interrupt")];
            }
            vec![]
        })
    });
    let mut config = base_loop_config(model.clone(), identity_convert());
    config.tool_execution = ToolExecutionMode::Sequential;
    config.get_steering_messages = Some(steering);

    let call_index = Arc::new(AtomicUsize::new(0));
    let saw_interrupt = Arc::new(AtomicUsize::new(0));
    let si = saw_interrupt.clone();
    let ci = call_index.clone();
    let model_stream = model.clone();
    let stream_fn: StreamFn = Arc::new(move |_model, ctx, _req| {
        if ci.load(Ordering::SeqCst) == 1 {
            let has_interrupt = ctx.messages.iter().any(|m| match m {
                Message::User(u) => {
                    matches!(&u.content, UserContent::Plain(s) if s == "interrupt")
                }
                _ => false,
            });
            if has_interrupt {
                si.fetch_add(1, Ordering::SeqCst);
            }
        }
        let idx = ci.fetch_add(1, Ordering::SeqCst);
        let first = assistant_with_tool_calls(
            &model_stream,
            vec![
                ToolCallContent {
                    id: "tool-1".into(),
                    name: "echo".into(),
                    arguments: json!({ "value": "first" }),
                },
                ToolCallContent {
                    id: "tool-2".into(),
                    name: "echo".into(),
                    arguments: json!({ "value": "second" }),
                },
            ],
        );
        let second = assistant_text("done", &model_stream);
        let msg = if idx == 0 { first } else { second.clone() };
        Box::pin(async move {
            let s = futures::stream::iter(vec![Ok(StreamEvent::Done { message: msg })]);
            Ok(Box::pin(s) as oh_my_agentloop::LlmEventStream)
        })
    });

    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![]),
        tools: vec![tool.clone()],
    };
    let cancel = CancellationToken::new();
    let rx = agent_loop(
        vec![user_message("start")],
        ctx,
        config,
        cancel,
        Arc::new(StreamFnAdapter(stream_fn)),
    );
    let events = collect_loop_events(rx).await;
    let ran: Vec<String> = executed
        .lock()
        .unwrap()
        .iter()
        .map(|p| {
            p.get("value")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string()
        })
        .collect();
    assert_eq!(ran, vec!["first".to_string(), "second".to_string()]);

    let sequence: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::MessageStart {
                message: AgentMessage::ToolResult(tr),
            } => Some(format!("tool:{}", tr.tool_call_id)),
            AgentEvent::MessageStart {
                message: AgentMessage::User(u),
            } => match &u.content {
                UserContent::Plain(s) => Some(s.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let i1 = sequence.iter().position(|s| s == "tool:tool-1").unwrap();
    let i2 = sequence.iter().position(|s| s == "tool:tool-2").unwrap();
    let i3 = sequence.iter().position(|s| s == "interrupt").unwrap();
    assert!(i1 < i3 && i2 < i3);
    assert_eq!(saw_interrupt.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn sequential_abort_stops_starting_later_tools() {
    let model = test_model();
    let started = Arc::new(Mutex::new(Vec::<String>::new()));
    let tool = Arc::new(AbortOnCancelTool {
        started: started.clone(),
    });
    let assistant = assistant_with_tool_calls(
        &model,
        vec![
            ToolCallContent {
                id: "tool-1".into(),
                name: "echo".into(),
                arguments: json!({ "value": "first" }),
            },
            ToolCallContent {
                id: "tool-2".into(),
                name: "echo".into(),
                arguments: json!({ "value": "second" }),
            },
        ],
    );
    let final_text = assistant_text("done", &model);
    let mut config = base_loop_config(model.clone(), identity_convert());
    config.tool_execution = ToolExecutionMode::Sequential;
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![]),
        tools: vec![tool],
    };
    let cancel = CancellationToken::new();
    let cancel_for_task = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        cancel_for_task.cancel();
    });

    let events = collect_loop_events(agent_loop(
        vec![user_message("start")],
        ctx,
        config,
        cancel,
        stream_two_rounds(assistant, final_text),
    ))
    .await;

    assert_eq!(started.lock().unwrap().as_slice(), ["tool-1"]);

    let started_ids: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionStart { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(started_ids, vec!["tool-1"]);
    assert!(matches!(events.last(), Some(AgentEvent::RunAborted { .. })));
}

#[tokio::test]
async fn parallel_abort_during_preparation_stops_later_starts_and_execution() {
    let model = test_model();
    let executed = Arc::new(Mutex::new(Vec::<Value>::new()));
    let tool = Arc::new(EchoTool {
        executed: executed.clone(),
    });
    let hook_calls = Arc::new(AtomicUsize::new(0));
    let before: oh_my_agentloop::BeforeToolCallHookFn = StdArc::new({
        let hook_calls = hook_calls.clone();
        move |_ctx: BeforeToolCallContext, cancel| {
            let hook_calls = hook_calls.clone();
            Box::pin(async move {
                if hook_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    cancel.cancel();
                }
                None
            })
        }
    });
    let assistant = assistant_with_tool_calls(
        &model,
        vec![
            ToolCallContent {
                id: "tool-1".into(),
                name: "echo".into(),
                arguments: json!({ "value": "first" }),
            },
            ToolCallContent {
                id: "tool-2".into(),
                name: "echo".into(),
                arguments: json!({ "value": "second" }),
            },
        ],
    );
    let final_text = assistant_text("done", &model);
    let mut config = base_loop_config(model.clone(), identity_convert());
    config.tool_execution = ToolExecutionMode::Parallel;
    config.before_tool_call = Some(before);
    let ctx = AgentContext {
        system_prompt: "".into(),
        messages: Arc::new(vec![]),
        tools: vec![tool],
    };

    let events = collect_loop_events(agent_loop(
        vec![user_message("start")],
        ctx,
        config,
        CancellationToken::new(),
        stream_two_rounds(assistant, final_text),
    ))
    .await;

    assert!(
        executed.lock().unwrap().is_empty(),
        "no tools should execute once cancel fires during preparation"
    );

    let started_ids: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionStart { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(started_ids, vec!["tool-1"]);

    let ended_ids: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(ended_ids, vec!["tool-1"]);
    assert!(matches!(events.last(), Some(AgentEvent::RunAborted { .. })));
}

#[tokio::test]
async fn sequential_listener_abort_on_tool_start_prevents_current_tool_execution() {
    let model = test_model();
    let executed = Arc::new(Mutex::new(Vec::<Value>::new()));
    let tool: Arc<dyn AgentTool> = Arc::new(EchoTool {
        executed: executed.clone(),
    });
    let assistant = assistant_with_tool_calls(
        &model,
        vec![ToolCallContent {
            id: "tool-1".into(),
            name: "echo".into(),
            arguments: json!({ "value": "first" }),
        }],
    );
    let final_text = assistant_text("done", &model);
    let agent = Agent::new(agent_options(
        stream_two_rounds(assistant, final_text),
        model,
        vec![tool],
        ToolExecutionMode::Sequential,
    ));
    let events = Arc::new(Mutex::new(Vec::<AgentEvent>::new()));
    let events_sub = events.clone();
    let agent_for_abort = agent.clone();
    let _sub = agent.subscribe(move |event, _cancel| {
        let events_sub = events_sub.clone();
        let agent_for_abort = agent_for_abort.clone();
        async move {
            events_sub.lock().unwrap().push(event.clone());
            if matches!(event, AgentEvent::ToolExecutionStart { .. }) {
                agent_for_abort.abort();
            }
        }
    });

    let err = agent.prompt_text("start", None).await.unwrap_err();
    assert!(matches!(err, AgentError::Aborted));
    assert!(
        executed.lock().unwrap().is_empty(),
        "tool should not execute after listener-triggered abort"
    );

    let started_ids: Vec<_> = events
        .lock()
        .unwrap()
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionStart { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(started_ids, vec!["tool-1"]);

    let ended_ids: Vec<_> = events
        .lock()
        .unwrap()
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(ended_ids, vec!["tool-1"]);
    assert!(matches!(
        events.lock().unwrap().last(),
        Some(AgentEvent::RunAborted { .. })
    ));
}
