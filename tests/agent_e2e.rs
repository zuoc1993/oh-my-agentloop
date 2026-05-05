//! End-to-end parity with pi-mono `packages/agent/test/e2e.test.ts` (non-proxy observable behavior).

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use common::{
    assistant_text, slow_stream_for_abort, stream_done_only, stream_two_rounds, test_model,
};
use oh_my_agentloop::{
    Agent, AgentError, AgentEvent, AgentMessage, AgentOptions, AgentTool, AgentToolResult,
    AssistantMessage, ContentBlock, InitialAgentState, LlmContext, Message, Model, StopReason,
    StreamFnAdapter, StreamProvider, TextContent, ThinkingContent, ThinkingLevel, ToolCallContent,
    ToolResultMessage, UserContent, UserMessage,
};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

fn agent_options(stream_provider: Arc<dyn StreamProvider>) -> AgentOptions {
    AgentOptions {
        initial_state: None,
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

fn text_from_blocks(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn text_from_agent_message(message: &AgentMessage) -> String {
    match message {
        AgentMessage::Assistant(a) => text_from_blocks(&a.content),
        AgentMessage::ToolResult(t) => text_from_blocks(&t.content),
        AgentMessage::User(u) => match &u.content {
            UserContent::Plain(s) => s.clone(),
            UserContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    oh_my_agentloop::UserContentBlock::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        },
        AgentMessage::Custom(_) => String::new(),
    }
}

fn llm_context_user_text_contains(ctx: &LlmContext, needle: &str) -> bool {
    ctx.messages.iter().any(|m| {
        if let Message::User(u) = m {
            match &u.content {
                UserContent::Plain(t) => t.contains(needle),
                UserContent::Blocks(blocks) => blocks.iter().any(|b| {
                    matches!(
                        b,
                        oh_my_agentloop::UserContentBlock::Text(t) if t.text.contains(needle)
                    )
                }),
            }
        } else {
            false
        }
    })
}

fn calculate_parameters_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "expression": { "type": "string", "description": "Arithmetic expression to evaluate" }
        },
        "required": ["expression"]
    })
}

/// Minimal calculator for e2e parity (`123 * 456`, `5 + 3`, etc.).
struct CalculateTool;

fn eval_simple_arithmetic(expression: &str) -> Option<i64> {
    let e = expression.replace(' ', "");
    for (sep, op) in [("*", 0i32), ("+", 1), ("-", 2)] {
        if let Some((a, b)) = e.split_once(sep) {
            let left: i64 = a.parse().ok()?;
            let right: i64 = b.parse().ok()?;
            return Some(match op {
                0 => left * right,
                1 => left + right,
                2 => left - right,
                _ => unreachable!(),
            });
        }
    }
    None
}

#[async_trait]
impl AgentTool for CalculateTool {
    fn name(&self) -> &str {
        "calculate"
    }
    fn label(&self) -> &str {
        "Calculator"
    }
    fn description(&self) -> &str {
        "Evaluate arithmetic expressions."
    }
    fn parameters(&self) -> Value {
        calculate_parameters_schema()
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _cancel: CancellationToken,
        _on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>>,
    ) -> Result<AgentToolResult, oh_my_agentloop::AgentError> {
        let expr = params
            .get("expression")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let value = eval_simple_arithmetic(expr).unwrap_or(0);
        let text = format!("{expr} = {value}");
        Ok(AgentToolResult {
            content: vec![ContentBlock::Text(TextContent {
                text,
                text_signature: None,
            })],
            details: Some(json!({ "expression": expr, "value": value })),
        })
    }
}

fn reasoning_model() -> Model {
    let mut m = test_model();
    m.reasoning = true;
    m
}

fn assistant_with_tool_calls(
    model: &Model,
    text: &str,
    calls: Vec<ToolCallContent>,
) -> AssistantMessage {
    let mut content: Vec<ContentBlock> = vec![ContentBlock::Text(TextContent {
        text: text.into(),
        text_signature: None,
    })];
    for tc in calls {
        content.push(ContentBlock::ToolCall(tc));
    }
    AssistantMessage {
        content,
        model: model.id.clone(),
        provider: model.provider.clone(),
        api: model.api.clone(),
        response_id: None,
        stop_reason: StopReason::ToolUse,
        error_message: None,
        usage: oh_my_agentloop::Usage::default(),
        timestamp: 0,
    }
}

#[tokio::test]
async fn multi_turn_context_is_preserved_across_prompts() {
    let model = test_model();
    let round = Arc::new(AtomicUsize::new(0));
    let r = round.clone();
    let stream_fn: oh_my_agentloop::StreamFn = Arc::new(move |m, ctx, _req| {
        let model = m.clone();
        let r = r.clone();
        Box::pin(async move {
            let n = r.fetch_add(1, Ordering::SeqCst);
            let msg = if n == 0 {
                assistant_text("Nice to meet you, Alice.", &model)
            } else if llm_context_user_text_contains(&ctx, "Alice") {
                assistant_text("Your name is Alice.", &model)
            } else {
                assistant_text("I do not know your name.", &model)
            };
            let s = futures::stream::iter(vec![Ok(oh_my_agentloop::StreamEvent::Done {
                message: msg,
            })]);
            Ok(Box::pin(s) as oh_my_agentloop::LlmEventStream)
        })
    });

    let mut opts = agent_options(Arc::new(StreamFnAdapter(stream_fn)));
    opts.initial_state = Some(InitialAgentState {
        system_prompt: Some("You are a helpful assistant.".into()),
        model: Some(model),
        thinking_level: Some(ThinkingLevel::Off),
        tools: Some(vec![]),
        messages: None,
    });
    let agent = Agent::new(opts);

    agent.prompt_text("My name is Alice.", None).await.unwrap();
    assert_eq!(agent.state().messages.len(), 2);

    agent.prompt_text("What is my name?", None).await.unwrap();
    assert_eq!(agent.state().messages.len(), 4);

    let state = agent.state();
    let last = state.messages.last().expect("assistant");
    assert_eq!(last.role(), "assistant");
    let tail_text = text_from_agent_message(last);
    assert!(
        tail_text.to_lowercase().contains("alice"),
        "expected name recall, got {:?}",
        tail_text
    );
}

#[tokio::test]
async fn thinking_blocks_are_preserved_in_stored_assistant_message() {
    let model = reasoning_model();
    let thinking = ThinkingContent {
        thinking: "step by step".into(),
        thinking_signature: None,
        redacted: None,
    };
    let final_msg = AssistantMessage {
        content: vec![
            ContentBlock::Thinking(thinking.clone()),
            ContentBlock::Text(TextContent {
                text: "4".into(),
                text_signature: None,
            }),
        ],
        model: model.id.clone(),
        provider: model.provider.clone(),
        api: model.api.clone(),
        response_id: None,
        stop_reason: StopReason::Stop,
        error_message: None,
        usage: oh_my_agentloop::Usage::default(),
        timestamp: 0,
    };
    let stream_fn = stream_done_only(final_msg.clone());

    let mut opts = agent_options(stream_fn);
    opts.initial_state = Some(InitialAgentState {
        system_prompt: Some("You are a helpful assistant.".into()),
        model: Some(model),
        thinking_level: Some(ThinkingLevel::Low),
        tools: Some(vec![]),
        messages: None,
    });
    let agent = Agent::new(opts);
    agent.prompt_text("What is 2+2?", None).await.unwrap();

    let state = agent.state();
    let assistant_msg = state.messages.get(1).expect("assistant");
    let AgentMessage::Assistant(am) = assistant_msg else {
        panic!("expected assistant");
    };
    assert_eq!(am.content.len(), 2);
    assert!(matches!(
        &am.content[0],
        ContentBlock::Thinking(t) if t.thinking == "step by step"
    ));
    assert!(matches!(
        &am.content[1],
        ContentBlock::Text(t) if t.text == "4"
    ));
}

#[tokio::test]
async fn continue_from_tool_result_produces_final_assistant_answer() {
    let model = test_model();
    let stream_fn = stream_done_only(assistant_text("The answer is 8.", &model));

    let mut opts = agent_options(stream_fn);
    opts.initial_state = Some(InitialAgentState {
        system_prompt: Some(
            "You are a helpful assistant. After getting a calculation result, state the answer clearly."
                .into(),
        ),
        model: Some(model.clone()),
        thinking_level: Some(ThinkingLevel::Off),
        tools: Some(vec![Arc::new(CalculateTool)]),
        messages: None,
    });
    let agent = Agent::new(opts);

    let user_message = UserMessage {
        content: UserContent::Plain("What is 5 + 3?".into()),
        timestamp: 0,
    };
    let assistant_message = assistant_with_tool_calls(
        &model,
        "Let me calculate that.",
        vec![ToolCallContent {
            id: "calc-1".into(),
            name: "calculate".into(),
            arguments: json!({ "expression": "5 + 3" }),
        }],
    );
    let tool_result = ToolResultMessage {
        tool_call_id: "calc-1".into(),
        tool_name: "calculate".into(),
        content: vec![ContentBlock::Text(TextContent {
            text: "5 + 3 = 8".into(),
            text_signature: None,
        })],
        details: None,
        is_error: false,
        timestamp: 0,
    };

    agent.set_messages(vec![
        AgentMessage::User(user_message),
        AgentMessage::Assistant(assistant_message),
        AgentMessage::ToolResult(tool_result),
    ]);

    agent.continue_run().await.unwrap();
    assert!(!agent.state().is_streaming);
    assert!(agent.state().messages.len() >= 4);

    let state = agent.state();
    let last = state.messages.last().expect("tail");
    assert_eq!(last.role(), "assistant");
    assert!(text_from_agent_message(last).contains('8'));
}

#[tokio::test]
async fn pending_tool_calls_tracks_active_execution_then_clears() {
    let model = test_model();
    let first = assistant_with_tool_calls(
        &model,
        "Let me calculate that.",
        vec![ToolCallContent {
            id: "calc-1".into(),
            name: "calculate".into(),
            arguments: json!({ "expression": "123 * 456" }),
        }],
    );
    let second = assistant_text("The result is 56088.", &model);
    let stream_fn = stream_two_rounds(first, second);

    #[allow(clippy::type_complexity)]
    let snapshots: Arc<Mutex<Vec<(String, Vec<String>)>>> = Arc::new(Mutex::new(Vec::new()));

    let mut opts = agent_options(stream_fn);
    opts.initial_state = Some(InitialAgentState {
        system_prompt: Some(
            "You are a helpful assistant. Always use the calculator tool for math.".into(),
        ),
        model: Some(model),
        thinking_level: Some(ThinkingLevel::Off),
        tools: Some(vec![Arc::new(CalculateTool)]),
        messages: None,
    });
    let agent = Agent::new(opts);
    let agent_listen = agent.clone();
    let snapshots_listen = snapshots.clone();

    let _sub = agent_listen.clone().subscribe(move |event, _cancel| {
        let agent_c = agent_listen.clone();
        let snapshots_c = snapshots_listen.clone();
        async move {
            match &event {
                AgentEvent::ToolExecutionStart { .. } | AgentEvent::ToolExecutionEnd { .. } => {
                    let mut ids: Vec<_> = agent_c.pending_tool_calls().into_iter().collect();
                    ids.sort();
                    let tag = match event {
                        AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
                        AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
                        _ => return,
                    };
                    snapshots_c.lock().unwrap().push((tag.to_string(), ids));
                }
                _ => {}
            }
        }
    });

    agent
        .prompt_text("Calculate 123 * 456 using the calculator tool.", None)
        .await
        .unwrap();

    assert!(!agent.state().is_streaming);
    assert!(agent.state().messages.len() >= 4);
    let state = agent.state();
    let tool_result_msg = state
        .messages
        .iter()
        .find(|m| m.role() == "toolResult")
        .expect("tool result");
    assert!(text_from_agent_message(tool_result_msg).contains("123 * 456 = 56088"));

    let final_message = state.messages.last().expect("final");
    assert_eq!(final_message.role(), "assistant");
    assert!(text_from_agent_message(final_message).contains("56088"));
    assert!(agent.state().pending_tool_calls.is_empty());

    assert_eq!(
        &*snapshots.lock().unwrap(),
        &vec![
            (
                "tool_execution_start".to_string(),
                vec!["calc-1".to_string()]
            ),
            ("tool_execution_end".to_string(), vec![]),
        ]
    );
}

#[tokio::test]
async fn abort_during_streaming_sets_aborted_terminal_assistant_state() {
    let model = test_model();
    let mut opts = agent_options(slow_stream_for_abort(model.clone()));
    opts.initial_state = Some(InitialAgentState {
        system_prompt: Some("You are a helpful assistant.".into()),
        model: Some(model),
        thinking_level: Some(ThinkingLevel::Off),
        tools: Some(vec![]),
        messages: None,
    });
    let agent = Agent::new(opts);
    let events = Arc::new(Mutex::new(Vec::<AgentEvent>::new()));
    let events_sub = events.clone();
    let _sub = agent.subscribe(move |event, _cancel| {
        let events_sub = events_sub.clone();
        async move {
            events_sub.lock().unwrap().push(event);
        }
    });

    let run = tokio::spawn({
        let a = agent.clone();
        async move { a.prompt_text("Count slowly from 1 to 20.", None).await }
    });

    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    agent.abort();
    let err = run.await.unwrap().unwrap_err();
    assert!(matches!(err, AgentError::Aborted));

    assert!(!agent.state().is_streaming);
    assert!(agent.state().messages.len() >= 2);
    assert!(matches!(
        events.lock().unwrap().last(),
        Some(AgentEvent::RunAborted { .. })
    ));
    let state = agent.state();
    let last = state.messages.last().expect("last");
    let AgentMessage::Assistant(am) = last else {
        panic!("expected assistant");
    };
    assert_eq!(am.stop_reason, StopReason::Aborted);
    assert!(am.error_message.is_some());
    assert_eq!(agent.state().error_message, am.error_message);
}
