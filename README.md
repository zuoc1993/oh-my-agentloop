# oh-my-agentloop

Rust port of `@mariozechner/pi-agent-core` — a generic, embeddable agent loop runtime.

## Origin

This project is a Rust port of [pi-agent-core](https://github.com/badlogic/pi-mono/tree/main/packages/agent) by [Mario Zechner](https://github.com/badlogic), originally part of the `pi-mono` monorepo. It is distributed under the [MIT License](LICENSE).

## What it does

`oh-my-agentloop` combines LLM streaming, tool orchestration, and event-driven state management into a single runtime that you can embed in any Rust application. You bring the model provider and tools; the runtime handles the loop.

**Core capabilities:**

- **Provider-agnostic LLM streaming** via an injected `StreamFn` — works with any provider.
- **Tool call orchestration** with JSON Schema validation, before/after hooks, and parallel or sequential execution.
- **Dual-layer API** — use the low-level `run_agent_loop` for full control, or the high-level `Agent` for managed state, subscriptions, and queue-based intervention.
- **Runtime intervention** — inject `steer` messages mid-run or queue `follow_up` tasks for after the agent would otherwise stop.
- **Event-driven state sync** — every state change is an `AgentEvent`; subscribe for UI updates, persistence, or auditing.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
oh-my-agentloop = { path = "../oh-my-agentloop" }
```

## Quick start

### 1. Define a model and stream function

```rust
use std::sync::Arc;
use oh_my_agentloop::{
    Agent, AgentOptions, AssistantMessage, ContentBlock, InitialAgentState,
    Model, ModelCost, StopReason, StreamEvent, StreamFn, TextContent,
    ThinkingLevel, Usage,
};

fn my_model() -> Model {
    Model {
        id: "gpt-4o".into(),
        name: "GPT-4o".into(),
        api: "openai-responses".into(),
        provider: "openai".into(),
        base_url: "https://api.openai.com".into(),
        reasoning: false,
        input: vec!["text".into()],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 4096,
    }
}

fn my_stream_fn() -> StreamFn {
    Arc::new(move |model, _ctx, _req| {
        Box::pin(async move {
            // Replace with your actual LLM provider call.
            let message = AssistantMessage {
                content: vec![ContentBlock::Text(TextContent {
                    text: "Hello from the agent!".into(),
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
            let stream = futures::stream::iter(vec![
                Ok(StreamEvent::Done { message }),
            ]);
            Ok(Box::pin(stream) as oh_my_agentloop::LlmEventStream)
        })
    })
}
```

### 2. Build and run the agent

```rust
#[tokio::main]
async fn main() {
    let agent = Agent::new(AgentOptions {
        initial_state: Some(InitialAgentState {
            system_prompt: Some("You are a helpful assistant.".into()),
            model: Some(my_model()),
            thinking_level: Some(ThinkingLevel::Off),
            tools: Some(vec![]),
            messages: None,
        }),
        stream_fn: my_stream_fn(),
        // All other fields default to None.
        convert_to_llm: None, transform_context: None,
        get_api_key: None, before_tool_call: None, after_tool_call: None,
        steering_mode: None, follow_up_mode: None, session_id: None,
        transport: None, tool_execution: None, api_key: None,
        temperature: None, max_tokens: None, thinking_budgets: None,
        max_retry_delay_ms: None, on_payload: None,
    });

    agent.prompt_text("Hello!", None).await.unwrap();

    let state = agent.state();
    for msg in &state.messages {
        println!("[{}] ...", msg.role());
    }
}
```

### 3. Subscribe to events

```rust
let _sub = agent.subscribe(|event, _cancel| async move {
    match event {
        oh_my_agentloop::AgentEvent::MessageUpdate { .. } => {
            // Push streaming tokens to your UI.
        }
        oh_my_agentloop::AgentEvent::ToolExecutionStart { tool_name, .. } => {
            println!("Running tool: {tool_name}");
        }
        oh_my_agentloop::AgentEvent::RunCompleted { .. }
        | oh_my_agentloop::AgentEvent::RunFailed { .. }
        | oh_my_agentloop::AgentEvent::RunAborted { .. } => {
            // Persist the final transcript.
        }
        _ => {}
    }
});
```

### 4. Add a tool

```rust
use async_trait::async_trait;
use oh_my_agentloop::{AgentError, AgentTool, AgentToolResult, ContentBlock, TextContent};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

struct EchoTool;

#[async_trait]
impl AgentTool for EchoTool {
    fn name(&self) -> &str { "echo" }
    fn label(&self) -> &str { "Echo" }
    fn description(&self) -> &str { "Echo the input back" }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _cancel: CancellationToken,
        _on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>>,
    ) -> Result<AgentToolResult, AgentError> {
        let value = params["value"].as_str().unwrap_or_default();
        Ok(AgentToolResult {
            content: vec![ContentBlock::Text(TextContent {
                text: format!("echo: {value}"),
                text_signature: None,
            })],
            details: None,
        })
    }
}
```

## Architecture

```text
┌──────────────────────────────────────────────────────────┐
│ High-Level API: Agent                                    │
│ - Owns transcript, tools, queues, listeners              │
│ - prompt / continue_run / abort / subscribe              │
│ - Reduces AgentEvent into AgentState                     │
└───────────────────────┬──────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────┐
│ Low-Level Runtime: run_agent_loop                        │
│ - Drives LLM streaming responses                        │
│ - Executes tool call pipeline                            │
│ - Emits lifecycle events via EventEmitter                │
└───────────────────────┬──────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────┐
│ Integration Boundary (you provide these)                 │
│ - StreamFn: connect to any LLM provider                  │
│ - AgentTool: implement external capabilities             │
│ - convert_to_llm / transform_context / hooks             │
└──────────────────────────────────────────────────────────┘
```

The low-level loop handles deterministic execution; the high-level `Agent` handles state, concurrency protection, and subscriptions. All observable behavior flows through the `AgentEvent` bus.

For full architectural details, see [docs/AGENT_ARCHITECTURE.md](docs/AGENT_ARCHITECTURE.md).

## Key concepts

| Concept | Description |
|---------|-------------|
| `Message` vs `AgentMessage` | `Message` is what the LLM understands (user/assistant/tool-result). `AgentMessage` adds a `Custom` variant for application-level messages, filtered out by `convert_to_llm` before LLM calls. |
| `StreamFn` | Your LLM provider adapter. Takes a `Model`, `LlmContext`, and `StreamRequest`; returns a stream of `StreamEvent`s. |
| `AgentTool` | A trait for tools the agent can call. Includes JSON Schema `parameters()`, optional `prepare_arguments()`, and `execute()`. |
| `AgentEvent` | Events emitted during a run — `AgentStart`, `TurnStart/End`, `MessageStart/Update/End`, `ToolExecution*`, `RunCompleted/Failed/Aborted`. |
| `steer()` / `follow_up()` | Queue-based runtime intervention. Steering injects messages before the next LLM call; follow-up runs only after the agent would otherwise stop. |
| `RunOutcome` | The authoritative result of a run: `Completed`, `Failed`, or `Aborted`. |

## Extension points

- **`convert_to_llm`** — Transform `AgentMessage[]` to `Message[]` before each LLM call. Required if you use `AgentMessage::Custom`.
- **`transform_context`** — Trim, summarize, or augment the message history before conversion. Essential for long conversations.
- **`before_tool_call` / `after_tool_call`** — Hooks for policy enforcement, auditing, argument mutation, result transformation.
- **`on_payload`** — Inspect or rewrite the raw provider request payload.
- **`get_api_key`** — Dynamic API key resolution per provider (useful for rotating/expiring tokens).

## Testing

```bash
cargo test
```

66 integration tests covering event sequencing, tool pipeline stages, cancellation paths, queue semantics, serialization parity, and end-to-end agent behavior.

## License

See the repository root for license information.
