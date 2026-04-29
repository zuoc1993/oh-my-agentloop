//! Generic stateful agent loop — Rust port of `@mariozechner/pi-agent-core`.
//!
//! # Overview
//!
//! This crate provides two API layers:
//!
//! - The **high-level [`Agent`]** owns state (messages, tools, model, queues) and
//!   coordinates streaming runs, steering, follow-ups, and cancellation. Use it to
//!   build a chat/agent UI.
//! - The **low-level [`run_agent_loop`] / [`mod@agent_loop`]** drives a single run over
//!   an arbitrary [`AgentContext`] with explicit [`EventEmitter`] wiring. Use it to
//!   embed the loop inside your own state machine.
//!
//! Message types in [`types`] follow the `@mariozechner/pi-ai` JSON shape
//! (camelCase fields, lowercase `StopReason`, text/thinking signatures, optional
//! tool-result `details`, etc.).
//!
//! # Quick start
//!
//! ```ignore
//! use oh_my_agentloop::{Agent, AgentOptions, InitialAgentState, ThinkingLevel};
//!
//! #[tokio::main]
//! async fn main() {
//!     let options = AgentOptions::builder(my_stream_fn())
//!         .initial_state(InitialAgentState {
//!             system_prompt: Some("You are a helpful assistant.".into()),
//!             model: Some(my_model()),
//!             thinking_level: Some(ThinkingLevel::Off),
//!             tools: Some(vec![]),
//!             messages: None,
//!         })
//!         .build();
//!     let agent = Agent::new(options);
//!     agent.prompt_text("Hello!", None).await.unwrap();
//! }
//! ```
//!
//! # Cargo features
//!
//! - `tracing` (*default*): emit structured `tracing` spans and events for runs,
//!   LLM streaming, and tool execution.
//!
//! # Observability
//!
//! Under the `tracing` feature, the following spans are emitted:
//!
//! | span                  | level  | key fields                                   |
//! |-----------------------|--------|----------------------------------------------|
//! | `agent.run`           | info   | `model`, `prompts`                           |
//! | `agent.run.continue`  | info   | `model`, `ctx_messages`                      |
//! | `agent.run.loop`      | debug  | `model`                                      |
//! | `agent.llm.stream`    | debug  | `model`                                      |
//! | `agent.tools.execute` | debug  | `mode`, `tool_calls`                         |
//! | `agent.tool.call`     | debug  | `tool`, `tool_call_id`                       |
//!
//! Wire up any `tracing-subscriber` to ingest these spans.

#![cfg_attr(docsrs, feature(doc_auto_cfg))]

pub mod agent;
pub mod agent_loop;
pub mod types;

pub use agent::{
    Agent, AgentOptions, AgentOptionsBuilder, InitialAgentState, QueueMode, Subscription,
};
pub use agent_loop::{agent_loop, agent_loop_continue, run_agent_loop, run_agent_loop_continue};
pub use types::*;
