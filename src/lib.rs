//! Stateful agent loop — Rust port of `@mariozechner/pi-agent-core` (non-proxy surface).
//!
//! LLM message types in [`types`] follow `@mariozechner/pi-ai` JSON field names and enums
//! (`StopReason`, `UserContent`, signatures on text/thinking blocks, optional `details`, etc.).

pub mod agent;
pub mod agent_loop;
pub mod types;

pub use agent::{Agent, AgentOptions, InitialAgentState, QueueMode, Subscription};
pub use agent_loop::{agent_loop, agent_loop_continue, run_agent_loop, run_agent_loop_continue};
pub use types::*;
