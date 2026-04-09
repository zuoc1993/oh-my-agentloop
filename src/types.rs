use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

// ============================================================
// LLM Types (pi-ai equivalent)
// ============================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    pub data: String,
    pub mime_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    pub thinking: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redacted: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallContent {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ContentBlock {
    Text(TextContent),
    Image(ImageContent),
    Thinking(ThinkingContent),
    ToolCall(ToolCallContent),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
}

impl Serialize for StopReason {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(match self {
            StopReason::Stop => "stop",
            StopReason::Length => "length",
            StopReason::ToolUse => "toolUse",
            StopReason::Error => "error",
            StopReason::Aborted => "aborted",
        })
    }
}

impl<'de> Deserialize<'de> for StopReason {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StopReasonVisitor;

        impl serde::de::Visitor<'_> for StopReasonVisitor {
            type Value = StopReason;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    f,
                    "a stop reason string: stop, length, toolUse, error, or aborted"
                )
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<StopReason, E> {
                match v {
                    "stop" => Ok(StopReason::Stop),
                    "length" => Ok(StopReason::Length),
                    "toolUse" => Ok(StopReason::ToolUse),
                    "error" => Ok(StopReason::Error),
                    "aborted" => Ok(StopReason::Aborted),
                    other => Err(E::unknown_variant(
                        other,
                        &["stop", "length", "toolUse", "error", "aborted"],
                    )),
                }
            }
        }

        deserializer.deserialize_str(StopReasonVisitor)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Cost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub total_tokens: u64,
    pub cost: Cost,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    Sse,
    Ws,
}

impl Default for Transport {
    fn default() -> Self {
        Transport::Sse
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: String,
    pub provider: String,
    pub base_url: String,
    pub reasoning: bool,
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub cost: ModelCost,
    #[serde(default)]
    pub context_window: u64,
    #[serde(default)]
    pub max_tokens: u64,
}

impl Default for Model {
    fn default() -> Self {
        Model {
            id: "unknown".into(),
            name: "unknown".into(),
            api: "unknown".into(),
            provider: "unknown".into(),
            base_url: String::new(),
            reasoning: false,
            input: Vec::new(),
            cost: ModelCost::default(),
            context_window: 0,
            max_tokens: 0,
        }
    }
}

/// Thinking/reasoning level for models that support it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    #[serde(rename = "xhigh")]
    XHigh,
}

impl Default for ThinkingLevel {
    fn default() -> Self {
        ThinkingLevel::Off
    }
}

// ============================================================
// LLM Messages
// ============================================================

/// A single user content block (`TextContent` | `ImageContent` from pi-ai).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum UserContentBlock {
    Text(TextContent),
    Image(ImageContent),
}

/// User message body: plain string or structured blocks (`pi-ai` `UserMessage["content"]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Plain(String),
    Blocks(Vec<UserContentBlock>),
}

/// Building [`UserContent`] from [`ContentBlock`] failed: user messages only allow text and image blocks.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UserContentBuildError {
    #[error("user message content cannot include a thinking block")]
    ThinkingBlock,
    #[error("user message content cannot include a tool call block")]
    ToolCallBlock,
}

impl UserContent {
    /// Converts LLM [`ContentBlock`]s into user message content.
    ///
    /// Returns an error if any block is not text or image (no data is dropped).
    pub fn try_from_llm_blocks(blocks: Vec<ContentBlock>) -> Result<Self, UserContentBuildError> {
        if blocks.is_empty() {
            return Ok(UserContent::Blocks(vec![]));
        }

        let mut out = Vec::new();
        for b in blocks {
            match b {
                ContentBlock::Text(t) => out.push(UserContentBlock::Text(t)),
                ContentBlock::Image(i) => out.push(UserContentBlock::Image(i)),
                ContentBlock::Thinking(_) => return Err(UserContentBuildError::ThinkingBlock),
                ContentBlock::ToolCall(_) => return Err(UserContentBuildError::ToolCallBlock),
            }
        }

        if out.len() == 1 {
            if let UserContentBlock::Text(t) = &out[0] {
                if t.text_signature.is_none() {
                    return Ok(UserContent::Plain(t.text.clone()));
                }
            }
        }

        Ok(UserContent::Blocks(out))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessage {
    pub content: UserContent,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub provider: String,
    pub api: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    pub stop_reason: StopReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    pub usage: Usage,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<ContentBlock>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    pub is_error: bool,
    pub timestamp: i64,
}

/// Standard LLM message — what the model understands.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
}

impl Message {
    pub fn role(&self) -> &str {
        match self {
            Message::User(_) => "user",
            Message::Assistant(_) => "assistant",
            Message::ToolResult(_) => "toolResult",
        }
    }
}

/// Tool schema definition sent to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Context passed to the LLM stream function.
#[derive(Debug, Clone)]
pub struct LlmContext {
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
}

// ============================================================
// Stream Types — LLM streaming interface
// ============================================================

/// Events yielded by the LLM stream function.
///
/// Mirrors TypeScript `AssistantMessageEvent`: each incremental event carries the
/// provider’s current `partial` [`AssistantMessage`]; the loop forwards that shape
/// to [`AgentEvent::MessageUpdate`] instead of reconstructing content locally.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Start {
        partial: AssistantMessage,
    },
    TextStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    TextDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    TextEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    ThinkingStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    ThinkingDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    ThinkingEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    ToolCallStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    ToolCallDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    ToolCallEnd {
        content_index: usize,
        tool_call: ToolCallContent,
        partial: AssistantMessage,
    },
    Done {
        message: AssistantMessage,
    },
    Error {
        message: AssistantMessage,
    },
}

/// Per-level thinking token budgets.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ThinkingBudgets {
    pub minimal: Option<u64>,
    pub low: Option<u64>,
    pub medium: Option<u64>,
    pub high: Option<u64>,
    pub xhigh: Option<u64>,
}

/// Options forwarded to the LLM stream function.
/// Mirrors TypeScript `SimpleStreamOptions`.
#[derive(Clone)]
pub struct StreamOptions {
    pub api_key: Option<String>,
    pub reasoning: Option<ThinkingLevel>,
    pub session_id: Option<String>,
    pub transport: Transport,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub max_retry_delay_ms: Option<u64>,
    pub on_payload: Option<OnPayloadFn>,
}

impl fmt::Debug for StreamOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamOptions")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("reasoning", &self.reasoning)
            .field("session_id", &self.session_id)
            .field("transport", &self.transport)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("thinking_budgets", &self.thinking_budgets)
            .field("max_retry_delay_ms", &self.max_retry_delay_ms)
            .field("on_payload", &self.on_payload.as_ref().map(|_| "<fn>"))
            .finish()
    }
}

#[derive(Clone)]
pub struct StreamRequest {
    pub options: StreamOptions,
    pub cancel: CancellationToken,
}

impl fmt::Debug for StreamRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamRequest")
            .field("options", &self.options)
            .field("cancel", &"<CancellationToken>")
            .finish()
    }
}

pub type LlmEventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, AgentError>> + Send>>;

/// Stream function injected by the user. Connects to any LLM provider.
///
/// Contract (mirrors TypeScript):
/// - Must not panic for request/model/runtime failures.
/// - Failures must be encoded in the stream via `StreamEvent::Error`.
pub type StreamFn = Arc<
    dyn Fn(
            Model,
            LlmContext,
            StreamRequest,
        ) -> Pin<Box<dyn Future<Output = Result<LlmEventStream, AgentError>> + Send>>
        + Send
        + Sync,
>;

// ============================================================
// Agent Message — extends LLM Message with custom types
// ============================================================

/// Custom application message. Filtered out by `convert_to_llm` before LLM calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomMessage {
    pub role: String,
    pub data: serde_json::Value,
    pub timestamp: i64,
}

/// Union of LLM messages + custom application messages.
///
/// Rust equivalent of TypeScript's `AgentMessage = Message | CustomAgentMessages[...]`.
/// Extension is done via the `Custom` variant rather than declaration merging.
#[derive(Debug, Clone)]
pub enum AgentMessage {
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
    Custom(CustomMessage),
}

impl AgentMessage {
    pub fn role(&self) -> &str {
        match self {
            AgentMessage::User(_) => "user",
            AgentMessage::Assistant(_) => "assistant",
            AgentMessage::ToolResult(_) => "toolResult",
            AgentMessage::Custom(c) => &c.role,
        }
    }

    pub fn as_assistant(&self) -> Option<&AssistantMessage> {
        match self {
            AgentMessage::Assistant(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_tool_result(&self) -> Option<&ToolResultMessage> {
        match self {
            AgentMessage::ToolResult(m) => Some(m),
            _ => None,
        }
    }

    pub fn into_message(self) -> Option<Message> {
        match self {
            AgentMessage::User(m) => Some(Message::User(m)),
            AgentMessage::Assistant(m) => Some(Message::Assistant(m)),
            AgentMessage::ToolResult(m) => Some(Message::ToolResult(m)),
            AgentMessage::Custom(_) => None,
        }
    }
}

impl From<Message> for AgentMessage {
    fn from(m: Message) -> Self {
        match m {
            Message::User(u) => AgentMessage::User(u),
            Message::Assistant(a) => AgentMessage::Assistant(a),
            Message::ToolResult(t) => AgentMessage::ToolResult(t),
        }
    }
}

// ============================================================
// Agent Tool
// ============================================================

/// Final or partial result produced by a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentToolResult {
    pub content: Vec<ContentBlock>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

/// Tool definition used by the agent runtime.
///
/// Mirrors TypeScript `AgentTool<TParameters, TDetails>`.
#[async_trait]
pub trait AgentTool: Send + Sync {
    fn name(&self) -> &str;
    fn label(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> serde_json::Value;

    /// Optional compatibility shim for raw tool-call arguments before validation.
    fn prepare_arguments(&self, args: serde_json::Value) -> serde_json::Value {
        args
    }

    /// Execute the tool call.
    async fn execute(
        &self,
        tool_call_id: &str,
        params: serde_json::Value,
        cancel: CancellationToken,
        on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>>,
    ) -> Result<AgentToolResult, AgentError>;

    fn as_tool_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters(),
        }
    }
}

// ============================================================
// Agent Context
// ============================================================

/// Context snapshot passed into the low-level agent loop.
pub struct AgentContext {
    pub system_prompt: String,
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<Arc<dyn AgentTool>>,
}

impl fmt::Debug for AgentContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentContext")
            .field("system_prompt", &self.system_prompt)
            .field("messages_count", &self.messages.len())
            .field("tools_count", &self.tools.len())
            .finish()
    }
}

// ============================================================
// Tool Execution Config
// ============================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolExecutionMode {
    Sequential,
    Parallel,
}

impl Default for ToolExecutionMode {
    fn default() -> Self {
        ToolExecutionMode::Parallel
    }
}

/// Snapshot of the agent context passed to tool hooks.
pub struct AgentContextSnapshot {
    pub system_prompt: String,
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<Arc<dyn AgentTool>>,
}

/// Context passed to `before_tool_call`.
///
/// `args` holds validated arguments (after `prepare_arguments` and JSON Schema validation).
/// The hook may mutate the inner JSON value in place; those mutations are used for execution
/// without a second validation pass (mirrors in-place mutation of the validated object in TS).
pub struct BeforeToolCallContext {
    pub assistant_message: AssistantMessage,
    pub tool_call: ToolCallContent,
    pub args: Arc<Mutex<serde_json::Value>>,
    pub context: AgentContextSnapshot,
}

/// Result from `before_tool_call`. Return `block: true` to prevent execution.
pub struct BeforeToolCallResult {
    pub block: bool,
    pub reason: Option<String>,
}

/// Context passed to `after_tool_call`.
pub struct AfterToolCallContext {
    pub assistant_message: AssistantMessage,
    pub tool_call: ToolCallContent,
    pub args: serde_json::Value,
    pub result: AgentToolResult,
    pub is_error: bool,
    pub context: AgentContextSnapshot,
}

/// Partial override returned from `after_tool_call`.
/// Omitted fields keep original values. No deep merge.
pub struct AfterToolCallResult {
    pub content: Option<Vec<ContentBlock>>,
    pub details: Option<serde_json::Value>,
    pub is_error: Option<bool>,
}

// ============================================================
// Callback Type Aliases
// ============================================================

/// Converts `AgentMessage[]` to LLM-compatible `Message[]` before each LLM call.
pub type ConvertToLlmFn = Arc<
    dyn Fn(Vec<AgentMessage>) -> Pin<Box<dyn Future<Output = Vec<Message>> + Send>> + Send + Sync,
>;

/// Optional transform applied to context before `convert_to_llm`.
pub type TransformContextFn = Arc<
    dyn Fn(
            Vec<AgentMessage>,
            CancellationToken,
        ) -> Pin<Box<dyn Future<Output = Vec<AgentMessage>> + Send>>
        + Send
        + Sync,
>;

/// Resolves an API key dynamically for each LLM call.
pub type GetApiKeyFn =
    Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = Option<String>> + Send>> + Send + Sync>;

/// Optional hook to inspect or replace provider request payloads before sending.
pub type OnPayloadFn = Arc<
    dyn Fn(
            serde_json::Value,
            Model,
        ) -> Pin<Box<dyn Future<Output = Option<serde_json::Value>> + Send>>
        + Send
        + Sync,
>;

/// Returns steering or follow-up messages.
pub type GetMessagesFn =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Vec<AgentMessage>> + Send>> + Send + Sync>;

/// Before-tool-call hook.
pub type BeforeToolCallHookFn = Arc<
    dyn Fn(
            BeforeToolCallContext,
            CancellationToken,
        ) -> Pin<Box<dyn Future<Output = Option<BeforeToolCallResult>> + Send>>
        + Send
        + Sync,
>;

/// After-tool-call hook.
pub type AfterToolCallHookFn = Arc<
    dyn Fn(
            AfterToolCallContext,
            CancellationToken,
        ) -> Pin<Box<dyn Future<Output = Option<AfterToolCallResult>> + Send>>
        + Send
        + Sync,
>;

// ============================================================
// Agent Loop Config
// ============================================================

/// Configuration for the low-level agent loop.
///
/// Mirrors TypeScript `AgentLoopConfig extends SimpleStreamOptions`.
pub struct AgentLoopConfig {
    pub model: Model,
    pub reasoning: Option<ThinkingLevel>,
    pub convert_to_llm: ConvertToLlmFn,
    pub transform_context: Option<TransformContextFn>,
    pub get_api_key: Option<GetApiKeyFn>,
    pub get_steering_messages: Option<GetMessagesFn>,
    pub get_follow_up_messages: Option<GetMessagesFn>,
    pub tool_execution: ToolExecutionMode,
    pub before_tool_call: Option<BeforeToolCallHookFn>,
    pub after_tool_call: Option<AfterToolCallHookFn>,
    pub api_key: Option<String>,
    pub session_id: Option<String>,
    pub transport: Transport,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub max_retry_delay_ms: Option<u64>,
    pub on_payload: Option<OnPayloadFn>,
}

// ============================================================
// Agent Events — emitted for UI / state updates
// ============================================================

/// Events emitted by the agent loop for UI updates and state synchronization.
///
/// Each run ends with exactly one explicit terminal event.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    AgentStart,
    RunCompleted {
        messages: Vec<AgentMessage>,
    },
    RunFailed {
        messages: Vec<AgentMessage>,
        error_message: String,
    },
    RunAborted {
        messages: Vec<AgentMessage>,
    },
    TurnStart,
    TurnEnd {
        message: AgentMessage,
        tool_results: Vec<ToolResultMessage>,
    },
    MessageStart {
        message: AgentMessage,
    },
    MessageUpdate {
        message: AgentMessage,
        stream_event: StreamEvent,
    },
    MessageEnd {
        message: AgentMessage,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
        partial_result: AgentToolResult,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: AgentToolResult,
        is_error: bool,
    },
}

// ============================================================
// Errors
// ============================================================

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Agent is already processing a prompt. Use steer() or follow_up() to queue messages.")]
    AlreadyProcessing,

    #[error("Cannot continue: no messages in context")]
    NoMessages,

    #[error("Cannot continue from message role: assistant")]
    ContinueFromAssistant,

    #[error("Run aborted")]
    Aborted,

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Task join error: {0}")]
    JoinError(String),

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

#[derive(Debug)]
pub enum RunOutcome {
    Completed {
        new_messages: Vec<AgentMessage>,
    },
    Failed {
        new_messages: Vec<AgentMessage>,
        error: AgentError,
    },
    Aborted {
        new_messages: Vec<AgentMessage>,
    },
}

// ============================================================
// Internal: Event Emitter
// ============================================================

/// Event emitter used by the agent loop.
/// Wraps a callback that processes `AgentEvent`s.
pub struct EventEmitter {
    f: Arc<dyn Fn(AgentEvent) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>,
}

impl EventEmitter {
    pub fn new<F, Fut>(f: F) -> Self
    where
        F: Fn(AgentEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        EventEmitter {
            f: Arc::new(move |event| Box::pin(f(event))),
        }
    }

    pub async fn emit(&self, event: AgentEvent) {
        (self.f)(event).await;
    }
}

impl Clone for EventEmitter {
    fn clone(&self) -> Self {
        EventEmitter { f: self.f.clone() }
    }
}

// ============================================================
// Public Agent State (for Agent high-level API)
// ============================================================

/// Public agent state exposed by the high-level `Agent`.
///
/// `is_streaming` stays true until the current run finishes and subscribed listeners have settled
/// for all emitted events (including the terminal run event), matching
/// `@mariozechner/agent` `AgentState`.
#[derive(Debug, Clone)]
pub struct AgentState {
    pub system_prompt: String,
    pub model: Model,
    pub thinking_level: ThinkingLevel,
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub messages: Vec<AgentMessage>,
    /// True while a prompt/continuation is running and until event listeners have completed.
    pub is_streaming: bool,
    pub streaming_message: Option<AgentMessage>,
    pub pending_tool_calls: std::collections::HashSet<String>,
    pub error_message: Option<String>,
}

impl fmt::Debug for dyn AgentTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentTool")
            .field("name", &self.name())
            .field("label", &self.label())
            .finish()
    }
}

// ============================================================
// Helpers
// ============================================================

pub fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Default `convert_to_llm`: passes through standard messages, drops custom.
pub fn default_convert_to_llm(messages: Vec<AgentMessage>) -> Vec<Message> {
    messages
        .into_iter()
        .filter_map(|m| m.into_message())
        .collect()
}

pub(crate) fn create_error_tool_result(message: &str) -> AgentToolResult {
    AgentToolResult {
        content: vec![ContentBlock::Text(TextContent {
            text: message.to_string(),
            text_signature: None,
        })],
        details: None,
    }
}
