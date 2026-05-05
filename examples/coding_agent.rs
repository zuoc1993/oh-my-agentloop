use std::process::Stdio;
use std::sync::Arc;

use async_trait::async_trait;
use oh_my_agentloop::{
    Agent, AgentError, AgentEvent, AgentMessage, AgentOptions, AgentTool, AgentToolResult,
    AssistantMessage, ContentBlock, InitialAgentState, LlmContext, LlmEventStream, Message, Model,
    ModelCost, StopReason, StreamEvent, StreamProvider, StreamRequest, TextContent, ThinkingLevel,
    ToolCallContent, Usage, UserContent, UserContentBlock,
};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

// ============================================================
// 0. RTK Pre-flight Check
// ============================================================

fn check_rtk() {
    match std::process::Command::new("rtk").arg("--version").output() {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            println!("✓ RTK detected: {version}\n");
        }
        _ => {
            eprintln!("❌ RTK is not installed or not found in PATH.");
            eprintln!();
            eprintln!("Please install RTK first:");
            eprintln!("  brew install rtk");
            eprintln!("  # or");
            eprintln!("  cargo install --git https://github.com/rtk-ai/rtk");
            eprintln!("  # or");
            eprintln!(
                "  curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh"
            );
            std::process::exit(1);
        }
    }
}

// ============================================================
// 1. Kimi K2.6 Stream Provider (OpenAI-compatible, non-streaming)
// ============================================================

struct KimiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl KimiProvider {
    fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: "https://api.moonshot.cn/v1".into(),
        }
    }
}

#[async_trait]
impl StreamProvider for KimiProvider {
    async fn stream(
        &self,
        model: Model,
        ctx: LlmContext,
        _req: StreamRequest,
    ) -> Result<LlmEventStream, AgentError> {
        let url = format!("{}/chat/completions", self.base_url);

        // Convert oh-my-agentloop Message[] to OpenAI format
        let messages: Vec<Value> = ctx
            .messages
            .into_iter()
            .map(|m| match m {
                Message::User(u) => {
                    let content = match &u.content {
                        UserContent::Plain(text) => Value::String(text.clone()),
                        UserContent::Blocks(blocks) => Value::Array(
                            blocks
                                .iter()
                                .map(|b| match b {
                                    UserContentBlock::Text(t) => {
                                        json!({"type": "text", "text": t.text})
                                    }
                                    UserContentBlock::Image(i) => json!({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": format!("data:{};base64,{}", i.mime_type, i.data)
                                        }
                                    }),
                                })
                                .collect(),
                        ),
                    };
                    json!({"role": "user", "content": content})
                }
                Message::Assistant(a) => {
                    let mut text_parts = Vec::new();
                    let mut tool_calls_json = Vec::new();
                    for block in &a.content {
                        match block {
                            ContentBlock::Text(t) => text_parts.push(t.text.clone()),
                            ContentBlock::ToolCall(tc) => {
                                tool_calls_json.push(json!({
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": tc.arguments.to_string()
                                    }
                                }));
                            }
                            _ => {}
                        }
                    }
                    let text = text_parts.join("");
                    let mut msg = json!({"role": "assistant"});
                    if !text.is_empty() {
                        msg["content"] = Value::String(text);
                    }
                    if !tool_calls_json.is_empty() {
                        msg["tool_calls"] = Value::Array(tool_calls_json);
                    }
                    msg
                }
                Message::ToolResult(tr) => {
                    let content = tr
                        .content
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text(t) => Some(t.text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    json!({
                        "role": "tool",
                        "tool_call_id": tr.tool_call_id,
                        "content": content
                    })
                }
            })
            .collect();

        // Convert tools to OpenAI function-calling format
        let tools: Vec<Value> = ctx
            .tools
            .into_iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters
                    }
                })
            })
            .collect();

        let mut body = json!({
            "model": model.id,
            "messages": messages,
            "thinking": { "type": "disabled" },
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentError::Stream(format!("HTTP request failed: {e}")))?;

        let status = response.status();
        let response_json: Value = response
            .json()
            .await
            .map_err(|e| AgentError::Stream(format!("Failed to parse JSON response: {e}")))?;

        if !status.is_success() {
            let error_msg = response_json
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown API error");
            return Err(AgentError::Stream(format!("API error: {error_msg}")));
        }

        // Parse response into AssistantMessage
        let choice = response_json
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .ok_or_else(|| AgentError::Stream("No choices in API response".into()))?;

        let message = choice
            .get("message")
            .ok_or_else(|| AgentError::Stream("No message in API choice".into()))?;

        let mut content_blocks = Vec::new();

        if let Some(content) = message.get("content").and_then(|c| c.as_str()) {
            if !content.is_empty() {
                content_blocks.push(ContentBlock::Text(TextContent {
                    text: content.into(),
                    text_signature: None,
                }));
            }
        }

        if let Some(tool_calls) = message.get("tool_calls").and_then(|c| c.as_array()) {
            for tc in tool_calls {
                let id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string();
                let name = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                let args_str = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|a| a.as_str())
                    .unwrap_or("{}");
                let arguments =
                    serde_json::from_str(args_str).unwrap_or_else(|_| json!({}));

                content_blocks.push(ContentBlock::ToolCall(ToolCallContent {
                    id,
                    name,
                    arguments,
                }));
            }
        }

        let stop_reason = choice
            .get("finish_reason")
            .and_then(|f| f.as_str())
            .map(|r| match r {
                "stop" => StopReason::Stop,
                "length" => StopReason::Length,
                "tool_calls" => StopReason::ToolUse,
                _ => StopReason::Stop,
            })
            .unwrap_or(StopReason::Stop);

        let assistant_message = AssistantMessage {
            content: content_blocks,
            model: model.id.clone(),
            provider: model.provider.clone(),
            api: model.api.clone(),
            response_id: response_json.get("id").and_then(|i| i.as_str()).map(|s| s.into()),
            stop_reason,
            error_message: None,
            usage: Usage::default(),
            timestamp: oh_my_agentloop::now_millis(),
        };

        let stream = futures::stream::iter(vec![
            Ok(StreamEvent::Start {
                partial: assistant_message.clone(),
            }),
            Ok(StreamEvent::Done {
                message: assistant_message,
            }),
        ]);

        Ok(Box::pin(stream) as LlmEventStream)
    }
}

// ============================================================
// 2. File Operation Tools
// ============================================================

/// Read a file using RTK for token-optimized output.
struct ReadFileTool;

#[async_trait]
impl AgentTool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }
    fn label(&self) -> &str {
        "Read File"
    }
    fn description(&self) -> &str {
        "Read the contents of a file at the given path. Uses RTK for token-optimized output. \
         Supports filtering levels: none (full), minimal (no comments/blanks), aggressive (signatures only)."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "level": {
                    "type": "string",
                    "enum": ["none", "minimal", "aggressive"],
                    "description": "Filter level: none (default, full content), minimal (strip comments/blanks), aggressive (signatures only)",
                    "default": "none"
                }
            },
            "required": ["path"],
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
        let path = params["path"].as_str().unwrap_or_default();
        let level = params.get("level").and_then(|v| v.as_str()).unwrap_or("none");

        let output = tokio::process::Command::new("rtk")
            .arg("read")
            .arg(path)
            .arg("-l")
            .arg(level)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| AgentError::Stream(format!("Failed to run rtk read: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let (text, is_error) = if output.status.success() {
            (stdout.into_owned(), false)
        } else {
            let msg = if stderr.is_empty() {
                stdout.into_owned()
            } else {
                format!("{stderr}\n{stdout}")
            };
            (msg, true)
        };

        Ok(AgentToolResult {
            content: vec![ContentBlock::Text(TextContent {
                text,
                text_signature: None,
            })],
            details: Some(json!({ "is_error": is_error })),
        })
    }
}

/// Write content to a file.
struct WriteFileTool;

#[async_trait]
impl AgentTool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }
    fn label(&self) -> &str {
        "Write File"
    }
    fn description(&self) -> &str {
        "Write content to a file at the given path. Creates the file if it does not exist, overwrites it if it does."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to write" },
                "content": { "type": "string", "description": "Content to write to the file" }
            },
            "required": ["path", "content"],
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
        let path = params["path"].as_str().unwrap_or_default();
        let content = params["content"].as_str().unwrap_or_default();

        match tokio::fs::write(path, content).await {
            Ok(()) => Ok(AgentToolResult {
                content: vec![ContentBlock::Text(TextContent {
                    text: format!("Successfully wrote to '{path}'"),
                    text_signature: None,
                })],
                details: None,
            }),
            Err(e) => Ok(AgentToolResult {
                content: vec![ContentBlock::Text(TextContent {
                    text: format!("Error writing file '{path}': {e}"),
                    text_signature: None,
                })],
                details: Some(json!({ "error": e.to_string() })),
            }),
        }
    }
}

/// Edit a file by replacing an exact string.
struct EditFileTool;

#[async_trait]
impl AgentTool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }
    fn label(&self) -> &str {
        "Edit File"
    }
    fn description(&self) -> &str {
        "Edit a file by replacing an exact string with another string. The old_string must match exactly \
         (including whitespace and newlines). Only the first occurrence is replaced. Returns success or error message."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to edit" },
                "old_string": { "type": "string", "description": "Exact text to replace. Must match exactly including whitespace and newlines." },
                "new_string": { "type": "string", "description": "Text to replace with" }
            },
            "required": ["path", "old_string", "new_string"],
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
        let path = params["path"].as_str().unwrap_or_default();
        let old_string = params["old_string"].as_str().unwrap_or_default();
        let new_string = params["new_string"].as_str().unwrap_or_default();

        if old_string.is_empty() {
            return Ok(AgentToolResult {
                content: vec![ContentBlock::Text(TextContent {
                    text: "Error: old_string cannot be empty".into(),
                    text_signature: None,
                })],
                details: Some(json!({ "error": "old_string is empty" })),
            });
        }

        let content = match tokio::fs::read_to_string(path).await {
            Ok(c) => c,
            Err(e) => {
                return Ok(AgentToolResult {
                    content: vec![ContentBlock::Text(TextContent {
                        text: format!("Error reading file '{path}': {e}"),
                        text_signature: None,
                    })],
                    details: Some(json!({ "error": e.to_string() })),
                });
            }
        };

        if !content.contains(old_string) {
            return Ok(AgentToolResult {
                content: vec![ContentBlock::Text(TextContent {
                    text: format!("Error: old_string not found in '{path}'"),
                    text_signature: None,
                })],
                details: Some(json!({ "error": "old_string not found" })),
            });
        }

        let new_content = content.replacen(old_string, new_string, 1);

        match tokio::fs::write(path, new_content).await {
            Ok(()) => Ok(AgentToolResult {
                content: vec![ContentBlock::Text(TextContent {
                    text: format!("Successfully edited '{path}'"),
                    text_signature: None,
                })],
                details: None,
            }),
            Err(e) => Ok(AgentToolResult {
                content: vec![ContentBlock::Text(TextContent {
                    text: format!("Error writing file '{path}': {e}"),
                    text_signature: None,
                })],
                details: Some(json!({ "error": e.to_string() })),
            }),
        }
    }
}

/// Find files using RTK for token-optimized output.
struct FindFilesTool;

#[async_trait]
impl AgentTool for FindFilesTool {
    fn name(&self) -> &str {
        "find_files"
    }
    fn label(&self) -> &str {
        "Find Files"
    }
    fn description(&self) -> &str {
        "Find files matching a pattern using RTK. Returns a compact, token-optimized list of matching file paths grouped by directory."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern or find arguments (e.g. '*.rs', '-name *.rs -type f')"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in",
                    "default": "."
                }
            },
            "required": ["pattern"],
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
        let pattern = params["pattern"].as_str().unwrap_or_default();
        let path = params.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let output = tokio::process::Command::new("rtk")
            .arg("find")
            .arg(pattern)
            .arg(path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| AgentError::Stream(format!("Failed to run rtk find: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let (text, is_error) = if output.status.success() {
            (stdout.into_owned(), false)
        } else {
            let msg = if stderr.is_empty() {
                stdout.into_owned()
            } else {
                format!("{stderr}\n{stdout}")
            };
            (msg, true)
        };

        Ok(AgentToolResult {
            content: vec![ContentBlock::Text(TextContent {
                text,
                text_signature: None,
            })],
            details: Some(json!({ "is_error": is_error })),
        })
    }
}

/// Grep files using RTK for token-optimized output.
struct GrepTool;

#[async_trait]
impl AgentTool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }
    fn label(&self) -> &str {
        "Grep"
    }
    fn description(&self) -> &str {
        "Search for a pattern in file contents using RTK. Returns grouped, truncated, token-optimized search results with file paths and line numbers."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Text pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in",
                    "default": "."
                },
                "file_type": {
                    "type": "string",
                    "description": "Filter by file type/extension (e.g. 'rs', 'py', 'ts')",
                    "default": ""
                }
            },
            "required": ["pattern"],
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
        let pattern = params["pattern"].as_str().unwrap_or_default();
        let path = params.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let file_type = params.get("file_type").and_then(|v| v.as_str()).unwrap_or("");

        let mut cmd = tokio::process::Command::new("rtk");
        cmd.arg("grep").arg(pattern).arg(path);
        if !file_type.is_empty() {
            cmd.arg("-t").arg(file_type);
        }

        let output = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| AgentError::Stream(format!("Failed to run rtk grep: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let (text, is_error) = if output.status.success() {
            (stdout.into_owned(), false)
        } else {
            let msg = if stderr.is_empty() {
                stdout.into_owned()
            } else {
                format!("{stderr}\n{stdout}")
            };
            (msg, true)
        };

        Ok(AgentToolResult {
            content: vec![ContentBlock::Text(TextContent {
                text,
                text_signature: None,
            })],
            details: Some(json!({ "is_error": is_error })),
        })
    }
}

// ============================================================
// 3. Helpers
// ============================================================

fn model() -> Model {
    let model_id = std::env::var("KIMI_MODEL").unwrap_or_else(|_| "kimi-k2-6".into());
    Model {
        id: model_id,
        name: "Kimi K2.6".into(),
        api: "openai-chat".into(),
        provider: "moonshot".into(),
        base_url: "https://api.moonshot.cn".into(),
        reasoning: false,
        input: vec!["text".into()],
        cost: ModelCost::default(),
        context_window: 256_000,
        max_tokens: 8192,
    }
}

fn print_assistant_reply(state: &oh_my_agentloop::AgentState) {
    if let Some(AgentMessage::Assistant(a)) = state
        .messages
        .iter()
        .rev()
        .find(|m| matches!(m, AgentMessage::Assistant(_)))
    {
        let mut has_text = false;
        for block in &a.content {
            if let ContentBlock::Text(t) = block {
                if !t.text.trim().is_empty() {
                    println!("🤖 {}", t.text.trim());
                    has_text = true;
                }
            }
        }
        if !has_text && a.stop_reason == StopReason::ToolUse {
            println!("🤖 (using tools...)");
        }
    }
}

// ============================================================
// 4. Main
// ============================================================

#[tokio::main]
async fn main() {
    check_rtk();

    let api_key = std::env::var("KIMI_API_KEY").unwrap_or_else(|_| {
        eprintln!("❌ KIMI_API_KEY environment variable is not set.");
        eprintln!("Please set it to your Moonshot AI API key:");
        eprintln!("  export KIMI_API_KEY='your-api-key'");
        eprintln!();
        eprintln!("Optionally set KIMI_MODEL to override the default model (default: kimi-k2-6).");
        std::process::exit(1);
    });

    let tools: Vec<Arc<dyn AgentTool>> = vec![
        Arc::new(ReadFileTool),
        Arc::new(WriteFileTool),
        Arc::new(EditFileTool),
        Arc::new(FindFilesTool),
        Arc::new(GrepTool),
    ];

    let provider = Arc::new(KimiProvider::new(api_key));
    let options = AgentOptions::builder(provider)
        .initial_state(InitialAgentState {
            system_prompt: Some(
                "You are a helpful coding assistant with access to file system tools. \
                 You can read files, write files, edit files, find files, and search file contents. \
                 Use these tools to help users understand and modify their codebase. \
                 Always think step by step, explain what you are doing, and use tools when needed. \
                 When editing files, make sure the old_string matches exactly."
                    .into(),
            ),
            model: Some(model()),
            thinking_level: Some(ThinkingLevel::Off),
            tools: Some(tools),
            messages: None,
        })
        .build();

    let agent = Agent::new(options);

    // Subscribe to events for real-time feedback
    let _sub = agent.subscribe(|event, _cancel| async move {
        match event {
            AgentEvent::ToolExecutionStart { tool_name, .. } => {
                println!("🔧  Using tool: {tool_name}...");
            }
            AgentEvent::ToolExecutionEnd {
                tool_name,
                is_error,
                ..
            } => {
                if is_error {
                    println!("❌  Tool {tool_name} failed.");
                } else {
                    println!("✅  Tool {tool_name} completed.");
                }
            }
            AgentEvent::RunCompleted { .. } => {
                println!("🏁  Done.\n");
            }
            AgentEvent::RunFailed { error_message, .. } => {
                println!("💥  Run failed: {error_message}\n");
            }
            _ => {}
        }
    });

    println!("╔═════════════════════════════════════════════════════════════════╗");
    println!("║             Coding Agent Example (Kimi K2.6 + RTK)              ║");
    println!("╠═════════════════════════════════════════════════════════════════╣");
    println!("║  Tools: read_file | write_file | edit_file | find_files | grep  ║");
    println!("║  Type your request, or 'exit' to quit.                          ║");
    println!("╚═════════════════════════════════════════════════════════════════╝\n");

    let stdin = std::io::stdin();
    let mut buffer = String::new();

    loop {
        print!("> ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        buffer.clear();
        if stdin.read_line(&mut buffer).is_err() {
            break;
        }
        let prompt = buffer.trim();
        if prompt.eq_ignore_ascii_case("exit") || prompt.eq_ignore_ascii_case("quit") {
            println!("👋  Goodbye!");
            break;
        }
        if prompt.is_empty() {
            continue;
        }

        if let Err(e) = agent.prompt_text(prompt, None).await {
            eprintln!("Error: {e}");
            continue;
        }

        print_assistant_reply(&agent.state());
    }
}
