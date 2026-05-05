use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use async_openai::{config::OpenAIConfig, Client};
use futures::StreamExt;
use oh_my_agentloop::{
    Agent, AgentError, AgentEvent, AgentOptions, AgentTool, AgentToolResult,
    AssistantMessage, ContentBlock, InitialAgentState, LlmContext, LlmEventStream, Message, Model,
    ModelCost, StopReason, StreamEvent, StreamProvider, StreamRequest, TextContent, ThinkingContent,
    ThinkingLevel, ToolCallContent, Usage, UserContent, UserContentBlock,
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
// 1. Kimi K2.6 Stream Provider (async-openai BYOT + streaming + thinking)
// ============================================================

struct KimiProvider {
    client: Client<OpenAIConfig>,
}

impl KimiProvider {
    fn new(api_key: String) -> Self {
        let config = OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base("https://api.moonshot.cn/v1");
        Self {
            client: Client::with_config(config),
        }
    }
}

fn convert_messages(messages: Vec<Message>) -> Vec<Value> {
    messages
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
                let mut reasoning_parts = Vec::new();
                let mut tool_calls_json = Vec::new();
                for block in &a.content {
                    match block {
                        ContentBlock::Text(t) => text_parts.push(t.text.clone()),
                        ContentBlock::Thinking(t) => reasoning_parts.push(t.thinking.clone()),
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
                let reasoning = reasoning_parts.join("");
                let mut msg = json!({"role": "assistant"});
                if !text.is_empty() {
                    msg["content"] = Value::String(text);
                }
                if !reasoning.is_empty() {
                    msg["reasoning_content"] = Value::String(reasoning);
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
        .collect()
}

#[async_trait]
impl StreamProvider for KimiProvider {
    async fn stream(
        &self,
        model: Model,
        ctx: LlmContext,
        _req: StreamRequest,
    ) -> Result<LlmEventStream, AgentError> {
        let messages = convert_messages(ctx.messages);
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
            "stream": true,
            "thinking": { "type": "enabled" },
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }

        let mut openai_stream = self
            .client
            .chat()
            .create_stream_byot(body)
            .await
            .map_err(|e| AgentError::Stream(format!("API request failed: {e}")))?;

        let model_clone = model.clone();
        let (tx, rx) =
            futures::channel::mpsc::unbounded::<Result<StreamEvent, AgentError>>();

        tokio::spawn(async move {
            let mut text_buf = String::new();
            let mut reasoning_buf = String::new();
            let mut tool_call_parts: Vec<(String, String, String)> = Vec::new();
            let mut response_id: Option<String> = None;
            let mut finish_reason: Option<String> = None;
            let mut has_sent_start = false;
            let mut has_sent_thinking_start = false;
            let mut has_sent_text_start = false;

            let build_partial =
                |text: &str,
                 reasoning: &str,
                 tool_calls: &[(String, String, String)],
                 response_id: Option<String>|
                 -> AssistantMessage {
                    let mut content = Vec::new();
                    if !reasoning.is_empty() {
                        content.push(ContentBlock::Thinking(ThinkingContent {
                            thinking: reasoning.into(),
                            thinking_signature: None,
                            redacted: Some(false),
                        }));
                    }
                    if !text.is_empty() {
                        content.push(ContentBlock::Text(TextContent {
                            text: text.into(),
                            text_signature: None,
                        }));
                    }
                    for (id, name, args_str) in tool_calls {
                        if !id.is_empty() && !name.is_empty() {
                            let arguments =
                                serde_json::from_str(args_str).unwrap_or_else(|_| json!({}));
                            content.push(ContentBlock::ToolCall(ToolCallContent {
                                id: id.clone(),
                                name: name.clone(),
                                arguments,
                            }));
                        }
                    }
                    AssistantMessage {
                        content,
                        model: model_clone.id.clone(),
                        provider: model_clone.provider.clone(),
                        api: model_clone.api.clone(),
                        response_id,
                        stop_reason: StopReason::Stop,
                        error_message: None,
                        usage: Usage::default(),
                        timestamp: oh_my_agentloop::now_millis(),
                    }
                };

            while let Some(chunk_result) = openai_stream.next().await {
                let chunk_result: Result<Value, _> = chunk_result;
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.unbounded_send(Err(AgentError::Stream(format!(
                            "API stream error: {e}"
                        ))));
                        return;
                    }
                };

                if let Some(id) = chunk.get("id").and_then(|i| i.as_str()) {
                    response_id = Some(id.to_string());
                }

                let choice = match chunk
                    .get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|a| a.first())
                {
                    Some(c) => c,
                    None => continue,
                };

                if let Some(reason) = choice.get("finish_reason").and_then(|f| f.as_str()) {
                    finish_reason = Some(reason.to_string());
                }

                let delta = match choice.get("delta") {
                    Some(d) => d,
                    None => continue,
                };

                if let Some(rc) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
                    if !rc.is_empty() {
                        if !has_sent_start {
                            let partial = build_partial(
                                &text_buf,
                                &reasoning_buf,
                                &tool_call_parts,
                                response_id.clone(),
                            );
                            let _ = tx.unbounded_send(Ok(StreamEvent::Start { partial }));
                            has_sent_start = true;
                        }
                        if !has_sent_thinking_start {
                            let partial = build_partial(
                                &text_buf,
                                &reasoning_buf,
                                &tool_call_parts,
                                response_id.clone(),
                            );
                            let _ = tx.unbounded_send(Ok(StreamEvent::ThinkingStart {
                                content_index: 0,
                                partial,
                            }));
                            has_sent_thinking_start = true;
                        }
                        reasoning_buf.push_str(rc);
                        let partial = build_partial(
                            &text_buf,
                            &reasoning_buf,
                            &tool_call_parts,
                            response_id.clone(),
                        );
                        let _ = tx.unbounded_send(Ok(StreamEvent::ThinkingDelta {
                            content_index: 0,
                            delta: rc.to_string(),
                            partial,
                        }));
                    }
                }

                if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                    if !content.is_empty() {
                        if !has_sent_start {
                            let partial = build_partial(
                                &text_buf,
                                &reasoning_buf,
                                &tool_call_parts,
                                response_id.clone(),
                            );
                            let _ = tx.unbounded_send(Ok(StreamEvent::Start { partial }));
                            has_sent_start = true;
                        }
                        if !has_sent_text_start {
                            let partial = build_partial(
                                &text_buf,
                                &reasoning_buf,
                                &tool_call_parts,
                                response_id.clone(),
                            );
                            let idx = if reasoning_buf.is_empty() { 0 } else { 1 };
                            let _ = tx.unbounded_send(Ok(StreamEvent::TextStart {
                                content_index: idx,
                                partial,
                            }));
                            has_sent_text_start = true;
                        }
                        text_buf.push_str(content);
                        let partial = build_partial(
                            &text_buf,
                            &reasoning_buf,
                            &tool_call_parts,
                            response_id.clone(),
                        );
                        let idx = if reasoning_buf.is_empty() { 0 } else { 1 };
                        let _ = tx.unbounded_send(Ok(StreamEvent::TextDelta {
                            content_index: idx,
                            delta: content.to_string(),
                            partial,
                        }));
                    }
                }

                if let Some(tc_deltas) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                    for tc_delta in tc_deltas {
                        let index =
                            tc_delta.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                        while tool_call_parts.len() <= index {
                            tool_call_parts.push((String::new(), String::new(), String::new()));
                        }
                        if let Some(id) = tc_delta.get("id").and_then(|i| i.as_str()) {
                            tool_call_parts[index].0 = id.to_string();
                        }
                        if let Some(func) = tc_delta.get("function") {
                            if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                                tool_call_parts[index].1 = name.to_string();
                            }
                            if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
                                tool_call_parts[index].2.push_str(args);
                            }
                        }
                    }
                }
            }

            let stop_reason = finish_reason
                .map(|r| match r.as_str() {
                    "stop" => StopReason::Stop,
                    "length" => StopReason::Length,
                    "tool_calls" => StopReason::ToolUse,
                    _ => StopReason::Stop,
                })
                .unwrap_or(StopReason::Stop);

            if has_sent_thinking_start {
                let partial = build_partial(
                    &text_buf,
                    &reasoning_buf,
                    &tool_call_parts,
                    response_id.clone(),
                );
                let _ = tx.unbounded_send(Ok(StreamEvent::ThinkingEnd {
                    content_index: 0,
                    content: reasoning_buf.clone(),
                    partial,
                }));
            }
            if has_sent_text_start {
                let partial = build_partial(
                    &text_buf,
                    &reasoning_buf,
                    &tool_call_parts,
                    response_id.clone(),
                );
                let _ = tx.unbounded_send(Ok(StreamEvent::TextEnd {
                    content_index: if reasoning_buf.is_empty() { 0 } else { 1 },
                    content: text_buf.clone(),
                    partial,
                }));
            }

            let mut content = Vec::new();
            if !reasoning_buf.is_empty() {
                content.push(ContentBlock::Thinking(ThinkingContent {
                    thinking: reasoning_buf,
                    thinking_signature: None,
                    redacted: Some(false),
                }));
            }
            if !text_buf.is_empty() {
                content.push(ContentBlock::Text(TextContent {
                    text: text_buf,
                    text_signature: None,
                }));
            }
            for (id, name, args_str) in tool_call_parts {
                if !id.is_empty() && !name.is_empty() {
                    let arguments =
                        serde_json::from_str(&args_str).unwrap_or_else(|_| json!({}));
                    content.push(ContentBlock::ToolCall(ToolCallContent {
                        id,
                        name,
                        arguments,
                    }));
                }
            }

            let message = AssistantMessage {
                content,
                model: model_clone.id,
                provider: model_clone.provider,
                api: model_clone.api,
                response_id,
                stop_reason,
                error_message: None,
                usage: Usage::default(),
                timestamp: oh_my_agentloop::now_millis(),
            };

            let _ = tx.unbounded_send(Ok(StreamEvent::Done { message }));
        });

        Ok(Box::pin(rx) as LlmEventStream)
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
    let has_thinking = Arc::new(AtomicBool::new(false));

    let _sub = agent.subscribe({
        let has_thinking = Arc::clone(&has_thinking);
        move |event, _cancel| {
            let has_thinking = Arc::clone(&has_thinking);
            async move {
                match event {
                    AgentEvent::MessageUpdate { stream_event, .. } => match stream_event {
                        StreamEvent::Start { .. } => print!("🤖 "),
                        StreamEvent::ThinkingStart { .. } => {
                            has_thinking.store(true, Ordering::Relaxed);
                        }
                        StreamEvent::ThinkingDelta { delta, .. } => {
                            eprint!("\x1b[90m{delta}\x1b[0m");
                            let _ = std::io::Write::flush(&mut std::io::stderr());
                        }
                        StreamEvent::TextStart { .. } => {
                            if has_thinking.swap(false, Ordering::Relaxed) {
                                println!();
                            }
                        }
                        StreamEvent::TextDelta { delta, .. } => {
                            print!("{delta}");
                            let _ = std::io::Write::flush(&mut std::io::stdout());
                        }
                        StreamEvent::Done { .. } => println!(),
                        _ => {}
                    },
            AgentEvent::ToolExecutionStart { tool_name, .. } => {
                println!("\n🔧  Using tool: {tool_name}...");
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
                println!();
            }
            AgentEvent::RunFailed { error_message, .. } => {
                println!("\n💥  Run failed: {error_message}");
            }
            _ => {}
        }
    }
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
    }
}
