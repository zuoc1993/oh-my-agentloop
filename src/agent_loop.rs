//! Low-level agent loop that works with AgentMessage throughout.
//! Transforms to Message[] only at the LLM call boundary.
//!
//! Direct port of pi-agent-core/src/agent-loop.ts

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use jsonschema::validator_for;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::types::*;

// ============================================================
// Stream-returning API (mirrors agentLoop / agentLoopContinue in TypeScript)
// ============================================================

/// Start an agent loop and return a stream of events via a channel receiver.
///
/// Events are pushed without backpressure (unbounded), matching the TypeScript
/// `EventStream` semantics where `stream.push()` is synchronous.
pub fn agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    cancel: CancellationToken,
    stream_provider: Arc<dyn StreamProvider>,
) -> mpsc::UnboundedReceiver<AgentEvent> {
    let (tx, rx) = mpsc::unbounded_channel();

    tokio::spawn(async move {
        let emitter = EventEmitter::new(move |event| {
            let tx = tx.clone();
            async move {
                let _ = tx.send(event);
            }
        });

        let _ = run_agent_loop(prompts, context, config, &emitter, cancel, &*stream_provider).await;
    });

    rx
}

/// Continue an agent loop and return a stream of events via a channel receiver.
pub fn agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel: CancellationToken,
    stream_provider: Arc<dyn StreamProvider>,
) -> Result<mpsc::UnboundedReceiver<AgentEvent>, AgentError> {
    if context.messages.is_empty() {
        return Err(AgentError::NoMessages);
    }
    if context.messages.last().map(|m| m.role()) == Some("assistant") {
        return Err(AgentError::ContinueFromAssistant);
    }

    let (tx, rx) = mpsc::unbounded_channel();

    tokio::spawn(async move {
        let emitter = EventEmitter::new(move |event| {
            let tx = tx.clone();
            async move {
                let _ = tx.send(event);
            }
        });

        let _ = run_agent_loop_continue(context, config, &emitter, cancel, &*stream_provider).await;
    });

    Ok(rx)
}

// ============================================================
// Callback-based API (used by Agent)
// ============================================================

enum LoopExit {
    Completed,
    Aborted,
    Failed(AgentError),
}

async fn append_failure_message(
    model: &Model,
    current_context: &mut AgentContext,
    new_messages: &mut Arc<Vec<AgentMessage>>,
    emitter: &EventEmitter,
    error: &AgentError,
) {
    let failure_message =
        AgentMessage::Assistant(create_error_assistant_message(model, &error.to_string()));

    Arc::make_mut(&mut current_context.messages).push(failure_message.clone());
    Arc::make_mut(new_messages).push(failure_message.clone());

    emitter
        .emit(AgentEvent::MessageStart {
            message: failure_message.clone(),
        })
        .await;
    emitter
        .emit(AgentEvent::MessageEnd {
            message: failure_message,
        })
        .await;
}

async fn emit_terminal_event(emitter: &EventEmitter, outcome: &RunOutcome) {
    match outcome {
        RunOutcome::Completed { new_messages } => {
            emitter
                .emit(AgentEvent::RunCompleted {
                    messages: new_messages.to_vec(),
                })
                .await;
        }
        RunOutcome::Failed {
            new_messages,
            error,
        } => {
            emitter
                .emit(AgentEvent::RunFailed {
                    messages: new_messages.to_vec(),
                    error_message: terminal_error_message(new_messages, StopReason::Error)
                        .unwrap_or_else(|| error.to_string()),
                })
                .await;
        }
        RunOutcome::Aborted { new_messages } => {
            emitter
                .emit(AgentEvent::RunAborted {
                    messages: new_messages.to_vec(),
                })
                .await;
        }
    }
}

fn terminal_error_message(messages: &[AgentMessage], stop_reason: StopReason) -> Option<String> {
    messages.iter().rev().find_map(|message| match message {
        AgentMessage::Assistant(assistant) if assistant.stop_reason == stop_reason => {
            assistant.error_message.clone()
        }
        _ => None,
    })
}

fn failure_from_terminal_message(message: &AssistantMessage) -> AgentError {
    AgentError::Stream(
        message
            .error_message
            .clone()
            .unwrap_or_else(|| "Unknown stream error".to_string()),
    )
}

/// Start an agent loop with new prompt messages.
/// The prompts are added to the context and events are emitted for them.
///
/// Returns the new messages produced during this run.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "info",
        name = "agent.run",
        skip(context, config, emitter, cancel, stream_provider),
        fields(model = %config.model.id, prompts = prompts.len())
    )
)]
pub async fn run_agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    emitter: &EventEmitter,
    cancel: CancellationToken,
    stream_provider: &dyn StreamProvider,
) -> Result<RunOutcome, AgentError> {
    let new_messages: Arc<Vec<AgentMessage>> = Arc::new(prompts.clone());
    let mut current_context = AgentContext {
        system_prompt: context.system_prompt,
        messages: {
            let mut msgs = Arc::unwrap_or_clone(context.messages);
            msgs.extend(prompts.iter().cloned());
            Arc::new(msgs)
        },
        tools: context.tools,
    };

    emitter.emit(AgentEvent::AgentStart).await;
    emitter.emit(AgentEvent::TurnStart).await;

    for prompt in &prompts {
        emitter
            .emit(AgentEvent::MessageStart {
                message: prompt.clone(),
            })
            .await;
        emitter
            .emit(AgentEvent::MessageEnd {
                message: prompt.clone(),
            })
            .await;
    }

    let mut new_messages = new_messages;
    match run_loop(
        &mut current_context,
        &mut new_messages,
        &config,
        &cancel,
        emitter,
        stream_provider,
    )
    .await
    {
        Ok(LoopExit::Completed) => {
            let outcome = RunOutcome::Completed { new_messages };
            emit_terminal_event(emitter, &outcome).await;
            Ok(outcome)
        }
        Ok(LoopExit::Aborted) => {
            let outcome = RunOutcome::Aborted { new_messages };
            emit_terminal_event(emitter, &outcome).await;
            Ok(outcome)
        }
        Ok(LoopExit::Failed(error)) => {
            let outcome = RunOutcome::Failed {
                new_messages,
                error,
            };
            emit_terminal_event(emitter, &outcome).await;
            Ok(outcome)
        }
        Err(error) => {
            append_failure_message(
                &config.model,
                &mut current_context,
                &mut new_messages,
                emitter,
                &error,
            )
            .await;

            let failure_message = new_messages.last().cloned().ok_or_else(|| {
                AgentError::Internal(
                    "run_agent_loop: failure assistant message was not appended".to_string(),
                )
            })?;
            emitter
                .emit(AgentEvent::TurnEnd {
                    message: failure_message,
                    tool_results: vec![],
                })
                .await;
            let outcome = RunOutcome::Failed {
                new_messages,
                error,
            };
            emit_terminal_event(emitter, &outcome).await;
            Ok(outcome)
        }
    }
}

/// Continue an agent loop from the current context without adding a new message.
/// Used for retries — context already has user message or tool results.
///
/// The last message in context must convert to a `user` or `toolResult` message
/// via `convert_to_llm`. If it doesn't, the LLM provider will reject the request.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "info",
        name = "agent.run.continue",
        skip(context, config, emitter, cancel, stream_provider),
        fields(model = %config.model.id, ctx_messages = context.messages.len())
    )
)]
pub async fn run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emitter: &EventEmitter,
    cancel: CancellationToken,
    stream_provider: &dyn StreamProvider,
) -> Result<RunOutcome, AgentError> {
    if context.messages.is_empty() {
        return Err(AgentError::NoMessages);
    }

    if context.messages.last().map(|m| m.role()) == Some("assistant") {
        return Err(AgentError::ContinueFromAssistant);
    }

    let mut current_context = AgentContext {
        system_prompt: context.system_prompt,
        messages: context.messages,
        tools: context.tools,
    };
    let mut new_messages: Arc<Vec<AgentMessage>> = Arc::new(Vec::new());

    emitter.emit(AgentEvent::AgentStart).await;
    emitter.emit(AgentEvent::TurnStart).await;

    match run_loop(
        &mut current_context,
        &mut new_messages,
        &config,
        &cancel,
        emitter,
        stream_provider,
    )
    .await
    {
        Ok(LoopExit::Completed) => {
            let outcome = RunOutcome::Completed { new_messages };
            emit_terminal_event(emitter, &outcome).await;
            Ok(outcome)
        }
        Ok(LoopExit::Aborted) => {
            let outcome = RunOutcome::Aborted { new_messages };
            emit_terminal_event(emitter, &outcome).await;
            Ok(outcome)
        }
        Ok(LoopExit::Failed(error)) => {
            let outcome = RunOutcome::Failed {
                new_messages,
                error,
            };
            emit_terminal_event(emitter, &outcome).await;
            Ok(outcome)
        }
        Err(error) => {
            append_failure_message(
                &config.model,
                &mut current_context,
                &mut new_messages,
                emitter,
                &error,
            )
            .await;

            let failure_message = new_messages.last().cloned().ok_or_else(|| {
                AgentError::Internal(
                    "run_agent_loop_continue: failure assistant message was not appended"
                        .to_string(),
                )
            })?;
            emitter
                .emit(AgentEvent::TurnEnd {
                    message: failure_message,
                    tool_results: vec![],
                })
                .await;
            let outcome = RunOutcome::Failed {
                new_messages,
                error,
            };
            emit_terminal_event(emitter, &outcome).await;
            Ok(outcome)
        }
    }
}

// ============================================================
// Main Loop Logic (mirrors agent-loop.ts runLoop)
// ============================================================

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "debug",
        name = "agent.run.loop",
        skip_all,
        fields(model = %config.model.id)
    )
)]
async fn run_loop(
    current_context: &mut AgentContext,
    new_messages: &mut Arc<Vec<AgentMessage>>,
    config: &AgentLoopConfig,
    cancel: &CancellationToken,
    emitter: &EventEmitter,
    stream_provider: &dyn StreamProvider,
) -> Result<LoopExit, AgentError> {
    let mut first_turn = true;

    let mut pending_messages: Vec<AgentMessage> =
        if let Some(ref get_steering) = config.get_steering_messages {
            get_steering().await
        } else {
            Vec::new()
        };

    // Outer loop: continues when queued follow-up messages arrive
    loop {
        let mut has_more_tool_calls = true;

        // Inner loop: process tool calls and steering messages
        while has_more_tool_calls || !pending_messages.is_empty() {
            if cancel.is_cancelled() {
                return Ok(LoopExit::Aborted);
            }

            if !first_turn {
                emitter.emit(AgentEvent::TurnStart).await;
            } else {
                first_turn = false;
            }

            // Process pending messages (inject before next assistant response)
            if !pending_messages.is_empty() {
                for message in pending_messages.drain(..) {
                    emitter
                        .emit(AgentEvent::MessageStart {
                            message: message.clone(),
                        })
                        .await;
                    emitter
                        .emit(AgentEvent::MessageEnd {
                            message: message.clone(),
                        })
                        .await;
                    Arc::make_mut(&mut current_context.messages).push(message.clone());
                    Arc::make_mut(new_messages).push(message);
                }
            }

            // Stream assistant response
            let message =
                stream_assistant_response(current_context, config, cancel, emitter, stream_provider)
                    .await?;
            Arc::make_mut(new_messages).push(AgentMessage::Assistant(message.clone()));

            if message.stop_reason == StopReason::Error
                || message.stop_reason == StopReason::Aborted
            {
                let loop_exit = if message.stop_reason == StopReason::Error {
                    LoopExit::Failed(failure_from_terminal_message(&message))
                } else {
                    LoopExit::Aborted
                };
                emitter
                    .emit(AgentEvent::TurnEnd {
                        message: AgentMessage::Assistant(message),
                        tool_results: vec![],
                    })
                    .await;
                return Ok(loop_exit);
            }

            // Check for tool calls
            let tool_calls: Vec<ToolCallContent> = message
                .content
                .iter()
                .filter_map(|c| match c {
                    ContentBlock::ToolCall(tc) => Some(tc.clone()),
                    _ => None,
                })
                .collect();
            has_more_tool_calls = !tool_calls.is_empty();

            let mut tool_results: Vec<ToolResultMessage> = Vec::new();
            if has_more_tool_calls {
                tool_results = execute_tool_calls(
                    current_context,
                    &message,
                    &tool_calls,
                    config,
                    cancel,
                    emitter,
                )
                .await?;

                for result in &tool_results {
                    Arc::make_mut(&mut current_context.messages)
                        .push(AgentMessage::ToolResult(result.clone()));
                    Arc::make_mut(new_messages).push(AgentMessage::ToolResult(result.clone()));
                }

                if cancel.is_cancelled() {
                    emitter
                        .emit(AgentEvent::TurnEnd {
                            message: AgentMessage::Assistant(message.clone()),
                            tool_results,
                        })
                        .await;
                    return Ok(LoopExit::Aborted);
                }
            }

            emitter
                .emit(AgentEvent::TurnEnd {
                    message: AgentMessage::Assistant(message),
                    tool_results,
                })
                .await;

            // Check steering
            pending_messages = if let Some(ref get_steering) = config.get_steering_messages {
                get_steering().await
            } else {
                Vec::new()
            };
        }

        // Agent would stop here. Check for follow-up messages.
        let follow_up_messages = if let Some(ref get_follow_up) = config.get_follow_up_messages {
            get_follow_up().await
        } else {
            Vec::new()
        };

        if !follow_up_messages.is_empty() {
            pending_messages = follow_up_messages;
            continue;
        }

        break;
    }

    Ok(LoopExit::Completed)
}

// ============================================================
// Stream Assistant Response (mirrors agent-loop.ts streamAssistantResponse)
// ============================================================

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "debug",
        name = "agent.llm.stream",
        skip_all,
        fields(model = %config.model.id)
    )
)]
async fn stream_assistant_response(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    cancel: &CancellationToken,
    emitter: &EventEmitter,
    stream_provider: &dyn StreamProvider,
) -> Result<AssistantMessage, AgentError> {
    // Apply context transform if configured (AgentMessage[] → AgentMessage[])
    let messages = if let Some(ref transform) = config.transform_context {
        Arc::new(transform(Arc::unwrap_or_clone(context.messages.clone()), cancel.clone()).await)
    } else {
        context.messages.clone()
    };

    // Convert to LLM-compatible messages (AgentMessage[] → Message[])
    let llm_messages = (config.convert_to_llm)(messages.to_vec()).await;

    // Build tool definitions
    let tool_definitions: Vec<ToolDefinition> = context
        .tools
        .iter()
        .map(|t| t.as_tool_definition())
        .collect();

    let llm_context = LlmContext {
        system_prompt: context.system_prompt.clone(),
        messages: llm_messages,
        tools: tool_definitions,
    };

    // Resolve API key (important for expiring tokens)
    let resolved_api_key = if let Some(ref get_key) = config.get_api_key {
        let key = get_key(config.model.provider.clone()).await;
        key.or_else(|| config.api_key.clone())
    } else {
        config.api_key.clone()
    };

    let stream_options = StreamOptions {
        api_key: resolved_api_key,
        reasoning: config.reasoning.clone(),
        session_id: config.session_id.clone(),
        transport: config.transport.clone(),
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        thinking_budgets: config.thinking_budgets.clone(),
        max_retry_delay_ms: config.max_retry_delay_ms,
        on_payload: config.on_payload.clone(),
    };

    let stream_request = StreamRequest {
        options: stream_options,
        cancel: cancel.clone(),
    };

    // Call stream provider
    let mut stream = match stream_provider
        .stream(config.model.clone(), llm_context, stream_request)
        .await
    {
        Ok(stream) => stream,
        Err(AgentError::Aborted) => {
            let abort_msg = create_abort_message(&config.model, None);
            finalize_stream_message(context, &abort_msg, false, emitter).await;
            return Ok(abort_msg);
        }
        Err(error) => return Err(error),
    };

    let mut partial_message: Option<AssistantMessage> = None;
    let mut added_partial = false;

    loop {
        let event_result = tokio::select! {
            biased;
            event_result = stream.next() => event_result,
            _ = cancel.cancelled() => {
                let abort_msg = create_abort_message(&config.model, partial_message.as_ref());
                finalize_stream_message(context, &abort_msg, added_partial, emitter).await;
                return Ok(abort_msg);
            }
        };

        let Some(event_result) = event_result else {
            break;
        };

        let event = match event_result {
            Ok(event) => event,
            Err(AgentError::Aborted) => {
                let abort_msg = create_abort_message(&config.model, partial_message.as_ref());
                finalize_stream_message(context, &abort_msg, added_partial, emitter).await;
                return Ok(abort_msg);
            }
            Err(error) => return Err(error),
        };

        match event {
            StreamEvent::Start { partial } => {
                if cancel.is_cancelled() {
                    let abort_msg = create_abort_message(&config.model, Some(&partial));
                    finalize_stream_message(context, &abort_msg, added_partial, emitter).await;
                    return Ok(abort_msg);
                }

                partial_message = Some(partial.clone());
                Arc::make_mut(&mut context.messages)
                    .push(AgentMessage::Assistant(partial.clone()));
                added_partial = true;
                emitter
                    .emit(AgentEvent::MessageStart {
                        message: AgentMessage::Assistant(partial),
                    })
                    .await;
            }

            StreamEvent::Done { message } => {
                let message = normalize_terminal_assistant_message(message);
                finalize_stream_message(context, &message, added_partial, emitter).await;
                return Ok(message);
            }

            StreamEvent::Error { mut message } => {
                if message.stop_reason != StopReason::Aborted {
                    message.stop_reason = StopReason::Error;
                }
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    model = %config.model.id,
                    stop_reason = ?message.stop_reason,
                    "provider stream reported an error"
                );
                let message = normalize_terminal_assistant_message(message);
                finalize_stream_message(context, &message, added_partial, emitter).await;
                return Ok(message);
            }

            other => {
                let abort_partial =
                    stream_event_partial_for_update(&other).or(partial_message.as_ref());
                if cancel.is_cancelled() {
                    let abort_msg = create_abort_message(&config.model, abort_partial);
                    finalize_stream_message(context, &abort_msg, added_partial, emitter).await;
                    return Ok(abort_msg);
                }

                if !added_partial {
                    continue;
                }
                if let Some(partial) = stream_event_partial_for_update(&other) {
                    partial_message = Some(partial.clone());
                    update_context_message(context, partial);
                    emitter
                        .emit(AgentEvent::MessageUpdate {
                            message: AgentMessage::Assistant(partial.clone()),
                            stream_event: other.clone(),
                        })
                        .await;
                }
            }
        }
    }

    // Fallback: stream ended without Done/Error
    let final_message =
        create_error_assistant_message(&config.model, "Stream ended without terminal event");
    finalize_stream_message(context, &final_message, added_partial, emitter).await;
    Ok(final_message)
}

// ============================================================
// Tool Execution (mirrors agent-loop.ts executeToolCalls*)
// ============================================================

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "debug",
        name = "agent.tools.execute",
        skip_all,
        fields(mode = ?config.tool_execution, tool_calls = tool_calls.len())
    )
)]
async fn execute_tool_calls(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCallContent],
    config: &AgentLoopConfig,
    cancel: &CancellationToken,
    emitter: &EventEmitter,
) -> Result<Vec<ToolResultMessage>, AgentError> {
    if config.tool_execution == ToolExecutionMode::Sequential {
        execute_tool_calls_sequential(
            context,
            assistant_message,
            tool_calls,
            config,
            cancel,
            emitter,
        )
        .await
    } else {
        execute_tool_calls_parallel(
            context,
            assistant_message,
            tool_calls,
            config,
            cancel,
            emitter,
        )
        .await
    }
}

async fn execute_tool_calls_sequential(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCallContent],
    config: &AgentLoopConfig,
    cancel: &CancellationToken,
    emitter: &EventEmitter,
) -> Result<Vec<ToolResultMessage>, AgentError> {
    let mut results: Vec<ToolResultMessage> = Vec::new();

    for tool_call in tool_calls {
        if cancel.is_cancelled() {
            break;
        }

        emitter
            .emit(AgentEvent::ToolExecutionStart {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                args: tool_call.arguments.clone(),
            })
            .await;

        if cancel.is_cancelled() {
            results.push(
                emit_tool_call_outcome(
                    tool_call,
                    create_error_tool_result(&AgentError::Aborted.to_string()),
                    true,
                    emitter,
                )
                .await,
            );
            break;
        }

        let preparation =
            prepare_tool_call(context, assistant_message, tool_call, config, cancel).await;

        match preparation {
            ToolCallPreparation::Immediate { result, is_error } => {
                results.push(emit_tool_call_outcome(tool_call, result, is_error, emitter).await);
                if cancel.is_cancelled() {
                    break;
                }
            }
            ToolCallPreparation::Prepared(prepared) => {
                if cancel.is_cancelled() {
                    results.push(
                        emit_tool_call_outcome(
                            &prepared.tool_call,
                            create_error_tool_result(&AgentError::Aborted.to_string()),
                            true,
                            emitter,
                        )
                        .await,
                    );
                    break;
                }

                let executed = execute_prepared_tool_call(&prepared, cancel, emitter).await;
                results.push(
                    finalize_executed_tool_call(
                        context,
                        assistant_message,
                        &prepared,
                        executed,
                        config,
                        cancel,
                        emitter,
                    )
                    .await,
                );
            }
        }
    }

    Ok(results)
}

async fn execute_tool_calls_parallel(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCallContent],
    config: &AgentLoopConfig,
    cancel: &CancellationToken,
    emitter: &EventEmitter,
) -> Result<Vec<ToolResultMessage>, AgentError> {
    let mut results: Vec<ToolResultMessage> = Vec::new();
    let mut runnable_calls: Vec<PreparedToolCall> = Vec::new();

    // Phase 1: Prepare all tool calls (sequential, like TypeScript)
    for tool_call in tool_calls {
        if cancel.is_cancelled() {
            break;
        }

        emitter
            .emit(AgentEvent::ToolExecutionStart {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                args: tool_call.arguments.clone(),
            })
            .await;

        if cancel.is_cancelled() {
            results.push(
                emit_tool_call_outcome(
                    tool_call,
                    create_error_tool_result(&AgentError::Aborted.to_string()),
                    true,
                    emitter,
                )
                .await,
            );
            break;
        }

        let preparation =
            prepare_tool_call(context, assistant_message, tool_call, config, cancel).await;

        match preparation {
            ToolCallPreparation::Immediate { result, is_error } => {
                results.push(emit_tool_call_outcome(tool_call, result, is_error, emitter).await);
                if cancel.is_cancelled() {
                    break;
                }
            }
            ToolCallPreparation::Prepared(prepared) => {
                if cancel.is_cancelled() {
                    results.push(
                        emit_tool_call_outcome(
                            &prepared.tool_call,
                            create_error_tool_result(&AgentError::Aborted.to_string()),
                            true,
                            emitter,
                        )
                        .await,
                    );
                    break;
                }

                runnable_calls.push(prepared);
            }
        }
    }

    // Phase 2: Execute all runnable tools concurrently via tokio::spawn
    let mut handles = Vec::new();
    let mut spawned_count = 0usize;
    for prepared in &runnable_calls {
        if cancel.is_cancelled() {
            break;
        }

        let tool = prepared.tool.clone();
        let tool_call = prepared.tool_call.clone();
        let args = prepared.args.clone();
        let cancel = cancel.clone();
        let emitter = emitter.clone();

        handles.push(tokio::spawn(async move {
            execute_tool_call_core(&*tool, &tool_call, args, cancel, emitter).await
        }));
        spawned_count += 1;
    }

    // Phase 3: Finalize in original order (sequential).
    // Any prepared calls that were never spawned because of cancellation are
    // closed out as aborted tool results so their `ToolExecutionStart` events
    // do not remain dangling in state.
    let mut handle_iter = handles.into_iter();
    for (index, prepared) in runnable_calls.iter().enumerate() {
        if index < spawned_count {
            let handle = handle_iter.next().ok_or_else(|| {
                AgentError::Internal(
                    "execute_tool_calls_parallel: spawned handles and prepared calls desynchronized"
                        .to_string(),
                )
            })?;
            let executed = handle
                .await
                .map_err(|e| AgentError::JoinError(e.to_string()))?;
            results.push(
                finalize_executed_tool_call(
                    context,
                    assistant_message,
                    prepared,
                    executed,
                    config,
                    cancel,
                    emitter,
                )
                .await,
            );
        } else {
            results.push(
                emit_tool_call_outcome(
                    &prepared.tool_call,
                    create_error_tool_result(&AgentError::Aborted.to_string()),
                    true,
                    emitter,
                )
                .await,
            );
        }
    }

    Ok(results)
}

// ============================================================
// Tool Call Pipeline Types
// ============================================================

struct PreparedToolCall {
    tool_call: ToolCallContent,
    tool: Arc<dyn AgentTool>,
    args: serde_json::Value,
}

enum ToolCallPreparation {
    Immediate {
        result: AgentToolResult,
        is_error: bool,
    },
    Prepared(PreparedToolCall),
}

struct ExecutedToolCallOutcome {
    result: AgentToolResult,
    is_error: bool,
}

// ============================================================
// Tool Call Pipeline Functions
// ============================================================

fn validate_tool_arguments(
    tool: &dyn AgentTool,
    arguments: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let schema = tool.parameters();
    let args = arguments.clone();
    let validator = validator_for(&schema)
        .map_err(|e| format!("Invalid JSON Schema for tool \"{}\": {}", tool.name(), e))?;
    if validator.is_valid(&args) {
        return Ok(args);
    }
    let mut lines: Vec<String> = Vec::new();
    for err in validator.iter_errors(&args) {
        let path = err.instance_path().to_string();
        let path = path.strip_prefix('/').unwrap_or(path.as_str());
        let path_display = if path.is_empty() { "root" } else { path };
        lines.push(format!("  - {path_display}: {err}"));
    }
    let errors = if lines.is_empty() {
        "Unknown validation error".to_string()
    } else {
        lines.join("\n")
    };
    Err(format!(
        "Validation failed for tool \"{}\":\n{}\n\nReceived arguments:\n{}",
        tool.name(),
        errors,
        serde_json::to_string_pretty(arguments).unwrap_or_else(|_| arguments.to_string())
    ))
}

async fn prepare_tool_call(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_call: &ToolCallContent,
    config: &AgentLoopConfig,
    cancel: &CancellationToken,
) -> ToolCallPreparation {
    let tool = context.tools.iter().find(|t| t.name() == tool_call.name);
    let tool = match tool {
        Some(t) => t.clone(),
        None => {
            return ToolCallPreparation::Immediate {
                result: create_error_tool_result(&format!("Tool {} not found", tool_call.name)),
                is_error: true,
            };
        }
    };

    let prepared_arguments = tool.prepare_arguments(tool_call.arguments.clone());

    let validated_args = match validate_tool_arguments(&*tool, &prepared_arguments) {
        Ok(v) => v,
        Err(msg) => {
            return ToolCallPreparation::Immediate {
                result: create_error_tool_result(&msg),
                is_error: true,
            };
        }
    };

    let args_cell = Arc::new(Mutex::new(validated_args));

    if let Some(ref before_hook) = config.before_tool_call {
        let hook_context = BeforeToolCallContext {
            assistant_message: assistant_message.clone(),
            tool_call: tool_call.clone(),
            args: args_cell.clone(),
            context: AgentContextSnapshot {
                system_prompt: context.system_prompt.clone(),
                messages: Arc::clone(&context.messages),
                tools: context.tools.clone(),
            },
        };

        if let Some(BeforeToolCallResult {
            block: true,
            reason,
        }) = before_hook(hook_context, cancel.clone()).await
        {
            let reason_text = reason.unwrap_or_else(|| "Tool execution was blocked".to_string());
            return ToolCallPreparation::Immediate {
                result: create_error_tool_result(&reason_text),
                is_error: true,
            };
        }
    }

    let final_args = match args_cell.lock() {
        Ok(g) => g.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    };

    ToolCallPreparation::Prepared(PreparedToolCall {
        tool_call: tool_call.clone(),
        tool,
        args: final_args,
    })
}

/// Core tool execution — no agent context. Safe to spawn; uses `emitter` for `on_update` events.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "debug",
        name = "agent.tool.call",
        skip_all,
        fields(tool = %tool.name(), tool_call_id = %tool_call.id)
    )
)]
async fn execute_tool_call_core(
    tool: &dyn AgentTool,
    tool_call: &ToolCallContent,
    args: serde_json::Value,
    cancel: CancellationToken,
    emitter: EventEmitter,
) -> ExecutedToolCallOutcome {
    let tool_call_id = tool_call.id.clone();
    let update_handles: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>> =
        Arc::new(Mutex::new(Vec::new()));
    let handles_for_cb = update_handles.clone();
    let emitter_for_cb = emitter.clone();
    let tool_call_for_cb = tool_call.clone();

    let on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>> =
        Some(Box::new(move |partial_result: AgentToolResult| {
            let emitter = emitter_for_cb.clone();
            let tc = tool_call_for_cb.clone();
            let mut guard = match handles_for_cb.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            guard.push(tokio::spawn(async move {
                emitter
                    .emit(AgentEvent::ToolExecutionUpdate {
                        tool_call_id: tc.id,
                        tool_name: tc.name,
                        args: tc.arguments,
                        partial_result,
                    })
                    .await;
            }));
        }));

    let outcome = match tool.execute(&tool_call_id, args, cancel, on_update).await {
        Ok(result) => ExecutedToolCallOutcome {
            result,
            is_error: false,
        },
        Err(e) => ExecutedToolCallOutcome {
            result: create_error_tool_result(&e.to_string()),
            is_error: true,
        },
    };

    let handles = match update_handles.lock() {
        Ok(mut g) => std::mem::take(&mut *g),
        Err(p) => std::mem::take(&mut *p.into_inner()),
    };
    for h in handles {
        let _ = h.await;
    }

    outcome
}

async fn execute_prepared_tool_call(
    prepared: &PreparedToolCall,
    cancel: &CancellationToken,
    emitter: &EventEmitter,
) -> ExecutedToolCallOutcome {
    execute_tool_call_core(
        &*prepared.tool,
        &prepared.tool_call,
        prepared.args.clone(),
        cancel.clone(),
        emitter.clone(),
    )
    .await
}

async fn finalize_executed_tool_call(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    prepared: &PreparedToolCall,
    mut executed: ExecutedToolCallOutcome,
    config: &AgentLoopConfig,
    cancel: &CancellationToken,
    emitter: &EventEmitter,
) -> ToolResultMessage {
    // afterToolCall hook
    if let Some(ref after_hook) = config.after_tool_call {
        let hook_context = AfterToolCallContext {
            assistant_message: assistant_message.clone(),
            tool_call: prepared.tool_call.clone(),
            args: prepared.args.clone(),
            result: executed.result.clone(),
            is_error: executed.is_error,
            context: AgentContextSnapshot {
                system_prompt: context.system_prompt.clone(),
                messages: Arc::clone(&context.messages),
                tools: context.tools.clone(),
            },
        };

        if let Some(after_result) = after_hook(hook_context, cancel.clone()).await {
            if let Some(content) = after_result.content {
                executed.result.content = content;
            }
            if let Some(details) = after_result.details {
                executed.result.details = Some(details);
            }
            if let Some(is_error) = after_result.is_error {
                executed.is_error = is_error;
            }
        }
    }

    emit_tool_call_outcome(
        &prepared.tool_call,
        executed.result,
        executed.is_error,
        emitter,
    )
    .await
}

async fn emit_tool_call_outcome(
    tool_call: &ToolCallContent,
    result: AgentToolResult,
    is_error: bool,
    emitter: &EventEmitter,
) -> ToolResultMessage {
    emitter
        .emit(AgentEvent::ToolExecutionEnd {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            result: result.clone(),
            is_error,
        })
        .await;

    let tool_result_message = ToolResultMessage {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        content: result.content,
        details: result.details,
        is_error,
        timestamp: now_millis(),
    };

    emitter
        .emit(AgentEvent::MessageStart {
            message: AgentMessage::ToolResult(tool_result_message.clone()),
        })
        .await;
    emitter
        .emit(AgentEvent::MessageEnd {
            message: AgentMessage::ToolResult(tool_result_message.clone()),
        })
        .await;

    tool_result_message
}

// ============================================================
// Helpers
// ============================================================

fn stream_event_partial_for_update(ev: &StreamEvent) -> Option<&AssistantMessage> {
    match ev {
        StreamEvent::TextStart { partial, .. }
        | StreamEvent::TextDelta { partial, .. }
        | StreamEvent::TextEnd { partial, .. }
        | StreamEvent::ThinkingStart { partial, .. }
        | StreamEvent::ThinkingDelta { partial, .. }
        | StreamEvent::ThinkingEnd { partial, .. }
        | StreamEvent::ToolCallStart { partial, .. }
        | StreamEvent::ToolCallDelta { partial, .. }
        | StreamEvent::ToolCallEnd { partial, .. } => Some(partial),
        StreamEvent::Start { .. } | StreamEvent::Done { .. } | StreamEvent::Error { .. } => None,
    }
}

fn update_context_message(context: &mut AgentContext, message: &AssistantMessage) {
    let msgs = Arc::make_mut(&mut context.messages);
    if let Some(last) = msgs.last_mut() {
        *last = AgentMessage::Assistant(message.clone());
    }
}

async fn finalize_stream_message(
    context: &mut AgentContext,
    message: &AssistantMessage,
    added_partial: bool,
    emitter: &EventEmitter,
) {
    let msgs = Arc::make_mut(&mut context.messages);
    if added_partial {
        if let Some(last) = msgs.last_mut() {
            *last = AgentMessage::Assistant(message.clone());
        }
    } else {
        msgs.push(AgentMessage::Assistant(message.clone()));
        emitter
            .emit(AgentEvent::MessageStart {
                message: AgentMessage::Assistant(message.clone()),
            })
            .await;
    }
    emitter
        .emit(AgentEvent::MessageEnd {
            message: AgentMessage::Assistant(message.clone()),
        })
        .await;
}

fn create_abort_message(model: &Model, partial: Option<&AssistantMessage>) -> AssistantMessage {
    if let Some(pm) = partial {
        AssistantMessage {
            stop_reason: StopReason::Aborted,
            error_message: Some("Aborted".to_string()),
            ..pm.clone()
        }
    } else {
        AssistantMessage {
            stop_reason: StopReason::Aborted,
            ..create_error_assistant_message(model, "Aborted")
        }
    }
}

fn normalize_terminal_assistant_message(mut message: AssistantMessage) -> AssistantMessage {
    if message.error_message.is_none() {
        message.error_message = Some(match message.stop_reason {
            StopReason::Aborted => "Aborted".to_string(),
            StopReason::Error => "Unknown stream error".to_string(),
            _ => return message,
        });
    }
    message
}

fn create_error_assistant_message(model: &Model, error: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![ContentBlock::Text(TextContent {
            text: String::new(),
            text_signature: None,
        })],
        model: model.id.clone(),
        provider: model.provider.clone(),
        api: model.api.clone(),
        response_id: None,
        stop_reason: StopReason::Error,
        error_message: Some(error.to_string()),
        usage: Usage::default(),
        timestamp: now_millis(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn mk_model() -> Model {
        Model {
            id: "m".into(),
            name: "m".into(),
            api: "openai-responses".into(),
            provider: "mock".into(),
            base_url: "https://example.invalid".into(),
            reasoning: false,
            input: vec!["text".into()],
            cost: ModelCost::default(),
            context_window: 1,
            max_tokens: 1,
        }
    }

    fn mk_assistant(stop: StopReason, err: Option<&str>) -> AssistantMessage {
        AssistantMessage {
            content: vec![],
            model: "m".into(),
            provider: "mock".into(),
            api: "openai-responses".into(),
            response_id: None,
            stop_reason: stop,
            error_message: err.map(|s| s.to_string()),
            usage: Usage::default(),
            timestamp: 0,
        }
    }

    #[test]
    fn normalize_aborted_adds_default_error_message() {
        let msg = normalize_terminal_assistant_message(mk_assistant(StopReason::Aborted, None));
        assert_eq!(msg.error_message.as_deref(), Some("Aborted"));
    }

    #[test]
    fn normalize_error_adds_default_error_message() {
        let msg = normalize_terminal_assistant_message(mk_assistant(StopReason::Error, None));
        assert_eq!(msg.error_message.as_deref(), Some("Unknown stream error"));
    }

    #[test]
    fn normalize_preserves_existing_error_message() {
        let msg =
            normalize_terminal_assistant_message(mk_assistant(StopReason::Error, Some("custom")));
        assert_eq!(msg.error_message.as_deref(), Some("custom"));
    }

    #[test]
    fn normalize_noop_for_stop_without_error() {
        let msg = normalize_terminal_assistant_message(mk_assistant(StopReason::Stop, None));
        assert!(msg.error_message.is_none());
    }

    #[test]
    fn terminal_error_message_finds_trailing_error() {
        let model = mk_model();
        let messages = vec![AgentMessage::Assistant(create_error_assistant_message(
            &model, "boom",
        ))];
        assert_eq!(
            terminal_error_message(&messages, StopReason::Error).as_deref(),
            Some("boom"),
        );
    }

    #[test]
    fn terminal_error_message_returns_none_for_non_terminal() {
        let messages = vec![AgentMessage::Assistant(mk_assistant(
            StopReason::Stop,
            None,
        ))];
        assert!(terminal_error_message(&messages, StopReason::Error).is_none());
    }

    struct RejectingTool;

    #[async_trait::async_trait]
    impl AgentTool for RejectingTool {
        fn name(&self) -> &str {
            "nope"
        }
        fn label(&self) -> &str {
            "nope"
        }
        fn description(&self) -> &str {
            "no"
        }
        fn parameters(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": { "x": { "type": "integer" } },
                "required": ["x"],
                "additionalProperties": false,
            })
        }
        async fn execute(
            &self,
            _tool_call_id: &str,
            _params: serde_json::Value,
            _cancel: CancellationToken,
            _on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>>,
        ) -> Result<AgentToolResult, AgentError> {
            unreachable!()
        }
    }

    #[test]
    fn validate_tool_arguments_accepts_valid() {
        let tool = RejectingTool;
        let ok = validate_tool_arguments(&tool, &json!({ "x": 42 }));
        assert_eq!(ok.unwrap(), json!({ "x": 42 }));
    }

    #[test]
    fn validate_tool_arguments_rejects_missing_required_field() {
        let tool = RejectingTool;
        let err = validate_tool_arguments(&tool, &json!({})).unwrap_err();
        assert!(
            err.contains("Validation failed for tool \"nope\"")
                && err.contains("required property"),
            "unexpected error: {err}"
        );
    }
}
