//! High-level stateful Agent wrapper around the low-level agent loop.
//!
//! Direct port of pi-agent-core/src/agent.ts

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::agent_loop::{run_agent_loop, run_agent_loop_continue};
use crate::types::*;

/// Lock a `std::sync::Mutex`, unwrapping poison.
///
/// All critical sections in this module are short and do not panic, so a poisoned
/// lock indicates a prior bug — unwrapping is acceptable and centralises the
/// single site where we do so.
#[inline]
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| {
        #[cfg(feature = "tracing")]
        tracing::error!(error = %e, "agent: mutex poisoned, recovering");
        #[cfg(not(feature = "tracing"))]
        let _ = &e;
        e.into_inner()
    })
}

// ============================================================
// Queue Types
// ============================================================

#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueueMode {
    All,
    #[default]
    OneAtATime,
}

struct PendingMessageQueue {
    messages: Vec<AgentMessage>,
    mode: QueueMode,
}

impl PendingMessageQueue {
    fn new(mode: QueueMode) -> Self {
        PendingMessageQueue {
            messages: Vec::new(),
            mode,
        }
    }

    fn enqueue(&mut self, message: AgentMessage) {
        self.messages.push(message);
    }

    fn has_items(&self) -> bool {
        !self.messages.is_empty()
    }

    fn drain(&mut self) -> Vec<AgentMessage> {
        if self.mode == QueueMode::All {
            std::mem::take(&mut self.messages)
        } else {
            if self.messages.is_empty() {
                return Vec::new();
            }
            vec![self.messages.remove(0)]
        }
    }

    fn clear(&mut self) {
        self.messages.clear();
    }
}

// ============================================================
// Mutable Agent State (internal)
// ============================================================

struct MutableAgentState {
    system_prompt: String,
    model: Model,
    thinking_level: ThinkingLevel,
    tools: Vec<Arc<dyn AgentTool>>,
    messages: Vec<AgentMessage>,
    is_streaming: bool,
    streaming_message: Option<AgentMessage>,
    pending_tool_calls: HashSet<String>,
    error_message: Option<String>,
}

impl MutableAgentState {
    fn snapshot(&self) -> AgentState {
        AgentState {
            system_prompt: self.system_prompt.clone(),
            model: self.model.clone(),
            thinking_level: self.thinking_level.clone(),
            tools: self.tools.clone(),
            messages: self.messages.clone(),
            is_streaming: self.is_streaming,
            streaming_message: self.streaming_message.clone(),
            pending_tool_calls: self.pending_tool_calls.clone(),
            error_message: self.error_message.clone(),
        }
    }
}

// ============================================================
// Agent Options
// ============================================================

/// Initial state for constructing an Agent.
#[derive(Default)]
pub struct InitialAgentState {
    pub system_prompt: Option<String>,
    pub model: Option<Model>,
    pub thinking_level: Option<ThinkingLevel>,
    pub tools: Option<Vec<Arc<dyn AgentTool>>>,
    pub messages: Option<Vec<AgentMessage>>,
}

/// Options for constructing an [`Agent`].
///
/// Prefer the [`AgentOptions::builder`] fluent API; the public fields remain
/// accessible for advanced use cases (e.g. serde-driven configuration).
pub struct AgentOptions {
    pub initial_state: Option<InitialAgentState>,
    pub convert_to_llm: Option<ConvertToLlmFn>,
    pub transform_context: Option<TransformContextFn>,
    pub stream_fn: StreamFn,
    pub get_api_key: Option<GetApiKeyFn>,
    pub before_tool_call: Option<BeforeToolCallHookFn>,
    pub after_tool_call: Option<AfterToolCallHookFn>,
    pub steering_mode: Option<QueueMode>,
    pub follow_up_mode: Option<QueueMode>,
    pub session_id: Option<String>,
    pub transport: Option<Transport>,
    pub tool_execution: Option<ToolExecutionMode>,
    pub api_key: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub max_retry_delay_ms: Option<u64>,
    /// Optional hook forwarded to each LLM request (`StreamOptions::on_payload`).
    pub on_payload: Option<OnPayloadFn>,
}

impl AgentOptions {
    /// Start building an [`AgentOptions`] value. `stream_fn` is the only required field.
    ///
    /// ```ignore
    /// let options = AgentOptions::builder(my_stream_fn())
    ///     .initial_state(initial)
    ///     .tool_execution(ToolExecutionMode::Parallel)
    ///     .build();
    /// ```
    pub fn builder(stream_fn: StreamFn) -> AgentOptionsBuilder {
        AgentOptionsBuilder::new(stream_fn)
    }
}

/// Fluent builder for [`AgentOptions`].
#[must_use = "call .build() to construct the AgentOptions"]
pub struct AgentOptionsBuilder {
    inner: AgentOptions,
}

impl AgentOptionsBuilder {
    /// Create a new builder with only the required `stream_fn` field set.
    pub fn new(stream_fn: StreamFn) -> Self {
        Self {
            inner: AgentOptions {
                initial_state: None,
                convert_to_llm: None,
                transform_context: None,
                stream_fn,
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
            },
        }
    }

    pub fn initial_state(mut self, state: InitialAgentState) -> Self {
        self.inner.initial_state = Some(state);
        self
    }
    pub fn convert_to_llm(mut self, f: ConvertToLlmFn) -> Self {
        self.inner.convert_to_llm = Some(f);
        self
    }
    pub fn transform_context(mut self, f: TransformContextFn) -> Self {
        self.inner.transform_context = Some(f);
        self
    }
    pub fn get_api_key(mut self, f: GetApiKeyFn) -> Self {
        self.inner.get_api_key = Some(f);
        self
    }
    pub fn before_tool_call(mut self, f: BeforeToolCallHookFn) -> Self {
        self.inner.before_tool_call = Some(f);
        self
    }
    pub fn after_tool_call(mut self, f: AfterToolCallHookFn) -> Self {
        self.inner.after_tool_call = Some(f);
        self
    }
    pub fn steering_mode(mut self, mode: QueueMode) -> Self {
        self.inner.steering_mode = Some(mode);
        self
    }
    pub fn follow_up_mode(mut self, mode: QueueMode) -> Self {
        self.inner.follow_up_mode = Some(mode);
        self
    }
    pub fn session_id(mut self, id: impl Into<String>) -> Self {
        self.inner.session_id = Some(id.into());
        self
    }
    pub fn transport(mut self, t: Transport) -> Self {
        self.inner.transport = Some(t);
        self
    }
    pub fn tool_execution(mut self, mode: ToolExecutionMode) -> Self {
        self.inner.tool_execution = Some(mode);
        self
    }
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.inner.api_key = Some(key.into());
        self
    }
    pub fn temperature(mut self, t: f64) -> Self {
        self.inner.temperature = Some(t);
        self
    }
    pub fn max_tokens(mut self, n: u64) -> Self {
        self.inner.max_tokens = Some(n);
        self
    }
    pub fn thinking_budgets(mut self, b: ThinkingBudgets) -> Self {
        self.inner.thinking_budgets = Some(b);
        self
    }
    pub fn max_retry_delay_ms(mut self, ms: u64) -> Self {
        self.inner.max_retry_delay_ms = Some(ms);
        self
    }
    pub fn on_payload(mut self, f: OnPayloadFn) -> Self {
        self.inner.on_payload = Some(f);
        self
    }

    /// Finish building the [`AgentOptions`] value.
    pub fn build(self) -> AgentOptions {
        self.inner
    }
}

// ============================================================
// Agent Listener
// ============================================================

type ListenerFn = Arc<
    dyn Fn(AgentEvent, CancellationToken) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync,
>;

/// Subscription handle. Dropping it unsubscribes.
///
/// Drop it only after you no longer need the event stream; keeping the handle alive
/// is the only way to receive further [`AgentEvent`]s.
#[must_use = "Subscription unsubscribes on drop; keep it alive to receive events"]
pub struct Subscription {
    inner: Arc<AgentInner>,
    listener: ListenerFn,
}

impl Drop for Subscription {
    fn drop(&mut self) {
        let mut listeners = self.inner.listeners.lock().unwrap();
        listeners.retain(|l| !Arc::ptr_eq(l, &self.listener));
    }
}

// ============================================================
// Agent Inner (shared state)
// ============================================================

struct AgentInner {
    state: Mutex<MutableAgentState>,
    steering_queue: Mutex<PendingMessageQueue>,
    follow_up_queue: Mutex<PendingMessageQueue>,
    listeners: Mutex<Vec<ListenerFn>>,
    cancel: Mutex<Option<CancellationToken>>,
    /// When `Some`, a run is in progress; `notify_waiters` is called when the run fully settles.
    run_complete: Mutex<Option<Arc<Notify>>>,

    // Config (cloneable, set at construction)
    convert_to_llm: ConvertToLlmFn,
    transform_context: Option<TransformContextFn>,
    stream_fn: StreamFn,
    get_api_key: Option<GetApiKeyFn>,
    before_tool_call: Mutex<Option<BeforeToolCallHookFn>>,
    after_tool_call: Mutex<Option<AfterToolCallHookFn>>,
    session_id: Mutex<Option<String>>,
    on_payload: Option<OnPayloadFn>,
    transport: Transport,
    tool_execution: ToolExecutionMode,
    api_key: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    thinking_budgets: Option<ThinkingBudgets>,
    max_retry_delay_ms: Option<u64>,
}

// ============================================================
// Agent
// ============================================================

/// Stateful wrapper around the low-level agent loop.
///
/// `Agent` owns the current transcript, emits lifecycle events, executes tools,
/// and exposes queueing APIs for steering and follow-up messages.
///
/// Mirrors TypeScript `Agent` class from pi-agent-core.
#[derive(Clone)]
pub struct Agent {
    inner: Arc<AgentInner>,
}

impl Agent {
    pub fn new(options: AgentOptions) -> Self {
        let initial = options.initial_state.unwrap_or_default();

        let state = MutableAgentState {
            system_prompt: initial.system_prompt.unwrap_or_default(),
            model: initial.model.unwrap_or_default(),
            thinking_level: initial.thinking_level.unwrap_or_default(),
            tools: initial.tools.unwrap_or_default(),
            messages: initial.messages.unwrap_or_default(),
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        };

        let convert_to_llm = options.convert_to_llm.unwrap_or_else(|| {
            Arc::new(|messages: Vec<AgentMessage>| {
                Box::pin(async move { default_convert_to_llm(messages) })
                    as Pin<Box<dyn Future<Output = Vec<Message>> + Send>>
            })
        });

        Agent {
            inner: Arc::new(AgentInner {
                state: Mutex::new(state),
                steering_queue: Mutex::new(PendingMessageQueue::new(
                    options.steering_mode.unwrap_or_default(),
                )),
                follow_up_queue: Mutex::new(PendingMessageQueue::new(
                    options.follow_up_mode.unwrap_or_default(),
                )),
                listeners: Mutex::new(Vec::new()),
                cancel: Mutex::new(None),
                run_complete: Mutex::new(None),
                convert_to_llm,
                transform_context: options.transform_context,
                stream_fn: options.stream_fn,
                get_api_key: options.get_api_key,
                before_tool_call: Mutex::new(options.before_tool_call),
                after_tool_call: Mutex::new(options.after_tool_call),
                session_id: Mutex::new(options.session_id),
                on_payload: options.on_payload,
                transport: options.transport.unwrap_or_default(),
                tool_execution: options.tool_execution.unwrap_or_default(),
                api_key: options.api_key,
                temperature: options.temperature,
                max_tokens: options.max_tokens,
                thinking_budgets: options.thinking_budgets,
                max_retry_delay_ms: options.max_retry_delay_ms,
            }),
        }
    }

    // ---- Public State Access ----

    pub fn state(&self) -> AgentState {
        self.with_state(MutableAgentState::snapshot)
    }

    pub fn is_streaming(&self) -> bool {
        self.with_state(|s| s.is_streaming)
    }

    pub fn pending_tool_calls(&self) -> HashSet<String> {
        self.with_state(|s| s.pending_tool_calls.clone())
    }

    // ---- Mutable State Setters ----
    //
    // Setters mutate state *without* emitting `AgentEvent`s. Subscribed UIs should
    // treat these as explicit, user-initiated changes and re-render on their own.

    pub fn set_system_prompt(&self, prompt: String) {
        self.with_state_mut(|s| s.system_prompt = prompt);
    }

    pub fn set_model(&self, model: Model) {
        self.with_state_mut(|s| s.model = model);
    }

    pub fn set_thinking_level(&self, level: ThinkingLevel) {
        self.with_state_mut(|s| s.thinking_level = level);
    }

    pub fn set_tools(&self, tools: Vec<Arc<dyn AgentTool>>) {
        self.with_state_mut(|s| s.tools = tools);
    }

    pub fn set_messages(&self, messages: Vec<AgentMessage>) {
        self.with_state_mut(|s| s.messages = messages);
    }

    pub fn set_before_tool_call(&self, hook: Option<BeforeToolCallHookFn>) {
        *lock(&self.inner.before_tool_call) = hook;
    }

    pub fn set_after_tool_call(&self, hook: Option<AfterToolCallHookFn>) {
        *lock(&self.inner.after_tool_call) = hook;
    }

    /// Update the session id forwarded to the stream function on subsequent runs.
    pub fn set_session_id(&self, session_id: String) {
        *lock(&self.inner.session_id) = Some(session_id);
    }

    // ---- Internal Locking Helpers ----

    fn with_state<R>(&self, f: impl FnOnce(&MutableAgentState) -> R) -> R {
        f(&lock(&self.inner.state))
    }

    fn with_state_mut<R>(&self, f: impl FnOnce(&mut MutableAgentState) -> R) -> R {
        f(&mut lock(&self.inner.state))
    }

    // ---- Queue Mode ----

    pub fn set_steering_mode(&self, mode: QueueMode) {
        self.inner.steering_queue.lock().unwrap().mode = mode;
    }

    pub fn steering_mode(&self) -> QueueMode {
        self.inner.steering_queue.lock().unwrap().mode.clone()
    }

    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        self.inner.follow_up_queue.lock().unwrap().mode = mode;
    }

    pub fn follow_up_mode(&self) -> QueueMode {
        self.inner.follow_up_queue.lock().unwrap().mode.clone()
    }

    // ---- Steering & Follow-up ----

    /// Queue a message to be injected after the current assistant turn finishes.
    pub fn steer(&self, message: AgentMessage) {
        self.inner.steering_queue.lock().unwrap().enqueue(message);
    }

    /// Queue a message to run only after the agent would otherwise stop.
    pub fn follow_up(&self, message: AgentMessage) {
        self.inner.follow_up_queue.lock().unwrap().enqueue(message);
    }

    pub fn clear_steering_queue(&self) {
        self.inner.steering_queue.lock().unwrap().clear();
    }

    pub fn clear_follow_up_queue(&self) {
        self.inner.follow_up_queue.lock().unwrap().clear();
    }

    pub fn clear_all_queues(&self) {
        self.clear_steering_queue();
        self.clear_follow_up_queue();
    }

    pub fn has_queued_messages(&self) -> bool {
        self.inner.steering_queue.lock().unwrap().has_items()
            || self.inner.follow_up_queue.lock().unwrap().has_items()
    }

    // ---- Lifecycle ----

    /// Active cancellation token for the current run, if any.
    pub fn signal(&self) -> Option<CancellationToken> {
        self.inner.cancel.lock().unwrap().clone()
    }

    /// Abort the current run, if one is active.
    pub fn abort(&self) {
        if let Some(ref cancel) = *self.inner.cancel.lock().unwrap() {
            cancel.cancel();
        }
    }

    /// Resolve when the current run and all awaited event listeners have finished.
    ///
    /// Matches TypeScript `waitForIdle()` tied to `activeRun.promise`: when no run is active,
    /// returns immediately. Callers that start a run concurrently should yield (or await the
    /// start of that run) before awaiting this, matching JS microtask ordering where `prompt()`
    /// registers `activeRun` before a subsequent `waitForIdle()` call runs.
    pub async fn wait_for_idle(&self) {
        loop {
            let notify = self.inner.run_complete.lock().unwrap().clone();
            match notify {
                Some(n) => n.notified().await,
                None => return,
            }
        }
    }

    /// Clear transcript state, runtime state, and queued messages.
    pub fn reset(&self) {
        let mut state = self.inner.state.lock().unwrap();
        state.messages.clear();
        state.is_streaming = false;
        state.streaming_message = None;
        state.pending_tool_calls.clear();
        state.error_message = None;
        drop(state);
        self.clear_all_queues();
    }

    // ---- Subscribe ----

    /// Subscribe to agent lifecycle events.
    /// Returns a Subscription handle; dropping it unsubscribes.
    pub fn subscribe<F, Fut>(&self, f: F) -> Subscription
    where
        F: Fn(AgentEvent, CancellationToken) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let listener: ListenerFn = Arc::new(move |event, cancel| Box::pin(f(event, cancel)));
        self.inner.listeners.lock().unwrap().push(listener.clone());
        Subscription {
            inner: self.inner.clone(),
            listener,
        }
    }

    // ---- Prompt & Continue ----

    /// Start a new prompt with a text string and optional images.
    pub async fn prompt_text(
        &self,
        text: &str,
        images: Option<Vec<ImageContent>>,
    ) -> Result<(), AgentError> {
        let mut content: Vec<ContentBlock> = vec![ContentBlock::Text(TextContent {
            text: text.to_string(),
            text_signature: None,
        })];
        if let Some(imgs) = images {
            for img in imgs {
                content.push(ContentBlock::Image(img));
            }
        }
        let message = AgentMessage::User(UserMessage {
            content: UserContent::try_from_llm_blocks(content)?,
            timestamp: now_millis(),
        });
        self.prompt(vec![message]).await
    }

    /// Start a new prompt from one or more messages.
    pub async fn prompt(&self, messages: Vec<AgentMessage>) -> Result<(), AgentError> {
        {
            let cancel = self.inner.cancel.lock().unwrap();
            if cancel.is_some() {
                return Err(AgentError::AlreadyProcessing);
            }
        }
        self.run_prompt_messages(messages, false).await
    }

    /// Continue from the current transcript.
    /// The last message must be a user or tool-result message.
    pub async fn continue_run(&self) -> Result<(), AgentError> {
        {
            let cancel = self.inner.cancel.lock().unwrap();
            if cancel.is_some() {
                return Err(AgentError::AlreadyProcessing);
            }
        }

        let last_role = {
            let state = self.inner.state.lock().unwrap();
            state.messages.last().map(|m| m.role().to_string())
        };

        match last_role.as_deref() {
            None => return Err(AgentError::NoMessages),
            Some("assistant") => {
                // Try draining steering/follow-up queues (like TypeScript)
                let steering = self.inner.steering_queue.lock().unwrap().drain();
                if !steering.is_empty() {
                    return self.run_prompt_messages(steering, true).await;
                }
                let follow_ups = self.inner.follow_up_queue.lock().unwrap().drain();
                if !follow_ups.is_empty() {
                    return self.run_prompt_messages(follow_ups, false).await;
                }
                return Err(AgentError::ContinueFromAssistant);
            }
            _ => {}
        }

        self.run_continuation().await
    }

    // ---- Internal Run Logic ----

    async fn run_prompt_messages(
        &self,
        messages: Vec<AgentMessage>,
        skip_initial_steering_poll: bool,
    ) -> Result<(), AgentError> {
        let outcome = self
            .run_with_lifecycle(|agent, cancel| {
                let messages = messages.clone();
                Box::pin(async move {
                    let context = agent.create_context_snapshot();
                    let config = agent.create_loop_config(skip_initial_steering_poll);
                    let stream_fn = agent.inner.stream_fn.clone();
                    let emitter = agent.create_emitter();
                    run_agent_loop(messages, context, config, &emitter, cancel, &stream_fn).await
                })
            })
            .await?;

        outcome_to_result(outcome)
    }

    async fn run_continuation(&self) -> Result<(), AgentError> {
        let outcome = self
            .run_with_lifecycle(|agent, cancel| {
                Box::pin(async move {
                    let context = agent.create_context_snapshot();
                    let config = agent.create_loop_config(false);
                    let stream_fn = agent.inner.stream_fn.clone();
                    let emitter = agent.create_emitter();
                    run_agent_loop_continue(context, config, &emitter, cancel, &stream_fn).await
                })
            })
            .await?;

        outcome_to_result(outcome)
    }

    async fn run_with_lifecycle<F>(&self, executor: F) -> Result<RunOutcome, AgentError>
    where
        F: FnOnce(
            Agent,
            CancellationToken,
        ) -> Pin<Box<dyn Future<Output = Result<RunOutcome, AgentError>> + Send>>,
    {
        let cancel = CancellationToken::new();
        let done = Arc::new(Notify::new());

        // Set active run (registration must complete before the first await — mirrors TS `activeRun`)
        {
            let mut cancel_guard = self.inner.cancel.lock().unwrap();
            if cancel_guard.is_some() {
                return Err(AgentError::AlreadyProcessing);
            }
            *self.inner.run_complete.lock().unwrap() = Some(done.clone());
            *cancel_guard = Some(cancel.clone());
        }

        // Mark streaming
        {
            let mut state = self.inner.state.lock().unwrap();
            state.is_streaming = true;
            state.streaming_message = None;
            state.error_message = None;
        }

        let result = executor(self.clone(), cancel).await;
        self.finish_run();
        result
    }

    fn finish_run(&self) {
        {
            let mut state = self.inner.state.lock().unwrap();
            state.is_streaming = false;
            state.streaming_message = None;
            state.pending_tool_calls.clear();
        }
        *self.inner.cancel.lock().unwrap() = None;
        let notify = self.inner.run_complete.lock().unwrap().take();
        if let Some(n) = notify {
            n.notify_waiters();
        }
    }

    fn create_context_snapshot(&self) -> AgentContext {
        let state = self.inner.state.lock().unwrap();
        AgentContext {
            system_prompt: state.system_prompt.clone(),
            messages: state.messages.clone(),
            tools: state.tools.clone(),
        }
    }

    fn create_loop_config(&self, skip_initial_steering_poll: bool) -> AgentLoopConfig {
        let state = self.inner.state.lock().unwrap();
        let model = state.model.clone();
        let thinking_level = state.thinking_level.clone();
        drop(state);

        let steering_inner = self.inner.clone();
        let skip_flag = Arc::new(std::sync::atomic::AtomicBool::new(
            skip_initial_steering_poll,
        ));

        let get_steering: GetMessagesFn = Arc::new(move || {
            let inner = steering_inner.clone();
            let should_skip = skip_flag.swap(false, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async move {
                if should_skip {
                    return Vec::new();
                }
                inner.steering_queue.lock().unwrap().drain()
            })
        });

        let follow_up_inner = self.inner.clone();
        let get_follow_up: GetMessagesFn = Arc::new(move || {
            let inner = follow_up_inner.clone();
            Box::pin(async move { inner.follow_up_queue.lock().unwrap().drain() })
        });

        let before = self.inner.before_tool_call.lock().unwrap().clone();
        let after = self.inner.after_tool_call.lock().unwrap().clone();

        AgentLoopConfig {
            model,
            reasoning: if thinking_level == ThinkingLevel::Off {
                None
            } else {
                Some(thinking_level)
            },
            convert_to_llm: self.inner.convert_to_llm.clone(),
            transform_context: self.inner.transform_context.clone(),
            get_api_key: self.inner.get_api_key.clone(),
            get_steering_messages: Some(get_steering),
            get_follow_up_messages: Some(get_follow_up),
            tool_execution: self.inner.tool_execution.clone(),
            before_tool_call: before,
            after_tool_call: after,
            api_key: self.inner.api_key.clone(),
            session_id: self.inner.session_id.lock().unwrap().clone(),
            transport: self.inner.transport.clone(),
            temperature: self.inner.temperature,
            max_tokens: self.inner.max_tokens,
            thinking_budgets: self.inner.thinking_budgets.clone(),
            max_retry_delay_ms: self.inner.max_retry_delay_ms,
            on_payload: self.inner.on_payload.clone(),
        }
    }

    fn create_emitter(&self) -> EventEmitter {
        let agent = self.clone();
        EventEmitter::new(move |event| {
            let agent = agent.clone();
            async move {
                agent.process_event(event).await;
            }
        })
    }

    /// Reduce internal state for a loop event, then await listeners.
    async fn process_event(&self, event: AgentEvent) {
        // Update state
        {
            let mut state = self.inner.state.lock().unwrap();
            match &event {
                AgentEvent::MessageStart { message } => {
                    state.streaming_message = Some(message.clone());
                }
                AgentEvent::MessageUpdate { message, .. } => {
                    state.streaming_message = Some(message.clone());
                }
                AgentEvent::MessageEnd { message } => {
                    state.streaming_message = None;
                    state.messages.push(message.clone());
                }
                AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                    state.pending_tool_calls.insert(tool_call_id.clone());
                }
                AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                    state.pending_tool_calls.remove(tool_call_id);
                }
                AgentEvent::RunCompleted { .. } => {
                    state.streaming_message = None;
                }
                AgentEvent::RunFailed {
                    messages,
                    error_message,
                } => {
                    state.streaming_message = None;
                    state.error_message = terminal_error_message(messages, StopReason::Error)
                        .or_else(|| Some(error_message.clone()));
                }
                AgentEvent::RunAborted { messages } => {
                    state.streaming_message = None;
                    state.error_message = terminal_error_message(messages, StopReason::Aborted)
                        .or_else(|| Some(AgentError::Aborted.to_string()));
                }
                _ => {}
            }
        }

        // Notify listeners (outside state lock)
        let cancel = self
            .inner
            .cancel
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_default();

        let listeners: Vec<ListenerFn> = self.inner.listeners.lock().unwrap().clone();
        for listener in &listeners {
            listener(event.clone(), cancel.clone()).await;
        }
    }
}

fn outcome_to_result(outcome: RunOutcome) -> Result<(), AgentError> {
    match outcome {
        RunOutcome::Completed { .. } => Ok(()),
        RunOutcome::Failed { error, .. } => Err(error),
        RunOutcome::Aborted { .. } => Err(AgentError::Aborted),
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

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("is_streaming", &self.is_streaming())
            .finish()
    }
}
