//! High-level `Agent` lifecycle parity with pi-mono `packages/agent` (agent.test.ts).

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{assistant_text, stream_done_only, stream_waits_for_cancel, test_model, user_message};
use futures::stream::{self, StreamExt};
use oh_my_agentloop::{
    Agent, AgentError, AgentEvent, AgentMessage, AgentOptions, InitialAgentState, Model, QueueMode,
    StreamEvent, StreamFn, UserContent, UserMessage,
};
use tokio::sync::Notify;

fn agent_options(stream_fn: StreamFn) -> AgentOptions {
    AgentOptions {
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
    }
}

fn hanging_partial_stream(model: Model) -> StreamFn {
    let partial = assistant_text("", &model);
    Arc::new(move |_m, _ctx, _req| {
        let partial = partial.clone();
        Box::pin(async move {
            let s = stream::repeat(()).then(move |_| {
                let partial = partial.clone();
                async move {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    Ok(StreamEvent::Start {
                        partial: partial.clone(),
                    })
                }
            });
            Ok(Box::pin(s) as oh_my_agentloop::LlmEventStream)
        })
    })
}

#[tokio::test]
async fn prompt_awaits_async_subscribers_before_completion() {
    let model = test_model();
    let stream_fn = stream_done_only(assistant_text("ok", &model));

    let barrier = Arc::new(Notify::new());
    let barrier_listen = barrier.clone();

    let agent = Agent::new(agent_options(stream_fn));

    let listener_done = Arc::new(AtomicBool::new(false));
    let ld = listener_done.clone();
    let _sub = agent.subscribe(move |event, _cancel| {
        let barrier_listen = barrier_listen.clone();
        let ld = ld.clone();
        async move {
            if matches!(
                event,
                AgentEvent::RunCompleted { .. }
                    | AgentEvent::RunFailed { .. }
                    | AgentEvent::RunAborted { .. }
            ) {
                barrier_listen.notified().await;
                ld.store(true, Ordering::SeqCst);
            }
        }
    });

    let prompt_done = Arc::new(AtomicBool::new(false));
    let pd = prompt_done.clone();
    let agent_clone = agent.clone();
    let prompt_task = tokio::spawn(async move {
        agent_clone.prompt_text("hello", None).await.unwrap();
        pd.store(true, Ordering::SeqCst);
    });

    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!prompt_done.load(Ordering::SeqCst));
    assert!(!listener_done.load(Ordering::SeqCst));
    assert!(agent.is_streaming());

    barrier.notify_waiters();
    prompt_task.await.unwrap();
    assert!(listener_done.load(Ordering::SeqCst));
    assert!(prompt_done.load(Ordering::SeqCst));
    assert!(!agent.is_streaming());
}

#[tokio::test]
async fn wait_for_idle_waits_for_async_subscribers_on_message_end() {
    let model = test_model();
    let stream_fn = stream_done_only(assistant_text("ok", &model));

    let barrier = Arc::new(Notify::new());
    let barrier_listen = barrier.clone();

    let agent = Agent::new(agent_options(stream_fn));

    let _sub = agent.subscribe(move |event, _cancel| {
        let barrier_listen = barrier_listen.clone();
        async move {
            if let AgentEvent::MessageEnd { message } = event {
                if message.role() == "assistant" {
                    barrier_listen.notified().await;
                }
            }
        }
    });

    let agent_clone = agent.clone();
    let prompt_task = tokio::spawn(async move {
        agent_clone.prompt_text("hello", None).await.unwrap();
    });

    // Match TS ordering: `prompt()` registers the active run before `waitForIdle()` observes it.
    tokio::task::yield_now().await;
    tokio::task::yield_now().await;

    let idle_done = Arc::new(AtomicBool::new(false));
    let id = idle_done.clone();
    let agent_idle = agent.clone();
    let idle_task = tokio::spawn(async move {
        agent_idle.wait_for_idle().await;
        id.store(true, Ordering::SeqCst);
    });

    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!idle_done.load(Ordering::SeqCst));
    assert!(agent.is_streaming());

    barrier.notify_waiters();
    let _ = tokio::join!(prompt_task, idle_task);
    assert!(idle_done.load(Ordering::SeqCst));
    assert!(!agent.is_streaming());
}

#[tokio::test]
async fn second_prompt_while_streaming_returns_already_processing() {
    let model = test_model();
    let agent = Agent::new(agent_options(hanging_partial_stream(model.clone())));

    let first = tokio::spawn({
        let a = agent.clone();
        async move { a.prompt_text("first", None).await }
    });

    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(agent.is_streaming());
    let err = agent.prompt_text("second", None).await.unwrap_err();
    assert!(matches!(err, AgentError::AlreadyProcessing));

    agent.abort();
    let err = first.await.unwrap().unwrap_err();
    assert!(matches!(err, AgentError::Aborted));
}

#[tokio::test]
async fn continue_while_streaming_returns_already_processing() {
    let model = test_model();
    let agent = Agent::new(agent_options(hanging_partial_stream(model.clone())));

    let first = tokio::spawn({
        let a = agent.clone();
        async move { a.prompt_text("first", None).await }
    });

    tokio::time::sleep(Duration::from_millis(30)).await;
    let err = agent.continue_run().await.unwrap_err();
    assert!(matches!(err, AgentError::AlreadyProcessing));

    agent.abort();
    let err = first.await.unwrap().unwrap_err();
    assert!(matches!(err, AgentError::Aborted));
}

#[tokio::test]
async fn continue_from_assistant_drains_follow_up_after_turn() {
    let model = test_model();
    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = call_count.clone();

    let stream_fn: StreamFn = Arc::new(move |model_arg, _ctx, _req| {
        let cc = cc.clone();
        Box::pin(async move {
            cc.fetch_add(1, Ordering::SeqCst);
            let msg = assistant_text("Processed", &model_arg);
            let s = futures::stream::iter(vec![Ok(StreamEvent::Done { message: msg })]);
            Ok(Box::pin(s) as oh_my_agentloop::LlmEventStream)
        })
    });

    let mut opts = agent_options(stream_fn);
    opts.initial_state = Some(InitialAgentState {
        system_prompt: None,
        model: Some(model.clone()),
        thinking_level: None,
        tools: None,
        messages: Some(vec![
            user_message("Initial"),
            AgentMessage::Assistant(assistant_text("Initial response", &model)),
        ]),
    });

    let agent = Agent::new(opts);
    agent.follow_up(AgentMessage::User(UserMessage {
        content: UserContent::Plain("Queued follow-up".into()),
        timestamp: 1,
    }));

    agent.continue_run().await.unwrap();

    let msgs = agent.state().messages;
    let has_follow = msgs.iter().any(|m| {
        if let AgentMessage::User(u) = m {
            if let UserContent::Plain(t) = &u.content {
                return t == "Queued follow-up";
            }
        }
        false
    });
    assert!(has_follow);
    assert_eq!(msgs.last().map(|m| m.role()), Some("assistant"));
    assert_eq!(call_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn continue_from_assistant_one_at_a_time_steering_runs_two_llm_rounds() {
    let model = test_model();
    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = call_count.clone();

    let stream_fn: StreamFn = Arc::new(move |model_arg, _ctx, _req| {
        let cc = cc.clone();
        Box::pin(async move {
            let n = cc.fetch_add(1, Ordering::SeqCst) + 1;
            let msg = assistant_text(&format!("Processed {n}"), &model_arg);
            let s = futures::stream::iter(vec![Ok(StreamEvent::Done { message: msg })]);
            Ok(Box::pin(s) as oh_my_agentloop::LlmEventStream)
        })
    });

    let mut opts = agent_options(stream_fn);
    opts.steering_mode = Some(QueueMode::OneAtATime);
    opts.initial_state = Some(InitialAgentState {
        system_prompt: None,
        model: Some(model.clone()),
        thinking_level: None,
        tools: None,
        messages: Some(vec![
            user_message("Initial"),
            AgentMessage::Assistant(assistant_text("Initial response", &model)),
        ]),
    });

    let agent = Agent::new(opts);
    agent.steer(AgentMessage::User(UserMessage {
        content: UserContent::Plain("Steering 1".into()),
        timestamp: 1,
    }));
    agent.steer(AgentMessage::User(UserMessage {
        content: UserContent::Plain("Steering 2".into()),
        timestamp: 2,
    }));

    agent.continue_run().await.unwrap();

    assert_eq!(call_count.load(Ordering::SeqCst), 2);
    let msgs = agent.state().messages;
    let tail_roles: Vec<_> = msgs.iter().rev().take(4).map(|m| m.role()).collect();
    let mut tail_roles = tail_roles;
    tail_roles.reverse();
    assert_eq!(tail_roles, vec!["user", "assistant", "user", "assistant"]);
}

#[tokio::test]
async fn steering_mode_all_drains_both_before_single_extra_llm_round() {
    let model = test_model();
    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = call_count.clone();

    let stream_fn: StreamFn = Arc::new(move |model_arg, _ctx, _req| {
        let cc = cc.clone();
        Box::pin(async move {
            cc.fetch_add(1, Ordering::SeqCst);
            let msg = assistant_text("Once", &model_arg);
            let s = futures::stream::iter(vec![Ok(StreamEvent::Done { message: msg })]);
            Ok(Box::pin(s) as oh_my_agentloop::LlmEventStream)
        })
    });

    let mut opts = agent_options(stream_fn);
    opts.steering_mode = Some(QueueMode::All);
    opts.initial_state = Some(InitialAgentState {
        system_prompt: None,
        model: Some(model.clone()),
        thinking_level: None,
        tools: None,
        messages: Some(vec![
            user_message("Initial"),
            AgentMessage::Assistant(assistant_text("Initial response", &model)),
        ]),
    });

    let agent = Agent::new(opts);
    agent.steer(AgentMessage::User(UserMessage {
        content: UserContent::Plain("A".into()),
        timestamp: 1,
    }));
    agent.steer(AgentMessage::User(UserMessage {
        content: UserContent::Plain("B".into()),
        timestamp: 2,
    }));

    agent.continue_run().await.unwrap();

    assert_eq!(call_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn reset_clears_messages_queues_and_runtime_fields() {
    let model = test_model();
    let stream_fn = stream_done_only(assistant_text("x", &model));
    let mut opts = agent_options(stream_fn);
    opts.initial_state = Some(InitialAgentState {
        system_prompt: Some("sys".into()),
        model: Some(model.clone()),
        thinking_level: None,
        tools: None,
        messages: Some(vec![user_message("keep sys")]),
    });
    let agent = Agent::new(opts);

    agent.steer(user_message("q"));
    agent.follow_up(user_message("f"));

    agent.reset();

    let s = agent.state();
    assert!(s.messages.is_empty());
    assert!(!s.is_streaming);
    assert!(s.streaming_message.is_none());
    assert!(s.pending_tool_calls.is_empty());
    assert!(s.error_message.is_none());
    assert!(!agent.has_queued_messages());
    assert_eq!(s.system_prompt, "sys");
}

#[tokio::test]
async fn forwards_session_id_to_stream_options() {
    let seen = Arc::new(Mutex::new(None::<String>));
    let seen_clone = seen.clone();

    let stream_fn: StreamFn = Arc::new(move |model_arg, _ctx, req| {
        let seen_clone = seen_clone.clone();
        Box::pin(async move {
            *seen_clone.lock().unwrap() = req.options.session_id.clone();
            let msg = assistant_text("ok", &model_arg);
            let s = futures::stream::iter(vec![Ok(StreamEvent::Done { message: msg })]);
            Ok(Box::pin(s) as oh_my_agentloop::LlmEventStream)
        })
    });

    let mut opts = agent_options(stream_fn);
    opts.session_id = Some("session-abc".into());
    let agent = Agent::new(opts);
    agent.prompt_text("hello", None).await.unwrap();
    assert_eq!(seen.lock().unwrap().as_deref(), Some("session-abc"));

    agent.set_session_id("session-def".into());
    agent.prompt_text("again", None).await.unwrap();
    assert_eq!(seen.lock().unwrap().as_deref(), Some("session-def"));
}

#[tokio::test]
async fn forwards_on_payload_to_stream_options() {
    let touched = Arc::new(AtomicBool::new(false));
    let touched_hook = touched.clone();

    let hook: oh_my_agentloop::OnPayloadFn = Arc::new(move |_payload, _model| {
        let touched_hook = touched_hook.clone();
        Box::pin(async move {
            touched_hook.store(true, Ordering::SeqCst);
            None
        })
    });

    let stream_fn: StreamFn = Arc::new(move |model_arg, _ctx, req| {
        Box::pin(async move {
            assert!(
                req.options.on_payload.is_some(),
                "on_payload should be forwarded into StreamOptions"
            );
            if let Some(ref h) = req.options.on_payload {
                let _ = h(serde_json::json!({}), model_arg.clone()).await;
            }
            let msg = assistant_text("ok", &model_arg);
            let s = futures::stream::iter(vec![Ok(StreamEvent::Done { message: msg })]);
            Ok(Box::pin(s) as oh_my_agentloop::LlmEventStream)
        })
    });

    let mut opts = agent_options(stream_fn);
    opts.on_payload = Some(hook);
    let agent = Agent::new(opts);
    agent.prompt_text("hello", None).await.unwrap();
    assert!(touched.load(Ordering::SeqCst));
}

#[tokio::test]
async fn abort_cancels_in_flight_run_and_clears_signal() {
    let model = test_model();
    let agent = Agent::new(agent_options(hanging_partial_stream(model.clone())));

    let run = tokio::spawn({
        let a = agent.clone();
        async move { a.prompt_text("hi", None).await }
    });

    common::wait_for(Duration::from_secs(1), "run has started", || {
        agent.signal().is_some()
    })
    .await;
    assert!(!agent.signal().unwrap().is_cancelled());

    agent.abort();
    assert!(agent.signal().unwrap().is_cancelled());

    let err = run.await.unwrap().unwrap_err();
    assert!(matches!(err, AgentError::Aborted));

    common::wait_for(Duration::from_secs(1), "signal is cleared", || {
        agent.signal().is_none() && !agent.is_streaming()
    })
    .await;
}

#[tokio::test]
async fn abort_propagates_through_stream_request_cancel() {
    let model = test_model();
    let agent = Agent::new(agent_options(stream_waits_for_cancel(model.clone())));

    let run = tokio::spawn({
        let agent = agent.clone();
        async move { agent.prompt_text("hello", None).await }
    });

    common::wait_for(Duration::from_secs(1), "agent is streaming", || {
        agent.is_streaming()
    })
    .await;
    agent.abort();

    let err = run.await.unwrap().unwrap_err();
    assert!(matches!(err, AgentError::Aborted));
}
