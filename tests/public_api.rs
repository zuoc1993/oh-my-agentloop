//! Crate-root exports are usable without module-qualified paths.
//!
//! Also acts as a basic public-API smoke check: each test below imports and
//! touches a specific slice of the crate-root surface. Adding/removing a
//! public item should produce a corresponding test change.

use oh_my_agentloop::{
    default_convert_to_llm, Agent, AgentError, AgentEvent, AgentMessage, AgentOptions,
    AgentOptionsBuilder, ContentBlock, ImageContent, InitialAgentState, Message, QueueMode,
    RunOutcome, StopReason, Subscription, TextContent, ThinkingLevel, ToolExecutionMode, Transport,
    UserContent, UserMessage,
};
use std::marker::PhantomData;

#[test]
fn crate_root_user_content_and_convert_to_llm() {
    let blocks = vec![
        ContentBlock::Text(TextContent {
            text: "hello".into(),
            text_signature: None,
        }),
        ContentBlock::Image(ImageContent {
            data: "d".into(),
            mime_type: "image/png".into(),
        }),
    ];
    let content = UserContent::try_from_llm_blocks(blocks).expect("text and image only");
    assert!(matches!(content, UserContent::Blocks(ref v) if v.len() == 2));

    let single = UserContent::try_from_llm_blocks(vec![ContentBlock::Text(TextContent {
        text: "only".into(),
        text_signature: None,
    })])
    .expect("single text");
    assert_eq!(single, UserContent::Plain("only".into()));

    let llm = default_convert_to_llm(vec![AgentMessage::User(UserMessage {
        content,
        timestamp: 0,
    })]);
    assert_eq!(llm.len(), 1);
    match &llm[0] {
        Message::User(u) => assert!(matches!(&u.content, UserContent::Blocks(bs) if bs.len() == 2)),
        _ => panic!("expected User message"),
    }
}

/// Sanity test: key public items exist and have expected shapes. If any of
/// these fail to compile, it is a semver-breaking change.
#[test]
fn crate_root_public_api_shape_is_stable() {
    #[allow(dead_code)]
    struct _PublicTypes<'a> {
        agent: PhantomData<Agent>,
        options: PhantomData<AgentOptions>,
        builder: PhantomData<AgentOptionsBuilder>,
        initial: PhantomData<InitialAgentState>,
        subscription: PhantomData<Subscription>,
        queue_mode: PhantomData<QueueMode>,
        transport: PhantomData<Transport>,
        tool_exec: PhantomData<ToolExecutionMode>,
        stop_reason: PhantomData<StopReason>,
        thinking: PhantomData<ThinkingLevel>,
        outcome: PhantomData<RunOutcome>,
        error: PhantomData<AgentError>,
        event: PhantomData<AgentEvent>,
        _marker: PhantomData<&'a ()>,
    }
    // All re-exports compile — that is the entire point of this test.
}
