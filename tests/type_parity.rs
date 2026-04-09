//! JSON / wire-shape parity with `@mariozechner/pi-ai` message types.

mod common;

use common::json_value;
use oh_my_agentloop::{
    AssistantMessage, ContentBlock, ImageContent, Message, StopReason, TextContent,
    ThinkingContent, ToolResultMessage, Usage, UserContent, UserContentBuildError, UserMessage,
};
use serde_json::json;

#[test]
fn user_content_try_from_llm_blocks_rejects_non_user_variants() {
    let thinking =
        UserContent::try_from_llm_blocks(vec![ContentBlock::Thinking(ThinkingContent {
            thinking: "x".into(),
            thinking_signature: None,
            redacted: None,
        })]);
    assert_eq!(thinking, Err(UserContentBuildError::ThinkingBlock));

    let tool = UserContent::try_from_llm_blocks(vec![ContentBlock::ToolCall(
        oh_my_agentloop::ToolCallContent {
            id: "1".into(),
            name: "n".into(),
            arguments: serde_json::json!({}),
        },
    )]);
    assert_eq!(tool, Err(UserContentBuildError::ToolCallBlock));
}

#[test]
fn stop_reason_serializes_to_pi_ai_strings() {
    assert_eq!(json_value(&StopReason::Stop), json!("stop"));
    assert_eq!(json_value(&StopReason::Length), json!("length"));
    assert_eq!(json_value(&StopReason::ToolUse), json!("toolUse"));
    assert_eq!(json_value(&StopReason::Error), json!("error"));
    assert_eq!(json_value(&StopReason::Aborted), json!("aborted"));
}

#[test]
fn stop_reason_deserializes_from_pi_ai_strings() {
    let v: Vec<StopReason> = vec![
        serde_json::from_value(json!("stop")).unwrap(),
        serde_json::from_value(json!("length")).unwrap(),
        serde_json::from_value(json!("toolUse")).unwrap(),
        serde_json::from_value(json!("error")).unwrap(),
        serde_json::from_value(json!("aborted")).unwrap(),
    ];
    assert_eq!(
        v,
        vec![
            StopReason::Stop,
            StopReason::Length,
            StopReason::ToolUse,
            StopReason::Error,
            StopReason::Aborted,
        ]
    );
}

#[test]
fn user_message_serializes_plain_string_content() {
    let m = Message::User(UserMessage {
        content: UserContent::Plain("hello".into()),
        timestamp: 42,
    });
    let v = json_value(&m);
    assert_eq!(
        v,
        json!({
            "role": "user",
            "content": "hello",
            "timestamp": 42
        })
    );
}

#[test]
fn user_message_serializes_block_array_content() {
    let m = Message::User(UserMessage {
        content: UserContent::Blocks(vec![
            oh_my_agentloop::UserContentBlock::Text(TextContent {
                text: "a".into(),
                text_signature: Some("sig".into()),
            }),
            oh_my_agentloop::UserContentBlock::Image(ImageContent {
                data: "YmFzZTY0".into(),
                mime_type: "image/png".into(),
            }),
        ]),
        timestamp: 1,
    });
    let v = json_value(&m);
    assert_eq!(
        v,
        json!({
            "role": "user",
            "content": [
                { "type": "text", "text": "a", "textSignature": "sig" },
                { "type": "image", "data": "YmFzZTY0", "mimeType": "image/png" }
            ],
            "timestamp": 1
        })
    );
}

#[test]
fn user_message_deserializes_string_or_array_content() {
    let m1: Message = serde_json::from_value(json!({
        "role": "user",
        "content": "plain",
        "timestamp": 9
    }))
    .unwrap();
    match m1 {
        Message::User(u) => {
            assert_eq!(u.content, UserContent::Plain("plain".into()));
            assert_eq!(u.timestamp, 9);
        }
        _ => panic!("expected user message"),
    }

    let m2: Message = serde_json::from_value(json!({
        "role": "user",
        "content": [{ "type": "text", "text": "x" }],
        "timestamp": 2
    }))
    .unwrap();
    match m2 {
        Message::User(u) => match u.content {
            UserContent::Blocks(b) => {
                assert_eq!(b.len(), 1);
            }
            UserContent::Plain(_) => panic!("expected blocks"),
        },
        _ => panic!("expected user message"),
    }
}

#[test]
fn assistant_message_optional_response_id_uses_camel_case() {
    let with_id = AssistantMessage {
        content: vec![ContentBlock::Text(TextContent {
            text: "t".into(),
            text_signature: Some("tsig".into()),
        })],
        model: "m".into(),
        provider: "openai".into(),
        api: "openai-responses".into(),
        response_id: Some("resp_abc".into()),
        stop_reason: StopReason::Stop,
        error_message: None,
        usage: Usage::default(),
        timestamp: 0,
    };
    let v = json_value(&with_id);
    assert_eq!(
        v.get("responseId").and_then(|x| x.as_str()),
        Some("resp_abc")
    );
    let text = &v["content"][0];
    assert_eq!(text["textSignature"], json!("tsig"));

    let without_id = AssistantMessage {
        response_id: None,
        ..with_id.clone()
    };
    let v2 = json_value(&without_id);
    assert!(v2.get("responseId").is_none());
}

#[test]
fn thinking_block_uses_thinking_signature_and_optional_redacted() {
    let m = AssistantMessage {
        content: vec![ContentBlock::Thinking(ThinkingContent {
            thinking: "reason".into(),
            thinking_signature: Some("rid".into()),
            redacted: Some(true),
        })],
        model: "m".into(),
        provider: "p".into(),
        api: "openai-responses".into(),
        response_id: None,
        stop_reason: StopReason::Stop,
        error_message: None,
        usage: Usage::default(),
        timestamp: 0,
    };
    let v = json_value(&m);
    let th = &v["content"][0];
    assert_eq!(th["thinkingSignature"], json!("rid"));
    assert_eq!(th["redacted"], json!(true));
}

#[test]
fn tool_result_message_omits_details_when_none() {
    let m = ToolResultMessage {
        tool_call_id: "c1".into(),
        tool_name: "foo".into(),
        content: vec![ContentBlock::Text(TextContent {
            text: "ok".into(),
            text_signature: None,
        })],
        details: None,
        is_error: false,
        timestamp: 3,
    };
    let v = json_value(&m);
    assert!(v.get("details").is_none());
}

#[test]
fn tool_result_message_serializes_details_when_some() {
    let m = ToolResultMessage {
        tool_call_id: "c1".into(),
        tool_name: "foo".into(),
        content: vec![],
        details: Some(json!({ "k": 1 })),
        is_error: false,
        timestamp: 3,
    };
    let v = json_value(&m);
    assert_eq!(v["details"], json!({ "k": 1 }));
}

#[test]
fn message_enum_user_variant_roundtrips_string_content() {
    let m = Message::User(UserMessage {
        content: UserContent::Plain("x".into()),
        timestamp: 1,
    });
    let v = json_value(&m);
    let back: Message = serde_json::from_value(v).unwrap();
    match back {
        Message::User(u) => assert_eq!(u.content, UserContent::Plain("x".into())),
        _ => panic!("expected user"),
    }
}
