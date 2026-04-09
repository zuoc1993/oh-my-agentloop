//! Crate-root exports are usable without module-qualified paths.

use oh_my_agentloop::{
    default_convert_to_llm, AgentMessage, ContentBlock, ImageContent, Message, TextContent,
    UserContent, UserMessage,
};

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
