# oh-my-agentloop 架构文档

> 面向当前 Rust 实现的说明文档，不再描述上游 TypeScript 或未实现的 `proxy` 能力。

---

## 1. 项目定位

`oh-my-agentloop` 是一个可嵌入的 Rust Agent Loop Runtime，用来把以下能力组合成统一运行时：

- LLM 流式调用抽象：通过注入 `StreamFn` 适配任意 Provider。
- 消息模型分层：应用层使用 `AgentMessage`，LLM 边界使用 `Message`。
- 工具调用编排：支持参数预处理、JSON Schema 校验、前后置 Hook、流式更新。
- 事件驱动状态同步：低层循环发事件，高层 `Agent` 消费事件并维护状态。
- 运行中干预：支持 `steer` 与 `follow_up` 两类队列消息。

当前 crate 聚焦“非 proxy 核心能力”，由调用方提供模型接入和工具实现。

---

## 2. 总体架构

```text
┌──────────────────────────────────────────────────────────┐
│ High-Level API: Agent                                   │
│ - 持有 transcript / tools / queues / listeners          │
│ - 提供 prompt_text / prompt / continue_run / abort      │
│ - 将 AgentEvent 规约为 AgentState                       │
└───────────────────────┬──────────────────────────────────┘
                        │ 调用
┌───────────────────────▼──────────────────────────────────┐
│ Low-Level Runtime: agent_loop / run_agent_loop          │
│ - 组装上下文                                             │
│ - 驱动 LLM 流式响应                                       │
│ - 执行工具调用                                            │
│ - 发出完整生命周期事件                                    │
└───────────────────────┬──────────────────────────────────┘
                        │ 注入
┌───────────────────────▼──────────────────────────────────┐
│ Integration Boundary                                     │
│ - StreamFn: 连接具体 LLM Provider                        │
│ - AgentTool: 执行外部能力                                │
│ - convert_to_llm / transform_context / hooks             │
└──────────────────────────────────────────────────────────┘
```

设计核心是“双层 API”：

- 低层负责确定性运行时流程。
- 高层负责状态、并发保护、订阅和队列管理。

这使得库既能作为纯运行时内核使用，也能作为带状态的 Agent 对象直接嵌入业务代码。

---

## 3. 代码结构

```text
oh-my-agentloop/
├── src/
│   ├── lib.rs         # 导出入口
│   ├── types.rs       # 公共类型、回调、错误、事件
│   ├── agent_loop.rs  # 低层循环与工具编排
│   └── agent.rs       # 高层有状态 Agent 封装
├── tests/             # 对外行为验证
└── docs/              # 设计、计划与架构文档
```

各模块职责如下：

| 模块 | 职责 |
|------|------|
| `types.rs` | 统一建模消息、内容块、流事件、工具接口、Hook 上下文、错误和配置 |
| `agent_loop.rs` | 执行一轮或多轮 agent loop，处理 LLM 响应、工具调用、队列注入 |
| `agent.rs` | 封装运行状态、监听器、取消控制、队列及高层 API |
| `lib.rs` | 对外 re-export，暴露 crate 主入口 |

---

## 4. 核心数据模型

### 4.1 消息分层

项目把消息分为两层：

1. `Message`
   纯 LLM 可理解消息，只包含：
   - `User`
   - `Assistant`
   - `ToolResult`

2. `AgentMessage`
   运行时内部消息抽象，在 `Message` 之外增加：
   - `Custom(CustomMessage)`

这层分离的意义是：应用侧可以携带自定义消息，但在真正调用 LLM 前，必须通过 `convert_to_llm` 明确转换或过滤。

消息转换链路如下：

```text
AgentContext.messages
  -> transform_context (可选)
  -> convert_to_llm (必需)
  -> LlmContext.messages
  -> StreamFn
```

### 4.2 内容块模型

Assistant / ToolResult 的 `content` 使用统一的 `ContentBlock`：

- `Text`
- `Image`
- `Thinking`
- `ToolCall`

这让一次 assistant 响应可以混合“自然语言 + 推理片段 + 工具调用声明”。

### 4.3 工具抽象

工具由 `AgentTool` trait 表达，核心接口包括：

- 元数据：`name()`、`label()`、`description()`、`parameters()`
- 参数兼容层：`prepare_arguments()`
- 实际执行：`execute()`

其中 `parameters()` 返回 JSON Schema，运行时会在调用前做校验。

### 4.4 配置与扩展回调

低层循环通过 `AgentLoopConfig` 注入扩展点：

- `convert_to_llm`
- `transform_context`
- `get_api_key`
- `get_steering_messages`
- `get_follow_up_messages`
- `before_tool_call`
- `after_tool_call`
- `on_payload`

这些扩展点基本覆盖了“上下文治理、认证、消息注入、工具审计、请求改写”等主要集成需求。

---

## 5. 低层运行时设计

`agent_loop.rs` 是项目的核心执行引擎，对外提供两套接口：

- Stream 风格：
  - `agent_loop`
  - `agent_loop_continue`
- Callback 风格：
  - `run_agent_loop`
  - `run_agent_loop_continue`

前者通过 `tokio::mpsc::UnboundedReceiver<AgentEvent>` 向外推事件，后者直接把事件发给 `EventEmitter`，供高层 `Agent` 使用。

### 5.1 Runtime Terminal Contract

低层 runtime 的终止契约已经收敛为显式 terminal outcome，不再依赖历史上的 `AgentEnd` 事件或 receiver close 作为“本次 run 结束”的判断依据。

- 每次运行恰好发出一个 terminal event：`RunCompleted` / `RunFailed` / `RunAborted`
- `agent_loop()` / `agent_loop_continue()` 只是把这些事件写入 channel；receiver close 只表示发送端生命周期结束，不再承载业务终止语义
- `run_agent_loop()` / `run_agent_loop_continue()` 返回 `Result<RunOutcome, AgentError>`
- 高层调用方应把这个 `Result` 当作控制流真相；UI / persistence 则以 transcript 和 `AgentEvent` 事件流作为可观测真相

### 5.2 两种启动模式

| 函数 | 用途 |
|------|------|
| `agent_loop` / `run_agent_loop` | 从新的 prompt 开始，把 prompt 追加到上下文 |
| `agent_loop_continue` / `run_agent_loop_continue` | 从现有 transcript 继续，不追加新 prompt |

继续运行时有两个保护：

- 空上下文直接报错 `NoMessages`
- 末尾是 assistant 消息时报错 `ContinueFromAssistant`

### 5.3 主循环结构

低层循环是“两层 while”结构：

- 内层：处理 assistant 响应、工具调用、steering 注入
- 外层：当 agent 准备结束时，检查 follow-up 队列并决定是否继续

简化流程如下：

```text
初始化 pending steering
while true:
  while 有工具调用 或 有待注入消息:
    注入 steering 消息
    调用 LLM，流式接收 assistant 消息
    如果 assistant 含 tool calls:
      执行工具
    轮次结束后再次检查 steering

  检查 follow-up
  若存在 follow-up:
    转入下一次外层循环
  否则:
    结束 agent
```

这个结构保证了两种注入语义：

- `steering` 在当前任务未完全结束时，插入下一轮推理前。
- `follow_up` 只有在 agent 原本会停止时才执行。

---

## 6. LLM 流式响应处理

### 6.1 调用边界

每次进入 `stream_assistant_response()`，会依次完成：

1. 对 `AgentMessage[]` 做 `transform_context`
2. 通过 `convert_to_llm` 变成 `Message[]`
3. 基于工具定义构造 `LlmContext`
4. 解析动态或静态 API Key
5. 组装 `StreamOptions`
6. 调用注入的 `StreamFn`

### 6.2 流式事件语义

`StreamEvent` 的设计与 provider 流式响应对齐，每个增量事件可携带当前 `partial AssistantMessage`。运行时据此维护“正在生成中的 assistant 消息”。

关键事件包括：

- `Start`
- `TextStart` / `TextDelta` / `TextEnd`
- `ThinkingStart` / `ThinkingDelta` / `ThinkingEnd`
- `ToolCallStart` / `ToolCallDelta` / `ToolCallEnd`
- `Done`
- `Error`

### 6.3 partial message 的处理方式

低层循环不会自己重建文本，而是直接信任 `StreamEvent` 携带的 `partial`：

- `Start` 时将 partial assistant 插入上下文，并发出 `MessageStart`
- 各类 delta/update 事件时，用新的 partial 覆盖上下文末尾消息，并发出 `MessageUpdate`
- `Done` 或 `Error` 时用最终消息替换 partial，并发出 `MessageEnd`

这种设计的优点：

- Provider 侧的 partial 结构可以原样向上传递
- UI 层能直接消费近实时状态
- 运行时避免重复实现 provider 级别的增量拼装逻辑

### 6.4 取消与异常

如果 `CancellationToken` 被触发：

- 当前流会被中断
- 运行时会构造一个 `StopReason::Aborted` 的 assistant 消息
- 最终仍按统一消息结束流程收口

如果 stream 返回错误事件或异常，也会转换成带 `StopReason::Error` 的终结消息，而不是让外层状态失配。

---

## 7. 事件模型

整个运行时以 `AgentEvent` 为统一事件总线，核心事件如下：

- `AgentStart`
- `RunCompleted`
- `RunFailed`
- `RunAborted`
- `TurnStart`
- `TurnEnd`
- `MessageStart`
- `MessageUpdate`
- `MessageEnd`
- `ToolExecutionStart`
- `ToolExecutionUpdate`
- `ToolExecutionEnd`

低层已经不再提供“靠 `AgentEnd` 或 receiver close 推断完成”的终止模型；terminal event 本身就是运行完成语义。

### 7.1 典型无工具流程

```text
AgentStart
TurnStart
MessageStart(user)
MessageEnd(user)
MessageStart(assistant partial/final)
MessageUpdate(0..n)
MessageEnd(assistant)
TurnEnd
RunCompleted
```

### 7.2 典型有工具流程

```text
AgentStart
TurnStart
MessageStart(user)
MessageEnd(user)
MessageStart(assistant)
MessageUpdate(...)
MessageEnd(assistant with tool calls)
ToolExecutionStart
ToolExecutionUpdate(0..n)
ToolExecutionEnd
MessageStart(tool result)
MessageEnd(tool result)
TurnEnd
TurnStart
... 下一轮 assistant ...
RunCompleted
```

这些事件既是 UI 更新来源，也是高层 `Agent` 做状态归约的依据。

---

## 8. 工具执行架构

工具执行管道是本项目最重要的运行时特性之一。

### 8.1 执行步骤

单个工具调用经历以下阶段：

```text
查找工具
-> prepare_arguments
-> JSON Schema 校验
-> before_tool_call
-> execute
-> after_tool_call
-> 生成 ToolResultMessage 并发事件
```

### 8.2 参数准备与校验

工具可以通过 `prepare_arguments()` 对模型产出的原始 JSON 参数做兼容性修正。

随后运行时使用 `jsonschema` 做校验：

- schema 非法时返回错误结果
- 参数不匹配时返回错误结果
- 不会直接 panic 或中断整个 loop

这意味着“错误参数”被当作一种可观测的工具失败，而不是运行时崩溃。

### 8.3 Hook 语义

`before_tool_call`

- 可以读取 assistant 消息、tool call、上下文快照
- 拿到的是 `Arc<Mutex<serde_json::Value>>` 形式的已校验参数
- 可以原地改参数
- 返回 `block = true` 可阻止执行

`after_tool_call`

- 可读取最终结果、错误标记和上下文
- 可覆盖：
  - `content`
  - `details`
  - `is_error`

这个设计让调用方可以实现策略拦截、审计、脱敏、后处理和统一错误包装。

### 8.4 流式工具更新

工具执行时可以通过 `on_update` 上报中间结果。运行时会把这些更新转成 `ToolExecutionUpdate` 事件。

因此工具和 assistant 一样，都具备“流式可视化”的能力。

### 8.5 顺序与并发

支持两种模式：

| 模式 | 语义 |
|------|------|
| `Sequential` | 逐个准备并逐个执行 |
| `Parallel` | 先顺序准备全部工具，再并发执行可运行项，最后按原始顺序收束结果 |

`Parallel` 的关键点不是“谁先完成就先返回”，而是：

- 并发提升吞吐
- 结果顺序仍与 assistant 原始 tool call 顺序一致

这样既保留并发收益，也避免后续对话上下文因为工具结果乱序而失真。

---

## 9. 高层 Agent 设计

`agent.rs` 提供面向业务的状态化接口，是低层循环的包装器。

### 9.1 Agent 持有的状态

`AgentInner` 主要包含：

- `state`: 当前公开状态的可变内部表示
- `steering_queue`
- `follow_up_queue`
- `listeners`
- `cancel`
- `run_complete`
- 与运行时相关的配置副本

公开态由 `AgentState` 表达，主要字段包括：

- `system_prompt`
- `model`
- `thinking_level`
- `tools`
- `messages`
- `is_streaming`
- `streaming_message`
- `pending_tool_calls`
- `error_message`

### 9.2 生命周期入口

对外 API 主要有：

- `prompt_text()`
- `prompt()`
- `continue_run()`
- `abort()`
- `wait_for_idle()`
- `reset()`
- `subscribe()`
- `steer()`
- `follow_up()`

其中：

- `prompt()` 在已有活跃运行时会拒绝并返回 `AlreadyProcessing`
- `continue_run()` 在 assistant 结尾时，会优先尝试消化队列中的 steering / follow-up
- `abort()` 通过取消 token 中断当前执行

### 9.3 状态同步机制

高层 `Agent` 通过 `process_event()` 把 `AgentEvent` 规约成 `AgentState`：

- `MessageStart` / `MessageUpdate` 更新 `streaming_message`
- `MessageEnd` 把最终消息写入 transcript
- `ToolExecutionStart` / `ToolExecutionEnd` 增删 `pending_tool_calls`
- `RunFailed` 从 runtime error 回填 `error_message`
- `RunCompleted` / `RunFailed` / `RunAborted` 收口 streaming 状态

这使得高层状态并不是独立维护的另一套流程，而是“事件驱动的视图”。

### 9.4 监听器与 idle 语义

`subscribe()` 允许外部注册异步监听器。

重要语义是：

- `process_event()` 会在状态更新后，顺序等待每个监听器完成
- `wait_for_idle()` 只有在当前 run 完成且所有监听器对最终事件都处理完成后才返回

这个语义非常适合接持久化、日志、UI side effect 等需要“最终一致”的场景。

---

## 10. 队列与干预机制

项目实现了两类待处理消息队列：

- `steering_queue`
- `follow_up_queue`

内部使用 `PendingMessageQueue`，支持两种 drain 模式：

- `QueueMode::All`
- `QueueMode::OneAtATime`

### 10.1 Steering

`steer()` 用于在当前 Agent 正在工作时插入新的用户意图。

特点：

- 不会打断当前已经开始的 assistant 流
- 会在当前一轮结束、下一次调用 LLM 之前注入

### 10.2 Follow-up

`follow_up()` 用于安排“agent 原本结束后再做的事”。

特点：

- 只有当没有更多工具调用且没有 steering 待处理时才会被提取
- 适合排队式后续任务、总结、收尾动作

### 10.3 队列模式

`OneAtATime`：

- 每次只弹出一条消息
- 更接近交互式逐条推进

`All`：

- 一次性清空队列
- 更适合批量注入上下文

---

## 11. 并发与线程安全

当前实现主要基于以下并发原语：

- `Arc`
- `Mutex`
- `tokio::spawn`
- `tokio::sync::Notify`
- `tokio_util::sync::CancellationToken`

### 11.1 并发边界

- `Agent` 自身通过 `cancel: Mutex<Option<CancellationToken>>` 防止重入运行
- 工具并行模式下使用 `tokio::spawn` 并发执行
- 工具更新事件也可能通过独立 task 异步发射

### 11.2 一致性策略

实现上偏向“状态快照 + 事件顺序”而不是复杂锁分层：

- 状态修改集中在 `process_event()`
- 对监听器的调用放在状态锁外执行
- 对外公开状态使用 clone/snapshot

这让代码更容易维持行为一致性，也降低了业务方订阅器造成死锁的风险。

---

## 12. 关键设计取舍

### 12.1 用 `AgentMessage` 承载扩展消息

优点：

- 不把应用消息强塞进 LLM 协议
- 扩展能力明确

代价：

- 调用方必须认真实现 `convert_to_llm`

### 12.2 让 provider 负责 partial message 形状

优点：

- 低层循环更薄
- 更贴近真实 provider 响应

代价：

- `StreamFn` 的契约更严格，调用方必须提供合法 partial

### 12.3 并发执行、顺序收束工具结果

优点：

- 有并发收益
- 不破坏对话上下文顺序

代价：

- 最慢的工具仍会拖住整个回合的最终完成时间

### 12.4 将错误包装成消息

优点：

- 外部更容易统一处理失败场景
- transcript 语义完整

代价：

- 调用方需要区分“模型自然回复”与“运行时错误终结消息”

---

## 13. 典型扩展点

### 13.1 自定义消息类型

通过 `AgentMessage::Custom` 承载业务消息，再在 `convert_to_llm` 中过滤或转换。

适用场景：

- UI 专用消息
- 系统通知
- 会话元数据
- 压缩摘要

### 13.2 上下文裁剪

通过 `transform_context` 实现：

- token budget 控制
- 摘要压缩
- 长会话修剪
- 注入外部检索上下文

### 13.3 工具审计或策略拦截

通过 `before_tool_call` / `after_tool_call` 可实现：

- 参数白名单
- 权限判断
- 结果脱敏
- 审计日志
- 错误包装

### 13.4 Provider 适配

调用方只要实现 `StreamFn`，即可接入不同模型服务。

运行时并不绑定任何具体厂商 SDK，这是项目最关键的可移植性来源。

---

## 14. 测试与验证现状

仓库下已有较完整的测试目录，覆盖方向包括：

- `public_api`
- `type_parity`
- `agent_loop_core`
- `agent_loop_tools`
- `agent_state`
- `agent_e2e`

从目录命名看，当前测试策略主要围绕“对外行为 parity”而不是私有实现细节，这与本项目架构分层是匹配的。

---

## 15. 模块调用时序图

这一节从“真实接入时最常见的几条路径”来解释模块之间如何协作。

### 15.1 普通 prompt 时序

```text
调用方
  -> Agent::prompt_text / prompt
  -> Agent::run_with_lifecycle
  -> Agent::create_context_snapshot
  -> Agent::create_loop_config
  -> run_agent_loop
  -> run_loop
  -> stream_assistant_response
  -> StreamFn(model, llm_context, stream_options)
  -> StreamEvent::Start / ... / Done
  -> EventEmitter.emit(AgentEvent)
  -> Agent::process_event
  -> listeners
  -> AgentState 收口
```

关键点：

- 高层 `Agent` 不直接操作模型，而是把所有执行委托给低层 loop。
- 低层 loop 不直接维护公开状态，而是只发事件。
- `AgentState` 是 `AgentEvent` 的归约结果。

### 15.2 带工具调用的完整时序

```text
User prompt
  -> assistant response (含 ToolCall)
  -> execute_tool_calls
  -> prepare_tool_call
  -> tool.prepare_arguments
  -> validate_tool_arguments
  -> before_tool_call
  -> tool.execute
  -> ToolExecutionUpdate*
  -> after_tool_call
  -> emit ToolExecutionEnd
  -> emit ToolResult MessageStart/MessageEnd
  -> 下一轮 LLM 调用
  -> assistant 总结工具结果
```

其中 `ToolExecutionUpdate*` 表示 0 到多次中间更新。

### 15.3 steering / follow-up 时序

```text
Agent 正在运行
  -> 外部调用 steer() 或 follow_up()
  -> message 进入对应队列
  -> 当前轮 assistant / tool call 完成
  -> run_loop 检查 steering 队列
  -> 注入 steering 消息并继续下一轮
  -> 当内层循环本应结束时
  -> run_loop 检查 follow-up 队列
  -> 若有 follow-up，则开启新一轮外层循环
```

语义差异：

- `steer` 偏“立即干预”
- `follow_up` 偏“尾部排队”

### 15.4 abort 时序

```text
外部调用 Agent::abort()
  -> CancellationToken.cancel()
  -> stream_assistant_response / tool.execute 感知取消
  -> 生成 Aborted assistant message
  -> MessageEnd
  -> TurnEnd
  -> RunAborted
  -> Agent::finish_run
```

这保证了即使被取消，最终 transcript 和状态仍然闭环。

---

## 16. 示例代码接入说明

这一节给出更贴近真实接入的说明，目标是让调用方知道“代码应该放哪、先实现什么、哪些点最容易出错”。

### 16.1 最小 Agent 组装

推荐的初始化思路：

1. 先定义 `Model`
2. 先让 `StreamFn` 支持最简单的 `Done` 终结事件
3. 再补工具和 streaming 增量事件
4. 最后再加 `transform_context`、hooks 和 payload 改写

最小组装示例：

```rust
use std::sync::Arc;

use oh_my_agentloop::{
    Agent, AgentOptions, AssistantMessage, ContentBlock, InitialAgentState, Model, ModelCost,
    StopReason, StreamEvent, StreamFn, TextContent, ThinkingLevel, Usage,
};

fn model() -> Model {
    Model {
        id: "mock".into(),
        name: "mock".into(),
        api: "openai-responses".into(),
        provider: "openai".into(),
        base_url: "https://example.invalid".into(),
        reasoning: false,
        input: vec!["text".into()],
        cost: ModelCost::default(),
        context_window: 8192,
        max_tokens: 2048,
    }
}

fn stream_fn() -> StreamFn {
    Arc::new(move |model, _ctx, _opts| {
        Box::pin(async move {
            let final_message = AssistantMessage {
                content: vec![ContentBlock::Text(TextContent {
                    text: "hello".into(),
                    text_signature: None,
                })],
                model: model.id.clone(),
                provider: model.provider.clone(),
                api: model.api.clone(),
                response_id: None,
                stop_reason: StopReason::Stop,
                error_message: None,
                usage: Usage::default(),
                timestamp: 0,
            };

            let stream = futures::stream::iter(vec![Ok(StreamEvent::Done {
                message: final_message,
            })]);

            Ok(Box::pin(stream) as oh_my_agentloop::LlmEventStream)
        })
    })
}

async fn build_agent() -> Agent {
    Agent::new(AgentOptions {
        initial_state: Some(InitialAgentState {
            system_prompt: Some("You are helpful.".into()),
            model: Some(model()),
            thinking_level: Some(ThinkingLevel::Off),
            tools: Some(vec![]),
            messages: None,
        }),
        convert_to_llm: None,
        transform_context: None,
        stream_fn: stream_fn(),
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
    })
}
```

### 16.2 自定义 `convert_to_llm`

如果你的 transcript 中会混入业务消息，必须自定义转换函数：

```rust
use std::sync::Arc;
use oh_my_agentloop::{AgentMessage, ConvertToLlmFn, Message};

fn convert_to_llm() -> ConvertToLlmFn {
    Arc::new(|messages: Vec<AgentMessage>| {
        Box::pin(async move {
            messages
                .into_iter()
                .filter_map(|m| m.into_message())
                .collect::<Vec<Message>>()
        })
    })
}
```

这也是把 `AgentMessage::Custom` 过滤出 LLM 边界的标准方式。

### 16.3 订阅事件做 UI / 持久化

接入方最推荐的扩展点是事件订阅，而不是侵入内部执行流程。

```rust
let _sub = agent.subscribe(|event, _cancel| async move {
    match event {
        oh_my_agentloop::AgentEvent::MessageUpdate { .. } => {
            // 推送流式 UI
        }
        oh_my_agentloop::AgentEvent::ToolExecutionStart { tool_name, .. } => {
            println!("tool start: {}", tool_name);
        }
        oh_my_agentloop::AgentEvent::RunCompleted { .. }
        | oh_my_agentloop::AgentEvent::RunFailed { .. }
        | oh_my_agentloop::AgentEvent::RunAborted { .. } => {
            println!("persist final transcript");
        }
        _ => {}
    }
});
```

### 16.4 工具接入示例

最典型的工具是纯 JSON 输入、文本输出：

```rust
use async_trait::async_trait;
use oh_my_agentloop::{AgentError, AgentTool, AgentToolResult, ContentBlock, TextContent};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

struct EchoTool;

#[async_trait]
impl AgentTool for EchoTool {
    fn name(&self) -> &str { "echo" }
    fn label(&self) -> &str { "Echo" }
    fn description(&self) -> &str { "Echo input" }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "value": { "type": "string" }
            },
            "required": ["value"],
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
        let value = params["value"].as_str().unwrap_or_default();
        Ok(AgentToolResult {
            content: vec![ContentBlock::Text(TextContent {
                text: format!("echo: {value}"),
                text_signature: None,
            })],
            details: Some(json!({ "echoed": value })),
        })
    }
}
```

### 16.5 带流式更新的工具

如果工具执行时间长，建议上报中间结果：

```rust
async fn execute(
    &self,
    _tool_call_id: &str,
    params: Value,
    _cancel: CancellationToken,
    on_update: Option<Box<dyn Fn(AgentToolResult) + Send + Sync>>,
) -> Result<AgentToolResult, AgentError> {
    if let Some(on_update) = on_update {
        on_update(AgentToolResult {
            content: vec![ContentBlock::Text(TextContent {
                text: "processing...".into(),
                text_signature: None,
            })],
            details: None,
        });
    }

    Ok(AgentToolResult {
        content: vec![ContentBlock::Text(TextContent {
            text: format!("done: {}", params),
            text_signature: None,
        })],
        details: None,
    })
}
```

运行时会自动把它映射成 `ToolExecutionUpdate` 事件。

### 16.6 长会话建议

当 transcript 越来越长时，建议尽早接入 `transform_context`：

- 保留最近 N 轮
- 把更早消息压缩成摘要
- 注入检索结果而不是盲目堆满历史

否则随着 `messages` 增长，provider 请求成本和延迟都会持续上升。

---

## 17. 总结

`oh-my-agentloop` 的整体接入方式可以概括为：

- 用 `StreamFn` 接模型
- 用 `AgentTool` 接能力
- 用 `Agent` 做运行时管理
- 用 `AgentEvent` 做观测和副作用

如果你只记住一句话，那就是：

“低层 loop 负责执行，高层 Agent 负责状态，所有可观测行为都尽量走事件总线。”

---

## 18. 演进方向

`oh-my-agentloop` 的本质不是一个“带模型能力的应用”，而是一个“可嵌入的 agent runtime 内核”。

它通过以下三层分工保证可扩展性：

- `types.rs` 定义清晰的协议与边界
- `agent_loop.rs` 实现确定性的低层执行时序
- `agent.rs` 提供有状态、高可用的业务接入接口

如果后续要继续演进，最自然的方向有两个：

- 新增 provider / proxy 侧能力，但保持 `StreamFn` 边界稳定
- 在不破坏事件顺序与队列语义的前提下扩展更高层的 session、persistence 或 orchestration 能力

---

*文档更新时间：2026-04-09*
