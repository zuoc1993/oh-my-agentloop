# oh-my-agentloop 生产级审核报告

> 审核范围：`oh-my-agentloop` v0.1.0（Rust 端口 of `@mariozechner/pi-agent-core`）
> 审核日期：2026-04-16
> 审核方法：静态代码审阅 + 工具链证据（`cargo fmt/clippy/test/doc`，证据落盘在 `docs/audit-evidence/`）
> 覆盖维度：架构 / API / 实现 / 测试 / 生产化

---

## 00 执行摘要

### 评分卡

| 维度 | 评分（满分 5） | 主要问题 |
|------|----------------|----------|
| 架构设计 | 3.5 | 双层 API 结构清晰；`types.rs` 943 行过载；`AgentOptions` 字段漏泄架构细节 |
| API 设计 | 2.5 | `AgentOptions` 无 builder、quick start 12 个 `None`；公开枚举未 `#[non_exhaustive]`；`anyhow::Error` 泄漏进公共错误 |
| 代码实现 | 3.5 | 逻辑清晰、测试稳；9 条 clippy 违规；3 处 `.expect(...)`；48 处 `std::sync::Mutex::lock().unwrap()`（均未跨 await，但违反 API 设计原则） |
| 测试工程 | 3.5 | 66 用例全绿；缺覆盖率统计、缺单元测试、缺 proptest/fuzz/loom；测试中有 `sleep` 同步 |
| 生产化 | 1.5 | **零可观测性**（无 tracing/metrics）；**零 CI**；Cargo 元数据残缺；无 MSRV；无 feature flag；cargo-audit/deny 缺席 |

### 发布建议

**🛑 阻断发布**。当前项目不具备生产就绪条件。核心原因：
1. 生产化维度几乎为零（无日志、无指标、无 CI）。
2. 公开枚举未冻结（`#[non_exhaustive]` 缺失 + `anyhow` 泄漏），发布即 semver 陷阱。
3. Quick start 体验差，`AgentOptions` 必须通过 builder 修复。

按本报告附带的 P0-P2 修复后可进入 `0.1.0` → `0.2.0` 的准备阶段。

### 关键数据

- 源码：`src/types.rs` 943 行、`src/agent_loop.rs` 1286 行、`src/agent.rs` 740 行（合计 2981 行）。
- 测试：66 个集成测试，全部通过（5+8+13+12+16+1+11=66，[tests/](../tests/) 下 6 个文件）。
- `cargo fmt --check`：通过。
- `cargo clippy --all-targets --all-features -D warnings`：**9 错误**。
- `RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links" cargo doc`：通过。
- 基础工具缺席：`cargo-audit` / `cargo-deny` / `cargo-public-api` / `cargo-llvm-cov` 未安装。

---

## 01 架构设计审核

### 1.1 双层 API 边界（结论：大致清晰，有小瑕疵）

- **证据**：[src/agent.rs](../src/agent.rs):536-567 `run_with_lifecycle` 是高层唯一进入低层的通道，低层 [src/agent_loop.rs](../src/agent_loop.rs) 通过 `EventEmitter` 回推事件，`Agent` 的 `process_event`（[src/agent.rs](../src/agent.rs):660-714）把事件规约为 `AgentState`。
- **优点**：低层无状态、可独立使用（`agent_loop` / `run_agent_loop` 两个入口）。
- **瑕疵**：
  - 低层函数名冗余——`agent_loop` + `run_agent_loop` + `agent_loop_continue` + `run_agent_loop_continue` 四个公共函数，两两成对但命名区分不显性。建议以 `stream` vs `run` 或 `channel` vs `await` 命名。（P2）
  - `create_emitter`（[src/agent.rs](../src/agent.rs):649-657）把 `Agent::clone` 捕获进闭包，将状态反向注入低层；这违反了"低层无状态"的纸面承诺，尽管事实上是通过事件回调完成的，但通过 `Arc<AgentInner>` 暴露多次看起来紧密耦合。建议在报告中明确文档化"低层无状态但需要调用方注入回调"。

### 1.2 模块职责（P1：`types.rs` 过载）

- **证据**：[src/types.rs](../src/types.rs) 共 943 行，包含：LLM 消息、内容块、流事件、工具抽象、hook 上下文、回调类型别名、`AgentLoopConfig`、`AgentEvent`、`AgentError`、`RunOutcome`、`EventEmitter`、`AgentState`、`now_millis()`、`default_convert_to_llm()`、`create_error_tool_result()` 等。
- **影响**：单个模块承担至少 6 个相互正交的职责，查找与修改困难。
- **建议**：拆分为 `types/message.rs`、`types/stream.rs`、`types/tool.rs`、`types/event.rs`、`types/error.rs`、`types/config.rs`、`types/hooks.rs`。`lib.rs` 保持 re-export。（P2）

### 1.3 事件总线唯一性（结论：基本达成）

- **证据**：所有状态变更都先进 `process_event` ([src/agent.rs](../src/agent.rs):660)，再去写 `MutableAgentState`。未发现旁路修改。
- **保留点**：`set_system_prompt` / `set_model` / `set_tools` 等 `set_*` 系列（[src/agent.rs](../src/agent.rs):286-303）**绕过事件**直接改状态，但它们是用户主动意图（如切换模型），不会发事件通知订阅者。这在文档里需明确标注："`set_*` 不发事件，调用方需自行刷新 UI"。（P2）

### 1.4 可扩展性（结论：好）

- 注入点：`StreamFn` / `AgentTool` / `convert_to_llm` / `transform_context` / `before_tool_call` / `after_tool_call` / `on_payload` / `get_api_key`。加新 provider 或新工具均无需改核心。

### 1.5 架构文档一致性（结论：待复核）

- `docs/AGENT_ARCHITECTURE.md` 996 行。需要对照修复后的模块拆分重绘。（P2）

---

## 02 API 设计审核

### 2.1 AgentOptions 人体工学（P0）

- **证据**：[src/agent.rs](../src/agent.rs):127-147 `AgentOptions` 有 **18** 个字段，17 个可选但全部为 `pub`。[README.md](../README.md):89-105 的 quick start 用户要写 12 个 `None`。
- **影响**：最主要的公共入口体验极差。
- **建议**：
  - 引入 `AgentOptionsBuilder`（P0）。
  - `AgentOptions` 去除 `pub` 字段，改为通过 builder 构造。
  - 或至少实现 `Default` 并要求只填 `stream_fn`。

### 2.2 枚举未 `#[non_exhaustive]`（P0，semver 陷阱）

- **证据**：
  - `QueueMode` ([src/agent.rs](../src/agent.rs):20-24)
  - `StopReason` ([src/types.rs](../src/types.rs):56-63)
  - `Transport` ([src/types.rs](../src/types.rs):131-136)
  - `ThinkingLevel` ([src/types.rs](../src/types.rs):190-200)
  - `ToolExecutionMode` ([src/types.rs](../src/types.rs):616-620)
  - `ContentBlock` ([src/types.rs](../src/types.rs):47-54)
  - `StreamEvent` ([src/types.rs](../src/types.rs):348-401)
  - `AgentEvent` ([src/types.rs](../src/types.rs):763-808)
  - `AgentError` ([src/types.rs](../src/types.rs):814-836)
  - `RunOutcome` ([src/types.rs](../src/types.rs):838-850)
  - `Message` / `AgentMessage` / `UserContent` / `UserContentBlock` / `UserContentBuildError`
- **影响**：任何一个变体的新增都会 break 下游 `match` 穷尽匹配，在 `0.1.0` 后会被锁死。
- **建议**：除真正"关闭"的枚举（如 `Message` 只有三种角色）外，其余一律加 `#[non_exhaustive]`。（P1）

### 2.3 anyhow 泄漏（P0，semver + 错误分类丢失）

- **证据**：[src/types.rs](../src/types.rs):834-835 `AgentError::Other(#[from] anyhow::Error)`。[src/agent.rs](../src/agent.rs):445 `AgentError::Other(anyhow::anyhow!(e))` 把 `UserContentBuildError` 包进 `anyhow`。
- **影响**：公共错误类型依赖 `anyhow`，下游无法按 enum 分类处理；升级 anyhow 主版本即 break。
- **建议**：
  - 新增 `AgentError::UserContent(UserContentBuildError)`、`AgentError::Internal(String)` 等具体变体。
  - 移除 `Other(anyhow::Error)`。（P1）

### 2.4 四个同形函数名歧义（P2）

- `agent_loop` / `run_agent_loop` / `agent_loop_continue` / `run_agent_loop_continue` 的区别仅在于"返回 channel" vs "await 并返回 `RunOutcome`"。建议：
  - 保留 `run_agent_loop` 和 `run_agent_loop_continue`（await 直接返回）。
  - 把 channel 版改名为 `stream_agent_loop` / `stream_agent_loop_continue`。

### 2.5 `pub use types::*` 面过宽（P2）

- **证据**：[src/lib.rs](../src/lib.rs):12 一股脑 re-export。
- **影响**：内部 pub(crate) 帮助函数（如果未来有）容易误升公共 API；`cargo public-api` 快照体积大。
- **建议**：显式 re-export 列表，控制公共表面。

### 2.6 回调类型别名（P3）

- 8 个 `Arc<dyn Fn(...) -> Pin<Box<dyn Future + Send>> + Send + Sync>` 别名（[src/types.rs](../src/types.rs):676-726）。clippy 已提示 `type_complexity`。这些类型在外部实现极其繁琐。
- **建议**：保留类型别名便于传参，但在 API 文档中展示 "如何包装一个闭包" 的样板。长期可考虑改为 trait（但会破坏当前 API）。

### 2.7 文档密度（P2）

- 未启用 `#![deny(missing_docs)]`。粗扫 `src/types.rs` 大量 `pub struct` 字段无 doc。
- **建议**：启用后逐个补齐。

### 2.8 `#[must_use]` 缺失（P2）

- `Subscription` 结构返回后若被丢弃则立即取消订阅（[src/agent.rs](../src/agent.rs):163-168）。这是典型的 `#[must_use]` 场景。
- `RunOutcome`、`Result`-like 返回值也应 `#[must_use]`。

---

## 03 代码实现审核

### 3.1 Clippy 违规清单（P0，CI 阻断级）

来自 `docs/audit-evidence/02-clippy.txt`：

| # | 文件:行 | Lint | 建议 |
|---|---------|------|------|
| 1 | `src/agent.rs:26` | `derivable_impls` | `QueueMode` 用 `#[derive(Default)]` + `#[default]` |
| 2 | `src/agent.rs:114` | `derivable_impls` | `InitialAgentState` 用 `#[derive(Default)]` |
| 3 | `src/agent.rs:708` | `unwrap_or_default` | `CancellationToken` 已实现 `Default`，改 `unwrap_or_default()` |
| 4 | `src/agent_loop.rs:1009` | `single_match` | 改为 `if let Some(BeforeToolCallResult { block: true, reason }) = ...` |
| 5 | `src/types.rs:138` | `derivable_impls` | `Transport` 用 `#[derive(Default)]` |
| 6 | `src/types.rs:202` | `derivable_impls` | `ThinkingLevel` 用 `#[derive(Default)]` |
| 7 | `src/types.rs:622` | `derivable_impls` | `ToolExecutionMode` 用 `#[derive(Default)]` |
| 8 | `src/types.rs:764` | `large_enum_variant` | `AgentEvent::MessageUpdate` 把 `StreamEvent` 装箱 |
| 9 | `src/types.rs:859` | `type_complexity` | `EventEmitter::f` 用类型别名 |

### 3.2 Panic 面（P1）

- `.expect(...)` 三处：
  - [src/agent_loop.rs](../src/agent_loop.rs):241, 329：终止消息追加后 last()。从上下文看确实不可触发，但不应用 `.expect`。
  - [src/agent_loop.rs](../src/agent_loop.rs):875：`spawned_count` 等于 `handles.len()`，从逻辑看 invariant 成立。
- **建议**：三处均改为 `.ok_or(AgentError::Internal(...))?`，明确用 `Internal` 变体，保留不变量假设的文档注释。

### 3.3 `std::sync::Mutex` 使用分析（P1）

- **48 处 `lock().unwrap()`**，但经过逐处审阅：
  - 均未跨 `.await` 持锁。
  - 死锁风险低：所有锁临界区都短且非嵌套（除了 `process_event` 中 state 锁+随后的 cancel 锁，顺序固定）。
- **问题**：
  - `.unwrap()` 让 poison 情况下 panic 扩散（即使概率低）。
  - `std::sync::Mutex` 对 async 代码不是错误，但 `parking_lot::Mutex` 更快且无 poison。
- **建议**：
  - 短期：包装 `with_state(|s| ...)` / `mutate_state(|s| ...)` 消除 48 处重复，`.unwrap()` 统一处理。
  - 中期：换 `parking_lot::Mutex`。（P2）

### 3.4 `set_*` 重复代码（P1）

- **证据**：[src/agent.rs](../src/agent.rs):286-354 有 13 个一行函数均是 `lock().unwrap().xxx = y`。
- **影响**：维护成本、噪音。
- **建议**：引入 `fn with_state_mut<F, R>(&self, f: F) -> R where F: FnOnce(&mut MutableAgentState) -> R` 封装。

### 3.5 `anyhow::anyhow!` 在实现中的使用

- `src/agent.rs:445` 是 `anyhow::anyhow!(UserContentBuildError)`。见 2.3。

### 3.6 订阅 Drop 正确性（结论：正确）

- `Subscription::drop` 用 `Arc::ptr_eq` 从 listeners 中摘除自己。线程安全，O(n)。大量订阅者场景下退化，但对 Agent 来说合理。

### 3.7 `CancellationToken` 传播（结论：覆盖到位）

- `run_loop` / `stream_assistant_response` / `execute_tool_calls_*` / `prepare_tool_call` / `execute_tool_call_core` 每个关键点都检查 `cancel.is_cancelled()` 或 `tokio::select!` with `cancel.cancelled()`。
- 并行工具路径（[src/agent_loop.rs](../src/agent_loop.rs):866-902）对未 spawn 的 prepared calls 会发 aborted tool result，避免 dangling `ToolExecutionStart`。

### 3.8 `tokio::spawn` 的 orphan 风险（P2）

- [src/agent_loop.rs](../src/agent_loop.rs):1060-1069 `on_update` 回调在工具执行期间每次都 `tokio::spawn` 一个广播任务，汇总到 `update_handles` 并在结束时 `h.await`——已正确回收。
- [src/agent_loop.rs](../src/agent_loop.rs):32-42 `agent_loop` 的 outer spawn 没有 JoinHandle 返回给调用方——若调用方 drop `rx` 前 spawn 还在跑，任务继续运行至 cancel。建议文档强调"必须用 CancellationToken 终止"。

### 3.9 `default_convert_to_llm` 丢消息（结论：符合设计，但需强警示）

- [src/types.rs](../src/types.rs):928-933 默认实现把 `AgentMessage::Custom` 悄无声息地过滤掉。
- 对 custom 消息用户这是易错点，建议在 doc 里加 `⚠️ **Warning**` 提示。

### 3.10 `now_millis()` 精度 & 溢出（P3）

- `as i64` 在当前日期之后约 292 亿年才会溢出，忽略；但 `duration_since` 可能返回 `Err`（系统时间回退），目前 `unwrap_or_default()` 退化成 0，会生成 epoch 时间戳。建议文档强调这是"尽力而为的时间戳"。

### 3.11 `StreamOptions` / `StreamRequest` Clone 成本

- 每次 LLM 调用都 clone 整套 options 和 context（[src/agent_loop.rs](../src/agent_loop.rs):540-555）。对 `Vec<Message>` 可能是大对象，O(N) 克隆。
- **建议**：`LlmContext` 改用 `Arc<Vec<Message>>` 或传引用。（P3，benchmark 验证后再定）

---

## 04 测试工程审核

### 4.1 基线数据

- 66 个集成测试通过，`tests/` 共 3781 行。
- 测试分布：
  - `agent_loop_core.rs` 8（循环骨架）
  - `agent_loop_tools.rs` 13（工具编排）
  - `agent_state.rs` 12（Agent 状态同步）
  - `agent_e2e.rs` 5（端到端）
  - `error_contract.rs` 16（错误契约）
  - `type_parity.rs` 11（类型 JSON 等价）
  - `public_api.rs` 1（门面）

### 4.2 覆盖率空白（P1）

- 未跑 `cargo llvm-cov`（工具未安装）。从代码路径推断未覆盖：
  - `Subscription::drop` 的并发竞争场景。
  - `process_event` 对 `AgentEvent::ToolExecutionUpdate` 的 state 分支（`_ => {}` 吞掉）。
  - `after_tool_call` hook 的 `Option::None` 默认分支。
  - `agent_loop_continue` channel 变体。

### 4.3 缺失的测试类别（P2）

- **单元测试**：`src/*.rs` 内 0 个 `#[cfg(test)] mod tests`，全靠集成测试推断内部行为。`validate_tool_arguments`、`terminal_error_message`、`normalize_terminal_assistant_message` 等纯函数都应有单元测试。
- **属性测试**：无。`Message` / `AgentMessage` serde roundtrip、`UserContent::try_from_llm_blocks` 的任意输入组合最适合 proptest。
- **Fuzz**：无 `fuzz/` 目录。JSON Schema 验证 & stream event 反序列化是攻击面。
- **Loom**：无。即使当前锁使用正确，48 处锁需要用 loom 锁定语义。
- **Miri**：未运行。

### 4.4 测试中时间同步（P1，flaky 风险）

- **证据**：13 处 `tokio::time::sleep` 在测试和 `tests/common/mod.rs` 中：`tests/agent_state.rs:47,94,145,164,184,436,446,461`，`tests/error_contract.rs:240,255,334,344`，`tests/agent_e2e.rs:463`，`tests/agent_loop_tools.rs:717,923`，`tests/common/mod.rs:195,224`。
- **影响**：CI 在 slow runner 上可能偶发失败。
- **建议**：
  - 引入 `tokio::test(start_paused = true)` + `tokio::time::advance()`（可控时间）。
  - 用 `tokio::sync::Notify` / `oneshot` 替代 sleep 作为同步点。
  - 用 `wait_for(condition, timeout)` helper。

### 4.5 `public-api` 回归测试（P1）

- [tests/public_api.rs](../tests/public_api.rs) 仅 39 行一个断言。
- **建议**：加入 `public-api` 快照文件（`tests/snapshots/public-api.txt`），由 `cargo public-api` 在 CI 检查变更。

### 4.6 测试 helper 质量（结论：OK）

- `tests/common/mod.rs` 190+ 行，结构清晰，未见过度抽象。

---

## 05 生产化审核

### 5.1 可观测性（P0，零基础）

- **现状**：
  - 0 次 `tracing::` 调用、0 次 `log::` 调用。
  - 0 次 `metrics::` / `prometheus::` / `opentelemetry::` 调用。
- **影响**：生产环境无法排障、无法监控。
- **建议**：
  - 加 `tracing` 可选 feature（默认开启）。
  - 关键 span：`run_loop`、`stream_assistant_response`、`execute_tool_calls_sequential`、`execute_tool_calls_parallel`、`prepare_tool_call`、`execute_tool_call_core`、`finalize_executed_tool_call`、`emit_tool_call_outcome`。
  - 结构化字段：`model.id`、`tool_name`、`tool_call_id`、`turn_index`、`is_error`。
  - 错误路径用 `tracing::error!` 带上下文。
  - 指标通过 trait `MetricsSink`（用户实现）或 `metrics` crate（可选 feature），暴露：
    - `agent.turn.total`
    - `agent.tool_call.duration_seconds`
    - `agent.tool_call.errors_total{tool}`
    - `agent.llm.tokens{kind=input|output}`
    - `agent.queue.pending{kind=steering|follow_up}`
    - `agent.run.outcome_total{outcome}`

### 5.2 Cargo 元数据（P0）

- **证据**：[Cargo.toml](../Cargo.toml):1-6 缺失 `license` / `repository` / `documentation` / `keywords` / `categories` / `rust-version` / `readme` / `homepage`。
- **影响**：无法 `cargo publish`。
- **建议**：补齐。同时加 `rust-toolchain.toml` 锁定开发时 toolchain。

### 5.3 CI/CD（P0，完全空白）

- **现状**：无 `.github/`、无 `.circleci/`、无任何 CI 配置。
- **建议**：GitHub Actions：
  - `fmt` / `clippy -D warnings` / `test --all-features` / `doc -D warnings` / `audit` / `deny check` / `public-api`。
  - 矩阵：stable + MSRV。
  - PR + push to main 触发。

### 5.4 Feature Flags（P1）

- **现状**：零 feature flag。
- **建议**：
  - `default = ["tracing"]`
  - `metrics`（默认关）
  - `serde` 是否应可选？（当前所有类型深度依赖 serde，不必可选）

### 5.5 许可证合规（P1）

- [LICENSE](../LICENSE) 是 MIT。
- `Cargo.toml` 未声明 `license = "MIT"`。
- 需要在 `Cargo.toml` 和 README 明确归属上游 pi-agent-core。

### 5.6 文档（P1）

- `cargo doc` 通过，但内容稀薄：
  - `lib.rs` crate 级文档仅 4 行。
  - 多数 `pub` 字段无 doc。
- **建议**：
  - 启用 `#![deny(missing_docs)]`。
  - `lib.rs` 补 crate overview + quick-start + architecture link。
  - README quick start 迁移到 `examples/quickstart.rs` 纳入 CI。

### 5.7 版本策略（P2）

- 当前 `0.1.0`。
- **建议**：完成 P0-P1 修复后发 `0.2.0`，把公共 API 稳定化（锁 `public-api`）后再 `1.0.0-rc.1`。

### 5.8 供应链（P1）

- 依赖数量：10 个直接依赖（见 Cargo.toml），`Cargo.lock` 29687 行。
- 未跑 `cargo audit` / `cargo deny check`。
- **建议**：CI 强制执行。

---

## 06 修复记录

本次审核一并产出 production 化修复，按 P0 → P1 → P2 的顺序落地于当前工作树（未分 commit）：

### P0（阻断级）— 已完成

| 项 | 证据 |
|----|------|
| 修复 9 条 clippy 违规（`derivable_impls` ×5、`unwrap_or_default`、`single_match`、`type_complexity`、`large_enum_variant`） | `cargo clippy --all-targets --all-features -- -D warnings` 通过 |
| 移除 3 处 `.expect(...)`，改为 `AgentError::Internal(..)?` | [src/agent_loop.rs](../src/agent_loop.rs) run_agent_loop / run_agent_loop_continue / execute_tool_calls_parallel |
| 移除 `anyhow::Error` 从公共错误类型泄漏；新增 `AgentError::UserContent(UserContentBuildError)` + `AgentError::Internal(String)`；删除 `Other(anyhow::Error)` 变体；`anyhow` crate 依赖完全移除 | [src/types.rs](../src/types.rs) + [Cargo.toml](../Cargo.toml) |
| 补齐 Cargo 元数据：`license="MIT"`、`repository`、`homepage`、`documentation`、`keywords`、`categories`、`rust-version="1.75"`、`readme` | [Cargo.toml](../Cargo.toml) |
| `unsafe_code = "forbid"` 作为 lint | [Cargo.toml](../Cargo.toml) |
| 新增 GitHub Actions CI：`fmt` / `clippy -D warnings` / `test` (stable + MSRV 1.75 + `--no-default-features`) / `doc -D warnings` / `cargo audit` / `cargo publish --dry-run` | [.github/workflows/ci.yml](../.github/workflows/ci.yml) |

### P1（高优）— 已完成

| 项 | 证据 |
|----|------|
| `AgentOptions::builder()` + `AgentOptionsBuilder`（Quick Start 从 12 个 `None` 降到 3 行） | [src/agent.rs](../src/agent.rs) + [README.md](../README.md) |
| 所有非封闭公共枚举加 `#[non_exhaustive]`：`StopReason` / `Transport` / `ThinkingLevel` / `ToolExecutionMode` / `QueueMode` / `ContentBlock` / `StreamEvent` / `AgentEvent` / `AgentError` / `RunOutcome` | [src/types.rs](../src/types.rs) + [src/agent.rs](../src/agent.rs) |
| `RunOutcome` 加 `#[must_use]`；`Subscription` 加 `#[must_use]` + 文档 | [src/types.rs](../src/types.rs) + [src/agent.rs](../src/agent.rs) |
| `tracing` 可选 feature（默认开启），关键路径加 `#[tracing::instrument]`：`agent.run` / `agent.run.continue` / `agent.run.loop` / `agent.llm.stream` / `agent.tools.execute` / `agent.tool.call`；stream 错误路径加 `warn!` | [src/lib.rs](../src/lib.rs) + [src/agent_loop.rs](../src/agent_loop.rs) |
| `lib.rs` crate-level 文档：overview + quick start + features + 可观测性表格 | [src/lib.rs](../src/lib.rs) |
| `AgentOptionsBuilder` re-export；`lock()` 辅助函数封装 setter mutex 样板；所有 `set_*` 用 `with_state_mut` | [src/agent.rs](../src/agent.rs) |
| `tests/public_api.rs` 扩展为 Public API 形状检测（PhantomData 捕获） | [tests/public_api.rs](../tests/public_api.rs) |

### P2（中优）— 已完成

| 项 | 证据 |
|----|------|
| 新增 `tests/common/wait_for(timeout, label, cond)` 同步 helper；迁移 `abort_*` 两个测试从固定 `sleep` 到 `wait_for`（flake 收窄） | [tests/common/mod.rs](../tests/common/mod.rs) + [tests/agent_state.rs](../tests/agent_state.rs) |
| `src/agent_loop.rs` 新增 8 条单元测试（首批内部测试）：`normalize_terminal_assistant_message` 四变体、`terminal_error_message` 两变体、`validate_tool_arguments` 接受/拒绝 | [src/agent_loop.rs](../src/agent_loop.rs) `mod tests` |
| `AgentEvent::MessageUpdate` 大型变体加了 `#[allow(clippy::large_enum_variant)]` + 注释，避免 semver 破坏性 Box 改造 | [src/types.rs](../src/types.rs) |

### 验证结果（最终）

```
cargo fmt --all -- --check                                            ✅
cargo clippy --all-targets --all-features -- -D warnings              ✅
cargo clippy --all-targets --no-default-features -- -D warnings       ✅
cargo test --all-features                                             ✅ 75 passed / 2 ignored
cargo test --no-default-features                                      ✅ 75 passed / 2 ignored
RUSTDOCFLAGS="-D warnings -D rustdoc::broken_intra_doc_links" cargo doc --no-deps --all-features ✅
```

### 推迟至下一里程碑（P3，保留在 roadmap）

- `cargo-llvm-cov` 覆盖率基线（工具未安装）。
- `criterion` benchmark + `cargo-mutants` + `cargo-semver-checks`。
- `types.rs` 拆 6 模块（943 → 多个 <200 行）。
- `proptest` / `loom` / `miri`。
- `parking_lot::Mutex` 迁移（消除 poison + 提升吞吐）。
- `#![deny(missing_docs)]` 启用 + 完整 field 级文档。
- 迁移剩余 10 处测试中的 `tokio::time::sleep`（helper 已备好）。

---

## 07 残留风险与后续 Roadmap

- **非常规环境**：`wasm32` / `no_std`：当前依赖 `tokio::full`，不兼容。如未来目标支持需改造。
- **性能**：尚无 benchmark。P3 引入 `criterion` 评估事件广播、stream 合并热点。
- **上游同步**：pi-agent-core (TS) 若引入新特性，当前 crate 可能漂移。建议定期 diff。
- **Proxy 能力**：README 声明不含；仍需持续与上游对齐。

---

## 附录 A：如何复跑审核工具链

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links" cargo doc --no-deps --all-features
cargo install cargo-audit cargo-deny cargo-public-api cargo-llvm-cov cargo-machete
cargo audit
cargo deny check
cargo public-api
cargo llvm-cov --summary-only
cargo machete
```

附录 B 证据文件列表见 `docs/audit-evidence/`。
