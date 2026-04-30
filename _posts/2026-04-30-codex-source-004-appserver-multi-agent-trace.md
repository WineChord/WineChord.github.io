---
classes: wide2
title: "Codex 源码剖析：004. App Server、多 Agent 与 Trace"
excerpt: "从 App Server 协议生命周期看 Codex 如何把 TUI、远程客户端、多 Agent、审批请求和 trace 统一成可路由的线程系统。"
categories:
  - LLM
  - Agent
tags:
  - Codex
  - Coding Agent
  - Source
toc: true
toc_sticky: true
mathjax: true
---

第一篇已经提到，本地 TUI 默认也走 in-process app-server。这一篇继续往下看：App Server 不只是把 UI 请求转给 core，它还把 thread、turn、多 Agent、审批、远程 client、事件回放、trace 全部放到一个协议化生命周期里。

这里最值得看的不是某个具体 JSON-RPC method，而是整体形态：

> Codex 正在把“一个 CLI agent”拆成“一个可被多个 client 驱动的 thread/turn runtime”。

这也是为什么 OpenAI 会单独写 App Server 的文章。client 和 server 先 `initialize`，协商能力、协议版本、feature flags，然后 client 创建 thread、启动 turn，server 回 progress notifications 和 approval requests（见 [Unlocking the Codex harness](https://openai.com/index/unlocking-the-codex-harness/)）。

## App Server Lifecycle

### MessageProcessor：连接、初始化和请求分发

App Server 的外层事件循环在 [`app-server/src/lib.rs`][app-server-loop]。它维护 connection map，监听 transport events，并在 connection 打开时创建 `ConnectionSessionState`。

每个连接需要先完成 initialize。connection state 里有：

- origin
- rpc gate
- initialized state
- experimental API 是否 enabled
- opted-out notifications
- client name / version

这些字段在 [`ConnectionSessionState`][connection-state]。

请求进来以后，`MessageProcessor::process_request` 负责把 JSON-RPC request 分给不同 API：

- config API
- device key API
- fs API
- external agent config API
- Codex thread/turn API

这张图先把 app-server、core、多 Agent 和 trace 放在一起。重点不是“多一层服务”，而是所有 client 都先变成 thread/turn lifecycle，再由 core 事件回流。

![Codex App Server、多 Agent 和 rollout trace 总览：连接、thread、turn、AgentControl、mailbox、state DB 和 trace graph 的关系](/assets/images/posts/2026-04-30-codex-source/appserver-agent-trace.svg)

*图 1. App Server 接住 client 请求，thread/turn 把请求送进 core，多 Agent 通过 AgentControl 和 mailbox 形成线程树，trace 再把这些运行证据还原成图。*

`MessageProcessor` 自身持有几类对象，见 [`MessageProcessor`][message-processor]：

- Codex thread/turn API：`CodexMessageProcessor`、`ThreadManager`
- 其他 API：ConfigApi、FsApi、AuthManager
- 运行时辅助：FsWatchManager、request serialization queues

这层的职责有点像应用网关。它不是模型循环，也不是 UI，但它决定了哪些请求可以进来、怎么被序列化、哪些 connection 能收到哪些 notification。

这里可以顺手和 OpenClaw 对照一下。OpenClaw 的 Gateway 文档把 WebSocket Gateway 明确写成 control plane 和 node transport，client 在 handshake 时声明 `operator` / `node` role 和 scope，见 [`protocol.md`][openclaw-gateway-protocol]。Codex App Server 也在做协议边界，但它更贴着 Codex 自己的 thread/turn/core event；OpenClaw 则更像先定义一层通用网关，再把 UI、节点、sandbox backend、ACP runtime 挂到这个控制面上。

### Protocol v2：Rust core 与 client 之间的翻译层

App Server protocol v2 里有不少“翻译层”代码。比如 `v2_enum_from_core!` 宏会把 core enum 映射成 camelCase API v2 enum，见 [`v2_enum_from_core`][v2-enum]。

错误类型也是类似。`CodexErrorInfo` 在 v2 里用 camelCase，并保留 HTTP status code、response stream connection/disconnection、active turn not steerable 等细节，见 [`CodexErrorInfo`][v2-error-info]。

为什么要这么做？

因为 core 内部类型要服务 Rust 运行时，protocol 类型要服务外部 client。两边如果直接共用所有结构，短期省事，长期会让内部重构和外部兼容绑死。Codex 现在用 app-server-protocol crate 把这层明确拆出来。

这也解释了为什么源码里会有很多看似重复的类型转换。那不是纯粹 boilerplate，而是给“内部可变、外部稳定”留空间。

### thread/start 与 turn/start：所有 client 进入 core 的同一条门

真正进入 core 的大多数请求，都在 `CodexMessageProcessor`。

它持有：

- `ThreadManager`
- `thread_store`
- `thread_state_manager`
- `thread_watch_manager`
- command exec manager
- workspace settings cache
- feedback / log db 等

见 [`CodexMessageProcessor`][codex-message-processor]。

`TurnStart` 分支最终进入 `turn_start`。这里做的事情上一篇讲过一部分：校验 input，检查 thread，映射 v2 input 到 core input，处理 turn overrides，最后构造 `Op::UserInputWithTurnContext` 或 `Op::UserInput`。源码在 [`turn_start`][turn-start]。

更重要的是，这条门不只服务 TUI。VS Code、remote control、其他 client 只要实现 protocol，就可以用同样的 thread/turn 生命周期驱动 core。

这和传统 terminal app 差别很大。传统 CLI 是“用户在一个进程里操作一个会话”。Codex app-server 形态是“多个 client 连接一个 thread runtime，thread runtime 再管理 core session”。不过这仍是一个迁移中的结构：TUI 侧的 [`app_server_adapter.rs`][tui-app-server-adapter] 明确说它是 hybrid migration period 的临时层，TUI 还保留 direct-core 行为；本篇讨论的是 app-server-backed flows 和它所代表的演进方向。

## Event Routing

### EventMsg 到 ServerNotification / ServerRequest

core 发出来的是 `EventMsg`。App Server 不能原样扔给 client，因为 client 需要的是 protocol v2 里的 notification/request。

翻译发生在 `apply_bespoke_event_handling`。它会把不同事件转成对应 app-server 事件，例如：

- `TurnStarted` -> `ServerNotification::TurnStarted`
- `AgentMessageContentDelta` -> `AgentMessageDelta`
- `RequestUserInput` -> `ServerRequest::ToolRequestUserInput`
- `TurnComplete` -> `TurnCompleted`

这个边界在 [`bespoke_event_handling.rs`][bespoke-event-handling]。

为什么 approval 要走 `ServerRequest`，而不是普通 notification？

因为 approval 需要 client 回答。core 这边已经计算出某个 command / patch / MCP tool 需要审批；app-server 把它转成 request，UI 展示给用户，用户决策再回填 core。这样审批不是 UI 自己推断出来的，而是 core policy 触发、protocol request 承载、client decision 回流。

这点对安全很关键。UI 只是呈现和收集决策，不应该重新实现一遍权限判断。

### TUI 侧：按 thread id 分流事件

TUI 收 app-server event 的入口是 [`handle_app_server_event`][tui-handle-app-server-event]。它根据 thread id 把 notification/request 投递到对应 thread channel。active thread 的事件再进入 `handle_thread_event_now`，最后交给 `ChatWidget::handle_server_notification` 或 `handle_server_request`。

这一层看起来只是 UI plumbing，但它支持两个能力：

1. 非 active thread 的事件可以先 buffer。
2. 切换 thread 后，UI 可以把该 thread 的 buffered events 重新消费。

这也是 thread/turn protocol 的价值：UI 不需要知道 core 内部 Session 怎么调度，只需要维护每个 thread 的事件流。

## Multi-Agent Runtime

### 多 Agent：不是函数递归，而是线程树

Codex 的 multi-agent v2 不是简单在当前 prompt 里塞“你现在是子 agent”。它更接近创建一个新的 child thread。

控制面是 `AgentControl`，它是 root session tree 共享的控制对象，内部持有 `AgentRegistry`，并通过 weak reference 回到 `ThreadManagerState` 避免循环引用，见 [`AgentControl`][agent-control]。

线程创建 / 恢复在 `ThreadManagerState`。fresh child 会继承父 trace context；resumed child 不继承，避免重复 `ThreadStarted` 事件。这个逻辑在 [`thread_manager.rs`][thread-manager-child]。

v2 identity 用 `AgentPath`。路径形如 `/root/task_name`，只允许小写、数字、下划线等安全字符，见 [`AgentPath`][agent-path]。这看起来只是命名规则，但它给多 Agent 树一个稳定、可显示、可路由的身份。

`spawn_agent` v2 handler 会解析 task、role、model、fork 配置，构造 child config，生成 `SessionSource::SubAgent(ThreadSpawn)`，再交给 `AgentControl::spawn_agent_with_metadata`，见 [`multi_agents_v2/spawn.rs`][spawn-agent-v2]。

这里的关键是：child agent 不是当前 turn 里的一个函数调用结果，而是一个有自己 session、history、tools、permissions、trace context 的执行实体。

![Codex 多 Agent trace edge 示例：parent spawn child、child 完成后返回 parent，并由 reducer 还原 InteractionEdge](/assets/images/posts/2026-04-30-codex-source/agent-edge-example.svg)

*图 2. 普通 transcript 只能看到“某段结果文本出现了”，trace graph 则能追到这段结果来自哪个 child thread，以及它是通过 spawn/result 边回到 parent 的。*

把 v2 spawn 的控制流压缩一下，大致是：

```text
parse spawn_agent args
resolve role / model / fork_context
build child SessionConfig
create child thread source = SubAgent(ThreadSpawn)
AgentControl.spawn_agent_with_metadata(child)
return child identity to parent turn
```

这段流程里真正改变系统形态的是“child thread source”。它不是在 parent history 里假装换了一个角色，而是创建一个可路由、可恢复、可 trace 的新执行实体。

### Mailbox：父子消息不直接写对方 history

父子 agent 通信走 `InterAgentCommunication`。这类消息先进入 mailbox，不是直接写对方 conversation history。mailbox drain 时再变成 assistant commentary message。

protocol 里的结构在 [`InterAgentCommunication`][inter-agent-communication]。Session 侧 drain / enqueue 逻辑可以从 [`session/mod.rs` 的 mailbox 路径][mailbox-drain] 看。

这个设计很克制。直接写对方 history 虽然简单，但会让信息来源、turn 边界、trace replay 都变得模糊。mailbox 相当于一个“跨线程消息缓冲层”：

- 消息可以触发新 turn，也可以等目标 idle。
- 消息来源可以被标记成 agent communication。
- trace 可以记录 spawn、send、followup、result 的边。
- UI 可以展示 subagent notification，而不是把它当普通用户消息。

child 完成后，v2 会向 parent 投递 completion envelope，并在 trace 里记录 `AgentResultObserved` edge，见 [`child completion`][agent-child-complete]。

### State DB：多 Agent 树需要可恢复关系

多 Agent 如果只存在内存里，就没法可靠 close descendant、resume tree、恢复父子关系。Codex state DB 有 `thread_spawn_edges`，用于记录 thread 树关系，见 [`state/src/runtime/threads.rs`][thread-spawn-edges]。

这也说明多 Agent 不是“并行起几个 task”这么简单。只要你希望它们可恢复、可关闭、可观察，就需要把父子边持久化。

### Rollout Trace：把运行时还原成图

上一篇区分了 rollout JSONL 和 rollout-trace。这里继续看 trace 的意义。

`rollout-trace/README.md` 的设计原则是 “observe first, interpret later”。运行时记录 raw events 和 payload references，离线 reducer 再还原成 semantic graph，见 [`rollout-trace README`][rollout-trace-readme]。

这个 graph 不只是 transcript。它包含：

- threads + turns
- conversation_items
- inference_calls
- tool_calls
- code_cells
- terminal operations
- compactions
- interaction_edges
- raw_payload refs

对多 Agent 来说，最关键的是 interaction edges。README 里有一段专门讲 multi-agent v2：child threads 和 root thread 共享 trace writer，一个 root bundle 可以 reduce 成包含 parent、child threads 和边的图，见 [`rollout trace multi-agent v2`][rollout-trace-multi-agent]。

trace reducer 还会把 spawn/send/followup/close/result 转成 `InteractionEdge`，尽量指向 recipient 侧真实 conversation item。如果 child 还没形成 task message，spawn edge fallback 到 thread。这个逻辑在 [`rollout-trace reducer agents.rs`][trace-agent-edges]。

这就是 trace 的价值：当模型、工具、终端、子 agent、UI notification 交错发生时，普通 transcript 只能告诉你“最后看到了什么文本”，trace graph 才能告诉你“这段文本从哪个 tool call、哪个 child thread、哪个 runtime object 流过来”。

## 和云端 coding agent 的关系

OpenAI 在 Codex 发布文章里强调过 cloud-based task 在独立 sandbox 环境里并行执行，并提供 terminal logs 和 test outputs 作为可验证证据。GitHub Copilot cloud agent 的官方文档也描述了类似方向：agent 可以在 GitHub Actions-powered ephemeral development environment 里研究仓库、制定计划、改分支、运行测试、再让用户 review diff / PR（见 [GitHub Copilot cloud agent](https://docs.github.com/en/copilot/concepts/agents/cloud-agent/about-cloud-agent)）。

这说明一个趋势：coding agent 正在从“IDE 里一个聊天助手”演进成“有独立执行环境、有生命周期、有日志、有分支、有审批、有多人协作入口的软件执行系统”。

Codex app-server / thread / turn / trace 这套结构，正是在为这种形态服务。它让本地配对和远程委托逐步收敛到同一个 core runtime 的不同 client 和不同部署方式，而不是永远维护两套互不相干的 agent。

## 小结

App Server 是 Codex harness 的协议边界，不只是 TUI 到 core 的 adapter。

它做了几件核心事情：

1. 用 `initialize` 管 client 能力和协议状态。
2. 用 thread/turn lifecycle 把多端请求统一进入 core。
3. 用 `ServerNotification` / `ServerRequest` 把 core 事件和审批请求翻译给 client。
4. 用 thread event buffer 支持 UI 的 thread 切换和事件回放。
5. 用 `AgentControl`、`ThreadManager`、`AgentPath`、mailbox 把多 Agent 做成线程树。
6. 用 state DB 保存 thread spawn edges，让多 Agent 关系可恢复。
7. 用 rollout trace 把运行时证据还原成 graph，解释信息如何在模型、工具、终端、子 agent 间流动。

所以 Codex 的 App Server 不是“多了一层服务端”。它是从单机 CLI agent 走向多端、多线程、多 Agent、可审计执行系统的关键中间层。

## 参考

- [OpenAI: Unlocking the Codex harness](https://openai.com/index/unlocking-the-codex-harness/)
- [OpenAI: Introducing Codex](https://openai.com/index/introducing-codex/)
- [GitHub Docs: About GitHub Copilot cloud agent](https://docs.github.com/en/copilot/concepts/agents/cloud-agent/about-cloud-agent)

[app-server-loop]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/lib.rs#L732-L930
[connection-state]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/message_processor.rs#L180-L246
[message-processor]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/message_processor.rs#L162-L178
[v2-enum]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/v2.rs#L123-L151
[v2-error-info]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/v2.rs#L162-L230
[codex-message-processor]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/codex_message_processor.rs#L520-L546
[turn-start]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/codex_message_processor.rs#L6529
[bespoke-event-handling]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/bespoke_event_handling.rs#L164
[tui-handle-app-server-event]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/app/app_server_adapter.rs#L125
[tui-app-server-adapter]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/app/app_server_adapter.rs#L1-L11
[agent-control]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/agent/control.rs#L129
[thread-manager-child]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/thread_manager.rs#L1058
[agent-path]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/agent_path.rs#L17
[spawn-agent-v2]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/handlers/multi_agents_v2/spawn.rs#L26
[inter-agent-communication]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/protocol.rs#L831
[mailbox-drain]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/mod.rs#L3113
[agent-child-complete]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/mod.rs#L1477
[thread-spawn-edges]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/state/src/runtime/threads.rs#L86
[rollout-trace-readme]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/rollout-trace/README.md#L9-L20
[rollout-trace-multi-agent]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/rollout-trace/README.md#L172-L197
[trace-agent-edges]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/rollout-trace/src/reducer/tool/agents.rs#L22
[openclaw-gateway-protocol]: https://github.com/openclaw/openclaw/blob/5d8ca42c7de8118b15782bad9cbac6240585e13a/docs/gateway/protocol.md#L12-L15
