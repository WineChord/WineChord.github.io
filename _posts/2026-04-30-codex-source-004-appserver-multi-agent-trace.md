---
classes:
  - wide2
  - codex-source-004-appserver-multi-agent-trace
title: "Codex 源码剖析：004. App Server、多 Agent 与 Trace"
excerpt: "从 App Server 的 JSON-RPC 生命周期出发，看 Codex 如何把 client、thread、turn、approval、多 Agent mailbox 和 rollout trace 放进同一套可路由的 harness。"
description: "这篇源码剖析自包含地解释 Codex App Server、thread/turn 协议、多 Agent mailbox、approval 回填和 rollout trace 的证据边界、协议形状与失败条件。"
locale: zh-CN
categories:
  - LLM
  - Agent
tags:
  - Codex
  - Coding Agent
  - Source
toc: true
toc_sticky: true
toc_label: "目录"
toc_levels: 2..4
mathjax: true
header:
  teaser: https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-appserver-runtime-cover.png
  og_image: https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-appserver-runtime-cover.png
---

App Server 是 Codex harness 里把“一个本地 CLI agent”推向“可被多个 client 驱动的 thread runtime”的那层协议边界。它不替代 core agent loop，也不替代 UI；它把 client request、thread/turn 生命周期、server notification、server request、approval 回填、多 Agent child thread 和 trace 证据都收束到一条可路由的执行通道里。

这篇从零开始读，不假设读者看过前文。可以先把 Codex 想成三层：

- **client surface**：TUI、IDE、桌面 App、远程控制入口，负责展示和收集用户输入。
- **App Server**：client-friendly 的 JSON-RPC/JSONL 协议层，负责连接、初始化、thread/turn request、server event 和 server-initiated request。
- **Codex core**：真正跑模型循环、工具、sandbox、审批策略、历史恢复和多 Agent 执行的 runtime。

OpenAI 的 [Codex App Server 文档](https://developers.openai.com/codex/app-server)把集成路径写成三步：启动 `codex app-server`，连接 transport，先发 `initialize` 和 `initialized`，再 `thread/start`、`turn/start` 并持续读取 notifications。源码里对应的是 [`app-server-protocol`][app-server-protocol-common] 的 request/notification 类型、[`MessageProcessor`][message-processor] 的连接状态和请求分发，以及 core 里的 thread/session 执行链。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-appserver-runtime-cover.png" alt="Codex App Server runtime 封面图：TUI、VS Code、remote client 进入 App Server，再由 thread/turn runtime 路由到 core 和 client UI">
  <figcaption>图 1. App Server 是多端进入同一套 thread/turn runtime 的协议边界，不是模型循环本身。</figcaption>
</figure>

## 阅读契约

这篇要回答四个问题：

1. `initialize`、`thread/start`、`turn/start` 如何把不同 client 接进同一个 core runtime？
2. `ServerNotification` 和 `ServerRequest` 为什么要分开，尤其是 approval 为什么必须有回填通道？
3. 多 Agent 为什么是 child thread、mailbox 和 state DB 组成的线程树，而不是 parent prompt 里的函数递归？
4. rollout trace 为什么能还原运行图谱，却不能被误读成生产恢复路径？

读完后，应该能把一个事件放回它所在的边界：连接状态、thread/turn、core event、client request、child agent edge、rollout JSONL，或离线 reducer 还原出的 trace graph。

## 证据边界

本文把证据分成三层。

**官方契约**来自 OpenAI 的 [Codex App Server 文档](https://developers.openai.com/codex/app-server)、[Codex CLI 文档](https://developers.openai.com/codex/cli)、[sandbox 概念文档](https://developers.openai.com/codex/concepts/sandboxing)和 [agent approvals/security 文档](https://developers.openai.com/codex/agent-approvals-security)。这些链接只说明公开产品和协议契约，不证明服务端内部实现细节。

**源码事实**固定在 [`openai/codex@ac4332c05b11e00ae775a24cb762edc05c5b5932`][codex-pinned-commit]。后文所有文件、函数、struct、enum、line anchor 都按这个 commit 读；如果未来 `main` 改了，本文仍是在解释这个快照。

**有界推断**只用于连接可见契约和源码边界：比如“App Server 让多端共享同一 thread/turn lifecycle”是由官方文档和源码请求形状共同支持的；但“OpenAI 托管服务内部怎样调度容器、怎样保存私有状态”不在可见源码内，本文不会把它写成事实。

## 一、App Server 是协议边界

下图把 App Server、core、多 Agent 和 trace 放在一起。重点不是“多一层服务”，而是所有 client 都先变成协议里的 connection、thread、turn 和 event stream，再由 core 执行模型循环。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-appserver-agent-trace.png" alt="Codex App Server、多 Agent 和 rollout trace 总览：连接、thread、turn、AgentControl、mailbox、state DB 和只记录证据的 trace graph 边界">
  <figcaption>图 2. App Server 接住 client 请求，thread/turn 把请求送进 core，多 Agent 通过 child session 和 mailbox 形成线程树，trace 再把运行证据还原成图。</figcaption>
</figure>

### 1.1 MessageProcessor 管连接、初始化和请求分发

App Server 的外层事件循环在 [`app-server/src/lib.rs`][app-server-loop]。它维护 connection map，监听 transport events，并在 connection 打开时创建 `ConnectionSessionState`。

#### initialize 建立连接状态

每个连接需要先完成 `initialize`。`ConnectionSessionState` 里保存 origin、rpc gate、initialized state、experimental API 是否 enabled、opted-out notifications、client name/version 等字段，见 [`ConnectionSessionState`][connection-state]。`InitializeParams` 的公开形状来自 v1 protocol：`clientInfo` 是必需字段，`capabilities.experimentalApi` 和 `optOutNotificationMethods` 是连接级能力声明，见 [`InitializeParams`][initialize-params]。

下面是 shape-level 示例；字段名来自公开 protocol，值为了说明而简化：

```json
{
  "method": "initialize",
  "id": 1,
  "params": {
    "clientInfo": {
      "name": "codex_desktop",
      "title": "Codex Desktop",
      "version": "0.1.0"
    },
    "capabilities": {
      "experimentalApi": true,
      "optOutNotificationMethods": ["item/agentMessage/delta"]
    }
  }
}
```

`initialize` 成功后，in-process / transport 层会让连接进入可发 outbound 的状态；官方文档也要求 client 随后发送 `initialized` notification 再开始 thread/turn 生命周期。这个握手的意义不是“问候一下”，而是先确定：这个 client 能不能接实验字段、是否拒收某些 notification、后续请求能否过 initialized gate。

#### 请求分发不是模型循环

请求进来以后，`MessageProcessor::process_request` 负责把 JSON-RPC request 分给不同 API：

- config API
- device key API
- fs API
- external agent config API
- Codex thread/turn API

`MessageProcessor` 自身持有 Codex thread/turn API、ConfigApi、FsApi、AuthManager、FsWatchManager、request serialization queues 等对象，见 [`MessageProcessor`][message-processor]。

这层有点像应用网关。它不是模型循环，也不是 UI，但它决定哪些请求可以进来、怎么按 thread/command/fs watch 等 key 被序列化、哪些 connection 能收到哪些 notification。

### 1.2 Protocol v2 是内部类型和外部 API 的翻译层

App Server protocol v2 里有不少“翻译层”代码。比如 `v2_enum_from_core!` 宏会把 core enum 映射成 camelCase API v2 enum，见 [`v2_enum_from_core`][v2-enum]。

错误类型也是类似。`CodexErrorInfo` 在 v2 里用 camelCase，并保留 HTTP status code、response stream connection/disconnection、active turn not steerable 等细节，见 [`CodexErrorInfo`][v2-error-info]。

#### field gate 是兼容性的边界

`thread/start` 和 `turn/start` 在 `ClientRequest` 里都是 v2 方法，并且都打开 `inspect_params: true`，见 [`ClientRequest` 定义][client-request-defs]。这说明兼容性不是只按 method 名粗暴切分，还会检查参数里的实验字段。`ThreadStartParams` 里 `permissions`、`environments`、`dynamicTools`、`experimentalRawEvents` 等字段都有 experimental 标记；`TurnStartParams` 里 `responsesapiClientMetadata`、`environments`、`permissions`、`collaborationMode` 也有类似门控，见 [`ThreadStartParams`][thread-start-params] 和 [`TurnStartParams`][turn-start-params]。

这里有一个容易误读的 failure condition：官方文档会讲协议版本和 feature flags 的协商，但这个快照的 `InitializeParams` 没有一个叫 `protocolVersion` 的线协议字段。实际可见边界是 v1/v2 类型层、`experimentalApi` opt-in、字段级 experimental 标记和 notification opt-out。如果 client 没有 opt in，却发送或期待实验字段，App Server 可以在 request 入口拒绝或剥离对应字段；这不是 core agent loop 的失败，而是协议兼容层的失败。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-protocol-lifecycle.png" alt="App Server 协议生命周期图：initialize、thread/start、turn/start、server events、turn done 顺序组成 client 到 core 的状态机">
  <figcaption>图 3. 协议生命周期先于模型循环。连接能力、thread 状态、turn 输入和 server events 都要先在协议层站稳。</figcaption>
</figure>


### 1.3 `thread/start` 与 `turn/start` 是多端进入 core 的同一条门

真正进入 core 的大多数请求，都在 `CodexMessageProcessor`。它持有 `ThreadManager`、`thread_store`、`thread_state_manager`、`thread_watch_manager`、command exec manager、workspace settings cache、feedback/log db 等对象，见 [`CodexMessageProcessor`][codex-message-processor]。

#### thread/start 创建可订阅容器

`thread/start` 创建新的 Codex conversation container。官方 App Server 文档说明它会返回 thread，并发出 `thread/started` notification；源码里 `ThreadStartedNotification` 也只包一份 `Thread`，见 [`ServerNotification` 定义][server-notification-defs]。

```json
{
  "method": "thread/start",
  "id": 10,
  "params": {
    "model": "gpt-5.4",
    "cwd": "/workspace/project",
    "approvalPolicy": "on-request",
    "sandbox": "workspaceWrite",
    "serviceName": "ide_client"
  }
}
```

response 里返回 thread metadata；随后 event stream 会出现：

```json
{
  "method": "thread/started",
  "params": {
    "thread": {
      "id": "thr_123",
      "sessionId": "thr_123"
    }
  }
}
```

这一步建立的是“后续 turn 和 item event 应该挂在哪个 durable container 上”，不是一次模型调用。

#### turn/start 提交一次 agent work

`TurnStart` 分支最终进入 `turn_start`。这里会校验 input，检查 thread，映射 v2 input 到 core input，处理 turn overrides，最后构造 `Op::UserInputWithTurnContext` 或 `Op::UserInput`。源码在 [`turn_start`][turn-start]。

shape-level 的 request 很小：

```json
{
  "method": "turn/start",
  "id": 11,
  "params": {
    "threadId": "thr_123",
    "input": [
      { "type": "text", "text": "Run tests and summarize failures." }
    ]
  }
}
```

但一次 `turn/start` 后面会展开成多条 server notifications：

```text
turn/started
  item/started
  item/agentMessage/delta
  item/commandExecution/outputDelta
  item/completed
turn/completed
```

这个形状解释了为什么 App Server 不是普通 request/response API。client 只发一次 turn request，UI 却必须能流式渲染 reasoning、消息、工具、diff、approval 和最终完成状态。

#### TUI 仍有迁移边界

这条门不只服务 TUI。VS Code、desktop、远程控制和其他 client 只要实现 protocol，就可以用同样的 thread/turn 生命周期驱动 core。

不过这里要保留源码边界：TUI 侧的 [`app_server_adapter.rs`][tui-app-server-adapter] 明确说它处在 hybrid migration period，TUI 还保留 direct-core 行为。本篇讨论的是 app-server-backed flows 和它代表的演进方向，不把所有 TUI 路径都说成已经完全协议化。


## 二、事件路由：core 只吐事件，client 负责呈现和回填

### 2.1 `EventMsg` 翻译成 notification / request

core 发出来的是 `EventMsg`。App Server 不能原样扔给 client，因为 client 需要的是 protocol v2 里的 notification/request。

翻译发生在 [`apply_bespoke_event_handling`][bespoke-event-handling]。它会把不同事件转成对应 app-server 事件，例如：

- `TurnStarted` -> `ServerNotification::TurnStarted`
- `AgentMessageContentDelta` -> `AgentMessageDelta`
- `ExecApprovalRequest` -> `ServerRequestPayload::CommandExecutionRequestApproval`
- `ApplyPatchApprovalRequest` -> `ServerRequestPayload::FileChangeRequestApproval`
- `RequestUserInput` -> `ServerRequestPayload::ToolRequestUserInput`
- `TurnComplete` -> `ServerNotification::TurnCompleted`

#### notification 是单向事实

`ServerNotification` 是 server 单向告诉 client “发生了什么”。`turn/started`、`item/started`、`item/agentMessage/delta`、`turn/completed` 都属于这一类，见 [`ServerNotification` 定义][server-notification-defs]。client 可以展示、缓存、过滤、滚动到对应 item，但不需要对 notification 回答。

#### request 是暂停点

approval、MCP elicitation、request user input 和 permissions request 走 `ServerRequest`，见 [`ServerRequest` 定义][server-request-defs]。它们都有 `id`，也都有对应 response type。approval 之所以不能做成普通 notification，是因为 core 已经算出某个 command、patch、MCP tool 或权限变更需要用户/宿主确认；App Server 必须暂停等待 client 回答，再把 decision 交回 core。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-server-request-approval.png" alt="Approval ServerRequest 图：core 产生 command/file approval request，App Server 翻译成 approval ServerRequest，client UI 收集 allow/deny 后回到 core">
  <figcaption>图 4. Notification 只能通知；approval 需要 client 回答。UI 负责呈现风险和收集决策，不重新实现权限判断。</figcaption>
</figure>


```json
{
  "method": "item/commandExecution/requestApproval",
  "id": 91,
  "params": {
    "threadId": "thr_123",
    "turnId": "turn_1",
    "itemId": "call_abc",
    "reason": "command requires approval",
    "command": "cargo test -p codex-app-server",
    "cwd": "/workspace/project",
    "availableDecisions": ["accept", "acceptForSession", "decline"]
  }
}
```

client 的回填仍然是同一个 JSON-RPC lite 方法名：

```json
{
  "method": "item/commandExecution/requestApproval",
  "id": 91,
  "response": {
    "decision": "acceptForSession"
  }
}
```

App Server 随后可以发 `serverRequest/resolved`，让 UI 把 pending prompt 收起来。注意：UI 负责呈现风险、展示命令、收集 allow/deny；权限判断本身不应该在 UI 里重新实现一遍。

#### 回填失败要保守失败

这里的 counterexample 很重要：如果 client 断连、response 反序列化失败，或者 request future 返回错误，App Server 不能假装用户同意了。源码里 command approval 反序列化失败会落到 `Decline`，file change approval 也会落到 `Decline`，request user input 失败会构造空 answers 再提交回 core，见 [`approval response handling`][approval-response-handling] 和 [`request user input handling`][request-user-input-handling]。

这会带来一个 UI 层可见现象：用户可能以为“只是弹窗没回去”，core 看到的却是一次 denied/empty response。好的 client 要把 pending request 生命周期做清楚，否则 approval 的安全语义会和用户感知错位。


### 2.2 TUI 按 thread id 分流事件

TUI 收 app-server event 的入口是 [`handle_app_server_event`][tui-handle-app-server-event]。它根据 thread id 把 notification/request 投递到对应 thread channel。active thread 的事件再进入 `handle_thread_event_now`，最后交给 `ChatWidget::handle_server_notification` 或 `handle_server_request`。

#### 非 active thread 可以 buffer

这一层支持两个能力：

1. 非 active thread 的事件可以先 buffer。
2. 切换 thread 后，UI 可以把该 thread 的 buffered events 重新消费。

这就是 thread/turn protocol 的价值：UI 不需要知道 core 内部 Session 怎么调度，只需要维护每个 thread 的事件流。

#### 事件丢失不是 core 事实丢失

另一个边界是：client 可选择 opt out 某些 notification，也可能断线重连。App Server 的 event stream 是 UI-ready projection；生产恢复仍要回到 thread store/rollout 等持久层。把“某个 client 没看到 delta”解释成“core 没产生这个 item”，就是把 UI 订阅层和 runtime evidence 混在一起了。

## 三、多 Agent 不是函数递归，而是线程树

### 3.1 child agent 是新的 child thread

Codex 的 multi-agent v2 不是简单在当前 prompt 里塞“你现在是子 agent”。它更接近创建一个新的 child thread。

控制面是 `AgentControl`，它是 root session tree 共享的控制对象，内部持有 `AgentRegistry`，并通过 weak reference 回到 `ThreadManagerState` 避免循环引用，见 [`AgentControl`][agent-control]。

#### spawn_agent 的控制流

v2 identity 用 `AgentPath`。路径形如 `/root/reviewer`，`AgentPath` 会校验绝对路径和 segment，见 [`AgentPath`][agent-path]。

`spawn_agent` v2 handler 会解析 task、role、model、fork 配置，构造 child config，生成 `SessionSource::SubAgent(ThreadSpawn)`，再交给 `AgentControl::spawn_agent_with_metadata`，见 [`multi_agents_v2/spawn.rs`][spawn-agent-v2]。

把 v2 spawn 压成控制流：

```text
parse spawn_agent args
resolve role / model / fork_context
build child SessionConfig
create child source = SessionSource::SubAgent(ThreadSpawn)
AgentControl.spawn_agent_with_metadata(child)
return child AgentPath / thread identity to parent turn
```

这个流程真正改变系统形态的是“child thread source”。它不是在 parent history 里假装换了一个角色，而是创建一个可路由、可恢复、可 trace 的新执行实体。

#### resumed child 不重复写 trace start

fresh child 和 resumed child 的 trace 边界不同。`ThreadManagerState::parent_rollout_thread_trace_for_source` 的注释写得很明确：fresh v2 child 属于 parent 的 rollout tree，所以从 parent thread context 派生 child trace；resumed child 已经有过这个 thread id 的 `ThreadStarted` event，如果 resume 时再派生并写一次 start event，会让 bundle 不可 replay，见 [`parent_rollout_thread_trace_for_source`][parent-rollout-thread-trace]。

所以“child thread resume”不是普通 spawn 的重复播放。它要恢复执行实体，但不能在 trace 里伪造第二次出生。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-agent-edge-example.png" alt="Codex 多 Agent trace edge 示例：parent spawn child、child 完成后返回 parent，并由 reducer 还原 InteractionEdge">
  <figcaption>图 5. 普通 transcript 只能看到“某段结果文本出现了”；trace graph 则能追到这段结果来自哪个 child thread，以及它是通过 spawn/result 边回到 parent 的。</figcaption>
</figure>

### 3.2 Mailbox 让父子消息保持来源边界

父子 agent 通信走 `InterAgentCommunication`。这类消息先进入 mailbox，不是直接写对方 conversation history。Session 侧 drain / enqueue 逻辑可以从 [`session/mod.rs` 的 mailbox 路径][mailbox-drain] 看。

#### mailbox 的内部消息形状

`InterAgentCommunication` 的源码字段是 `author`、`recipient`、`other_recipients`、`content`、`trigger_turn`，见 [`InterAgentCommunication`][inter-agent-communication]。它最终会被转成 `ResponseInputItem::Message`，role 是 `assistant`，phase 是 `Commentary`，content 里放序列化后的 communication JSON。

shape-level 示例：

```json
{
  "author": "/root",
  "recipient": "/root/reviewer",
  "other_recipients": [],
  "content": "检查这个 diff 的并发风险。",
  "trigger_turn": true
}
```

这不是 app-server client 直接发送的公开 wire payload，而是 core runtime 内部把跨 agent 消息带进目标 thread model-visible input 的形状。

#### 为什么不直接写对方 history

直接写对方 history 虽然简单，但会让信息来源、turn 边界、trace replay 都变得模糊。mailbox 相当于一个跨线程消息缓冲层：

- 消息可以触发新 turn，也可以等目标 idle。
- 消息来源可以被标记成 agent communication。
- trace 可以记录 spawn、assign task、send message、result、close 等边。
- UI 可以展示 subagent notification，而不是把它当普通用户消息。

child 完成后，v2 会向 parent 投递 completion envelope，并在 trace 里记录 `AgentResult` edge。源码里 `forward_child_completion_to_parent` 会构造 `InterAgentCommunication`，`trigger_turn` 为 `false`，再调用 parent 的 `enqueue_mailbox_communication`，同时在 trace enabled 时记录 `AgentResultTracePayload`，见 [`child completion`][agent-child-complete]。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v4-child-thread-mailbox.png" alt="多 Agent child thread 与 mailbox 图：parent thread 和 child thread 通过 mailbox 交换消息，state DB 只持久化 spawn edge，completion envelope 经 mailbox 回到 parent 并进入 trace AgentResult">
  <figcaption>图 6. 多 Agent 是线程树，不是函数递归。mailbox 保留来源边界，state DB 保留可恢复的父子关系。</figcaption>
</figure>

### 3.3 State DB 保存可恢复关系

多 Agent 如果只存在内存里，就没法可靠 close descendant、resume tree、恢复父子关系。Codex state DB 有 `thread_spawn_edges`，用于记录 thread 树关系；`upsert_thread_spawn_edge` 会按 child thread id upsert parent-child edge 和 lifecycle status，见 [`state/src/runtime/threads.rs`][thread-spawn-edges]。

这说明多 Agent 不是“并行起几个 task”这么简单。只要希望它们可恢复、可关闭、可观察，就需要把父子边持久化。

#### 持久关系不是 prompt 关系

一个常见 counterexample 是：parent prompt 里写“请你作为 reviewer 回答”，模型也能产出类似子 agent 的文本，但那不是 child thread。它没有独立 thread id，没有 mailbox 边界，没有 state DB spawn edge，也不会在 rollout trace reducer 里形成 `InteractionEdge`。从读源码的角度看，判断 multi-agent 的标准不是“文风像不像另一个角色”，而是有没有可路由、可恢复、可 trace 的执行实体。

## 四、rollout trace 把运行时还原成图

`rollout-trace/README.md` 的设计原则是 “observe first, interpret later”。运行时记录 raw events 和 payload references，离线 reducer 再还原成 semantic graph，见 [`rollout-trace README`][rollout-trace-readme]。

### 4.1 raw evidence 和 reduced graph 是两层

trace bundle 包含 `manifest.json`、`trace.jsonl`、`payloads/*.json`，可选的 reducer 输出是 `state.json`。README 明确区分：

- `ConversationItem`：模型可见请求/响应里出现了什么。
- `ToolCall`、`CodeCell`、`TerminalOperation`、`InferenceCall`、`Compaction`：runtime/debug 边界。
- `InteractionEdge`：对象之间的信息流。
- `RawPayloadRef`：指回原始证据。

#### graph 不是生产恢复路径

这里的 failure boundary 必须说清楚：trace graph 很适合诊断交错运行证据，但生产恢复仍依赖 thread store/rollout JSONL 和 core 的恢复逻辑。trace 是 opt-in diagnostic path；README 也写明只有设置 `CODEX_ROLLOUT_TRACE_ROOT` 时才写本地 bundle，且记录可能包含 prompts、responses、tool inputs/outputs、terminal output 和路径。

如果把 reduced `state.json` 当成可以直接 resume 的 canonical session，就会越过源码边界。它是解释证据的图，不是驱动下一轮模型请求的权威历史。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-trace-graph-reducer.png" alt="rollout trace reducer 图：raw events 与 payload refs 写入 trace bundle，离线 reducer 再还原 threads、edges 和 tool evidence graph">
  <figcaption>图 7. Trace graph 很适合诊断交错运行证据；生产恢复仍依赖 rollout JSONL 和 thread/runtime 恢复路径。</figcaption>
</figure>

### 4.2 InteractionEdge 连接 parent、child 和 tool evidence

对多 Agent 来说，最关键的是 interaction edges。README 里有一段专门讲 multi-agent v2：child threads 和 root thread 共享 trace writer，一个 root bundle 可以 reduce 成包含 parent、child threads 和边的图，见 [`rollout trace multi-agent v2`][rollout-trace-multi-agent]。

reducer 的模型里 `InteractionEdgeKind` 包括 `spawn_agent`、`assign_agent_task`、`send_message`、`agent_result`、`close_agent`，见 [`InteractionEdgeKind`][interaction-edge-kind]。`rollout-trace reducer agents.rs` 会把 spawn/send/followup/close/result 转成这些 edge，尽量指向 recipient 侧真实 conversation item；如果 child 还没形成 task message，spawn edge fallback 到 thread，见 [`trace agent edges`][trace-agent-edges]。

shape-level 示例：

```json
{
  "interaction_edges": [
    {
      "edge_id": "edge:spawn:parent-thread:child-thread",
      "kind": "spawn_agent",
      "source": { "type": "tool_call", "tool_call_id": "call_spawn" },
      "target": { "type": "conversation_item", "item_id": "child_task_item" },
      "carried_raw_payload_ids": ["payload_spawn_end"]
    },
    {
      "edge_id": "edge:result:child-thread:parent-thread",
      "kind": "agent_result",
      "source": { "type": "conversation_item", "item_id": "child_answer" },
      "target": { "type": "conversation_item", "item_id": "parent_notice" }
    }
  ]
}
```

这个例子故意只保留 reducer 概念形状，不声称是完整 `state.json` schema。真正的价值是边界：普通 transcript 只能告诉你“parent 后来看到了一段文本”，trace graph 才能告诉你这段文本从哪个 tool call、哪个 child thread、哪个 result notification 流过来。

### 4.3 trace failure conditions

trace 也有自己的失败条件：

- 如果没设置 `CODEX_ROLLOUT_TRACE_ROOT`，trace context 可以 no-op，不能期待本地 bundle 存在。
- 如果 runtime payload 先到、model-visible delivery item 后到，reducer 会排队等待，不能把“当前 graph 还没目标”当成“消息没送达”。
- 如果 child 在 task message model-visible 前失败，spawn edge 可能 fallback 到 thread；这不是精度 bug，而是 reducer 对可见证据不足的降级。
- 如果把 runtime payload 当成模型一定看过的文本，就会混淆 debug evidence 和 model-visible conversation。README 明确说 runtime payload 是证据，不证明模型看到了同样字节。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v4-appserver-agent-trace-final-map.png" alt="Codex App Server final map：client protocol、thread/turn、ServerRequest、mailbox/state DB、rollout log recovery 和 trace graph diagnosis 的分离边界地图">
  <figcaption>图 8. 最终地图把“协议入口、runtime 执行、跨 agent 通信、生产恢复、诊断图谱”五层放在同一张边界图上；trace graph 不在 production recovery path 上。</figcaption>
</figure>


## 五、和云端 coding agent 的关系

### 5.1 只能做产品形态的相邻比较

OpenAI 的 [App Server 文档](https://developers.openai.com/codex/app-server)支撑的是较窄的公开契约：`codex app-server` 可以被 client 启动和连接，client 先完成 `initialize`/`initialized`，再通过 `thread/start`、`turn/start` 和 notifications 驱动 thread/turn 生命周期；源码能继续说明这些请求怎样进入本地 core runtime。它不证明 Codex Web 是否复用同一个 App Server binary，也不说明浏览器到服务端的 transport、后台 worker 或 task event schema。

把它放到云端 coding agent 趋势里时，更稳妥的证据来自各产品自己的文档。GitHub Copilot cloud agent 的官方文档描述了一个相邻方向：agent 可以在 GitHub Actions-powered ephemeral development environment 里研究仓库、制定计划、修改分支、执行测试/linters，并围绕分支和 pull request 让用户继续 review 与迭代（见 [GitHub Copilot cloud agent](https://docs.github.com/en/copilot/concepts/agents/cloud-agent/about-cloud-agent)）。

这只能说明产品形态的共同压力：coding agent 正在从“IDE 里一个聊天助手”演进成“有独立执行环境、有生命周期、有日志、有分支、有审批、有多人协作入口的软件执行系统”。对这个源码快照来说，可以确定的是 App Server 把本地/集成式 client 收束到协议化 core runtime 入口；托管云端服务内部如何部署和调度，不在当前证据边界内。

### 5.2 这里不能外推的部分

源码能说明 App Server 和 core 的公开 harness 形状，不能说明 OpenAI 托管 Codex Web 的全部内部调度策略。比如容器生命周期、后台 worker 可靠性、服务端任务队列、产品权限系统，本文只按官方文档做产品层引用，不把它们投射回开源 repo 里的某个私有实现。

## 六、源码阅读规则

上面的最终地图已经把 client protocol、thread/turn、`ServerNotification`/`ServerRequest`、child thread/mailbox/state DB、rollout JSONL 和 trace reducer graph 放回同一张边界图。下面的规则表只做最后压缩：protocol/runtime/mailbox/state DB 属于执行链，rollout JSONL 是生产恢复材料，trace bundle/reducer/graph 只属于诊断路径。


<table class="codex-mobile-table">
  <thead>
    <tr>
      <th>读到的模块</th>
      <th>先问什么</th>
      <th>正确边界</th>
      <th>常见失败条件</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>App Server</td>
      <td>它是在跑模型，还是在守协议？</td>
      <td>它是 protocol boundary，不是模型循环。</td>
      <td>未初始化、连接 opt-out、request serialization key 冲突。</td>
    </tr>
    <tr>
      <td>protocol v2</td>
      <td>为什么有类型转换和 experimental 标记？</td>
      <td>内部 Rust 类型和外部 client API 要解耦。</td>
      <td>client 没有 opt in 却发送或期待实验字段。</td>
    </tr>
    <tr>
      <td><code>ServerRequest</code></td>
      <td>为什么不是 notification？</td>
      <td>approval/user input/permissions 需要 client 回答。</td>
      <td>回填失败会按 conservative default 处理，如 decline 或 empty answers。</td>
    </tr>
    <tr>
      <td>TUI event channel</td>
      <td>为什么按 thread id 分流？</td>
      <td>UI 要能 buffer 非 active thread 事件。</td>
      <td>把 client 没收到 delta 误读成 core 没产生事件。</td>
    </tr>
    <tr>
      <td><code>spawn_agent</code></td>
      <td>它是不是函数递归？</td>
      <td>不是。它创建 child thread 和 <code>SessionSource::SubAgent(ThreadSpawn)</code>。</td>
      <td>resumed child 不能重复写 <code>ThreadStarted</code> trace event。</td>
    </tr>
    <tr>
      <td>mailbox</td>
      <td>为什么不直接写 history？</td>
      <td>要保留来源、turn 边界和 trace replay。</td>
      <td>把 mailbox message 当成普通用户消息，会丢掉 agent 来源边界。</td>
    </tr>
    <tr>
      <td>rollout trace</td>
      <td>它是不是生产恢复路径？</td>
      <td>不是。它是 opt-in 诊断图谱。</td>
      <td>把 reduced graph 当成 rollout JSONL 或 canonical model history。</td>
    </tr>
  </tbody>
</table>

## 小结

App Server 是 Codex harness 的协议边界，不只是 TUI 到 core 的 adapter。它用 `initialize` 管 client 能力和连接状态，用 thread/turn lifecycle 把多端请求统一送进 core，用 `ServerNotification`/`ServerRequest` 把 core 事件和审批请求翻译给 client，用 thread event buffer 支持 UI 的 thread 切换和事件回放，用 child thread、mailbox、state DB 把多 Agent 做成可恢复线程树，再用 rollout trace 把运行时证据还原成 graph。

所以 Codex 的 App Server 不是“多了一层服务端”。它是从单机 CLI agent 走向多端、多线程、多 Agent、可审计执行系统的关键中间层。

## 参考

- [OpenAI Developers: Codex App Server](https://developers.openai.com/codex/app-server)
- [OpenAI Developers: Codex CLI](https://developers.openai.com/codex/cli)
- [OpenAI Developers: Codex sandboxing](https://developers.openai.com/codex/concepts/sandboxing)
- [OpenAI Developers: Agent approvals & security](https://developers.openai.com/codex/agent-approvals-security)
- [GitHub Docs: About GitHub Copilot cloud agent](https://docs.github.com/en/copilot/concepts/agents/cloud-agent/about-cloud-agent)
- [`openai/codex@ac4332c05b11e00ae775a24cb762edc05c5b5932`][codex-pinned-commit]

[codex-pinned-commit]: https://github.com/openai/codex/tree/ac4332c05b11e00ae775a24cb762edc05c5b5932
[app-server-protocol-common]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/common.rs#L147-L340
[app-server-loop]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/lib.rs#L732-L930
[connection-state]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/message_processor.rs#L180-L246
[initialize-params]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/v1.rs#L28-L63
[message-processor]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/message_processor.rs#L162-L178
[v2-enum]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/v2.rs#L123-L151
[v2-error-info]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/v2.rs#L162-L230
[client-request-defs]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/common.rs#L431-L703
[thread-start-params]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/v2.rs#L3546-L3625
[turn-start-params]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/v2.rs#L5438-L5561
[codex-message-processor]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/codex_message_processor.rs#L520-L546
[server-notification-defs]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/common.rs#L1351-L1389
[turn-start]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/codex_message_processor.rs#L6529
[tui-app-server-adapter]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/app/app_server_adapter.rs#L1-L11
[bespoke-event-handling]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/bespoke_event_handling.rs#L164
[server-request-defs]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/common.rs#L1209-L1242
[approval-response-handling]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/bespoke_event_handling.rs#L2499-L2676
[request-user-input-handling]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/bespoke_event_handling.rs#L2209-L2258
[tui-handle-app-server-event]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/app/app_server_adapter.rs#L125
[agent-control]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/agent/control.rs#L129
[agent-path]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/agent_path.rs#L17
[spawn-agent-v2]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/handlers/multi_agents_v2/spawn.rs#L26
[parent-rollout-thread-trace]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/thread_manager.rs#L1172-L1191
[mailbox-drain]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/mod.rs#L3077-L3145
[inter-agent-communication]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/protocol.rs#L831-L866
[agent-child-complete]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/mod.rs#L1516-L1565
[thread-spawn-edges]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/state/src/runtime/threads.rs#L86-L112
[rollout-trace-readme]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/rollout-trace/README.md#L9-L20
[rollout-trace-multi-agent]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/rollout-trace/README.md#L172-L197
[interaction-edge-kind]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/rollout-trace/src/model/runtime.rs#L300-L329
[trace-agent-edges]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/rollout-trace/src/reducer/tool/agents.rs#L22-L58
