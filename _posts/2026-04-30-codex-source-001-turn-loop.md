---
classes: wide2
title: "Codex 源码剖析：001. 从 TUI 到 run_turn"
excerpt: "从一个用户输入出发，顺着 Codex CLI、TUI、in-process App Server、core Session 和 run_turn，把 Codex 的主循环走一遍。"
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

![Codex 源码剖析系列封面：终端、模型循环、工具和上下文线索交织在一起](/assets/images/posts/2026-04-30-codex-source/series-hero.png)

*图 0：这个系列关注的不是“模型会不会写代码”这一层，而是 Codex 如何把模型、工具、上下文、权限和 UI 放进一个可运行的工程系统。*

这个系列从源码角度看 Codex。

我用的 Codex 版本是 [`openai/codex@ac4332c05b11e00ae775a24cb762edc05c5b5932`](https://github.com/openai/codex/tree/ac4332c05b11e00ae775a24cb762edc05c5b5932)。如果后面源码演进了，具体文件名和行号可能会变，但本文关心的是当前这套 harness 的结构。

第一篇先不急着看工具、沙箱、MCP、memory 这些局部模块，而是从最朴素的问题开始：

> 我在终端里输入一句话之后，Codex 到底如何把它变成一次模型请求、工具调用、再把结果流式打回 TUI？

这个问题比看起来重要。很多 agent 项目源码读起来混乱，是因为 CLI、UI、模型循环、工具执行、权限判断全部揉在一个大循环里。Codex 现在的结构则不是这样。它把本地 TUI 也放到了 app-server protocol 后面，即使 server 和 TUI 在同一个进程内，也要走 `thread/start`、`turn/start`、notification/request 这套协议边界。

这也是我读完之后最想先讲的点：**Codex 的本地交互正在从简单 terminal app，迁向一个把 terminal 当作 client 的 agent harness**。

![Codex app-server-backed turn path：用户输入经过 TUI、in-process App Server、core Session、RegularTask 和模型流，再由事件回到界面](/assets/images/posts/2026-04-30-codex-source/main-loop.svg)

*图 1：读这张图时，从左侧 terminal 输入开始，顺着 `turn/start` 进到 core，再看右侧的 model stream 如何通过 event 反向回到 TUI。注意这里讲的是 app-server-backed turn path；TUI 源码里仍有一个临时 adapter，说明当前还在 hybrid migration period。*

![Codex 主循环术语图：Thread、Turn、Submission、ResponseItem 和 EventMsg 的关系](/assets/images/posts/2026-04-30-codex-source/turn-terms.svg)

*图 2：`thread` 是会话线，`turn` 是一次用户任务，`submission` 是进入 core 的操作，`ResponseItem` 是模型可见历史项，`EventMsg` 是 core 对外发出的事件。先把这几个词摆正，后面看源码会轻很多。*

## 为什么多了一层 App Server？

OpenAI 在 2025 年 5 月发布 Codex 时，把它描述成一个可以在云端独立环境里并行处理任务的软件工程 agent，能读写文件、运行测试、提交变更，并通过日志和测试结果提供可验证证据（见 [Introducing Codex](https://openai.com/index/introducing-codex/)）。后来他们又专门写了 App Server 这层 harness，解释 client 和 server 需要先做 `initialize` handshake，然后通过 thread/turn lifecycle 以及 progress notifications 协作（见 [Unlocking the Codex harness](https://openai.com/index/unlocking-the-codex-harness/)）。

所以回头看源码会发现，在已经迁到 app-server surface 的 turn path 上，Codex CLI 的 TUI 并没有直接调用 `codex-core` 里的 `run_turn()`。本地默认路径大致是：

```text
codex.js
  -> Rust CLI
  -> TUI
  -> in-process app-server
  -> app-server protocol ClientRequest::TurnStart
  -> core Op::UserInput / UserInputWithTurnContext
  -> Session
  -> RegularTask
  -> run_turn
  -> ModelClientSession::stream
  -> EventMsg
  -> app-server notification/request
  -> TUI render
```

这条链路看起来长，但它换来的东西很明确：TUI、VS Code、远程客户端、无头执行、未来的桌面端，都可以逐步共享同一套 thread/turn 语义。这里要保留一个源码事实：[`app_server_adapter.rs`][tui-app-server-adapter] 文件头写得很清楚，它是 TUI 和 app-server 之间的临时 adapter，当前仍处在 hybrid migration period，TUI 还保留已有 direct-core 行为。所以本文说“统一协议边界”，指的是这条 app-server-backed path 的方向和已经落地的主线，而不是说所有 TUI 行为都已经完全迁完。

所谓 harness，不只是“把模型包起来”，而是把模型、工具、审批、UI、恢复和观测都放到一个可复用协议里。

## CLI 入口：npm 只负责找到 native binary

从安装包入口看，`codex-cli/bin/codex.js` 的职责非常薄：根据平台和架构选择对应 optional dependency，找到 `codex` native binary，然后把控制权交给 Rust 程序。平台包映射就在 [`codex.js` 的 `PLATFORM_PACKAGE_BY_TARGET`][codex-js-platform] 里。

这意味着 Node 层不是业务运行时。真正的 CLI 入口在 Rust 的 [`codex-rs/cli/src/main.rs`][cli-root]。`MultitoolCli` 把 config override、feature toggles、remote options、interactive TUI 参数和 subcommand 都合在一起。没有 subcommand 时进入交互 TUI；`exec`、`review`、`mcp-server`、`app-server`、`sandbox` 等则是不同子命令分支。

这里有一个很容易忽略的设计点：**Codex CLI 不是一个单一命令，而是一个 multitool**。它既能作为人用的 TUI，也能作为无头执行器，还能启动 app-server、MCP server、sandbox 调试命令。把 npm 层做薄以后，这些模式都落在同一个 native 分发物里。

## TUI：ChatWidget 只负责交互状态，不负责跑 agent

TUI 入口是 [`codex-rs/tui/src/lib.rs` 的 `run_main`][tui-run-main]。在进入 ratatui 主循环前，它会决定 `AppServerTarget`：本地默认是 embedded，也就是 in-process app-server；远端则可以走 websocket。这个目标枚举在 [`AppServerTarget`][tui-app-server-target]。

`app-server/src/in_process.rs` 的注释把这点说得很直白：in-process server 保留 app-server 语义，只是去掉进程边界。也就是说，本地 TUI 和远程 client 的差别不是“是否走协议”，而只是“协议跑在进程内还是进程外”。

TUI 内部有两个关键角色：

1. [`App`][tui-app] 是总调度器，持有 active thread、thread event channels、pending requests 等状态。
2. [`ChatWidget`][tui-chatwidget] 是聊天界面状态机，处理键盘、粘贴、图片、mentions、底部输入框、流式文本渲染。

`ChatWidget` 的职责边界很重要。它会把用户行为转换成 `UserMessage` / `UserInput`，但不直接跑模型。真正提交时，它构造 `AppCommand::user_turn`，里面包含：

- `items`
- `cwd`
- approval policy
- permission profile
- model
- reasoning effort
- service tier
- collaboration mode
- personality

也就是说，用户输入和本 turn 的运行配置是一起被提交的。这个小细节很关键：agent 的行为不是只由文本 prompt 决定，还由 cwd、权限、沙箱、模型、协作模式等运行时上下文决定。

## UserInput：UI/core 之间的中间语言

Codex 不是把输入框里的字符串直接塞进模型。

底部输入框先产生 `InputResult::Submitted` 或其他命令结果；`ChatWidget` 再把它组装成 `UserInput`。`UserInput` 定义在 [`codex-rs/protocol/src/user_input.rs`][user-input]，里面不只是纯文本，还包括 remote image、local image、skill、structured mention 等类型。

到真正喂给模型之前，`Vec<UserInput>` 会转换成一条 user `ResponseInputItem::Message`，转换逻辑在 [`protocol/src/models.rs`][user-input-models]。但是 skill / mention 不会在这里简单展开成文本，因为它们后面还会被 core 解析成更精确的上下文注入，例如显式技能触发、插件能力、app connector 选择等。

这就是 Codex 的一个常见模式：**UI 层保留用户意图，core 层负责把意图解释成模型可见上下文和工具集合**。

## App Server：turn/start 是进入 core 的窄门

TUI 通过 active thread routing 把 `AppEvent::CodexOp` 发给 `AppServerSession::turn_start`，然后构造 app-server protocol 的 `ClientRequest::TurnStart`。协议中的 `TurnStartParams.input` 仍然是 `Vec<UserInput>`，v2 protocol 再把它映射回 core 的 input 类型。

app-server 的分发入口在 [`CodexMessageProcessor::process_request`][app-process-request]，`TurnStart` 分支最终进入 [`turn_start`][app-turn-start]。这里会做几件事：

1. 校验输入大小和 thread 状态。
2. 把 v2 input 转成 core input。
3. 处理 cwd、model、approval、sandbox 等 turn override。
4. 构造 core 的 `Op::UserInputWithTurnContext` 或 `Op::UserInput`。
5. 通过 `CodexThread::submit_with_trace` 投递给 core。

所以 app-server 的意义不是“传话”。它是 client 请求进入 core 前的 protocol boundary：校验、转换、路由、状态管理和兼容都在这里发生。

## Core：Codex 是一个 Submission/Event 队列对

到 core 之后，真正的抽象是 `Codex`。它在 [`core/src/session/mod.rs`][codex-core-queue] 里被描述为一个 queue pair：一边收 `Submission/Op`，一边吐 `Event`。

`Codex::spawn` 会创建 `Session` 并启动后台 `submission_loop`；`submit` 只是生成 submission id 并送入队列。`Session` 自己的注释更关键：一个 session 代表初始化后的 model agent，同一时刻最多只有一个 running task，可以被用户输入 interrupt 或 steer（见 [`Session` 结构体][session-struct]）。

这解释了为什么 Codex 对用户输入有两种处理：

- 如果当前没有 active turn，就创建新 turn 并启动 `RegularTask`。
- 如果当前已有 active turn，就尝试 `steer_input`，把新用户输入导入当前 active turn。

用户 turn 的处理在 [`user_input_or_turn_inner`][user-input-or-turn]。它先把 `Op` 拆成 `items + SessionSettingsUpdate`，应用 turn 配置；如果已有 active turn，就通过 `steer_input` 把新输入导入当前任务；如果没有 active turn，就 `spawn_task(... RegularTask::new())`。

## RegularTask：真正进入 run_turn

`RegularTask::run` 很短，但非常关键。它先发 `TurnStarted`，然后循环调用 [`run_turn`][regular-task]。

为什么是循环？因为一次用户 turn 可能不是一次模型请求。模型可能先返回一个 tool call，Codex 执行工具，把工具输出写回 history，然后再发下一次 sampling request。直到模型返回普通 assistant message，或者没有 follow-up 需要继续，turn 才算完成。

如果把 [`RegularTask::run`][regular-task] 和 `run_turn` 合起来看，它的形状可以简化成下面这样。这里不是逐字摘源码，而是把控制流压到最小：

```text
emit TurnStarted
loop:
  prompt = build_prompt(history, visible_tools, turn_settings)
  stream = model.stream(prompt)
  for item in stream:
    if item is assistant delta:
      emit EventMsg delta
    if item is tool call:
      run tool through ToolRouter / ToolRegistry
      append tool output to history
      mark needs_follow_up
  if needs_follow_up:
    continue
  emit TurnComplete
  break
```

这段伪代码里最重要的是 `needs_follow_up`。工具输出不是给 UI 看完就结束，而是会变成下一次模型请求的一部分，所以一个 turn 内部可以有多次 model sampling。

`run_turn` 源码注释其实就是 agent loop 的最小定义。它说明：

- 模型返回 function call 时，Codex 执行对应工具，并把工具输出加入下一次 sampling request。
- 模型只返回 assistant message 时，把消息加入 conversation history，并认为 turn 完成。

这和 ReAct 论文里“reasoning traces 与 actions 交错进行”的思想是一脉相承的（见 [ReAct](https://arxiv.org/abs/2210.03629)）。只是到了 Codex 这里，action 不再是 prompt 里的自由文本，而是有 schema、有分发、有审批、有沙箱、有事件回流的结构化运行时。

## build_prompt：模型请求长什么样

真正构造模型请求的是 [`build_prompt`][build-prompt]。它把下面几类东西组装成 `Prompt`：

- `input`：当前 history / 本轮新增输入。
- `tools`：`router.model_visible_specs()`，也就是模型本次可见的工具。
- `parallel_tool_calls`：由模型能力决定是否允许并行工具调用。
- `base_instructions`：模型基础指令。
- `personality`
- `output_schema`

注意这里的 `tools` 来自 `ToolRouter`，而不是 handler 直接暴露。模型看到的是工具 spec；core 维护的是 handler registry、并发策略、审批和沙箱。下一篇会展开这个工具系统。

模型流式调用在 [`try_run_sampling_request`][try-run-sampling]。它创建 stream 后循环读取 `ResponseEvent`：

- `OutputItemAdded`：通知 UI 某个 item 开始，例如 assistant message 或 custom tool call。
- `OutputTextDelta`：流式文本 delta。
- `OutputItemDone`：一个完整 item 结束，如果是工具调用，就进入工具分发。
- `Completed`：本次 sampling request 完成，更新 token usage，决定是否需要 follow-up。

工具调用的关键分支是 [`handle_output_item_done`][handle-output-item-done]。它调用 `ToolRouter::build_tool_call` 识别工具，然后创建 in-flight future。只要有工具调用，就设置 `needs_follow_up = true`，因为工具输出要反馈给模型继续推理。

这就是 Codex turn loop 的心跳：

```text
prompt -> model stream -> tool call -> tool output -> next prompt
```

## 事件如何回到 UI

core 不直接操作 TUI。它只发 `EventMsg`。app-server listener 循环读取 `conversation.next_event()`，再通过 `apply_bespoke_event_handling` 把 core 事件翻译成 app-server protocol 的 `ServerNotification` 或 `ServerRequest`。

例如：

- `TurnStarted` 变成 `ServerNotification::TurnStarted`
- `AgentMessageContentDelta` 变成 `AgentMessageDelta`
- `RequestUserInput` 变成 `ServerRequest::ToolRequestUserInput`
- `TurnComplete` 变成 `TurnCompleted`

TUI 的 [`handle_app_server_event`][tui-handle-server-event] 再根据 thread id 把事件投递到对应 thread channel。active thread 的事件最后进入 `ChatWidget::handle_server_notification` 或 `handle_server_request`，更新界面状态、追加流式文本、弹出 approval、或者收尾 turn。

这个方向上再看一遍，就能理解为什么 Codex 的 UI 层和 core 层分得很开：UI 看到的是 protocol event，而不是 core 内部函数调用。

## 这一篇的核心结论

Codex 主循环不是“CLI 里调一个模型 API，然后解析工具调用”这么简单。

它更像这样：

1. npm 层只负责分发 native binary。
2. Rust CLI 是 multitool 入口。
3. 本地 TUI 默认启动 in-process app-server。
4. 用户输入先变成 `UserInput`，再通过 `turn/start` 进入 core。
5. core 里 `Codex` 是 submission/event 队列对，`Session` 维护同一时刻一个 active task。
6. `RegularTask` 调 `run_turn`，`run_turn` 把 model stream、工具调用、history、token usage、compaction 串起来。
7. 所有 UI 更新都通过事件回流，而不是 core 直接操纵界面。

所以 Codex 现在的源码重心，其实不在“写一个 chat CLI”，而在“构建一个可复用的 agent harness”。这也是后面几篇看工具、上下文、沙箱、多 Agent、trace 时需要一直带着的主线。

## 参考

- [Introducing Codex](https://openai.com/index/introducing-codex/)
- [Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/)
- [Unlocking the Codex harness: how we built the App Server](https://openai.com/index/unlocking-the-codex-harness/)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

[codex-js-platform]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-cli/bin/codex.js#L15-L22
[cli-root]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/cli/src/main.rs#L70-L176
[tui-run-main]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/lib.rs#L678
[tui-app-server-target]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/lib.rs#L284
[tui-app-server-adapter]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/app/app_server_adapter.rs#L1-L11
[tui-app]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/app.rs#L510
[tui-chatwidget]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/chatwidget.rs#L784
[user-input]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/user_input.rs#L13
[user-input-models]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/models.rs#L1200
[app-process-request]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/codex_message_processor.rs#L971
[app-turn-start]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/codex_message_processor.rs#L6529
[codex-core-queue]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/mod.rs#L363
[session-struct]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/session.rs#L7-L32
[user-input-or-turn]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/handlers.rs#L123-L280
[regular-task]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tasks/regular.rs#L40-L86
[build-prompt]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L935-L952
[try-run-sampling]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L1809-L2105
[handle-output-item-done]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/stream_events_utils.rs#L219-L255
[tui-handle-server-event]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/app/app_server_adapter.rs#L125
