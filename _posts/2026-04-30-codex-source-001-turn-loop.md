---
classes: wide2
title: "Codex 源码剖析：001. 从 TUI 到 run_turn"
excerpt: "从用户输入、TUI、in-process App Server、core Session 到 run_turn，重建 Codex 一次 turn 的协议边界、模型上下文和事件回流。"
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

把 Codex 当成一个“终端里的聊天程序”，源码会很容易读错。用户在 TUI 里敲下一句话，表面上只是输入框提交、模型流式回复；真实路径却更像一个小型 agent runtime：TUI 是 client，App Server 是协议边界，core 维护 session 和 task，`run_turn` 再把模型流、工具调用、history 和下一次采样串成一个可恢复的用户任务。

本文使用的源码版本是 [`openai/codex@ac4332c`](https://github.com/openai/codex/tree/ac4332c05b11e00ae775a24cb762edc05c5b5932)。行号后续可能漂移，但这里关心的是更稳定的结构问题：

> 一次用户输入到底穿过了哪些边界，才变成一次可恢复、可审计、可继续执行的 agent turn？

![Codex 一次 turn 的封面图：TUI client、App Server、Core Session、run_turn 和 EventMsg 围成可恢复的执行闭环](/assets/images/posts/2026-04-30-codex-source/turn-loop-cover.png)

*图 1. 这篇文章只讨论 app-server-backed 主路径。TUI 仍处在 hybrid migration period，部分旧路径还没有完全迁出 direct-core 形态。*

## 阅读契约

读这篇时可以只抓四个问题：

1. TUI 到底在哪一层停止，core 又从哪一层开始？
2. `UserInput` 为什么不是一段直接喂给模型的字符串？
3. 一次 turn 为什么可能包含多次模型请求？
4. UI 为什么应该消费 protocol event，而不是自己推断 core 状态？

如果这四个问题能回答清楚，后面读工具、审批、沙箱、上下文压缩和多 Agent trace 时，就不会把“界面看到的聊天记录”和“模型下一次请求看到的上下文”混在一起。

## 一、先把 turn 看成一组边界

### 1.1 不要从“主函数”开始找

很多源码阅读会从“哪个函数调用模型 API”开始。这对 Codex 不够。`run_turn` 当然关键，但它不是用户输入的入口，也不是 UI 的出口。当前主路径可以压成这样：

```text
codex.js
  -> Rust CLI
  -> TUI
  -> in-process App Server
  -> ClientRequest::TurnStart
  -> Op::UserInput / UserInputWithTurnContext
  -> Session
  -> RegularTask
  -> run_turn
  -> ModelClientSession::stream
  -> EventMsg
  -> ServerNotification / ServerRequest
  -> TUI render
```

这条链路换来的不是“多绕几层”，而是统一生命周期：本地 TUI、远程 client、无头执行和未来其他前端，都可以逐步共享 thread/turn 语义。[`app_server_adapter.rs`][tui-app-server-adapter] 文件头也提醒我们：这仍是迁移期 adapter，不能把它误读成所有 TUI 行为都已经完全协议化。

![Codex app-server-backed turn path：用户输入经过 TUI、in-process App Server、core Session、RegularTask 和模型流，再由事件回到界面](/assets/images/posts/2026-04-30-codex-source/main-loop.png)

*图 2. 主路径不是一条函数调用栈，而是 client surface、protocol boundary、core queue 和 model/runtime loop 的组合。*

### 1.2 五个词必须分开

源码里最容易混的不是函数名，而是状态名。下面这张图把几个核心词放在同一条边界图里：

![Codex 主循环术语图：Thread、Turn、Submission、ResponseItem 和 EventMsg 的关系](/assets/images/posts/2026-04-30-codex-source/turn-terms.png)

*图 3. `Thread` 是会话线，`Turn` 是一次用户任务，`Submission/Op` 是进入 core 的操作，`ResponseItem` 是模型可见历史项，`EventMsg` 是 core 对外吐出的事件。*

| 概念 | 它回答的问题 | 常见误读 |
| --- | --- | --- |
| `Thread` | 这条会话线是谁？能不能切换、恢复、fork？ | 把它当成单轮 prompt。 |
| `Turn` | 当前用户任务的生命周期到哪里了？ | 把一次 turn 等同于一次模型请求。 |
| `Submission/Op` | 外部把什么操作投递给 core？ | 以为 UI 直接调用 `run_turn`。 |
| `ResponseItem` | 模型下一次请求能看到什么历史？ | 把 UI transcript 当成完整模型上下文。 |
| `EventMsg` | core 如何把进度、文本、审批、完成状态通知出去？ | 以为 core 直接操作 TUI。 |

后面所有细节都围绕一个判断：某个对象到底跨过的是 UI 边界、协议边界、core 队列边界，还是模型上下文边界。

## 二、从 CLI 到 App Server

### 2.1 npm 入口只负责找到 native binary

安装包里的 `codex-cli/bin/codex.js` 很薄。它根据平台和架构选择 optional dependency，找到对应的 `codex` native binary，然后把控制权交给 Rust 程序。平台包映射在 [`codex.js` 的 `PLATFORM_PACKAGE_BY_TARGET`][codex-js-platform]。

真正的 CLI 入口在 Rust 的 [`codex-rs/cli/src/main.rs`][cli-root]。`MultitoolCli` 把 interactive TUI、`exec`、`review`、`mcp-server`、`app-server`、`sandbox`、debug 等入口放在同一个 native 分发物里。这意味着 Codex CLI 不是一个单一聊天命令，而是一个 multitool。TUI 只是其中一个 client surface。

### 2.2 TUI 管交互状态，不管模型循环

TUI 入口是 [`codex-rs/tui/src/lib.rs` 的 `run_main`][tui-run-main]。进入 ratatui 主循环前，它会决定 `AppServerTarget`：本地默认是 embedded，也就是 in-process app-server；远端可以走 websocket。这个目标枚举在 [`AppServerTarget`][tui-app-server-target]。

TUI 内部可以先分成两个角色：

- [`App`][tui-app] 是总调度器，持有 active thread、thread event channels、pending requests 等状态。
- [`ChatWidget`][tui-chatwidget] 是聊天界面状态机，处理键盘、粘贴、图片、mentions、输入框和流式文本渲染。

`ChatWidget` 会把用户行为转换成 `UserMessage` / `UserInput`，但不直接跑模型。真正提交时，它构造 `AppCommand::user_turn`，一起带上 `cwd`、approval policy、permission profile、model、reasoning effort、service tier、collaboration mode、personality 等 turn settings。

所以，一次 turn 不是“用户文本”本身，而是“用户文本 + 当前运行约束 + 当前 client/thread 状态”的组合。

### 2.3 `UserInput` 是中间语言

Codex 不会把输入框里的字符串直接塞给模型。底部输入框先产生 `InputResult::Submitted` 或其他命令结果；`ChatWidget` 再组装成 `UserInput`。`UserInput` 定义在 [`codex-rs/protocol/src/user_input.rs`][user-input]，里面不只有文本，还包括 remote image、local image、skill、structured mention 等类型。

![UserInput 结构图：text、image、skill、mention 和 turn settings 共同组成 UserInputWithTurnContext](/assets/images/posts/2026-04-30-codex-source/input-structure.png)

*图 4. UI 层保留用户意图，core 层再把它解释成模型可见上下文、工具集合和运行约束。*

到真正喂给模型之前，`Vec<UserInput>` 会转换成 user `ResponseInputItem::Message`，转换逻辑在 [`protocol/src/models.rs`][user-input-models]。但 skill、mention 这类结构不会在 UI 层简单展开，因为 core 后面还要根据它们决定是否做技能注入、插件能力选择或 app connector 处理。

这是 Codex 常见的工程模式：越靠近 UI，越保留用户意图；越靠近模型请求，越需要把意图投影成 provider API 可以理解的 message、tool 和 instruction。

### 2.4 `turn/start` 是进入 core 的窄门

TUI 通过 active thread routing 把 `AppEvent::CodexOp` 发给 `AppServerSession::turn_start`，构造 app-server protocol 的 `ClientRequest::TurnStart`。协议中的 `TurnStartParams.input` 仍然是 `Vec<UserInput>`，v2 protocol 再映射回 core input 类型。

![turn/start 边界图：ClientRequest 经过 validate、override、Op 构造和 submit_with_trace 后进入 core](/assets/images/posts/2026-04-30-codex-source/turn-start-boundary.png)

*图 5. App Server 是 core 前的 protocol boundary：校验、转换、路由、兼容和状态管理都在这里发生。*

app-server 的分发入口是 [`CodexMessageProcessor::process_request`][app-process-request]，`TurnStart` 分支最终进入 [`turn_start`][app-turn-start]。这里会做几件事：

1. 校验输入大小和 thread 状态。
2. 把 v2 input 转成 core input。
3. 处理 cwd、model、approval、sandbox 等 turn override。
4. 构造 core 的 `Op::UserInputWithTurnContext` 或 `Op::UserInput`。
5. 通过 `CodexThread::submit_with_trace` 投递给 core。

App Server 因此不是传话筒。它是 client 请求进入 core 前的窄门，也是把不同 client surface 收束到同一套 turn 语义上的地方。OpenAI 的 [Codex App Server 文章][openai-app-server] 讲的是产品化集成视角；从源码读，它的核心价值就是把 UI surface 和 agent harness 分开。

## 三、core 内部如何跑完一次 turn

### 3.1 `Codex` 是 submission/event 队列对

进入 core 后，关键抽象是 `Codex`。它在 [`core/src/session/mod.rs`][codex-core-queue] 里被描述为一个 queue pair：一边收 `Submission/Op`，一边吐 `Event`。

`Codex::spawn` 创建 `Session` 并启动后台 `submission_loop`；`submit` 只负责生成 submission id 并送入队列。`Session` 的注释更关键：一个 session 代表初始化后的 model agent，同一时刻最多只有一个 running task，可以被用户输入 interrupt 或 steer，见 [`Session` 结构体][session-struct]。

这解释了用户输入的两种处理方式：

- 当前没有 active turn：创建新 turn 并启动 `RegularTask`。
- 当前已有 active turn：尝试 `steer_input`，把新用户输入导入当前 active turn。

用户 turn 的处理入口在 [`user_input_or_turn_inner`][user-input-or-turn]。它先拆出 `items + SessionSettingsUpdate`，应用 turn 配置；已有 active turn 时走 steering，没有 active turn 时才 `spawn_task(... RegularTask::new())`。

### 3.2 一次 turn 可以包含多次模型请求

[`RegularTask::run`][regular-task] 很短，却是理解 agent loop 的关键。它先发 `TurnStarted`，然后循环调用 `run_turn`。

为什么是循环？因为一次用户 turn 不一定只对应一次 sampling request。模型可能先返回 tool call，Codex 执行工具，把工具输出写回 history，再发下一次模型请求。直到模型返回普通 assistant message，或者没有 follow-up 需要继续，turn 才算完成。

![一次 turn 包含多次 sampling：TurnStarted、sampling A、tool output、history、sampling B、TurnComplete](/assets/images/posts/2026-04-30-codex-source/sampling-followup-loop.png)

*图 6. `needs_follow_up` 是读 `run_turn` 的关键：工具输出不是 UI 展示后就结束，而是会变成下一次模型请求的一部分。*

压成控制流：

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

这段伪代码的核心是 `needs_follow_up`。工具输出不是给 UI 看完就结束，而是会变成下一次模型请求的一部分。所以 Codex 的 turn 是“用户任务生命周期”，不是“模型请求次数”。OpenAI 的 [Codex agent loop][openai-codex-loop] 文章讲的也是这个闭环：模型、工具、环境反馈持续交替，直到任务有明确收束点。

### 3.3 `build_prompt` 决定模型本次能看见什么

真正构造模型请求的是 [`build_prompt`][build-prompt]。它把下面几类东西组装成 `Prompt`：

- `input`：当前 history / 本轮新增输入。
- `tools`：`router.model_visible_specs()`，也就是模型本次可见的工具。
- `parallel_tool_calls`：由模型能力决定是否允许并行工具调用。
- `base_instructions`：模型基础指令。
- `personality`
- `output_schema`

注意 `tools` 来自 `ToolRouter`，而不是 handler 直接暴露。模型看到的是工具 spec；core 维护的是 handler registry、并发策略、审批和沙箱。下一篇会专门拆这套工具系统。

模型流式调用在 [`try_run_sampling_request`][try-run-sampling]。它创建 stream 后循环读取 `ResponseEvent`：`OutputItemAdded` 表示 item 开始，`OutputTextDelta` 是流式文本，`OutputItemDone` 可能触发工具分发，`Completed` 则更新 token usage 并决定是否需要 follow-up。

工具调用的关键分支是 [`handle_output_item_done`][handle-output-item-done]。它调用 `ToolRouter::build_tool_call` 识别工具，然后创建 in-flight future。只要有工具调用，就设置 `needs_follow_up = true`，因为工具输出还要反馈给模型继续推理。

## 四、事件如何回到 UI

core 不直接操作 TUI。它只发 `EventMsg`。app-server listener 循环读取 `conversation.next_event()`，再通过 `apply_bespoke_event_handling` 把 core 事件翻译成 app-server protocol 的 `ServerNotification` 或 `ServerRequest`。

![事件回流图：EventMsg 经过 App Server 翻译成 ServerNotification/ServerRequest，再按 thread id 路由到 ChatWidget](/assets/images/posts/2026-04-30-codex-source/event-return-path.png)

*图 7. UI 消费的是 protocol event，而不是重新读取 core 内部 task 状态。approval 这类安全请求必须沿着同一条事件链回到界面。*

例如：

- `TurnStarted` 变成 `ServerNotification::TurnStarted`
- `AgentMessageContentDelta` 变成 `AgentMessageDelta`
- `RequestUserInput` 变成 `ServerRequest::ToolRequestUserInput`
- `TurnComplete` 变成 `TurnCompleted`

TUI 的 [`handle_app_server_event`][tui-handle-server-event] 再根据 thread id 把事件投递到对应 thread channel。active thread 的事件最后进入 `ChatWidget::handle_server_notification` 或 `handle_server_request`，更新界面状态、追加流式文本、弹出 approval，或者收尾 turn。

这条回流路径解释了一个设计原则：UI 只消费 protocol event，不应该重新推断 core 内部状态。尤其是 approval 这类安全相关请求，必须由 core policy 触发，再通过 protocol request 回到 UI。

## 五、源码阅读规则

| 读到的代码 | 先问什么 | 判断结果 |
| --- | --- | --- |
| TUI / `ChatWidget` | 它是在管理界面状态，还是在提交 turn？ | TUI 不直接跑模型，它把意图和 turn settings 交出去。 |
| `UserInput` | 这是原始文本，还是结构化用户意图？ | 图片、skill、mention 都要保留结构，不能提前拍平成字符串。 |
| `turn/start` | 这里做了哪些协议校验和类型映射？ | App Server 是 core 前的窄门，不是简单转发。 |
| `Session` | 当前是否已有 active task？ | 新输入可能启动新 turn，也可能 steer 当前 turn。 |
| `run_turn` | 这次 sampling 后是否需要 follow-up？ | 一次 turn 可以包含多次模型请求。 |
| `EventMsg` | 事件是给谁消费的？ | core 只吐事件，UI 通过 app-server protocol 渲染。 |

读 Codex 主循环时，不要追问“哪个函数就是整个 agent”。更稳的读法是沿着四条边界走：UI 边界、协议边界、core 队列边界、模型上下文边界。只要某个概念跨过边界，就先问它的形态有没有变：UI 看到的是 transcript，App Server 看到的是 protocol request，core 看到的是 `Op`，模型看到的是 `ResponseInputItem` 和 tool spec。

## 小结

Codex 主循环不是“CLI 调模型 API，然后解析工具调用”。它更像一个可复用 agent harness 的最小闭环：npm 层分发 native binary，Rust CLI 分发多种入口，TUI 通过 in-process app-server 提交 turn，core 用 submission/event 队列管理 session，`RegularTask` 和 `run_turn` 把模型流、工具调用和 history 串成一个可继续的用户任务。

这也是后面几篇继续看工具、审批、沙箱、上下文、App Server、多 Agent 和 trace 时需要带着的主线：Codex 的价值不只是会调用模型，而是把模型放进了一个有协议、有约束、有恢复路径的软件系统里。

## 参考

- [Introducing Codex](https://openai.com/index/introducing-codex/)
- [Unrolling the Codex agent loop][openai-codex-loop]
- [Unlocking the Codex harness: how we built the App Server][openai-app-server]
- [Using tools in the OpenAI API][openai-responses-tools]
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

[openai-codex-loop]: https://openai.com/index/unrolling-the-codex-agent-loop/
[openai-app-server]: https://openai.com/index/unlocking-the-codex-harness/
[openai-responses-tools]: https://developers.openai.com/api/docs/guides/tools
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
