---
classes:
  - wide2
  - codex-source-001-turn-loop
title: "Codex 源码剖析：001. 从 TUI 到 run_turn"
excerpt: "从用户输入、TUI、in-process App Server、core Session 到 run_turn，重建 Codex 一次 turn 的协议边界、模型上下文和事件回流。"
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
  og_image: https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v4-turn-loop-cover.png
  teaser: https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v4-turn-loop-cover.png
---

<div class="wc-turn-loop" markdown="1">

把 Codex 当成一个“终端里的聊天程序”，源码会很容易读错。用户在 TUI 里敲下一句话，表面上只是输入框提交、模型流式回复；真实路径却更像一个小型 agent runtime：TUI 是 client surface，App Server 是协议边界，core 维护 session 和 task，[`run_turn`][run-turn] 再把模型流、工具调用、history 和下一次采样串成一个可恢复的用户任务。

本文使用的源码版本是 [`openai/codex@ac4332c`](https://github.com/openai/codex/tree/ac4332c05b11e00ae775a24cb762edc05c5b5932)。这里的源码判断只依赖公开文档和 pinned source，不推断不可见的服务端内部实现；本文解释的是这个固定快照里的结构问题：

> 一次用户输入到底穿过了哪些边界，才变成一次可恢复、可审计、可继续执行的 agent turn？

![Codex 一次 turn 的执行闭环：TUI、App Server、Session、run_turn 和 EventMsg 串成可恢复 agent runtime](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v4-turn-loop-cover.png)

*图 1. 封面只保留本文要追踪的关键边界：TUI、App Server、core、模型流和事件回流。*

## 阅读契约

### 0.1 四个问题

读这篇时可以只抓四个问题：

1. TUI 到底在哪一层停止，core 又从哪一层开始？
2. `UserInput` 为什么不是一段直接喂给模型的字符串？
3. 一次 turn 为什么可能包含多次模型请求？
4. UI 为什么应该消费 protocol event，而不是自己推断 core 状态？

如果这四个问题能回答清楚，后面读工具、审批、沙箱、上下文压缩和多 Agent trace 时，就不会把“界面看到的聊天记录”和“模型下一次请求看到的上下文”混在一起。

### 0.2 本文边界

本文只讨论 app-server-backed 主路径。Codex TUI 仍处在 hybrid migration period，部分旧路径还没有完全迁出 direct-core 形态；[`app_server_adapter.rs`][tui-app-server-adapter] 文件头也明确提示它是迁移期 adapter。因此，下面的链路不是“所有历史路径都长这样”，而是当前源码中最值得作为后续文章基线的主路径。

## 一、先讲 provider 和 platform contract

源码链路之前，要先把公开平台契约分开。否则读者很容易从 `run_turn` 倒推，误以为 Codex CLI、App Server 和 OpenAI tools contract 都只是一个“大模型调用封装”。

![Codex App Server contract：多个 client surface 进入 App Server gate，再以 thread、turn、request 和 notification 语义连接 core queue](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v4-app-server-contract.png)

*图 2. App Server contract 图只保留 client surface、protocol boundary、agent harness 和 event stream 的关键关系。*

### 1.1 Codex CLI 的公开契约

OpenAI 的 [Codex CLI 文档][openai-codex-cli] 把 CLI 定义为本地终端里的 coding agent：它能在选定目录里读代码、改代码、运行命令。这个产品契约先于源码实现：用户拿到的是一个本地 native CLI，而不是一个需要自己拼装 protocol 的库。

#### CLI/native 边界

源码上，npm 包入口 [`codex-cli/bin/codex.js`][codex-js-platform] 只负责识别平台和架构，选择 optional dependency 里的 native binary，然后把控制权交给 Rust 程序。真正的 Rust CLI 入口在 [`codex-rs/cli/src/main.rs`][cli-root]，`MultitoolCli` 把 interactive TUI、`exec`、`mcp-server`、`app-server`、`sandbox`、debug 等入口放在同一个 native 分发物里。

![Codex CLI 到 TUI 的边界：codex.js 选择 native binary，Rust MultitoolCli 分发到 TUI、exec、app-server 和 sandbox，TUI 再构造 AppCommand::user_turn](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v4-cli-tui-boundary.png)

*图 3. CLI/TUI 边界图只画 native 分发物、入口分发和 TUI surface；模型循环还没有开始。*

#### TUI 只是一个 client surface

因此，TUI 不是 Codex 的全部，只是 `codex` native binary 里的一个交互入口。后面看到 `ChatWidget` 或 `App` 时，要先记住它们属于 client surface；它们管理交互状态、线程选择、输入框和渲染，而不是直接拥有模型循环。

### 1.2 App Server 的公开契约

OpenAI 的 [Codex App Server 文档][openai-app-server] 把 app-server 定义成 rich clients 使用的接口，用来承载 authentication、conversation history、approvals 和 streamed agent events。这个契约说明 App Server 不是“多绕一层转发”，而是把不同前端收束到同一套 thread/turn/event 语义上。

#### rich client contract

源码里的 `AppServerTarget` 正好落在这个契约上。TUI 入口 [`run_main`][tui-run-main] 会选择 [`AppServerTarget`][tui-app-server-target]：本地默认走 embedded，也就是 in-process app-server；远端可以走 websocket。TUI 和 App Server 即使跑在同一进程里，概念上也已经被 protocol boundary 隔开。

#### thread、turn、event 是契约词

App Server contract 里最重要的不是“HTTP 还是 websocket”，而是 `thread`、`turn`、`request`、`notification`、`approval` 这类词。后面读 `ClientRequest::TurnStart`、`ServerNotification` 和 `ServerRequest` 时，要把它们看成跨 client 的协议词，而不是某个 TUI widget 的局部事件。

### 1.3 OpenAI tools contract

OpenAI API 的 [tools 文档][openai-responses-tools] 和 [function calling 文档][openai-function-calling] 描述的是更底层的工具调用契约：应用把 tool specs 给模型，模型产生 tool call，应用执行工具，再把 tool output 作为后续输入交回模型。这个 contract 解释了为什么一次 Codex turn 可能包含多次 sampling request。

#### 工具调用不是一次请求内的魔法

从平台契约看，工具执行属于 application/runtime 责任，不是 provider 在隐藏环境里替应用完成。Codex core 的 `ToolRouter`、审批、沙箱和 follow-up loop，就是把这个 tools contract 落到本地 agent harness 的源码形态。

### 1.4 Codex cloud 与本地 runtime 的分工

[Codex web / cloud 文档][openai-codex-cloud] 讲的是把任务委派到云端环境。本文不分析 cloud implementation，只用它提醒一个边界：Codex 产品有多个 surface，本地 TUI 只是其中一个。源码阅读要避免把“本地 TUI 的渲染细节”误认为“Codex agent contract 的全部”。

## 二、用户输入如何离开 TUI

### 2.1 主路径不是一条函数调用栈

当前 app-server-backed 主路径可以压成这样：

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

这条链路换来的不是“多绕几层”，而是统一生命周期：本地 TUI、远程 client、无头执行和未来其他前端，都可以逐步共享 thread/turn 语义。

![Codex 主路径：TUI client 的 UserInput 经过 App Server turn/start、core Op queue、RegularTask、Model stream，再以 EventMsg 回到界面](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v4-main-loop.png)

*图 4. 主路径图只保留从 TUI 到模型流、再从事件回到 UI 的关键节点。*

### 2.2 五个词必须分开

源码里最容易混的不是函数名，而是状态名。下面几个词分别跨过不同边界：

<section class="wc-card-grid" aria-label="Codex turn loop terms">
  <article class="wc-card">
    <h4 class="no_toc"><code>Thread</code></h4>
    <p><span class="wc-label">回答的问题</span>这条会话线是谁，能不能切换、恢复、fork。</p>
    <p><span class="wc-label">常见误读</span>把它当成单轮 prompt。</p>
  </article>
  <article class="wc-card">
    <h4 class="no_toc"><code>Turn</code></h4>
    <p><span class="wc-label">回答的问题</span>当前用户任务的生命周期到哪里了。</p>
    <p><span class="wc-label">常见误读</span>把一次 turn 等同于一次模型请求。</p>
  </article>
  <article class="wc-card">
    <h4 class="no_toc"><code>Submission/Op</code></h4>
    <p><span class="wc-label">回答的问题</span>外部把什么操作投递给 core。</p>
    <p><span class="wc-label">常见误读</span>以为 UI 直接调用 <code>run_turn</code>。</p>
  </article>
  <article class="wc-card">
    <h4 class="no_toc"><code>ResponseItem</code></h4>
    <p><span class="wc-label">回答的问题</span>模型下一次请求能看到什么历史。</p>
    <p><span class="wc-label">常见误读</span>把 UI transcript 当成完整模型上下文。</p>
  </article>
  <article class="wc-card">
    <h4 class="no_toc"><code>EventMsg</code></h4>
    <p><span class="wc-label">回答的问题</span>core 如何把进度、文本、审批、完成状态通知出去。</p>
    <p><span class="wc-label">常见误读</span>以为 core 直接操作 TUI。</p>
  </article>
</section>

![Codex 五个术语跨三层边界：Thread、Turn、Submission/Op、ResponseItem 和 EventMsg 分别属于 UI、core 与 model view 的不同切面](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v4-turn-terms.png)

*图 5. 术语图只保留 Thread、Turn、Submission/Op、ResponseItem 和 EventMsg 的边界关系。*

### 2.3 TUI App 和 ChatWidget 的职责

#### `App` 与 `ChatWidget`

TUI 内部可以先分成两个角色：[`App`][tui-app] 是总调度器，持有 active thread、thread event channels、pending requests 等状态；[`ChatWidget`][tui-chatwidget] 是聊天界面状态机，处理键盘、粘贴、图片、mentions、输入框和流式文本渲染。

#### `InputResult::Submitted` 是输入框出口

底部输入框不会直接调用模型。用户按下提交后，`ChatWidget` 先得到 [`InputResult::Submitted`][input-result-submitted] 这类 UI 层结果，再由 [`ChatWidget` 的用户输入处理路径][chatwidget-user-input] 把它转成更结构化的用户意图。这个阶段仍在 TUI surface 内，关心的是“用户做了什么”，而不是“模型请求长什么样”。

#### `AppCommand::user_turn` 汇总 turn settings

真正准备提交 turn 时，TUI 会构造 [`AppCommand::user_turn`][appcommand-user-turn]。这里带的不只是用户文本，还包括 `cwd`、approval policy、permission profile、model、reasoning effort、service tier、collaboration mode、personality 等 turn settings。所以，一次 turn 不是“用户文本”本身，而是“用户文本 + 当前运行约束 + 当前 client/thread 状态”的组合。

#### `AppEvent::CodexOp` 把 UI 命令送到 App Server

`AppCommand::user_turn` 后续会沿着 app event 路由，形成 [`AppEvent::CodexOp`][app-event-codex-op]。这个名字很容易误导：它不是 core 已经执行了 `Op`，而是 TUI 把要进入 Codex 的操作交给 App Server adapter，由 adapter 决定如何发起 protocol request。

### 2.4 `UserInput` 是中间语言

Codex 不会把输入框里的字符串直接塞给模型。[`UserInput`][user-input] 不是单纯文本，而是可以包含 text、remote image、local image、skill、structured mention 等形态的结构化用户意图。到真正喂给模型之前，`Vec<UserInput>` 会在 [`protocol/src/models.rs`][user-input-models] 投影成 user `ResponseInputItem::Message`。

![UserInput 结构：用户意图与 turn settings、thread state 合并为 core Op::UserInputWithTurnContext，再经 build_prompt 投影成 model-visible input](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v6-input-structure.png)

*图 6. 输入结构图用少量标签标出 UI intent、turn settings、`UserInputWithTurnContext`、`build_prompt` 和 model-visible input 的分层。*

#### 越靠近 UI，越保留意图

skill、mention 这类结构不适合在 UI 层提前拍平成字符串，因为 core 后面还要根据它们决定技能注入、插件能力选择或 app connector 处理。Codex 的模式是：越靠近 UI，越保留用户意图；越靠近模型请求，越把意图投影成 provider API 可以理解的 message、tool 和 instruction。

#### `Op::UserInputWithTurnContext` 才是 core 入口形态

到 core 入口时，用户意图会落成 [`Op::UserInput` 或 `Op::UserInputWithTurnContext`][protocol-op-user-input]。后者说明这已经不是裸文本，而是包含 turn context 的 runtime 操作。读源码时要把 `UserInput`、`Op` 和 provider message 分成三层看。

## 三、`turn/start` 是进入 core 的窄门

### 3.1 shape-level request

TUI 通过 active thread routing 把用户 turn 送到 App Server 后，分发层会进入 [`ClientRequest::TurnStart`][client-request-turn-start] 分支。在 v2 protocol 中，[`TurnStartParams`][protocol-turn-start-params] 保留 `input`、工作目录、模型、approval、sandbox 等参数，并通过 `rename_all = "camelCase"` 暴露给协议调用方。下面是简化后的 shape-level 例子，字段只保留本文需要解释的边界：

```json
{
  "method": "turn/start",
  "params": {
    "threadId": "current-thread",
    "input": [
      { "type": "text", "text": "Explain this module" }
    ],
    "cwd": "repo-root",
    "model": "selected-model",
    "approvalPolicy": "on-request",
    "sandboxPolicy": {
      "type": "workspaceWrite",
      "networkAccess": false
    }
  }
}
```

这不是 provider API payload，而是 Codex App Server v2 protocol 的请求形态。示例里的值也是 shape-level 占位：真实客户端会发送实际工作目录和更完整的 sandbox 配置；这里要看的只是 `threadId`、`approvalPolicy`、`sandboxPolicy` 这类协议字段如何先进入 App Server。它还没有进入 `Session`，也还没有变成模型上下文。

![turn/start 边界：load_thread、map v2 input、apply overrides、build core Op、submit_core_op、submit_with_trace 和 core queue 的顺序](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v4-turn-start-boundary.png)

*图 7. turn/start 边界图按源码顺序压缩成七步：load_thread -> map v2 input -> apply overrides -> build core Op -> submit_core_op -> submit_with_trace -> core queue。*

### 3.2 `turn_start -> submit_core_op -> submit_with_trace`

app-server 的分发入口是 [`CodexMessageProcessor::process_request`][app-process-request]，`TurnStart` 分支最终进入 [`AppServerSession::turn_start`][app-turn-start]。这里会做输入大小、thread 状态、turn override、v2 input 到 core input 的转换，然后构造 core `Op`。

#### `AppServerSession::turn_start`

`turn_start` 的职责是把 client request 收束到 core 能理解的操作，而不是直接跑模型。它要处理 cwd、model、approval、sandbox 等覆盖项，并把协议里的 `TurnStartParams.input` 映射到 core 的用户输入类型。

#### `submit_core_op`

投递证据不能写成 `turn_start` 直接 `submit_with_trace`。当前路径应读成 `turn_start -> submit_core_op -> submit_with_trace`：[`submit_core_op`][submit-core-op] 是 App Server 内部的窄函数，负责把已经构造好的 core `Op` 继续交给 Codex thread。

#### `submit_with_trace`

`submit_with_trace` 属于更靠近 core thread 的投递动作。它重要，但不能越过 `submit_core_op` 去解释 `turn/start`。这个分层让 App Server 可以在进入 core 前做协议兼容、状态管理和 trace 绑定。

### 3.3 App Server 不是传话筒

App Server 的价值是把多个 client surface 收束到同一套 turn 语义上。`ClientRequest::TurnStart` 是外部请求，`TurnStartParams` 是协议数据，`Op::UserInputWithTurnContext` 是 core 操作，`submit_core_op` 是投递桥。把这几个词混起来，就会误读权限、cwd、模型选择和协作模式究竟在哪一层生效。

## 四、core 如何跑完一次 turn

### 4.1 `Codex` 是 submission/event 队列对

进入 core 后，关键抽象是 `Codex`。它在 [`core/src/session/mod.rs`][codex-core-queue] 里被描述为一个 queue pair：一边收 `Submission/Op`，一边吐 `Event`。`Codex::spawn` 创建 [`Session`][session-struct] 并启动后台 `submission_loop`；`submit` 只负责生成 submission id 并送入队列。

#### `Session` 维护 active task

`Session` 的注释说明，一个 session 代表初始化后的 model agent，同一时刻最多只有一个 running task，可以被用户输入 interrupt 或 steer。用户 turn 的处理入口在 [`user_input_or_turn_inner`][user-input-or-turn]：当前没有 active turn 时创建新 turn 并启动 `RegularTask`；当前已有 active turn 时尝试 `steer_input`，把新用户输入导入当前 active turn。

#### 新 turn 与 steer 不是同一件事

这解释了为什么 UI 看起来都是“又输入了一句话”，core 却可能有两种处理：启动一个新的用户任务，或者 steering 当前任务。这个区别不应由 TUI 猜测，而应由 `Session` 根据 active task 状态决定。

### 4.2 一次 turn 可以包含多次 sampling

[`RegularTask::run`][regular-task] 很短，却是理解 agent loop 的关键。它先发 `TurnStarted`，然后循环调用 [`run_turn`][run-turn]。为什么是循环？因为一次用户 turn 不一定只对应一次 model request：模型可能先返回 tool call，Codex 执行工具，把工具输出写回 history，再发下一次模型请求。`run_turn` 本体返回的是最后一条 agent message；标准的 `TurnComplete` envelope 由 [task wrapper][task-wrapper-turn-complete] 在任务返回后统一发出，而不是 `run_turn` 内部的 loop 终点。

![一次 turn 的 sampling follow-up loop：RegularTask::run 先发 TurnStarted，再进入 run_turn，run_turn 根据 needs_follow_up 继续 sampling 或返回，外层 task wrapper 最后发 TurnComplete](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v6-sampling-followup-loop.png)

*图 8. sampling follow-up 图把 `run_turn` 内部的一次或多次 sampling、tool output、history、`needs_follow_up` 和外层 task wrapper 发出的 `TurnComplete` 分开。*

#### `needs_follow_up` 是读 `run_turn` 的关键

[`run_turn`][run-turn] 的文件头注释直接把它描述成一个 sampling loop：模型要么返回工具调用，要么返回 assistant message；工具调用会被执行并在下一次 sampling request 里反馈给模型。源码上要把三层拆开：`RegularTask::run` 负责 turn 启动事件，`run_turn` 根据 [`SamplingRequestResult.needs_follow_up`][run-turn-followup] 决定继续或退出，provider stream 的 [`ResponseEvent::Completed`][response-event-completed-branch] 只会结束本次 sampling request 并带回 `needs_follow_up`。压成控制流：

```text
RegularTask::run:
  emit TurnStarted
  loop:
    last_agent_message = run_turn(next_input)
    if session has no pending input:
      return last_agent_message
    next_input = []

run_turn:
  loop:
    prompt = build_prompt(history, visible_tools, turn_settings)
    sampling = try_run_sampling_request(prompt)
    if sampling.needs_follow_up:
      continue
    return last_agent_message

task wrapper:
  emit TurnComplete envelope after task return
```

工具输出不是给 UI 看完就结束，而是会变成下一次模型请求的一部分。所以 Codex 的 turn 是“用户任务生命周期”，不是“模型请求次数”。这也和 OpenAI tools contract 一致：tool output 需要作为后续模型输入继续推进任务。

#### `handle_output_item_done` 触发工具分发

工具调用的关键分支是 [`handle_output_item_done`][handle-output-item-done]。它调用 [`ToolRouter::build_tool_call`][tool-router-build-tool-call] 识别工具，然后创建 in-flight future。只要有工具调用，就设置 `needs_follow_up = true`，因为工具输出还要反馈给模型继续推理。

### 4.3 `build_prompt` 决定模型本次能看见什么

真正构造模型请求的是 [`build_prompt`][build-prompt]。它把 history / 本轮新增输入、[`router.model_visible_specs()`][tool-router-model-visible-specs]、模型基础指令、personality 和 output schema 组装成 `Prompt`。注意 `tools` 来自 `ToolRouter`，而不是 handler 直接暴露。模型看到的是 tool spec；core 维护的是 handler registry、并发策略、审批和沙箱。

![build_prompt 投影模型视图：history ledger、base instructions、ToolRouter specs、personality 和 output_schema 汇入 Prompt，再交给 ModelClientSession::stream](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v4-build-prompt-model-view.png)

*图 9. build_prompt 图只保留 history、instructions、tool specs、output schema 和模型流入口。*

#### `ModelClientSession::stream`

模型流式调用在 [`try_run_sampling_request`][try-run-sampling]。这里创建 [`ModelClientSession::stream`][model-client-stream-call]，然后循环读取 [`ResponseEvent`][response-event]：`OutputItemAdded` 表示 item 开始，`OutputTextDelta` 是流式文本，`OutputItemDone` 可能触发工具分发，`Completed` 则更新 token usage 并决定是否需要 follow-up。

#### 模型视图不是 UI transcript

`build_prompt` 不是 UI transcript 的格式化函数，而是 runtime projection：它把 runtime 拥有的 history、tools、instructions 和输出约束投影成一次 provider sampling request。UI 看到的聊天记录、core 保存的 history、模型本次看到的 prompt，是三种不同视图。

## 五、事件如何回到 UI

### 5.1 core 只发 `EventMsg`

core 不直接操作 TUI。它只发 [`EventMsg`][protocol-event-msg]。`EventMsg` 是 protocol 层能消费的事件集合，包括 turn 开始、文本增量、工具事件、审批请求、用户输入请求、turn 完成等。这个设计让 UI 不需要读 core 内部 task 状态。

![事件回流路径：EventMsg 到 conversation.next_event，再到 apply_bespoke_event_handling，分成 ServerNotification to UI 与 ServerRequest to UI，回答再回到 core policy](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v4-event-return-path.png)

*图 10. 事件回流图只保留 EventMsg -> conversation.next_event -> apply_bespoke_event_handling -> ServerNotification to UI / ServerRequest to UI -> answer returns to core policy。*

#### `conversation.next_event()`

App Server listener 循环读取 [`conversation.next_event()`][conversation-next-event-call]。这一步把 core 的 event stream 拉回 App Server 层，但还没有直接进入 TUI。它仍需要经过 protocol 翻译，才能发给对应 client surface。

#### `apply_bespoke_event_handling`

同一段 listener 路径里的 [`apply_bespoke_event_handling`][apply-bespoke-event-handling-call] 会对部分 core event 做特定处理，然后映射成 app-server protocol 的 notification 或 request。比如用户输入请求会走 [`RequestUserInput` 对应的工具请求处理][tool-request-user-input]，而不是让 UI 自己猜测 core 卡在哪一步。

### 5.2 `ServerNotification` 与 `ServerRequest`

App Server adapter 明确导入 [`ServerNotification` 和 `ServerRequest`][adapter-imports]。这两个类型代表 UI 要消费的两类 server-originated protocol event：notification 是状态/内容更新，request 是需要 client 回答的交互请求。

#### notification path

[`ServerNotification` handling][adapter-server-notifications] 覆盖 `TurnStarted`、agent message delta、turn completed 等无需同步回答的事件。TUI adapter 会通过 [`handle_app_server_event`][tui-handle-server-event] 按 thread id 把这些事件路由到对应 thread channel。

#### request path

[`ServerRequest` handling][adapter-server-requests] 覆盖 approval、用户输入请求等需要 UI 参与的事件。安全相关交互必须沿着这条链路回到界面，因为发起请求的是 core policy，而不是 UI 自己临时判断。

### 5.3 ChatWidget 只消费协议事件

active thread 的事件最后进入 [`ChatWidget::handle_server_notification`][chatwidget-handle-notification] 或 [`ChatWidget::handle_server_request`][chatwidget-handle-request]。前者更新界面状态、追加流式文本、收尾 turn；后者处理需要用户回应的请求。

#### event return path 的设计原则

这条回流路径解释了一个设计原则：UI 消费 protocol event，不应该重新推断 core 内部状态。尤其是 approval 和 request user input 这类安全或交互相关事件，必须由 core/runtime 触发，再通过 protocol request 回到 UI。

## 六、常见误读

前面的术语卡片只能提醒局部混淆；真正读源码时，下面几类误读会让整条链路偏掉。它们都不是命名问题，而是把一个边界上的形态投射到了另一个边界。

<section class="wc-card-grid" aria-label="Codex turn loop common misreadings">
  <article class="wc-card">
    <h4 class="no_toc"><code>turn/start</code> 不是模型请求</h4>
    <p><span class="wc-label">误读</span>看到 <code>input</code>、<code>model</code>、<code>sandboxPolicy</code>，就把它当成 provider payload。</p>
    <p><span class="wc-label">校正</span>它是 App Server v2 protocol shape，进入 core 前还要 load thread、映射 input、应用 overrides、构造 <code>Op</code>。</p>
  </article>
  <article class="wc-card">
    <h4 class="no_toc"><code>run_turn</code> 不是 UI 入口</h4>
    <p><span class="wc-label">误读</span>从 TUI 提交动作一路追到 <code>run_turn</code>，以为 UI 直接调用模型循环。</p>
    <p><span class="wc-label">校正</span>TUI 提交的是 protocol operation；core 的 <code>Session</code> 接住 submission 后，才由 task 调用 <code>run_turn</code>。</p>
  </article>
  <article class="wc-card">
    <h4 class="no_toc"><code>Turn</code> 不是一次 sampling</h4>
    <p><span class="wc-label">误读</span>把一轮用户输入等同于一次模型流式请求。</p>
    <p><span class="wc-label">校正</span>工具调用会写回 history 并触发 follow-up；同一个 turn 可以包含多次 sampling request。</p>
  </article>
  <article class="wc-card">
    <h4 class="no_toc"><code>EventMsg</code> 不是 widget event</h4>
    <p><span class="wc-label">误读</span>把 core event 当成 TUI 私有状态更新。</p>
    <p><span class="wc-label">校正</span>core 只吐 protocol event；App Server 再翻译成 <code>ServerNotification</code> 或 <code>ServerRequest</code>。</p>
  </article>
  <article class="wc-card">
    <h4 class="no_toc"><code>build_prompt</code> 不是 transcript 格式化</h4>
    <p><span class="wc-label">误读</span>把界面聊天记录当成模型下一次请求的完整上下文。</p>
    <p><span class="wc-label">校正</span>模型看到的是 runtime projection：history、tool specs、instructions、personality 和 output schema 的组合。</p>
  </article>
  <article class="wc-card">
    <h4 class="no_toc"><code>submit_with_trace</code> 不是协议窄门</h4>
    <p><span class="wc-label">误读</span>跳过 <code>submit_core_op</code>，直接用底层提交动作解释 <code>turn/start</code>。</p>
    <p><span class="wc-label">校正</span><code>turn_start</code> 先完成协议兼容和 override 校验，再通过 <code>submit_core_op</code> 进入 core queue。</p>
  </article>
</section>

## 七、源码阅读规则

### 7.1 先问边界，再追函数

读 Codex 主循环时，不要追问“哪个函数就是整个 agent”。更稳的读法是沿着四条边界走：UI 边界、协议边界、core 队列边界、模型上下文边界。只要某个概念跨过边界，就先问它的形态有没有变。

<section class="wc-card-grid" aria-label="Codex turn loop reading rules">
  <article class="wc-card">
    <h4 class="no_toc">先判 owner</h4>
    <p><span class="wc-label">规则</span>每个状态先归到 UI、protocol、core queue、model view 或 evidence ledger。</p>
    <p><span class="wc-label">用途</span>owner 不同，字段形态、生命周期和可恢复性就不同。</p>
  </article>
  <article class="wc-card">
    <h4 class="no_toc">先看 shape</h4>
    <p><span class="wc-label">规则</span>看到 JSON 或 struct，先问它是 public protocol、core op，还是 provider request。</p>
    <p><span class="wc-label">用途</span><code>turn/start</code> 的 camelCase shape 不能拿来解释 Responses API payload。</p>
  </article>
  <article class="wc-card">
    <h4 class="no_toc">找转换点</h4>
    <p><span class="wc-label">规则</span>跨层概念要找转换函数或构造点，而不是只看同名字段。</p>
    <p><span class="wc-label">用途</span><code>TurnStartParams.input</code> 要经过 v2 input mapping，才会进入 core <code>Op</code>。</p>
  </article>
  <article class="wc-card">
    <h4 class="no_toc">按 loop 读 turn</h4>
    <p><span class="wc-label">规则</span>遇到工具调用、审批、用户输入请求时，先问是否会触发 follow-up。</p>
    <p><span class="wc-label">用途</span>一次 turn 的完成条件不是“收到一个模型响应”，而是没有继续采样的必要。</p>
  </article>
  <article class="wc-card">
    <h4 class="no_toc">按 event 读 UI</h4>
    <p><span class="wc-label">规则</span>界面变化要回到 <code>ServerNotification</code> 和 <code>ServerRequest</code>，不要倒推 core 内部状态。</p>
    <p><span class="wc-label">用途</span>approval 和 request user input 必须从 core policy 发起，再由 UI 回答。</p>
  </article>
  <article class="wc-card">
    <h4 class="no_toc">证据落同层</h4>
    <p><span class="wc-label">规则</span>协议字段链接协议 struct，函数行为链接定义或调用点，事件路由链接翻译点。</p>
    <p><span class="wc-label">用途</span><code>ModelClientSession::stream</code> 和 <code>apply_bespoke_event_handling</code> 都应落到实际调用点。</p>
  </article>
</section>

### 7.2 最终诊断图

![Codex turn runtime 最终诊断图：UI、protocol、core queue、model view 四条边界与 evidence ledger 帮助定位 run_turn、事件回流和 history ownership](https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v6-turn-runtime-route-map.png)

*图 11. 最终诊断图不再重复完整闭环，而是把四条边界和 evidence ledger 放在同一张 route map 上：先定位边界，再回到精确源码证据。*

一次 turn 的关键不是“哪一层调用模型”，而是输入路径、follow-up loop、事件回流和持久证据各自有 owner。缺任何一边，长期 agent 工作都会失去恢复、审计或继续执行的基础；把它们混成一条调用栈，则会读错权限、上下文和 UI 行为的生效位置。

## 小结

Codex 主循环不是“CLI 调模型 API，然后解析工具调用”。它更像一个可复用 agent harness 的最小闭环：npm 层分发 native binary，Rust CLI 分发多种入口，TUI 通过 in-process app-server 提交 turn，core 用 submission/event 队列管理 session，`RegularTask` 和 `run_turn` 把模型流、工具调用和 history 串成一个可继续的用户任务。

这也是后面继续看工具、审批、沙箱、上下文、App Server、多 Agent 和 trace 时需要带着的主线：Codex 的价值不只是会调用模型，而是把模型放进了一个有协议、有约束、有恢复路径的软件系统里。

</div>

## 参考

- [Codex][openai-codex]
- [Codex CLI][openai-codex-cli]
- [Codex App Server][openai-app-server]
- [Codex web / cloud][openai-codex-cloud]
- [Using tools in the OpenAI API][openai-responses-tools]
- [Function calling in the OpenAI API][openai-function-calling]
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

[openai-codex]: https://developers.openai.com/codex
[openai-codex-cli]: https://developers.openai.com/codex/cli
[openai-app-server]: https://developers.openai.com/codex/app-server
[openai-codex-cloud]: https://developers.openai.com/codex/cloud
[openai-responses-tools]: https://developers.openai.com/api/docs/guides/tools
[openai-function-calling]: https://developers.openai.com/api/docs/guides/function-calling
[codex-js-platform]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-cli/bin/codex.js#L15-L229
[cli-root]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/cli/src/main.rs#L70-L176
[tui-run-main]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/lib.rs#L678-L700
[tui-app-server-target]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/lib.rs#L284
[tui-app-server-adapter]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/app/app_server_adapter.rs#L1-L11
[tui-app]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/app.rs#L510
[tui-chatwidget]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/chatwidget.rs#L784
[input-result-submitted]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/chatwidget.rs#L5621-L5659
[chatwidget-user-input]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/chatwidget.rs#L6059-L6195
[appcommand-user-turn]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/chatwidget.rs#L6245-L6257
[user-input]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/user_input.rs#L13-L40
[user-input-models]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/models.rs#L1200-L1241
[app-event-codex-op]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/app_event.rs#L183-L185
[client-request-turn-start]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/codex_message_processor.rs#L1138-L1143
[app-process-request]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/codex_message_processor.rs#L971-L1142
[protocol-turn-start-params]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/v2.rs#L5436-L5566
[app-turn-start]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/codex_message_processor.rs#L6529-L6713
[submit-core-op]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/codex_message_processor.rs#L2637-L2644
[protocol-op-user-input]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/protocol.rs#L405-L452
[codex-core-queue]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/mod.rs#L363
[session-struct]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/session.rs#L7-L32
[user-input-or-turn]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/handlers.rs#L123-L280
[regular-task]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tasks/regular.rs#L40-L86
[run-turn]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L118-L142
[run-turn-followup]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L445-L503
[task-wrapper-turn-complete]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tasks/mod.rs#L728-L753
[build-prompt]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L935-L952
[tool-router-model-visible-specs]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/router.rs#L103-L112
[try-run-sampling]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L1809-L1875
[model-client-stream-call]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L1832-L1845
[handle-output-item-done]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/stream_events_utils.rs#L219-L255
[tool-router-build-tool-call]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/router.rs#L175-L267
[response-event]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/codex-api/src/common.rs#L68-L107
[response-event-completed-branch]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L2084-L2105
[protocol-event-msg]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/protocol.rs#L1314-L1505
[conversation-next-event-call]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/codex_message_processor.rs#L7598-L7605
[apply-bespoke-event-handling-call]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/codex_message_processor.rs#L7637-L7650
[tool-request-user-input]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/codex_message_processor.rs#L10789-L10804
[adapter-imports]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/app/app_server_adapter.rs#L25-L26
[tui-handle-server-event]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/app/app_server_adapter.rs#L125-L144
[adapter-server-notifications]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/app/app_server_adapter.rs#L155-L235
[adapter-server-requests]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/app/app_server_adapter.rs#L238-L322
[chatwidget-handle-request]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/chatwidget.rs#L6717-L6758
[chatwidget-handle-notification]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tui/src/chatwidget.rs#L6760-L6780
