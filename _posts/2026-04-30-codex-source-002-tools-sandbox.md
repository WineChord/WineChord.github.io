---
classes:
  - wide2
  - codex-source-002-tools-sandbox
title: "Codex 源码剖析：002. 工具、审批与沙箱"
excerpt: "从 ToolSpec、ResponseItem、ToolCall、ToolRegistry 到 approval/sandbox，追踪 Codex 如何把模型工具意图变成可审计、可恢复、可拒绝的受控执行。"
last_modified_at: 2026-06-15
categories:
  - LLM
  - Agent
tags:
  - Codex
  - Coding Agent
  - Source
locale: zh-CN
canonical_url: "https://www.wineandchord.com/llm/agent/codex-source-002-tools-sandbox/"
toc: true
toc_sticky: true
toc_label: "本文目录"
toc_levels: 2..4
mathjax: true
header:
  og_image: https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-tool-runtime-cover.png
  teaser: https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-tool-runtime-cover.png
  teaser_alt: "Codex 工具运行时总览：ToolSpec、approval、sandbox、handler、history 和 follow-up"
---

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-tool-runtime-cover.png" alt="Codex 工具运行时总览：模型工具意图穿过 ToolSpec、approval、sandbox、handler、observation、history 和 follow-up 边界。">
  <figcaption>图 1. 这张总览先把工具意图拆成 spec、call、handler、approval、sandbox 和 history 六个边界，帮助读者按执行责任而不是工具名字阅读后文。</figcaption>
</figure>

Coding agent 最危险、也最容易被低估的部分，不是模型会不会写代码，而是模型把一句自然语言意图变成真实环境动作的瞬间。一次 `npm test`、一次 `apply_patch`、一次 MCP 工具调用、一次 Apps connector 调用，看起来都是“模型调用工具”；在 Codex 源码里，它们其实会穿过四类边界：模型可见的工具描述、provider 返回的 tool call item、core 内部的执行票据、handler/runtime 最终写回模型的 observation。

本文只追踪这条链路：Codex 如何把模型的工具意图变成受控执行，并在每个可能越权、并发冲突、协议错配、沙箱失败的位置留下可恢复的边界。读完以后，你应该能回答三个问题：

1. `ToolSpec`、`ResponseItem`、`ToolCall`、`ResponseInputItem` 分别是谁的形状，为什么不能混用。
2. approval 和 sandbox 为什么不是 handler 私有逻辑，而是执行 attempt 的统一策略层。
3. Unified Exec、`apply_patch`、MCP / Apps 这些看似不同的工具，为什么最终都要回到同一条 history -> follow-up prompt 链路。

## 阅读契约

本文证据分三层。第一层是公开协议：OpenAI Responses API 的工具文档说明模型可见 `tools` 契约，MCP Tools spec 说明外部 server 如何暴露 tools。第二层是固定源码：Codex 引用均锁定在 `openai/codex@ac4332c05b11e00ae775a24cb762edc05c5b5932`，链接指向具体文件和行号。第三层是有界推断：Apps connector 的服务端内部实现不在这个仓库里，本文只根据 Codex app-server protocol、`codex_apps` MCP 元数据、connector state / tool policy 的源码推断 core 侧生命周期；不会把服务端未公开行为写成事实。OpenClaw 只作为架构对照，不作为 Codex 行为证据。

## 一、先把四种形状分开

工具系统最常见的误读，是把“模型看见的工具”和“本地真正执行的函数”当成同一个对象。Codex 刻意把这件事拆成四种形状：

| 形状 | 所属边界 | 谁生产 | 谁消费 | 关键字段 |
| --- | --- | --- | --- | --- |
| `ToolSpec` | model-visible schema | Codex core | Responses API / model | name、description、JSON schema、namespace、defer loading |
| `ResponseItem` | provider output | Responses API stream | Codex stream loop | function call、custom tool call、local shell call、tool search call |
| `ToolCall` | core execution ticket | `ToolRouter` | `ToolCallRuntime` / `ToolRegistry` | `call_id`、canonical tool name、typed payload |
| `ResponseInputItem` | next-turn input | handler/runtime | conversation history / next prompt | function output、custom output、shell output、reasoning/input items |

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-spec-router-handler.png" alt="ToolSpec、ResponseItem、ToolCall、ResponseInputItem 的形状转换图：schema、provider item、core ticket、follow-up input 四段互不混用。">
  <figcaption>图 2. 先把四种形状排成转换链，后面读每个源码分支时就能确认自己站在 schema、provider item、core ticket 还是 follow-up input 哪一层。</figcaption>
</figure>

### 1.1 `ToolSpec` 是模型可见 ABI

`ToolSpec` 是 Codex 给模型看的 ABI（Application Binary Interface 的类比）：它描述“你可以怎样提出一个工具请求”，但不等于本地函数指针。公开层面，OpenAI tools 文档把工具定义为模型可调用的能力描述；Codex 源码层面，`ToolSpec` 枚举覆盖普通 function、namespace、tool search、local shell、web search、freeform 等几类模型可见工具形态。[`ToolSpec` 枚举][tool-spec-enum] 和 Responses API tool 结构体把 name、description、strict、`defer_loading`、parameters、output schema 分开保存。[Responses API tool][responses-api-tool]

#### Before：模型可见的是 schema，不是 handler

简化后的模型可见工具长这样：

```json
{
  "type": "function",
  "name": "exec_command",
  "description": "Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
  "strict": true,
  "parameters": {
    "type": "object",
    "required": ["cmd"],
    "properties": {
      "cmd": { "type": "string" },
      "yield_time_ms": { "type": "integer" }
    }
  }
}
```

这里没有 Rust handler 类型、没有 approval policy、没有 sandbox 参数，也没有本地进程句柄。它只是一份 provider 能理解的 schema。

#### Source：`build_specs` 决定本轮暴露什么

Codex 在每轮 sampling 前构造工具列表：`run_sampling_request` 建 `ToolRouter`，随后 `build_prompt` 把 `router.model_visible_specs()` 塞进 `Prompt.tools`。[sampling request][run-sampling-request] 这条 prompt 构造路径见 [`build_prompt`][build-prompt]。真正的工具计划来自 `build_specs`，它按当前配置、approval/sandbox 能力、MCP server、Apps discoverable tools、deferred dynamic tools 等输入组装本轮可见工具。[tool spec build][build-specs]

这意味着“工具是否存在”是一个轮次内事实，而不是全局常量。某个 dynamic/deferred tool 本轮不可见时，模型不该直接调用它；如果调用了，后面会落到 unavailable / unsupported 的失败边界。

### 1.2 `ResponseItem` 是 provider 回来的 raw output

当模型决定调用工具，provider stream 回来的不是 Codex 内部的 `ToolCall`，而是 `ResponseItem`。Codex 的协议模型里，`ResponseItem` 同时包含文本、reasoning、function call、custom tool call、local shell call、tool search call 等 provider/runtime 输出变体。[response item shapes][response-item-tool-shapes] MCP approval 是后续 app/tool policy 路径上的暂停点，不是这里的 provider output item 变体。

#### Before：provider output 保留 provider 形状

以普通 function call 为例，stream 完成后的 item 可以简化成：

```json
{
  "type": "function_call",
  "call_id": "call_exec_1",
  "name": "exec_command",
  "arguments": "{\"cmd\":\"npm test\",\"yield_time_ms\":30000}"
}
```

MCP namespace call 则会把 namespace 和 name 放在 provider 侧的函数名结构里；local shell / custom tool call 又有自己的 item 形状。Codex 不能直接把它们交给 handler，因为 handler 需要的是 core 统一后的 execution ticket。

### 1.3 `ToolCall` 是 core 侧执行票据

`ToolRouter::build_tool_call` 负责把 `ResponseItem` 还原成 `ToolCall`。[router build_tool_call][build-tool-call] `ToolCall` 本身很小：`call_id`、`tool_name`、`payload`。[ToolCall struct][tool-call-struct] 关键是 payload 已经被归一化成 core 能分派的几类：function、custom、local shell、MCP、tool search 等。

#### After：`ResponseItem` 变成可分派的 `ToolCall`

普通 function call 会变成：

```json
{
  "call_id": "call_exec_1",
  "tool_name": "exec_command",
  "payload": {
    "type": "function",
    "name": "exec_command",
    "arguments": "{\"cmd\":\"npm test\",\"yield_time_ms\":30000}"
  }
}
```

MCP 的关键变化是“模型可见名”和“server 原始名”被重新接上：

```json
{
  "call_id": "call_mcp_1",
  "tool_name": "mcp__codex_apps__calendar_create_event",
  "payload": {
    "type": "mcp",
    "server": "codex_apps",
    "tool": "calendar_create_event",
    "raw_arguments": "{\"title\":\"Design review\"}"
  }
}
```

这不是语义装饰。MCP server 暴露的 raw name 可能包含 provider 不接受的字符，也可能和其他 server 的工具重名。Codex 在 `qualify_tools` 中把 raw server/tool name 转成模型可见的 `callable_namespace` / `callable_name`，同时保留 raw identity；清洗、去重、hash 后缀和 64 字节限制在同一段逻辑里完成。这里可以分别看 [`ToolInfo`][mcp-tool-info]、[`qualify_tools`][mcp-qualify-tools] 和 [name hash helper][mcp-name-hash]。对应测试还断言重复 raw 名不会重复暴露为可调用名。[MCP no duplicate test][tool-plan-no-duplicates-test]

### 1.4 Handler 是执行者，不是策略总线

`ToolHandler` trait 描述 handler 能做什么：声明 kind、判断 payload、声明是否 mutating、提供 pre/post hook、可选 diff consumer、执行 `handle`。[ToolHandler trait][tool-handler-trait] 这让 handler 负责“这个工具如何执行”，但并不让每个 handler 自己重新发明并发锁、approval、sandbox、history 写回。

## 二、一次工具调用在 core 里怎样穿过 gate

源码里的真实链路比“模型调用函数”多几层，因为 Codex 要同时处理 streaming、并行工具、mutating 互斥、post hook、history 持久化和下一轮 prompt。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-tool-gates.png" alt="一次工具调用穿过 ToolRouter、parallel lock、registry、mutating gate、handler、post hook 和 history 的 gate 图。">
  <figcaption>图 3. 先看整条 gate path，再读 2.1 到 2.5：router 只还原形状，runtime 与 registry 决定并发和 handler，history 决定下一轮模型能看见什么。</figcaption>
</figure>

### 2.1 Sampling 前：本轮工具列表先被固定

`turn.rs` 在进入 sampling 前先从 history 构造 `initial_input`，再调用 `run_sampling_request`。[turn sampling input][turn-loop-sampling-input] `run_sampling_request` 建 router、runtime、prompt，prompt 里同时包含 `input` 和本轮工具列表。[sampling request][run-sampling-request] 这一步固定了一个非常重要的不变量：

> 模型只能基于本轮 prompt 里实际暴露的 tool schema 产生合法工具调用。

如果一个 deferred dynamic tool 尚未被 `tool_search` 暴露，它不应该出现在 `Prompt.tools`。`build_specs` 对 unavailable called tools 会创建占位或隐藏策略，让模型之后能收到可解释的失败，而不是让 core 在未知状态下执行。[tool spec build][build-specs]

### 2.2 Router：只做形状还原，不做执行

stream loop 收到 completed output item 后，会尝试 `build_tool_call`。成功时，它把 tool future 放进 `in_flight`; 失败时，错误会作为可给模型看的输出进入 follow-up。[stream output done][handle-output-done] Router 的职责到这里结束：它不跑命令、不改文件、不请求审批，只把 provider item 变成 core ticket。

### 2.3 `ToolCallRuntime`：先判定本次调用能否并行

`ToolCallRuntime` 包住 router、registry 和当前 turn context。[runtime struct][tool-runtime-struct] 它调用 registry 前，先问 `router.tool_supports_parallel(&call)`，再按结果拿 `parallel_execution` 的读锁或写锁：支持并行的调用拿读锁，不支持并行的调用拿写锁。[runtime lock][tool-runtime-lock] 这一步是 router/spec 层面的并行门，不等同于 handler 的 mutating 判断。

随后进入 registry 后，`dispatch_any` 才会调用 handler 的 `is_mutating`，并在 mutating 为真时等待 turn 里的 `tool_call_gate`。[registry dispatch][registry-dispatch] 也就是说，Codex 有两道不同的并发边界：runtime 的 parallel support lock 先避免不支持并行的 tool call 交错；registry 的 mutating gate 再保护会修改环境的 handler 执行顺序。这条规则不是为了性能，而是为了让文件系统、进程状态、conversation history 的观察顺序可解释。

### 2.4 `ToolRegistry::dispatch_any` 的九步

`dispatch_any` 是这篇文章最值得精读的函数。[registry dispatch][registry-dispatch] 它把“找 handler、pre hook、mutating gate、执行、post hook、输出转换”压在一个地方。按源码行为拆开，可以读成九步：

#### 第 1 步：按 payload kind 找 handler

registry 先用 `ToolPayload::kind()` 找匹配 handler。找不到时，不会猜一个“最接近”的工具，而是返回 unsupported tool 的 model-visible observation。[registry dispatch][registry-dispatch]

#### 第 2 步：检查 handler 是否真的接受这个 payload

同名或同 kind 不够。`extract_payload` 失败会被视作 handler payload mismatch，是内部协议错配级别的问题，而不是模型自己能修复的普通参数错误。[registry dispatch][registry-dispatch]

#### 第 3 步：运行 pre hook

pre hook 可以记录事件、准备 diff consumer、做 shell hook 之类的前置处理。Unified Exec 就会在这里把命令、cwd、解析后的参数和权限信息组合成 hook payload。[unified exec pre/post][unified-exec-pre-post]

#### 第 4 步：如果 handler 判定 mutating，等待工具 gate

这一步发生在 registry 内部。handler 的 `is_mutating` 返回真时，`dispatch_any` 等待 `turn.tool_call_gate.wait_ready()`，保证会修改环境的工具不会和其他工具交错写状态。`apply_patch`、shell/unified exec 这类工具都可能在这里被串行化。

#### 第 5 步：执行 handler

handler 的 `handle` 返回 typed result，不直接写 history。对 shell 类工具来说，handler 内部还会再走 approval/sandbox orchestrator；对 MCP 来说，会请求 MCP approval、执行 server tool call、再把结果包成 MCP output。

#### 第 6 步：收集 telemetry 和 duration

registry 记录工具耗时、开始/结束事件和错误状态。这些不是模型可见主数据，但它们保证 UI / trace 能解释一次 tool call 为什么卡住或失败。

#### 第 7 步：运行 post hook

post hook 可以改写 observation，或者要求停止后续处理。这个 hook 让 handler 能处理“命令已经执行，但输出需要被裁剪/转换/补充 diff 摘要”的情况。

#### 第 8 步：把 typed result 转成 `ResponseInputItem`

`AnyToolResult::into_response` 调 `to_response_item`，把 handler result 变成下一轮 provider 能吃的 input item。[AnyToolResult][registry-any-result] `ToolCallRuntime::handle_tool_call` 最终返回的也是 `ResponseInputItem`。[runtime response][tool-runtime-response]

#### 第 9 步：失败也必须变成可解释边界

如果 handler 返回可恢复错误，runtime 会用 `failure_response` 构造 function/custom/tool-search 对应的输出 item。[runtime failure response][tool-runtime-failure-response] 这就是为什么 Codex 可以把很多失败交还给模型继续修正，而不是直接中断整个 turn。

### 2.5 handler output 怎样回到 follow-up prompt

工具调用只有进入下一轮 prompt，模型才真正“看见”执行结果。Codex 的证据链是：

1. stream loop 记录 provider completed item，并把工具执行放进 `in_flight`。[stream output done][handle-output-done]
2. 工具 future 结束后，`drain_in_flight_tool_calls` 把 `ResponseInputItem` 转成 `ResponseItem` 并写入 conversation items。[drain in-flight][drain-in-flight]
3. `record_conversation_items` 同时写 context manager history、rollout recorder 和 raw event。[record conversation items][record-conversation-items]
4. context manager 只记录 API message，并在下一次 `for_prompt` 时归一化可发给 provider 的 history。[context manager history][context-manager-record]
5. follow-up loop 判断是否需要继续，下一轮 `run_sampling_request` 再用 `history.for_prompt(...)` 和当前 tools 构造 prompt。[turn follow-up loop][turn-follow-up-loop]

#### Shape-level before / after

provider 先给 Codex 一个 call：

```json
{
  "type": "function_call",
  "call_id": "call_exec_1",
  "name": "exec_command",
  "arguments": "{\"cmd\":\"npm test\"}"
}
```

handler/runtime 执行后，给 history 的是 output：

```json
{
  "type": "function_call_output",
  "call_id": "call_exec_1",
  "output": "exit code: 0\nstdout:\n..."
}
```

下一轮 prompt 不是只发 output，而是发规范化后的 history：

```text
Prompt {
  input: [
    ...,
    FunctionCall(call_id = "call_exec_1"),
    FunctionCallOutput(call_id = "call_exec_1", output = "exit code: 0\n...")
  ],
  tools: router.model_visible_specs()
}
```

这条链路解释了一个重要现象：handler 不需要直接“调用模型”。它只要返回正确的 `ResponseInputItem`，turn loop 就会把结果带回下一轮 sampling。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-tool-output-followup.png" alt="handler output 写入 ResponseInputItem、history、follow-up build_prompt 的证据链图。">
  <figcaption>图 4. 这张图把 observation 从 handler 返回值一路追到 history 和 follow-up prompt，用来避免把工具执行结果误读成 handler 的私有状态。</figcaption>
</figure>

## 三、approval 和 sandbox 是执行 attempt 的策略

approval 和 sandbox 很容易被写成“shell handler 的 if/else”。Codex 源码把它们集中到 `ToolOrchestrator`，因为它们描述的是一次执行 attempt 的安全策略，而不是某个 handler 的业务逻辑。

### 3.1 `ExecApprovalRequirement` 把执行前状态归成三类

`default_approval_for` 根据 approval policy、sandbox policy、命令风险、cwd 写权限等信息，把一次执行归成三类：允许直接执行、需要用户/guardian approval、禁止执行。[default approval][default-approval] `ExecApprovalRequirement` 的枚举也体现了这三种状态。[exec approval requirement][exec-approval-requirement]

这一步的输入不是“模型说它需要 sudo”，而是 core 对命令、路径、sandbox 和 policy 的联合判断。模型可以请求 escalated permission，但是否允许请求、是否发起 approval、是否直接拒绝，仍由 Codex 控制。

### 3.2 `ToolOrchestrator` 的 approval -> sandbox -> retry

`ToolOrchestrator` 是 command-like 工具的 attempt 调度器。结构体本身持有 approval policy、sandbox policy、命令上下文和 hook。[orchestrator struct][tool-orchestrator] 运行时先判断 approval requirement，再选择初始 sandbox，执行一次 attempt；主流程见 [`run`][tool-orchestrator-run]，sandbox denial 后的升权重试再落到专门的 retry path。[orchestrator sandbox retry][tool-orchestrator-sandbox-retry]

#### Approval gate：跳过、发起、拒绝是三种不同路径

如果 requirement 是 `Skip`，orchestrator 可以直接进入 sandboxed attempt。如果是 `NeedsApproval`，它通过 permission request hook 请求 approval；如果是 `Forbidden`，或用户/guardian 拒绝，会走 reject path，不会偷偷降级成无审批执行。[orchestrator approval][tool-orchestrator-approval]

#### Sandbox attempt：先在受限环境里跑

当 sandbox policy 支持时，orchestrator 会优先在 sandbox 里执行命令。失败不等于自动升权；只有 sandbox denial 且 policy / approval 允许，才会进入重试决策。

#### Retry gate：sandbox denial 后仍要再次过 approval

sandbox 拒绝说明“命令需要的环境动作超出当前沙箱”，不说明“用户已经同意不带沙箱执行”。源码里 sandbox denial 后的无沙箱重试仍然要满足 approval 条件。[orchestrator sandbox retry][tool-orchestrator-sandbox-retry]

### 3.3 OpenClaw 对照：把策略层外置

OpenClaw 文档把 sandboxing 放在 gateway 层描述，backend 也把 sandbox 后端当成独立运行环境选择；公开文档和后端类型可以分别看 [OpenClaw sandboxing][openclaw-sandboxing] 与 [sandbox backend][openclaw-sandbox-backend]。这和 Codex 的细节不同，但方向一致：不要让每个工具 handler 私自决定安全策略。工具 handler 负责执行语义，策略层负责“能否执行、在哪里执行、失败后能否重试”。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-tool-sandbox.png" alt="approval、sandbox、retry 三个 gate 的状态机：Skip 进入 sandbox attempt，NeedsApproval 走 client decision，Forbidden 终止执行。">
  <figcaption>图 5. 这张状态机用来读安全路径：approval 决定有没有资格执行，sandbox 决定在哪个边界里执行，retry 则必须重新证明资格。</figcaption>
</figure>

## 四、三个 handler walkthrough

下面看三个 handler 家族：Unified Exec、`apply_patch`、MCP / Apps。它们覆盖了进程、文件、外部服务三类工具，正好能看出同一套 registry/runtime 规则如何落到不同执行面。

### 4.1 Unified Exec：交互式进程不是 `Command::new` 包装

Unified Exec 支持两种用户可见动作：启动命令和写入已有会话 stdin。参数结构里包含 `cmd`、`yield_time_ms`、`max_output_tokens`、`sandbox_permissions`、`prefix_rule`、`tty`、`workdir` 等字段。[unified exec args][unified-exec-args] 它不是一次性 `Command::new(...).output()`，而是通过 process manager 维护可复用会话。[UnifiedExec manager][unified-exec-manager]

#### 输入 shape：`exec_command` 与 `write_stdin`

启动命令的模型输入可以简化成：

```json
{
  "cmd": "npm test",
  "yield_time_ms": 30000,
  "max_output_tokens": 12000,
  "workdir": "/repo",
  "sandbox_permissions": "use_default"
}
```

写 stdin 则必须带已有 session id：

```json
{
  "session_id": 42,
  "chars": "q",
  "yield_time_ms": 1000
}
```

session id 是 runtime 状态，不是模型可以凭空发明的资源。写入不存在或已结束的 session，会作为工具失败返回给模型，而不是新建一个隐式进程。

#### mutating 判断和 hook

Unified Exec handler 通过命令和解析结果判断是否 mutating，并提供 pre/post hook。[unified exec pre/post][unified-exec-pre-post] hook 的意义是把 shell 命令这个“自由文本动作”转换成 trace 能理解的结构化事件，例如 cwd、命令参数、permission request 和输出裁剪。

#### `handle`：cwd、approval、process id、output budget

`handle` 会解析参数、计算 cwd、拿到 process id、配置输出预算，之后才进入 manager 执行。[unified exec handle][unified-exec-handle] 当模型请求 escalated permission，而当前 approval policy 不允许请求时，handler 会直接拒绝这次执行；这属于 approval 边界，不是 shell 失败。

#### `apply_patch` 拦截：同一个 shell 入口不能绕过文件语义

Unified Exec handler 中有 `intercept_apply_patch` 分支。[unified exec handle][unified-exec-handle] 如果模型试图通过 shell 形式调用 `apply_patch`，Codex 会把它转回 `apply_patch` 语义处理，而不是让 shell 随便改文件。这保证文件修改仍然经过 patch parser、权限、verification 和 diff 事件。

#### `UnifiedExecProcessManager`：进程生命周期属于 runtime

进程创建、PTY / 非 PTY、输出轮询、stdin 写入、session 保活都在 `UnifiedExecProcessManager` 侧维护；manager 定义和 runtime 执行路径分别见 [`UnifiedExecProcessManager`][unified-exec-manager] 与 [Unified Exec runtime][unified-exec-run]。handler 只把工具请求和策略上下文交给 manager；manager 返回的是可以转成 observation 的输出片段或 session id。

#### 失败边界

Unified Exec 至少有五个可恢复失败边界：approval 被拒绝、当前 policy 不允许请求 escalated permission、sandbox denial 且无权重试、session id 不存在或已关闭、handler payload 与工具 kind 不匹配。前四个通常可以作为 observation 让模型改计划；最后一个更像 core 内部协议错配，需要开发者修源码而不是让模型猜参数。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-unified-exec-lifecycle.png" alt="Unified Exec 生命周期图：exec_command、approval、sandbox、process manager、live PTY、write_stdin、output 和 history 的回路。">
  <figcaption>图 6. 这张生命周期图把 shell 请求读成 runtime 对象：命令要先过 hook、approval、sandbox 和 process manager，输出才回到可续写的 observation。</figcaption>
</figure>

### 4.2 `apply_patch`：文件修改必须保持 patch 语义

`apply_patch` 是 Codex 里最能体现“工具不是裸函数”的 handler。它同时承担三件事：解析 patch grammar、计算被触碰文件、验证修改是否按预期落地。[apply_patch handle][apply-patch-handle]

#### 一等 handler：streaming diff 和文件路径先于写入

`apply_patch` handler 提供 diff consumer，streaming 阶段就能消费 patch diff。[apply_patch diff consumer][apply-patch-diff-consumer] 它还会从 patch 中提取 file paths，做写权限检查。[apply_patch permissions][apply-patch-permissions] 这比“shell 执行一个 patch 命令”更强，因为 Codex 在真正写文件前已经知道影响面。

#### shell / unified exec 拦截：防止绕过 patch 边界

源码里有专门的 `intercept_apply_patch`，用于拦截 shell/unified exec 形式的 `apply_patch`。[apply_patch intercept][apply-patch-intercept] 这让模型不能靠“我从 shell 跑 patch”绕过 patch parser 和权限边界。

#### verification failure：失败要回到模型

patch 解析成功不代表最终文件一定符合预期。`apply_patch` 在执行后会验证结果，verification failure 会变成给模型的错误 observation；普通 handler 路径见 [`apply_patch` handler][apply-patch-handle]，shell/unified exec 拦截路径见 [`apply_patch` intercept][apply-patch-intercept]。这类失败非常适合模型自我修复：它知道 patch 没套上，可以重新读取文件并生成更小的 patch。

#### Shape-level before / after

模型输入是 freeform patch：

```text
*** Begin Patch
*** Update File: src/lib.rs
@@
-old_call();
+new_call();
*** End Patch
```

handler 输出不是“文件已改”的自然语言承诺，而是带着执行结果和错误信息的 observation：

```json
{
  "type": "function_call_output",
  "call_id": "call_patch_1",
  "output": "Success. Updated the following files:\nsrc/lib.rs"
}
```

如果 verification 失败，output 会描述失败原因，让模型重新构造 patch，而不是假装修改成功。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-apply-patch-boundary.png" alt="apply_patch 边界图：freeform patch、grammar parser、file path permission、apply workbench、verification failure 和 retry。">
  <figcaption>图 7. 这张图把文件修改从“任意 shell 副作用”收束成可验证 patch 事务，读者可以沿着 parser、permission、verification 三个点检查失败原因。</figcaption>
</figure>

### 4.3 MCP / Apps：外部能力也要先变成 core 工具

MCP 的公开协议层规定 server 暴露 tools，client 调用 `tools/list` 和 `tools/call`。[MCP Tools spec][mcp-tools-spec] Codex 的 MCP client 负责连接 server、列工具、保留 connector metadata，并把工具塞回 core 的 tool plan；连接生命周期和 tools/list 读取路径分别见 [rmcp lifecycle][rmcp-client-lifecycle] 与 [rmcp tools][rmcp-client-tools]。

#### MCP name 清洗：模型可见名不是 raw server name

MCP raw tool name 可能不满足 Responses API 的 name 约束，多个 server 也可能暴露同名工具。Codex `qualify_tools` 做三件事：保留 raw `server_name` / raw `tool.name`，生成模型可见 callable namespace/name，必要时加 hash 后缀保证唯一且不超长；字段形状见 [`ToolInfo`][mcp-tool-info]，转换逻辑见 [`qualify_tools`][mcp-qualify-tools] 和 [name hash helper][mcp-name-hash]。

这也是为什么 router 不能只看 provider 回来的字符串。它必须通过 session 解析 MCP tool info，再把 payload 还原为 raw server/tool 调用。[router build_tool_call][build-tool-call]

#### MCP approval：外部服务也可能需要用户确认

`mcp_tool_call.rs` 会先解析 raw arguments、读取 metadata、检查 app tool policy，然后决定是否请求 approval。[MCP call start][mcp-tool-call-start] approval 被拒绝、取消或被 safety monitor 拦截时，不会继续调用 server；结果会作为 MCP tool call 的 output 返回。[MCP approval][mcp-tool-approval]

#### Apps connector state：enabled、approval、destructive/open-world policy

Apps 是 Codex 通过 `codex_apps` MCP server 暴露的一类 connector 能力。`turn.rs` 会在 built tools 阶段读取 app state、discoverable apps/tools，并决定哪些工具进入 router。[Apps built tools][apps-built-tools] `connectors.rs` 只从 `codex_apps` 的 `ToolInfo` 中抽取可访问 connector；connector state 和 tool policy 分别见 [Apps connectors][apps-connectors-from-mcp] 与 [Apps tool policy][apps-tool-policy]。

这里能确认的是 core 侧状态机：disabled app 不应执行；需要 approval 的 app tool 必须先过 policy；destructive / open-world 工具会影响 approval 要求。connector 服务端怎样具体访问第三方系统，不在本文证据范围内。

#### app-server protocol：UI / connector 面和模型工具 schema 分层

Codex app-server protocol 里有 `app/list`、`mcpServer/tool/call`、`AppsConfig`、`AppToolsConfig`、`AppToolApproval`、`McpServerToolCallParams` 等类型。[app/list request][app-server-client-request] 和 [MCP server tool call request][app-server-mcp-tool-call-request] 说明 client/connector 调用面先进入 App Server；[AppsConfig][app-server-apps-config]、[AppToolsConfig][app-server-app-tools-config]、[AppToolApproval][app-server-app-tool-approval] 和 [MCP server tool call params][app-server-mcp-tool-call-params] 则说明配置、审批和调用参数被分层保存。这些类型说明 Apps 不是“把所有第三方 API 直接塞进 prompt”，而是先进入 app/server/config/policy 层，再由 MCP tool info 映射到模型可见工具。

#### 失败边界

MCP / Apps 失败边界至少包括：server 未连接或 list tools 失败、dynamic/deferred tool 尚未暴露、disabled app、MCP approval declined/cancelled、handler payload mismatch、server tool call 返回错误。前四类可以告诉模型换路径或请求用户操作；payload mismatch 仍然是 core/协议内部问题。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-mcp-apps-lifecycle.png" alt="MCP 与 Apps 生命周期图：tools/list、name qualify、codex_apps metadata、connector state、tool approval、server call、ResponseInputItem 回填。">
  <figcaption>图 8. 这张生命周期图说明外部能力不是直接进入 prompt，而是先被拉回 Codex 的命名、connector state、审批和 history 回填边界。</figcaption>
</figure>

## 五、失败边界不是附录

一个可靠的 agent runtime，失败路径必须和成功路径一样清楚。下面这张表把本文涉及的失败收束到“谁发现、模型看见什么、能否继续”三件事。

| 失败边界 | 发现位置 | 模型看到的结果 | 能否继续 |
| --- | --- | --- | --- |
| approval 被拒绝 | `ToolOrchestrator` permission request | 工具输出说明未获批准或被策略拒绝 | 可以，模型应改用不需审批的方案或询问用户 |
| 当前 policy 不允许 escalated request | Unified Exec handler | 工具输出说明该 approval policy 下不能请求升权 | 可以，模型应去掉升权或换命令 |
| sandbox denial | orchestrator / unified exec runtime | sandbox denied 相关输出；符合条件才可能请求无沙箱重试 | 可以，但无沙箱重试必须再过 approval |
| MCP approval declined / cancelled | `mcp_tool_call.rs` | MCP tool output 表示用户拒绝、取消或 safety monitor 拦截 | 可以，模型应选择不调用该外部能力的方案 |
| `apply_patch` verification failure | `apply_patch` handler / intercept path | patch verification failed 和原因 | 可以，模型应重新读取文件并生成更精确 patch |
| dynamic/deferred tool missing | `build_specs` / router / registry | unavailable 或 unsupported tool observation | 可以，模型应先 `tool_search` 或改用已暴露工具 |
| handler payload mismatch | `dispatch_any` payload extraction | fatal / incompatible payload 类错误 | 通常不能靠模型修复，说明 core schema 与 handler 不一致 |
| unsupported tool kind/name | `dispatch_any` handler lookup | unsupported tool observation | 可以，模型应停止调用该工具或查找正确工具 |
| post hook stop / replacement | handler post hook | 被 post hook 改写后的 observation | 可以，模型只基于最终 observation 继续 |

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-approval-sandbox-matrix.png" alt="失败边界矩阵图：approval、sandbox、MCP、apply_patch、dynamic tools、payload mismatch 分别落到可恢复或不可恢复轨道。">
  <figcaption>图 9. 这张矩阵把失败路径和成功路径放在同一张图里：失败 observation 也是下一轮 prompt 的输入，而不是附录里的异常。</figcaption>
</figure>

## 六、把规则压成可迁移表

如果要在自己的 agent runtime 里复用 Codex 的设计，不要先问“我要支持多少工具”，而要先问“每个工具意图穿过哪些边界”。下面这张机制表是本文的可迁移版本。

失败矩阵回答的是“坏事发生时谁先发现”；最终规则图回答的是“正常设计时 owner 应该如何分层”。这两张图不能合成一张，因为前者按 failure boundary 排列，后者按 execution ownership 排列。把它们分开，读者才能把可恢复错误和架构不变量分别带走。

换到自己的 runtime 里，也可以按同样顺序压测：先固定 model-visible schema，再固定 router 归一化，再固定并发和 policy gate，最后要求所有 handler output 都回到 history。只要其中任何一层让 handler 私自跨界，后面的审计和恢复就会变成偶然行为。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-tool-runtime-rule-map.png" alt="Codex 工具运行时规则图：schema owner、router owner、runtime lock、policy gate、handler output、history owner 的责任地图。">
  <figcaption>图 10. 这张最终规则图把机制表画成执行路径，用来检查每个 owner 是否只负责自己的边界，以及 observation 是否重新回到 history。</figcaption>
</figure>

| 机制 | Codex 里的 owner | 必须保持的不变量 | 反例风险 |
| --- | --- | --- | --- |
| 工具 schema | `build_specs` / `ToolSpec` | 模型只看 schema，不看本地 handler | prompt 暴露不存在或未授权能力 |
| provider output | `ResponseItem` | 保留 provider 形状直到 router 解析 | handler 直接吃 provider 字符串导致协议耦合 |
| execution ticket | `ToolRouter` / `ToolCall` | core 只分派 typed payload | MCP raw name 丢失、namespace 碰撞 |
| 并行控制 | `ToolCallRuntime` parallel support lock / registry mutating gate | 不支持并行的调用先独占；mutating handler 再等工具 gate | 把 router 并行能力和 handler 副作用判断混成一层，导致文件/进程状态交错 |
| 安全策略 | `ToolOrchestrator` | approval、sandbox、retry 集中决策 | 每个 handler 私自升权或绕过沙箱 |
| 文件修改 | `apply_patch` handler | patch grammar、文件路径、verification 先于“修改成功” | shell 副作用绕过审计 |
| 外部能力 | MCP / Apps connector policy | raw identity、model-visible name、approval policy 分层 | 第三方 API 直接暴露给模型且不可控 |
| observation | `ResponseInputItem` | 成功和失败都回填 history | 模型无法基于真实结果继续推理 |

## 小结

Codex 的工具系统不是一个大 `match tool_name`。它更像一条窄而硬的流水线：`ToolSpec` 只描述模型可见能力，`ResponseItem` 保留 provider 输出，`ToolRouter` 还原 core ticket，`ToolCallRuntime` 和 `ToolRegistry` 处理并发与 handler 生命周期，`ToolOrchestrator` 统一 approval/sandbox/retry，handler 只返回 `ResponseInputItem`，最后由 history 带回下一轮 prompt。

这套设计的价值不在“能接很多工具”，而在“每个工具失败时都知道失败发生在哪个边界”。approval 拒绝、sandbox denial、MCP approval、`apply_patch` verification failure、dynamic tool missing、payload mismatch 都不是含混的异常；它们是 agent 可以继续工作的结构化事实。

## 参考

- [OpenAI Developers: Tools][openai-tools]
- [Model Context Protocol: Tools][mcp-tools-spec]
- [Codex `ToolSpec` enum][tool-spec-enum]
- [Codex Responses API tool schema][responses-api-tool]
- [Codex `build_specs`][build-specs]
- [Codex `run_sampling_request`][run-sampling-request]
- [Codex `build_prompt`][build-prompt]
- [Codex turn sampling input][turn-loop-sampling-input]
- [Codex follow-up loop][turn-follow-up-loop]
- [Codex stream output handling][handle-output-done]
- [Codex drain in-flight tool calls][drain-in-flight]
- [Codex record conversation items][record-conversation-items]
- [Codex context manager history][context-manager-record]
- [Codex `ResponseInputItem`][response-input-item]
- [Codex `ResponseItem` tool call shapes][response-item-tool-shapes]
- [Codex `FunctionCallOutputPayload`][function-output-payload]
- [Codex `ToolRouter` struct][tool-router-struct]
- [Codex `ToolRouter::build_tool_call`][build-tool-call]
- [Codex `ToolCall` struct][tool-call-struct]
- [Codex `ToolRouter::dispatch_call`][router-dispatch]
- [Codex MCP `ToolInfo`][mcp-tool-info]
- [Codex MCP `qualify_tools`][mcp-qualify-tools]
- [Codex MCP name hash helpers][mcp-name-hash]
- [Codex MCP duplicate-name test][tool-plan-no-duplicates-test]
- [Codex `ToolHandler` trait][tool-handler-trait]
- [Codex `AnyToolResult`][registry-any-result]
- [Codex `ToolRegistry::dispatch_any`][registry-dispatch]
- [Codex `ToolCallRuntime` struct][tool-runtime-struct]
- [Codex `ToolCallRuntime` lock logic][tool-runtime-lock]
- [Codex runtime response conversion][tool-runtime-response]
- [Codex runtime failure response][tool-runtime-failure-response]
- [Codex `ToolOrchestrator` struct][tool-orchestrator]
- [Codex orchestrator approval run][tool-orchestrator-run]
- [Codex orchestrator sandbox retry][tool-orchestrator-sandbox-retry]
- [Codex orchestrator approval request][tool-orchestrator-approval]
- [Codex `ExecApprovalRequirement`][exec-approval-requirement]
- [Codex default approval logic][default-approval]
- [Codex Unified Exec args][unified-exec-args]
- [Codex Unified Exec pre/post hooks][unified-exec-pre-post]
- [Codex Unified Exec handler][unified-exec-handle]
- [Codex `UnifiedExecProcessManager`][unified-exec-manager]
- [Codex Unified Exec runtime][unified-exec-run]
- [Codex `apply_patch` diff consumer][apply-patch-diff-consumer]
- [Codex `apply_patch` permissions][apply-patch-permissions]
- [Codex `apply_patch` handler][apply-patch-handle]
- [Codex `apply_patch` intercept][apply-patch-intercept]
- [Codex MCP tool call start][mcp-tool-call-start]
- [Codex MCP approval path][mcp-tool-approval]
- [Codex MCP tool call execute][mcp-tool-call-execute]
- [Codex rmcp client lifecycle][rmcp-client-lifecycle]
- [Codex rmcp tools list][rmcp-client-tools]
- [Codex Apps built tools][apps-built-tools]
- [Codex Apps connectors from MCP][apps-connectors-from-mcp]
- [Codex Apps tool policy][apps-tool-policy]
- [Codex app-server protocol request types][app-server-client-request]
- [Codex app-server MCP tool call request][app-server-mcp-tool-call-request]
- [Codex app-server `AppsConfig`][app-server-apps-config]
- [Codex app-server `AppToolsConfig`][app-server-app-tools-config]
- [Codex app-server `AppToolApproval`][app-server-app-tool-approval]
- [Codex app-server `McpServerToolCallParams`][app-server-mcp-tool-call-params]
- [OpenClaw sandboxing design][openclaw-sandboxing]
- [OpenClaw sandbox backend][openclaw-sandbox-backend]

[openai-tools]: https://developers.openai.com/api/docs/guides/tools
[mcp-tools-spec]: https://modelcontextprotocol.io/specification/2025-06-18/server/tools
[tool-spec-enum]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tools/src/tool_spec.rs#L18-L58
[responses-api-tool]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tools/src/responses_api.rs#L25-L38
[build-specs]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/spec.rs#L71-L338
[run-sampling-request]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L963-L1017
[build-prompt]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L935-L952
[turn-loop-sampling-input]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L429-L455
[turn-follow-up-loop]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L458-L500
[handle-output-done]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/stream_events_utils.rs#L219-L255
[drain-in-flight]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L1775-L1799
[record-conversation-items]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/mod.rs#L2355-L2375
[context-manager-record]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/context_manager/history.rs#L98-L122
[response-input-item]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/models.rs#L659-L695
[response-item-tool-shapes]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/models.rs#L741-L845
[function-output-payload]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/models.rs#L1370-L1475
[tool-router-struct]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/router.rs#L39-L100
[build-tool-call]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/router.rs#L175-L267
[tool-call-struct]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/router.rs#L32-L37
[router-dispatch]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/router.rs#L269-L297
[mcp-tool-info]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/codex-mcp/src/tools.rs#L28-L54
[mcp-qualify-tools]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/codex-mcp/src/tools.rs#L133-L229
[mcp-name-hash]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/codex-mcp/src/tools.rs#L291-L369
[tool-plan-no-duplicates-test]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/tools/src/tool_registry_plan_tests.rs#L72-L83
[tool-handler-trait]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/registry.rs#L44-L92
[registry-any-result]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/registry.rs#L107-L123
[registry-dispatch]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/registry.rs#L265-L513
[tool-runtime-struct]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/parallel.rs#L27-L49
[tool-runtime-lock]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/parallel.rs#L82-L143
[tool-runtime-response]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/parallel.rs#L63-L78
[tool-runtime-failure-response]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/parallel.rs#L146-L172
[tool-orchestrator]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/orchestrator.rs#L40-L54
[tool-orchestrator-run]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/orchestrator.rs#L126-L213
[tool-orchestrator-sandbox-retry]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/orchestrator.rs#L215-L379
[tool-orchestrator-approval]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/orchestrator.rs#L382-L475
[exec-approval-requirement]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/sandboxing.rs#L159-L180
[default-approval]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/sandboxing.rs#L198-L239
[unified-exec-args]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/handlers/unified_exec.rs#L45-L68
[unified-exec-pre-post]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/handlers/unified_exec.rs#L101-L177
[unified-exec-handle]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/handlers/unified_exec.rs#L179-L380
[unified-exec-manager]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/unified_exec/process_manager.rs#L331-L380
[unified-exec-run]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/runtimes/unified_exec.rs#L241-L360
[apply-patch-diff-consumer]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/handlers/apply_patch.rs#L53-L124
[apply-patch-permissions]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/handlers/apply_patch.rs#L190-L244
[apply-patch-handle]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/handlers/apply_patch.rs#L341-L465
[apply-patch-intercept]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/handlers/apply_patch.rs#L468-L567
[mcp-tool-call-start]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/mcp_tool_call.rs#L85-L170
[mcp-tool-approval]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/mcp_tool_call.rs#L191-L272
[mcp-tool-call-execute]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/mcp_tool_call.rs#L291-L384
[rmcp-client-lifecycle]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/codex-mcp/src/rmcp_client.rs#L1-L7
[rmcp-client-tools]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/codex-mcp/src/rmcp_client.rs#L334-L388
[apps-built-tools]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L1108-L1243
[apps-connectors-from-mcp]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/connectors.rs#L514-L557
[apps-tool-policy]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/connectors.rs#L572-L723
[app-server-client-request]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/common.rs#L614-L618
[app-server-mcp-tool-call-request]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/common.rs#L808-L812
[app-server-apps-config]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/schema/typescript/v2/AppsConfig.ts#L1-L8
[app-server-app-tools-config]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/schema/typescript/v2/AppToolsConfig.ts#L1-L6
[app-server-app-tool-approval]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/schema/typescript/v2/AppToolApproval.ts#L1-L5
[app-server-mcp-tool-call-params]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/schema/typescript/v2/McpServerToolCallParams.ts#L1-L6
[openclaw-sandboxing]: https://github.com/openclaw/openclaw/blob/5d8ca42c7de8118b15782bad9cbac6240585e13a/docs/gateway/sandboxing.md#L10-L37
[openclaw-sandbox-backend]: https://github.com/openclaw/openclaw/blob/5d8ca42c7de8118b15782bad9cbac6240585e13a/src/agents/sandbox/backend.ts#L29-L54
