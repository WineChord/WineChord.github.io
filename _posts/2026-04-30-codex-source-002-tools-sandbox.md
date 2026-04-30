---
classes: wide2
title: "Codex 源码剖析：002. 工具、审批与沙箱"
excerpt: "拆开 Codex 的 ToolRouter、ToolRegistry、ToolCallRuntime、ToolOrchestrator，看模型工具调用如何变成受控执行。"
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

上一篇从 TUI 走到了 `run_turn`。这一篇看更核心的问题：模型输出一个工具调用之后，Codex 如何决定它到底是什么工具、能不能执行、能不能并行、要不要审批、是否应该进入 sandbox、最后如何把结果喂回模型。

如果只看概念，agent loop 很简单：

```text
model -> action -> observation -> model
```

ReAct 论文把 reasoning 和 acting 放到同一个轨迹里；SWE-agent 进一步强调 agent-computer interface 会影响软件工程任务表现。但工程上真正难的不是“让模型说我要运行测试”，而是：

- 模型看到的工具 schema 和真实执行实现如何分离？
- 一个工具会不会修改文件、启动进程、访问网络，谁说了算？
- 并行工具调用如何避免互相踩状态？
- `apply_patch` 为什么不能只是 shell 里的一段命令？
- MCP / Apps / Hooks 这种外部扩展如何接进来，又不把 core 搞成一堆特判？

Codex 当前的答案是：**工具调用先统一还原成内部 `ToolCall`，再走 router / runtime / registry / orchestrator / handler 分层**。

## Tool Call Pipeline

### 模型看到的是 spec，core 持有的是 handler

工具在 `build_prompt` 时注入模型请求。上一篇看过，`Prompt.tools` 来自 [`router.model_visible_specs()`][build-prompt]，不是直接来自某个 handler。

工具集合的构造在 [`build_specs_with_discoverable_tools`][build-specs]。这里会把几类工具来源统一规划：

- 内置工具，例如 `apply_patch`、`view_image`、`unified_exec`
- MCP / Apps 工具
- deferred MCP tools
- dynamic / discoverable tools
- 多 Agent 工具
- `tool_search` / `tool_suggest`

规划结果分成两类东西：

1. model-visible `ToolSpec`
2. core-side `ToolHandler`

这一层分离非常重要。模型只需要知道“我可以调用什么，参数 schema 是什么”。core 则需要知道“这个工具由谁处理、是否支持并行、是否可能修改环境、执行前后有哪些 hook、失败如何反馈给模型”。

这张图先给一个总览。后面每一节只拆其中一段，不要把工具系统理解成一个“大 switch”。

![Codex 工具系统总览：模型可见 ToolSpec 经过 Router、Runtime、Registry、Approval、Sandbox 后才进入具体 handler](/assets/images/posts/2026-04-30-codex-source/tool-sandbox.svg)

*图 1. 模型只看到工具 schema；core 侧才知道 handler、审批、沙箱、并发和 hook。读源码时可以先沿着 `ToolRouter::build_tool_call` 到 `ToolRegistry::dispatch_any` 这条线走。*

`ToolRouter` 保存的字段很清楚：registry、所有 spec、模型可见 spec、允许并行的 MCP server 集合，见 [`ToolRouter`][tool-router-struct]。`from_config` 里还会过滤 deferred dynamic tools，避免还没加载的工具直接暴露给模型。

这就解释了为什么 Codex 可以支持 lazy tool discovery。工具世界可以很大，但模型每次看到的工具集合应该尽量小、尽量确定。

### Router：把 ResponseItem 还原成 ToolCall

模型流里出现 `OutputItemDone` 时，Codex 调 [`handle_output_item_done`][handle-output]。这里先调用 `ToolRouter::build_tool_call`，把不同形态的 `ResponseItem` 还原为内部 `ToolCall`：

- `FunctionCall`：普通 function tool；如果命中 MCP tool info，则转成 `ToolPayload::Mcp`
- `ToolSearchCall`：客户端执行的 `tool_search`
- `CustomToolCall`：Responses API custom tool
- `LocalShellCall`：转换成 `local_shell`

对应逻辑在 [`ToolRouter::build_tool_call`][build-tool-call]。

这个转换层的意义在于：模型 API 可以有多种工具表达方式，但 core 不希望后面的执行层到处判断 `ResponseItem`。后面统一处理 `ToolCall { tool_name, call_id, payload }` 就够了。

`payload` 也很关键。MCP 工具的 model-visible name 可能经过清洗、去重、hash，以适配 API 命名限制；但执行时还必须知道原始 server/tool。Codex 在 router 阶段把它还原成 `server + raw tool name + raw arguments`，后面 MCP handler 才能准确调用对应 server。

### Runtime：并行不是“全都并行”

进入 handler 之前，工具调用还要过几道 gate。这个地方和 Claude Code 对照仓库里的 `ToolUseContext` 很不一样：Claude Code 更倾向于把工具执行所需状态集中携带在一个应用运行时上下文里；Codex 则把并发、mutating、approval、sandbox 拆成可复用的 runtime/policy 层。

![Codex 工具执行闸门：Router、Runtime lock、Registry、Approval 和 Sandbox 的顺序](/assets/images/posts/2026-04-30-codex-source/tool-gates.svg)

*图 2. 工具执行中几个关键 gate。模型给的是“我要调用某个工具”的意图，系统要在每一道门上重新确认“这个意图能不能以这种方式执行”。*

`ToolCallRuntime` 是工具执行的外层 runtime。它有一个 `parallel_execution: RwLock<()>`，见 [`ToolCallRuntime`][tool-runtime-struct]。

执行时先问 router：这个工具是否支持并行。支持并行的工具拿读锁，不支持并行的工具拿写锁，见 [`handle_tool_call_with_source`][tool-runtime-lock]。

这个设计很朴素，但非常实用：

- `list_dir`、某些只读 MCP 工具可以并行。
- shell、patch、会改文件的工具通常应该串行。
- 如果模型一次返回多个工具调用，runtime 不需要为每个工具写一套并发策略。

这里不是为了追求极致吞吐，而是为了维持一个可解释的状态边界：**读可以并发，写要保守串行**。

### Registry：统一处理 telemetry、hooks 和 mutating gate

`ToolRegistry::dispatch_any` 是工具分发中心。它做的事情很多，但边界比较清晰，源码可以从 [`dispatch_any`][registry-dispatch] 读起：

1. 找 handler。
2. 检查 payload kind 是否匹配。
3. 运行 pre-tool-use hooks。
4. 调用 `handler.is_mutating()` 判断是否可能修改环境。
5. mutating 工具等待 `tool_call_gate`。
6. 调用 handler。
7. 运行 post-tool-use hooks。
8. 记录 goal runtime、telemetry、trace。
9. 把 handler output 转成模型能吃的 `ResponseInputItem`。

`ToolHandler` trait 自身也把这些扩展点列出来了：`kind`、`matches_kind`、`is_mutating`、`pre_tool_use_payload`、`post_tool_use_payload`、`create_diff_consumer`、`handle`，见 [`ToolHandler`][tool-handler-trait]。

这里最值得注意的是：hooks 不是模型工具。hooks 是工具执行链旁边的策略扩展点。pre hook 可以阻止执行；post hook 可以追加上下文、停止流程、替换反馈给模型的结果。这样 hooks 能影响 agent 行为，但不需要被暴露成模型可调用工具。

这也是安全边界的一部分：模型工具调用是 untrusted input；hook command 是用户配置的 trusted policy surface。两者必须分开看。

把 [`dispatch_any`][registry-dispatch] 压缩成伪代码，大概是这样：

```text
handler = registry.lookup(tool_name)
ensure payload kind matches handler kind

pre_result = run_pre_tool_use_hooks(handler, payload)
if pre_result blocks:
  return blocked output

if handler.is_mutating():
  wait for tool_call_gate

output = handler.handle(payload, runtime_context)
post_result = run_post_tool_use_hooks(handler, output)
record telemetry / trace
return output as ResponseInputItem
```

这段伪代码的重点不是 hook 本身，而是 `is_mutating()` 和 `tool_call_gate`。Codex 没有假设所有工具都同样安全，也没有把“会改文件”和“只读查询”放进同一个并发池里。

## Execution Policy

### Shell 与 Unified Exec：命令执行不是一个函数

Codex 里现在有 shell 相关的多条路径。传统 shell handler 会先把参数转成 `ExecParams`，再进入后面的执行策略。

`ExecParams` 里比较关键的是：

- command / cwd / env
- network policy
- sandbox permissions
- Windows sandbox level

对应转换在 [`ShellHandler::to_exec_params`][shell-to-exec-params]。

`UnifiedExecHandler` 则面向更完整的终端交互场景：`exec_command` 可以启动一个命令会话，`write_stdin` 可以向已有 session 写入输入。见 [`ExecCommandArgs`][unified-exec-args]。

这里不要把它理解成“多传了几个 shell 参数”。这些字段可以分成几组：

- 命令形态：`cmd`、`shell`、`login`、`tty`
- 执行位置：`workdir`
- 输出预算：`yield_time_ms`、`max_output_tokens`
- 权限请求：`sandbox_permissions`、`additional_permissions`
- 用户说明：`justification`、`prefix_rule`

这不是简单包装 `Command::new("bash")`。`exec_command` 执行前会：

- 解析 cwd / workdir
- 可能触发 implicit skill invocation
- 分配 process id
- 派生实际 shell command
- 计算输出截断预算
- 应用已经批准的 turn permissions
- 校验额外权限请求
- 拦截 `apply_patch`
- 交给 `UnifiedExecProcessManager`

这条路径在 [`UnifiedExecHandler::handle`][unified-exec-handle]。

为什么要这么复杂？因为 coding agent 的 shell 不是普通 shell。它是模型输出影响真实工作区的主要入口。这里必须同时考虑可恢复交互、TTY、输出截断、审批、sandbox、权限升级、事件流和用户可见性。

### Approval 与 sandbox：集中到 orchestrator

如果每个工具都自己写“先判断审批、再尝试 sandbox、失败后提示用户、用户允许再重试”，最后一定会漂移。Codex 把这套东西集中到 `ToolOrchestrator` 和 `sandboxing.rs`。

`sandboxing.rs` 里有一个核心枚举：[`ExecApprovalRequirement`][exec-approval-requirement]。

它把执行前状态分成三类：

- `Skip`：不需要审批，可能还带 `bypass_sandbox`。
- `NeedsApproval`：需要用户或 guardian 审批。
- `Forbidden`：直接禁止。

默认判断在 [`default_exec_approval_requirement`][default-approval]。它结合 `AskForApproval` 和 filesystem sandbox policy 得出是否需要审批。比如 `Never`、`OnFailure` 默认不问；`OnRequest` 在 restricted filesystem 下需要问；`UnlessTrusted` 总是问；granular policy 还可能禁止某些 prompt。

这层设计有两个好处：

1. 审批逻辑是工具无关的，shell / unified exec / apply_patch 都能复用。
2. sandbox 不是“开或关”一个布尔值，而是一次执行尝试的策略：初始 attempt 是否 sandbox、失败后是否允许升级、额外权限是否被批准。

从用户体验看，这就是 Codex 会提示“是否允许这个命令越过当前 sandbox / 申请额外权限”的根源。从源码看，这不是 UI 行为，而是 core policy 计算的结果。

OpenClaw 的取舍更偏 gateway-first。它的文档把 Gateway 放在 host 侧，工具执行可以下沉到 sandbox backend；backend 抽象里再提供 exec spec、shell command 和 filesystem bridge 等能力，见 [`sandboxing.md`][openclaw-sandboxing] 和 [`backend.ts`][openclaw-sandbox-backend]。也就是说，Codex 更像“core 工具运行时统一算 approval/sandbox”，OpenClaw 更像“Gateway 控制面统一接入不同执行后端”。两者都不是把 shell 当普通函数，只是边界落点不同。

## Handler Case Studies

### apply_patch 为什么是一等工具

很多 agent 都会把文件编辑做成“让模型生成 shell 命令”。Codex 没这么做。

`apply_patch` 有独立 handler，主入口在 [`handlers/apply_patch.rs`][apply-patch-handler]。同时 shell / unified exec 中如果检测到 `apply_patch` 命令，还会走 [`intercept_apply_patch`][apply-patch-intercept]。

这个拦截很重要。否则模型可以把 patch 塞进 shell，绕过 patch 工具的语义、文件级审批、diff tracking 和安全评估。

`apply_patch` 一等化带来的好处是：

- patch 可以被解析成结构化文件修改。
- 审批可以按文件级别处理。
- TUI / app-server 可以展示 diff。
- rollout / trace 可以更准确记录“修改了什么”。
- shell 只是命令执行，不承担所有文件编辑语义。

换句话说，Codex 不是只关心“最终文件变了”，它还关心“这个变更是通过什么语义边界产生的”。

### MCP / Apps：外部工具不是直接塞进 handler

MCP 工具走的路径也很能体现 Codex 的边界感。

Router 会把 model-visible MCP tool name 还原成 `server/tool/raw_arguments`。MCP handler 本身不实现业务，而是把调用交给 MCP connection manager。MCP 调用的生命周期在 [`mcp_tool_call.rs`][mcp-tool-call]：开始事件、approval、执行、结束事件都在这里。

Apps 是特殊 MCP server，也就是 `codex_apps`。app connector 的可用性、是否 enabled、tool approval policy、tool search lazy loading，都不是模型自己决定的，而是由 config、connector state 和 app-server protocol 共同决定。

这点和传统“给模型一个 tools 数组”不同。Codex 需要处理的不是几十个静态函数，而是动态出现的插件、MCP server、connector、deferred tools。`tool_search` / `tool_suggest` 的存在，就是为了在工具空间很大时，仍然让模型只在需要时展开能力。

## 小结

Codex 的工具系统可以概括成一句话：

> 模型输出只是一份工具调用意图；真正能不能执行、如何执行、在哪个权限和 sandbox 下执行，由 core 的工具运行时决定。

关键分层是：

1. `ToolSpec`：模型可见 schema。
2. `ToolRouter`：把模型输出还原成内部 `ToolCall`。
3. `ToolCallRuntime`：处理取消和并行。
4. `ToolRegistry`：统一 handler 分发、hooks、telemetry、mutating gate。
5. `ToolOrchestrator / sandboxing`：集中审批和 sandbox attempt。
6. handler runtime：shell、unified exec、apply_patch、MCP、dynamic tools 各自执行。

这套结构的代价是文件多、调用链长；收益是安全策略、观测、并发和扩展都不需要散落到每个工具里。对 coding agent 来说，这个取舍很现实：工具能力越强，越需要把“能力”和“约束能力的边界”一起设计。

## 参考

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793)
- [Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/)

[build-prompt]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L935-L952
[build-specs]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/spec.rs#L71-L338
[tool-router-struct]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/router.rs#L39-L100
[handle-output]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/stream_events_utils.rs#L219-L255
[build-tool-call]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/router.rs#L175-L267
[tool-runtime-struct]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/parallel.rs#L27-L49
[tool-runtime-lock]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/parallel.rs#L82-L143
[registry-dispatch]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/registry.rs#L265-L513
[tool-handler-trait]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/registry.rs#L44-L92
[shell-to-exec-params]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/handlers/shell.rs#L92-L114
[unified-exec-args]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/handlers/unified_exec.rs#L45-L68
[unified-exec-handle]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/handlers/unified_exec.rs#L179-L380
[exec-approval-requirement]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/sandboxing.rs#L159-L180
[default-approval]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/sandboxing.rs#L198-L239
[apply-patch-handler]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/handlers/apply_patch.rs#L341
[apply-patch-intercept]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/handlers/apply_patch.rs#L468
[mcp-tool-call]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/mcp_tool_call.rs#L87
[openclaw-sandboxing]: https://github.com/openclaw/openclaw/blob/5d8ca42c7de8118b15782bad9cbac6240585e13a/docs/gateway/sandboxing.md#L10-L37
[openclaw-sandbox-backend]: https://github.com/openclaw/openclaw/blob/5d8ca42c7de8118b15782bad9cbac6240585e13a/src/agents/sandbox/backend.ts#L29-L54
