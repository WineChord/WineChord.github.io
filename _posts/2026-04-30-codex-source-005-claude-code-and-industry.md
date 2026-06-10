---
classes: wide2
title: "Codex 源码剖析：005. 和 Claude Code 以及行业形态对照"
excerpt: "把 Codex 和 WineChord/claude-code 对照仓库、OpenClaw 放在一起看：一个更像可复用 harness，一个更像集成式 TypeScript CLI runtime，一个更像 gateway-first control plane。"
categories:
  - LLM
  - Agent
tags:
  - Codex
  - Claude Code
  - Coding Agent
  - Source
toc: true
toc_sticky: true
mathjax: true
---

前四篇把 Codex 的主循环、工具系统、上下文工程、App Server、多 Agent 和 trace 都走了一遍。最后一篇做横向对照：Codex 和 `WineChord/claude-code` 对照仓库、OpenClaw 的源码组织有什么差异，这些差异背后反映了 coding agent 的什么演进方向。

源码版本：

- Codex: [`openai/codex@ac4332c`](https://github.com/openai/codex/tree/ac4332c05b11e00ae775a24cb762edc05c5b5932)
- Claude Code 对照仓库: [`WineChord/claude-code@5a774a2`](https://github.com/WineChord/claude-code/tree/5a774a2b62d7949c1d94e0b726281554d7893cfd)
- OpenClaw: [`openclaw/openclaw@5d8ca42`](https://github.com/openclaw/openclaw/tree/5d8ca42c7de8118b15782bad9cbac6240585e13a)

先把边界说清楚：`WineChord/claude-code` 是本文用来做公开源码链接和结构对照的仓库样本，不代表 Anthropic 官方内部实现的完整形态。Claude Code 的产品行为和公开能力，以 Anthropic 官方文档为准；这里的 TypeScript 结构只用来帮助理解一种 terminal-first runtime 的代码组织方式。

本文结论很简单：

> Claude Code 对照仓库更像集成式 CLI runtime；Codex 更像协议化 agent harness；OpenClaw 更像 gateway-first control plane。

这不是“谁更好”的排序，而是三种产品压力在源码边界上的投影。

![Codex、Claude Code 和 OpenClaw 对照：Claude Code 更像集成式 TypeScript CLI runtime，Codex 更像协议化 agent harness，OpenClaw 更像 gateway-first control plane](/assets/images/posts/2026-04-30-codex-source/comparison.png)

*图 1. 横向看，差异不是语言选择。Claude Code 对照仓库把很多应用状态放在 TypeScript CLI runtime 里；Codex 把 CLI、App Server、core、tools、sandbox、rollout、trace 拆成更硬的层；OpenClaw 则把 Gateway 作为 control plane。*

## 一、Claude Code：集成式 CLI runtime

Claude Code 官方文档把它描述成一个 agentic coding tool：能读代码库、编辑文件、运行命令，并集成到 terminal、IDE、desktop、browser 等开发工具里（见 [Claude Code overview](https://code.claude.com/docs/en/overview)）。它的工作方式文档也直接把工具分成 file operations、search、execution、web、code intelligence 等类别，并说明每次 tool use 的结果会反馈进下一步决策（见 [How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)）。

从对照仓库看，这种“完整 CLI 应用”的味道很明显。

入口在 [`src/entrypoints/cli.tsx`][claude-cli]。这个文件一上来就处理 fast path、remote-control、Chrome MCP、daemon worker、version、system prompt dump 等启动路径。它不是像 Codex 的 `codex.js` 那样只负责找 native binary，而是 TypeScript runtime 的真正 bootstrap。

模型循环核心集中在 [`src/QueryEngine.ts`][claude-query-engine] 和 [`src/query.ts`][claude-query]。`QueryEngineConfig` 里可以看到它直接持有几类运行时状态：

- 工作区和输入：`cwd`、initial messages、file caches
- 工具和扩展：tools、commands、MCP clients、agents
- 权限和 UI：`canUseTool`、AppState getter/setter、permission elicitation
- 模型设置：model / fallback model、thinking config、budget、json schema

`query.ts` 导出 async generator `query()`，内部 `queryLoop()` 维护 messages、toolUseContext、autoCompactTracking、turnCount、pendingToolUseSummary、stopHookActive 等 mutable state，见 [`query.ts` 的 query loop][claude-query-loop]。

这说明 Claude Code 对照仓库的核心循环更像“一个应用内的大型 async generator”。模型请求、tool use、compaction、memory prefetch、skill discovery、budget、stop hooks 都在同一条运行时轨迹里协作。

工具抽象也体现了集成式特点。[`ToolInputJSONSchema`][claude-tool-schema] 定义 schema 形状，[`ToolPermissionContext`][claude-tool-permission] 放权限状态，[`Tool` 类型][claude-tool-type] 的 `call` / `description` / permission 相关上下文又会接触到 UI、session state、MCP、agent、hooks、file state cache 等运行时信息。

这种结构的好处是功能接入路径短。要做一个 terminal-first 的完整产品，把输入、UI、工具、权限、hooks、memory 放在一个 TypeScript runtime 里，会很快，体验也容易做得整体。

代价是边界压力。随着支持的 surface 越来越多，模型循环会吸进越来越多职责。要把同一套 loop 拆给远程 server、无头 worker、IDE client、独立 sandbox、外部协议，就会遇到边界压力。

## 二、Codex：协议化 harness

Codex 的入口完全不同。

`codex-cli/bin/codex.js` 只做平台 binary 分发；Rust CLI 的 [`MultitoolCli`][codex-cli] 再分发到 TUI、exec、review、MCP server、app-server、sandbox、debug 等子命令。TUI 本地默认也启动 in-process app-server，走 thread/turn protocol。core 里的 `Codex` 是 submission/event 队列对，`Session` 维护 active task，`RegularTask` 再进入 `run_turn`。

工具层也不是“Tool 对象里什么都有”。Codex 拆成：

- [`ToolSpec` 构造][codex-tool-spec]
- [`ToolRouter`][codex-tool-router]
- [`ToolCallRuntime`][codex-tool-runtime]
- [`ToolRegistry`][codex-tool-registry]
- sandbox / approval
- handler runtime

上下文层继续拆成 `base_instructions`、contextual developer/user fragments、`reference_context_item`、`ContextManager`、compaction、rollout JSONL、rollout trace。

App Server 再把 external protocol 单独拆出去，v2 类型做 camelCase translation，core 错误类型映射成 client protocol 错误，见 [`app-server-protocol/v2.rs`][codex-v2-protocol]。

这套结构的好处不是代码少，而是边界稳定。TUI、VS Code、remote client、无头 exec、MCP server、cloud task、multi-agent child thread，都可以围绕同一个 core runtime 组合。

代价也有：读源码时更绕，跨 crate 多，类型转换多，很多地方看起来像“为什么不直接调用”。但如果目标是构建一个多端、多权限、多运行环境、可恢复、可审计的 agent harness，这些中间层就是必要成本。

## 三、OpenClaw：gateway-first control plane

OpenClaw 给了第三个参照系。它的 Gateway protocol 明确区分 operator / node role，control UI 和 TUI 都是 Gateway client；ACP runtime 又被抽象成 `ensureSession`、`runTurn`、cancel、close 等接口，甚至可以通过 `acpx` 启动外部命令，见 [`control-ui.md`][openclaw-control-ui]、[`gateway-chat.ts`][openclaw-tui-gateway] 和 [`acp runtime types`][openclaw-acp-runtime]。

这种结构比 Codex 更 gateway-first，也比 Claude Code 对照仓库的 in-process query loop 更像一个分布式控制面。它强调的是：先定义 operator、node、sandbox backend、ACP runtime 之间的控制边界，再把具体 agent 能力挂进去。

所以可以把三者粗略放成三类：

| 形态 | 主要边界 | 优点 | 压力 |
| --- | --- | --- | --- |
| 集成式 CLI runtime | 应用内 query loop | 产品体验聚合快 | 多端、多环境拆分压力大 |
| 协议化 agent harness | CLI / App Server / core / tools / trace | 多端复用、可恢复、可审计 | 调用链长、类型转换多 |
| gateway-first control plane | Gateway / node / sandbox / ACP runtime | 分布式控制边界清楚 | 本地单机体验要再组合 |

## 四、行业脉络：从补全到执行环境

这条线可以用一个简单坐标理解：越往右，越不只是模型能力，而是 execution environment 的工程能力。

![Coding agent 演进轴：从补全聊天到 ReAct、ACI，再到可约束、可恢复、可审计的 agent harness](/assets/images/posts/2026-04-30-codex-source/industry-axis.png)

*图 2. 行业演进的重心在右移。模型能力当然重要，但 coding agent 真正落地时，执行环境、权限、日志、恢复和 trace 会越来越像基础设施。*

第一步是补全和聊天：模型给建议，人自己复制、运行、修。这个阶段工具边界很弱，模型输出主要是文本。

第二步是 ReAct 这种 “reason + act” 循环：模型可以交错产生推理和动作，动作结果再反馈给模型。ReAct 论文强调 actions 能让模型接触外部环境，reasoning traces 则帮助跟踪和更新计划（见 [ReAct](https://arxiv.org/abs/2210.03629)）。

第三步是 agent-computer interface。SWE-agent 论文的核心观点是：语言模型 agent 也是一种新型 end user，也需要专门设计的接口；好的 ACI 会显著影响它浏览代码、编辑文件、运行测试的表现（见 [SWE-agent](https://arxiv.org/abs/2405.15793)）。

第四步就是现在这些 coding agent 产品正在做的事：把 agent 放进可审计的软件执行环境里。OpenAI Codex 的云端形态强调独立 sandbox、并行任务、日志、测试输出、PR workflow。GitHub Copilot cloud agent 也强调 agent 在 GitHub Actions-powered ephemeral development environment 中研究仓库、制定计划、改分支、运行测试，并让用户 review diff / PR（见 [GitHub Copilot cloud agent](https://docs.github.com/en/copilot/concepts/agents/cloud-agent/about-cloud-agent)）。

这说明行业焦点已经从“模型能不能写代码”转向“模型写代码的执行环境是否可靠”：

- 能否限制文件系统和网络？
- 能否复现每一步？
- 能否审计 tool call？
- 能否恢复长会话？
- 能否把多个 agent 的协作关系讲清楚？
- 能否把权限、记忆、上下文、日志、测试证据纳入统一生命周期？

Codex 这个 repo 最有价值的地方，就在于它把这些 harness 问题都显式工程化了。

## 五、从源码看未来几条线

第一，local pair 和 async delegation 会继续收敛。OpenAI 在 Codex 发布文章里也说过，实时协作和异步委托最终会收敛。源码上看，Codex 已经在朝这个方向走：TUI 本地通过 in-process app-server，远程 client 通过 websocket app-server，core 看到的都是 thread/turn protocol。

第二，工具系统会继续从“函数列表”变成“策略化工具空间”。MCP、Apps、plugins、dynamic tools、deferred tools、tool_search、hooks、approval、sandbox 放在一起看，工具已经不是一个 schema 数组，而是一套随上下文变化的能力市场。

第三，memory 会从“保存用户偏好”变成“长期工程上下文的受控管线”。Codex 把 memory read/write 拆开，并让写入走后台 Phase 1 / Phase 2 和隔离 internal agent。这说明可靠 memory 不能只是当前 agent 随手写几行，它需要选择、抽取、脱敏、合并、diff、审计。

第四，trace 会越来越重要。当一次任务里有模型请求、工具调用、终端进程、patch、MCP、子 agent、approval、compaction，普通聊天记录已经解释不了系统行为。Codex 的 rollout trace 把 raw evidence reduce 成 graph，这类能力会变成 debugging agent 的基础设施。

## 六、源码阅读规则

| 看到一种 agent 架构 | 先问什么 | 判断重点 |
| --- | --- | --- |
| 一个巨大 query loop | 它聚合了多少 UI、工具、权限状态？ | 体验接入快，但拆分压力可能上升。 |
| 很多 protocol 类型转换 | 它是在保护外部 API 稳定性吗？ | 不要把转换都当成无意义样板。 |
| 独立 App Server / Gateway | 它服务单端还是多端？ | 多端、多环境越多，协议边界越重要。 |
| 工具对象很胖 | 它是否直接触达 UI 和权限状态？ | 这是集成式 runtime 的常见信号。 |
| rollout / trace / state DB | 它解决恢复还是诊断？ | 恢复路径和诊断路径要分开。 |

## 小结

Claude Code 和 Codex 源码的差异，不只是 TypeScript 和 Rust 的差异。

Claude Code 对照仓库更像一个强集成 CLI runtime：入口、模型循环、工具、权限、UI、memory、hooks 在同一应用层里深度协作，适合快速做出 terminal-first 的完整体验。

Codex 更像一个 agent harness：CLI/TUI/app-server/core/tools/sandbox/context/rollout/trace 被拆成稳定边界，适合多端、多环境、多 Agent、可恢复、可审计的执行系统。

OpenClaw 则把 Gateway 放在更前面，强调 operator / node / sandbox / ACP runtime 的控制边界。

可以把这三者看成 coding agent 演进中的不同阶段和取舍：当产品主要是“人在终端里和 agent 结对”，集成式 runtime 很自然；当产品开始承担“后台任务、远程协作、分支提交、长会话恢复、多 Agent 调度、日志审计”，harness 化和 control-plane 化就会越来越重要。

## 参考

- [Claude Code overview](https://code.claude.com/docs/en/overview)
- [Claude Code: How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)
- [Claude Code: Permission modes](https://code.claude.com/docs/en/permission-modes)
- [Claude Code: How Claude remembers your project](https://code.claude.com/docs/en/memory)
- [OpenAI: Introducing Codex](https://openai.com/index/introducing-codex/)
- [GitHub Docs: About GitHub Copilot cloud agent](https://docs.github.com/en/copilot/concepts/agents/cloud-agent/about-cloud-agent)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793)

[claude-cli]: https://github.com/WineChord/claude-code/blob/5a774a2b62d7949c1d94e0b726281554d7893cfd/src/entrypoints/cli.tsx#L28-L160
[claude-query-engine]: https://github.com/WineChord/claude-code/blob/5a774a2b62d7949c1d94e0b726281554d7893cfd/src/QueryEngine.ts#L130-L180
[claude-query]: https://github.com/WineChord/claude-code/blob/5a774a2b62d7949c1d94e0b726281554d7893cfd/src/query.ts#L181-L220
[claude-query-loop]: https://github.com/WineChord/claude-code/blob/5a774a2b62d7949c1d94e0b726281554d7893cfd/src/query.ts#L241-L307
[claude-tool-schema]: https://github.com/WineChord/claude-code/blob/5a774a2b62d7949c1d94e0b726281554d7893cfd/src/Tool.ts#L15-L21
[claude-tool-permission]: https://github.com/WineChord/claude-code/blob/5a774a2b62d7949c1d94e0b726281554d7893cfd/src/Tool.ts#L123-L148
[claude-tool-type]: https://github.com/WineChord/claude-code/blob/5a774a2b62d7949c1d94e0b726281554d7893cfd/src/Tool.ts#L362-L390
[codex-js-platform]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-cli/bin/codex.js#L15-L22
[codex-cli]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/cli/src/main.rs#L70-L176
[codex-regular-task]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tasks/regular.rs#L40-L86
[codex-tool-spec]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/spec.rs#L71-L338
[codex-tool-router]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/router.rs#L39-L100
[codex-tool-runtime]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/parallel.rs#L27-L143
[codex-tool-registry]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/registry.rs#L44-L92
[codex-v2-protocol]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/v2.rs#L123-L230
[openclaw-control-ui]: https://github.com/openclaw/openclaw/blob/5d8ca42c7de8118b15782bad9cbac6240585e13a/docs/web/control-ui.md#L11-L17
[openclaw-tui-gateway]: https://github.com/openclaw/openclaw/blob/5d8ca42c7de8118b15782bad9cbac6240585e13a/src/tui/gateway-chat.ts#L153-L167
[openclaw-acp-runtime]: https://github.com/openclaw/openclaw/blob/5d8ca42c7de8118b15782bad9cbac6240585e13a/src/acp/runtime/types.ts#L118-L138
