---
classes: wide2
title: "Codex 源码剖析：005. 和 Claude Code 以及行业形态对照"
excerpt: "把 Codex 和 WineChord/claude-code 对照仓库的源码组织放在一起看：一个更像可复用 harness，一个更像集成式 TypeScript CLI runtime。"
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

![Codex 源码剖析系列封面：把 Codex 和 Claude Code 对照放在 coding agent 行业演进中理解](/assets/images/posts/2026-04-30-codex-source/series-hero.png)

*图 0：最后一篇不再只看 Codex 内部，而是把它放到 Claude Code 和 coding agent 产品形态演进里看。*

前四篇基本把 Codex 的主循环、工具系统、上下文工程、App Server、多 Agent 和 trace 都走了一遍。最后一篇做一个横向对照：Codex 和 `WineChord/claude-code` 源码结构有什么差异，这些差异背后反映了 coding agent 的什么演进方向。

源码版本：

- Codex: [`openai/codex@ac4332c05b11e00ae775a24cb762edc05c5b5932`](https://github.com/openai/codex/tree/ac4332c05b11e00ae775a24cb762edc05c5b5932)
- Claude Code 对照仓库: [`WineChord/claude-code@5a774a2b62d7949c1d94e0b726281554d7893cfd`](https://github.com/WineChord/claude-code/tree/5a774a2b62d7949c1d94e0b726281554d7893cfd)

这里先把边界说清楚：`WineChord/claude-code` 是本文用来做公开源码链接和结构对照的仓库样本，我不会把它当作 Anthropic 官方内部实现的完整代表。Claude Code 的产品行为和公开能力，以 Anthropic 的官方文档为准；仓库里的 TypeScript 结构只用来帮助我们理解一种 terminal-first runtime 的代码组织方式。

先说结论：

> Claude Code 更像一个集成式 TypeScript CLI runtime，把入口、模型循环、工具、权限、UI 状态、memory、hooks、subagent 都串在一个应用运行时里；Codex 更像一个协议化 harness，把 CLI / TUI / app-server protocol / core session-turn engine / tool router-registry / sandbox runtime / rollout / trace 拆成清晰边界。

这不是简单的“谁更好”。两种结构背后对应的是不同产品形态、不同部署方式、不同演进压力。

![Codex 和 Claude Code 对照：Claude Code 更像集成式 TypeScript CLI runtime，Codex 更像协议化 agent harness](/assets/images/posts/2026-04-30-codex-source/comparison.svg)

*图 1：横向看，两边不是语言差异这么简单，而是 runtime 边界差异。Claude Code 对照仓库把很多应用状态放在 TypeScript CLI runtime 里；Codex 把 CLI、App Server、core、tools、sandbox、rollout、trace 拆成更硬的层。*

![Coding agent 演进轴：从补全聊天到 ReAct、ACI，再到可约束、可恢复、可审计的 agent harness](/assets/images/posts/2026-04-30-codex-source/industry-axis.svg)

*图 2：行业演进的重心在右移。模型能力当然重要，但 coding agent 真正落地时，执行环境、权限、日志、恢复和 trace 会越来越像基础设施。*

## Claude Code：集成式 CLI runtime

Claude Code 官方文档里把它描述成一个 agentic coding tool：能读代码库、编辑文件、运行命令，并集成到 terminal、IDE、desktop、browser 等开发工具里（见 [Claude Code overview](https://code.claude.com/docs/en/overview)）。它的工作方式文档也直接把工具分成 file operations、search、execution、web、code intelligence 等类别，并说明每次 tool use 的结果会反馈进下一步决策（见 [How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)）。

源码上看，这种“完整 CLI 应用”的味道很明显。

入口在 [`src/entrypoints/cli.tsx`][claude-cli]。这个文件一上来就处理各种 fast path、remote-control、Chrome MCP、daemon worker、version、system prompt dump 等启动路径。它不是像 Codex 的 `codex.js` 那样只负责找 native binary，而是 TypeScript runtime 的真正 bootstrap。

模型循环核心集中在 [`src/QueryEngine.ts`][claude-query-engine] 和 [`src/query.ts`][claude-query]。`QueryEngineConfig` 里可以看到它直接持有：

- cwd
- tools
- commands
- MCP clients
- agents
- canUseTool
- AppState getter/setter
- initial messages
- file caches
- model / fallback model
- thinking config
- budget
- json schema
- permission elicitation 等

`query.ts` 则导出 async generator `query()`，内部 `queryLoop()` 维护 messages、toolUseContext、autoCompactTracking、turnCount、pendingToolUseSummary、stopHookActive 等 mutable state，见 [`query.ts` 的 query loop][claude-query-loop]。

这说明 Claude Code 的核心循环更像“一个应用内的大型 async generator”。模型请求、tool use、compaction、memory prefetch、skill discovery、budget、stop hooks 都在同一条运行时轨迹里协作。

工具抽象也体现了集成式特点。[`ToolInputJSONSchema`][claude-tool-schema] 定义 schema 形状，[`ToolPermissionContext`][claude-tool-permission] 放权限状态，[`Tool` 类型][claude-tool-type] 的 `call` / `description` / permission 相关上下文又会接触到 UI、session state、MCP、agent、hooks、file state cache 等运行时信息。也就是说，一个 Tool 的上下文不仅知道“如何执行”，还知道应用运行时的很多状态。

这种结构的好处是功能接入路径短。你要做一个 terminal-first 的产品，把输入、UI、工具、权限、hooks、memory 全放在一个 TypeScript runtime 里，会很快，体验也容易做得整体。

代价也很明显：随着支持的 surface 越来越多，模型循环会吸进越来越多职责。要把同一套 loop 拆给远程 server、无头 worker、IDE client、独立 sandbox、外部协议，就会遇到边界压力。

## Codex：协议化 harness

Codex 的入口完全不同。

`codex-cli/bin/codex.js` 只做平台 binary 分发；Rust CLI 的 [`MultitoolCli`][codex-cli] 再分发到 TUI、exec、review、MCP server、app-server、sandbox、debug 等子命令。TUI 本地默认也启动 in-process app-server，走 thread/turn protocol。core 里的 `Codex` 是 submission/event 队列对，`Session` 维护 active task，`RegularTask` 再进入 `run_turn`。

工具层也不是“Tool 对象里什么都有”。Codex 拆成：

- [`ToolSpec` 构造][codex-tool-spec]
- [`ToolRouter`][codex-tool-router]
- [`ToolCallRuntime`][codex-tool-runtime]
- [`ToolRegistry`][codex-tool-registry]
- sandbox / approval
- handler runtime

上下文层继续拆：

- `base_instructions`
- contextual developer/user fragments
- `reference_context_item`
- `ContextManager`
- compaction
- rollout JSONL
- rollout trace

App Server 再把 external protocol 单独拆出去，v2 类型做 camelCase translation，core 错误类型映射成 client protocol 错误，见 [`app-server-protocol/v2.rs`][codex-v2-protocol]。

这套结构的好处不是代码少，而是边界稳定。TUI、VS Code、remote client、无头 exec、MCP server、cloud task、multi-agent child thread，都可以围绕同一个 core runtime 组合。

代价也有：读源码时更绕，跨 crate 多，类型转换多，很多地方看起来像“为什么不直接调用”。但如果目标是构建一个多端、多权限、多运行环境、可恢复、可审计的 agent harness，这些中间层就是必要成本。

## 两者差异来自产品形态

用一句话概括：

- Claude Code 更像“terminal-first agent application”。
- Codex 更像“agent runtime + protocol + harness”。

这会影响源码组织的很多细节。

Claude Code 的 `query.ts` 把模型循环做成 async generator，天然适合把消息、工具、UI streaming、SDK/headless 输出放在同一个数据流里。工具上下文里带 AppState、setToolJSX、appendSystemMessage、sendOSNotification，这对 terminal/REPL 体验很直接。

Codex 则尽量让 core 不知道 TUI 怎么渲染。core 只发 `EventMsg`；app-server 翻译成 `ServerNotification` / `ServerRequest`；TUI 再根据 thread id buffer 和渲染。工具 handler 也不直接关心 UI，审批请求通过 protocol 回流。

所以 Claude Code 读起来更像“一个应用怎么完成用户任务”；Codex 读起来更像“一个 agent 系统怎么定义稳定边界”。

如果只做本地 CLI，集成式结构更快、更顺。如果要同时服务本地交互、远程任务、VS Code、cloud delegation、多 Agent、trace graph，Codex 这种分层会更有后劲。

把两边入口链路压缩成最小图，大概是：

```text
Claude Code 对照仓库:
  cli.tsx -> QueryEngine -> query() async generator -> Tool.call()

Codex:
  codex.js -> Rust CLI -> TUI / App Server -> Codex queue
           -> RegularTask / run_turn -> ToolRouter / ToolRegistry
```

这也是我说两边“产品形态不同”的原因：前者更像一个应用 runtime 内部的主数据流，后者更像一个可被多个 client 驱动的 protocol runtime。

| 观察点 | `WineChord/claude-code` 对照仓库 | Codex |
| --- | --- | --- |
| 入口 | [`cli.tsx`][claude-cli] 直接承担 TypeScript runtime bootstrap | [`codex.js`][codex-js-platform] 只找 native binary，Rust CLI 再分发 |
| 主循环 | [`query()`][claude-query-loop] 是大型 async generator | [`RegularTask`][codex-regular-task] 进入 `run_turn`，core 通过 event 回流 |
| 工具上下文 | [`Tool` 类型][claude-tool-type] 接触 UI、permission、MCP、agent 等应用状态 | `ToolSpec` / `ToolRouter` / `ToolRegistry` / sandbox 分层 |
| 外部协议 | 更像 terminal-first 应用向外延展 | App Server protocol 是一等边界 |
| 恢复与观测 | 重点在应用会话体验 | rollout JSONL + rollout trace graph 明确拆分 |

## 行业脉络：从补全到执行环境

coding agent 的演进可以粗略分成几步。

第一步是补全和聊天：模型给建议，人自己复制、运行、修。这个阶段工具边界很弱，模型输出主要是文本。

第二步是 ReAct 这种 “reason + act” 循环：模型可以交错产生推理和动作，动作结果再反馈给模型。ReAct 论文强调 actions 能让模型接触外部环境，reasoning traces 则帮助跟踪和更新计划（见 [ReAct](https://arxiv.org/abs/2210.03629)）。

第三步是 agent-computer interface。SWE-agent 论文的核心观点是：语言模型 agent 也是一种新型 end user，也需要专门设计的接口；好的 ACI 会显著影响它浏览代码、编辑文件、运行测试的表现（见 [SWE-agent](https://arxiv.org/abs/2405.15793)）。

第四步就是现在这些 coding agent 产品正在做的事：把 agent 放进可审计的软件执行环境里。

OpenAI Codex 的云端形态强调独立 sandbox、并行任务、日志、测试输出、PR workflow。GitHub Copilot cloud agent 也强调 agent 在 GitHub Actions-powered ephemeral development environment 中研究仓库、制定计划、改分支、运行测试，并让用户 review diff / PR（见 [GitHub Copilot cloud agent](https://docs.github.com/en/copilot/concepts/agents/cloud-agent/about-cloud-agent)）。

这说明行业焦点已经从“模型能不能写代码”转向“模型写代码的执行环境是否可靠”：

- 能否限制文件系统和网络？
- 能否复现每一步？
- 能否审计 tool call？
- 能否恢复长会话？
- 能否把多个 agent 的协作关系讲清楚？
- 能否把权限、记忆、上下文、日志、测试证据纳入统一生命周期？

Codex 这个 repo 最有价值的地方，就在于它把这些 harness 问题都显式工程化了。

## 从源码看未来几条线

第一，local pair 和 async delegation 会越来越合并。

OpenAI 在 Codex 发布文章里也说过，实时协作和异步委托最终会收敛。源码上看，Codex 已经在朝这个方向走：TUI 本地通过 in-process app-server，远程 client 通过 websocket app-server，core 看到的都是 thread/turn protocol。

第二，工具系统会继续从“函数列表”变成“策略化工具空间”。

MCP、Apps、plugins、dynamic tools、deferred tools、tool_search、hooks、approval、sandbox 这些东西放在一起看，工具已经不是一个 schema 数组，而是一套随上下文变化的能力市场。模型看到什么工具、工具能不能执行、执行结果如何进入下一轮，都需要 runtime 决策。

第三，memory 会从“保存用户偏好”变成“长期工程上下文的受控管线”。

Codex 把 memory read/write 拆开，并让写入走后台 Phase 1 / Phase 2 和隔离 internal agent。这说明真正可靠的 memory 不能只是当前 agent 随手写几行，它需要选择、抽取、脱敏、合并、diff、审计。

第四，trace 会越来越重要。

当一次任务里有模型请求、工具调用、终端进程、patch、MCP、子 agent、approval、compaction，普通聊天记录已经解释不了系统行为。Codex 的 rollout trace 把 raw evidence reduce 成 graph，这类能力未来会变成 debugging agent 的基础设施。

## 这一篇的核心结论

Claude Code 和 Codex 源码的差异，不只是 TS 和 Rust 的差异。

Claude Code 更像一个强集成 CLI runtime：入口、模型循环、工具、权限、UI、memory、hooks 在同一应用层里深度协作，适合快速做出 terminal-first 的完整体验。

Codex 更像一个 agent harness：CLI/TUI/app-server/core/tools/sandbox/context/rollout/trace 被拆成稳定边界，适合多端、多环境、多 Agent、可恢复、可审计的执行系统。

我个人更倾向于把这两者看成 coding agent 演进中的两种阶段：

- 当产品还主要是“人在终端里和 agent 结对”，集成式 runtime 很自然。
- 当产品开始承担“后台任务、远程协作、分支提交、长会话恢复、多 Agent 调度、日志审计”，harness 化就会越来越重要。

从这个角度看 Codex，最值得学的不是某个工具实现，而是它如何把模型能力放进一套可约束、可恢复、可观测的软件系统里。

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
