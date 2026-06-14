---
classes:
  - wide2
  - codex-source-005-claude-code-and-industry
title: "Codex 源码剖析：005. Claude Code、Codex 与行业形态"
excerpt: "从实践压力出发，对照 Claude Code 公开源码样本、Codex 和 OpenClaw：先看 provider/platform contract，再看 source execution chain，最后提炼 coding agent harness 的可迁移规则。"
last_modified_at: 2026-06-14
locale: zh-CN
canonical_url: "https://www.wineandchord.com/llm/agent/codex-source-005-claude-code-and-industry/"
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
toc_label: "目录"
toc_levels: 2..4
mathjax: true
header:
  og_image: https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-industry-comparison-cover.png
  teaser: https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-industry-comparison-cover.png
---

一个 coding agent 真正进入团队工作流以后，最先暴露的不是“模型会不会写代码”，而是执行环境能不能被信任：它在哪个目录里读写文件，什么时候能跑命令，失败后能不能恢复，审计时能不能还原哪次 tool call 造成了哪段 diff，多个 agent 并行时谁拥有上下文。

所以源码对照不应该从 TypeScript 和 Rust 的语言偏好开始。更好的切入点是实践压力：当一个 agent 要从单人终端扩展到 IDE、桌面、云端任务、GitHub review、多人协作和长会话恢复时，runtime 边界会被迫移动。

本文对照三个公开源码或公开协议样本：

- Codex: [`openai/codex@ac4332c`](https://github.com/openai/codex/tree/ac4332c05b11e00ae775a24cb762edc05c5b5932)
- Claude Code 对照仓库: [`WineChord/claude-code@5a774a2`](https://github.com/WineChord/claude-code/tree/5a774a2b62d7949c1d94e0b726281554d7893cfd)
- OpenClaw: [`openclaw/openclaw@5d8ca42`](https://github.com/openclaw/openclaw/tree/5d8ca42c7de8118b15782bad9cbac6240585e13a)

证据边界也要先说清楚：`WineChord/claude-code` 是本文用于公开源码链接和结构对照的仓库样本，不代表 Anthropic 官方内部实现的完整形态。Claude Code 的产品能力以 [Claude Code overview][claude-overview] 和 [How Claude Code works][claude-how] 为准；Codex 的产品与平台 contract 以 [Codex CLI][codex-cli-docs]、[Codex web][codex-cloud-docs]、[Codex IDE extension][codex-ide-docs]、[Sandbox][codex-sandbox-docs] 和 `openai/codex` 固定源码快照为准。OpenClaw 只作为 gateway-first control plane 的开源参照。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-industry-comparison-cover.png" alt="三种 coding agent 架构压力：Claude Code sample、Codex 和 OpenClaw 分别对应 integrated CLI runtime、agent harness 和 gateway-first control plane">
  <figcaption>图 1. 对照的核心不是语言选择，而是产品压力落在哪一层：集成体验、可恢复 harness，还是分布式 control plane。</figcaption>
</figure>

## 阅读契约

这篇文章只回答一个问题：当 coding agent 从“会调用工具的聊天模型”变成“能读写仓库、运行命令、恢复长会话、交付 PR、被多个 client 驱动的执行环境”时，源码边界会如何变化？

读完以后，应该能把一个 agent 架构放进五层判断里：

1. mental model：它的核心 owner 是应用 runtime、协议 harness，还是 gateway control plane？
2. provider/platform contract：官方文档和外部 API 先承诺了哪些执行能力？
3. source execution chain：一次用户任务从入口、loop、tools、context 到 durable record 怎么流动？
4. common misreadings：哪些源码差异容易被误读成语言偏好、样板代码或产品排名？
5. transferable rules：设计自己的 coding agent 时，哪些状态应该归 UI、runtime、provider、rollout 或 trace？

## 一、先建立 mental model：coding agent 是执行环境

### 1.1 从“模型循环”升到“runtime owner”

ReAct 把模型循环讲清楚了：reasoning trace 和 action 交错，外部动作结果再反馈给下一步推理，见 [ReAct][react-paper]。SWE-agent 进一步提出 ACI：语言模型 agent 需要专门设计的 computer interface，接口质量会影响它浏览代码、编辑文件、运行测试的能力，见 [SWE-agent][swe-agent-paper]。

但真实产品还要再往下走一层。coding agent 不是只执行 `think -> act -> observe`，它还要回答这些工程问题：

- 这一步动作在哪个机器或 sandbox 里执行？
- tool result 是完整进入模型、被 preview、被落盘，还是只进 trace？
- 权限是 per-action prompt、runtime policy，还是 gateway perimeter？
- 长会话压缩以后，resume 的 baseline 从哪里来？
- 多个 agent 协作时，父线程、子线程和 mailbox 谁拥有信息流？

这就是本文的 mental model：把 agent 看成一个执行环境，而不是一个 prompt 模板。

### 1.2 三个典型 owner

三份源码呈现的是三种 owner 选择。

Claude Code 对照仓库更像 integrated CLI runtime：入口、query loop、工具、权限、UI state、memory、hooks 都在 TypeScript 应用 runtime 里聚合，功能接入路径短。

Codex 更像 protocolized agent harness：JavaScript launcher、Rust CLI、App Server、core Session、tools runtime、context manager、[`rollout JSONL`][codex-rollout-recorder]、[`rollout trace`][codex-rollout-trace-readme] 被拆成稳定边界，适合多端复用、恢复和审计。

OpenClaw 更像 gateway-first control plane：先定义 operator、node、sandbox backend、ACP runtime 的控制边界，再把具体 agent 能力挂进去。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-comparison.png" alt="Codex、Claude Code 和 OpenClaw 的 owner 对照：Claude Code 是集成式 CLI runtime，Codex 是协议化 agent harness，OpenClaw 是 gateway-first control plane">
  <figcaption>图 2. 三者都能组织 agent work，但 owner 不同：应用 runtime 聚合体验，harness 稳住恢复和协议，gateway 先划分分布式控制面。</figcaption>
</figure>

## 二、Provider 与 platform contract 先约束源码边界

### 2.1 官方 contract 决定读源码的起点

不要先看类名。先看公开 contract 承诺了什么。

[Claude Code overview][claude-overview] 把 Claude Code 描述为能读代码库、编辑文件、运行命令，并出现在 terminal、IDE、desktop、browser 等 surface 的 agentic coding tool。[How Claude Code works][claude-how] 进一步把工具能力分成 file operations、search、execution、web、code intelligence 等类别，并说明每次 tool use 的结果会反馈给下一步决策。这解释了为什么一个 terminal-first runtime 会把 tool、permission、session、context 和 UI 操作放得很近。

OpenAI 侧的 contract 是多 surface 的组合。[Codex CLI][codex-cli-docs] 说明 CLI 是本地终端里的 coding agent，能在选定目录读、改、运行代码；[Codex web][codex-cloud-docs] 说明 cloud 任务可以在独立云环境后台运行，包括并行任务；[Codex IDE extension][codex-ide-docs] 说明 IDE 里既能 side-by-side pair，也能 delegate tasks to Codex Cloud；[Sandbox][codex-sandbox-docs] 则把 sandbox 明确称为让 Codex 自主行动但不获得默认无限制机器权限的边界。这些 contract 共同要求源码里有 client protocol、权限边界、执行隔离、恢复记录和跨 surface 状态。

OpenClaw 的公开源码从 Gateway protocol 入手。[`docs/gateway/protocol.md`][openclaw-gateway-protocol] 在 Roles + scopes 里把 `operator` 定义为 control plane client、把 `node` 定义为 capability host；[`control-ui.md`][openclaw-control-ui] 说明 Control UI 由 Gateway serving 并直接走 Gateway WebSocket，[`gateway-chat.ts`][openclaw-tui-gateway] 也把 TUI 注册成 Gateway client，ACP runtime 再提供 `ensureSession`、`runTurn`、cancel、close 等接口。这种 contract 先定义控制面，再接 agent runtime。

### 2.2 Contract 到 source shape 的映射

<div class="wc-responsive-table-wrap">
<table class="wc-responsive-table">
  <thead>
    <tr>
      <th>公开 contract</th>
      <th>源码后果</th>
      <th>保护的 invariant</th>
      <th>常见代价</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Terminal-first pair</td>
      <td>入口、loop、UI state、permission、tool context 靠近</td>
      <td>交互延迟低，功能接入短</td>
      <td>surface 增多后 loop 容易吸职责</td>
    </tr>
    <tr>
      <td>多端/云端/IDE 复用</td>
      <td>CLI、App Server、core、protocol types、tools、context 拆层</td>
      <td>外部 API 稳定，runtime 可恢复</td>
      <td>调用链更长，类型转换更多</td>
    </tr>
    <tr>
      <td>Sandbox 与 approval</td>
      <td>tool call 不直接等于系统调用，中间有 policy/runtime gate</td>
      <td>自主执行不等于无限权限</td>
      <td>边界调试比单进程调用复杂</td>
    </tr>
    <tr>
      <td>Gateway-first control plane</td>
      <td>client、operator、node、sandbox backend、ACP runtime 先分离</td>
      <td>分布式控制与路由清楚</td>
      <td>本地单机体验需要再组合</td>
    </tr>
  </tbody>
</table>
</div>

这个表先把源码阅读顺序固定下来：先找 contract，再找 owner，再看 execution chain。否则很容易把“为外部稳定性付出的转换层”误读成无意义样板。

## 三、Source execution chain：三个系统如何跑起来

### 3.1 Claude Code 对照仓库：integrated CLI runtime

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-claude-cli-runtime-loop.png" alt="Claude Code 对照仓库的集成式 CLI runtime：CLI bootstrap 进入 QueryEngine 和 query loop，再连接 UI state、tools、MCP、permissions、memory 和 skills">
  <figcaption>图 3. 集成式 runtime 的教学重点是“接入路径短”：工具、权限、UI 和 memory 在同一条 query loop 周围协作。</figcaption>
</figure>

#### CLI bootstrap 不是薄壳

Claude Code 对照仓库入口在 [`src/entrypoints/cli.tsx`][claude-cli]。这个文件处理 fast path、remote-control、Chrome MCP、daemon worker、version、system prompt dump 等启动路径。它不是纯粹寻找 native binary 的 shim，而是 TypeScript runtime 的真实 bootstrap。

模型循环核心集中在 [`src/QueryEngine.ts`][claude-query-engine] 和 [`src/query.ts`][claude-query]。`QueryEngineConfig` 里可以看到工作区、initial messages、file caches、tools、commands、MCP clients、agents、`canUseTool`、AppState getter/setter、model / fallback model、thinking config、budget、JSON schema 等状态。`query.ts` 导出的 async generator `query()` 继续进入 `queryLoop()`，在一条 runtime 轨迹里维护 messages、toolUseContext、autoCompactTracking、turnCount、pendingToolUseSummary、stopHookActive 等 mutable state，见 [`queryLoop`][claude-query-loop]。

#### Tool object 承载应用态

工具抽象也很集成。[`ToolInputJSONSchema`][claude-tool-schema] 定义输入 schema，[`ToolPermissionContext`][claude-tool-permission] 放权限状态，[`Tool` 类型][claude-tool-type] 的 `call`、`description` 和 permission 相关上下文会接触 UI、session state、MCP、agent、hooks、file state cache 等运行时信息。


#### 优势与边界压力

这种结构的优势很直接：terminal-first 产品要快速把工具、权限、UI、hooks、memory、skills 做成整体体验，把它们放在一个 TypeScript runtime 里很自然。

代价也同样清楚：当同一套 loop 要给远程 server、无头 worker、IDE client、独立 sandbox、外部 protocol 或多 agent child thread 复用时，原本方便的应用态会变成拆分压力。这里不是“TypeScript 不适合”，而是 owner 选择带来的边界成本。

### 3.2 Codex：protocolized agent harness

#### 从 JS launcher 到 Rust command fan-out

Codex 的入口完全不同。[`codex-cli/bin/codex.js`][codex-js-platform] 负责平台 target triple 和 native package binary 分发；Rust CLI 的 [`MultitoolCli`][codex-cli] 再把命令分发到 TUI、exec、review、MCP server、app-server、sandbox、debug 等子命令。进入 core 后，`Session` 维护 active task，`RegularTask` 再进入 [`run_turn`][codex-regular-task]。

这意味着 Codex 不是把所有能力塞进一个 CLI loop，而是先承认“会有很多入口”。CLI/TUI、无头 exec、MCP server、app-server、debug trace 都是进入 harness 的不同门。

#### Tools runtime 拆成 spec、router、registry 和 handler

工具层也不是“一个 Tool 对象里什么都有”。Codex 把工具拆成 [`ToolSpec` 构造][codex-tool-spec]、[`ToolRouter`][codex-tool-router]、[`ToolCallRuntime`][codex-tool-runtime]、[`ToolRegistry`][codex-tool-registry]、sandbox / approval 和 handler runtime。

这类拆分的好处是 tool call 可以同时服务模型请求、审批、sandbox、runtime handler 和 trace，而不是被 UI 对象或单个 query loop 绑定。

#### ContextManager 是恢复契约，不只是 prompt 拼接器

上下文层更能体现 harness 思路。[`ContextManager`][codex-context-manager] 在内存里保存 model-visible history、`history_version`、token usage，并有一个 [`reference_context_item`][codex-reference-context-item] 作为 settings diff 和 context reinjection 的 baseline。对应的 [`TurnContextItem`][codex-turn-context-item] 在 protocol 里记录 cwd、date/timezone、approval policy、sandbox policy、model、personality、user/developer instructions、truncation policy 等字段；源码注释说明它会在每个真实用户 turn 后持久化，也会在 mid-turn compaction 后重新建立 baseline，供 resume/fork replay 恢复。

rollout 是另一条 durable path。[`RolloutRecorder`][codex-rollout-recorder] 明确把 session rollout 作为 JSONL 持久化，以便 replay 或 inspect。恢复时，[`rollout_reconstruction.rs`][codex-rollout-reconstruction] 从新到旧扫描 rollout，找到 surviving `replacement_history` checkpoint 和最新 resume metadata，再只重放 surviving tail。trace 又是诊断路径：[`rollout-trace` README][codex-rollout-trace-readme] 说明它是 opt-in local diagnostic path，先记录 raw runtime evidence，再由 reducer 还原 semantic graph；[`replay_bundle()`][codex-rollout-trace-reducer] 从 `trace.jsonl` 和 payloads 读 raw events，reduce 成 `RolloutTrace`。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-codex-harness-boundaries.png" alt="Codex harness 边界：CLI/TUI、App Server、core Session、tools runtime、context system、rollout JSONL 和 rollout trace 形成稳定责任分层">
  <figcaption>图 4. Codex 的教学重点是“责任分层”：调用链变长，但 context、recovery、trace 和 client protocol 各有 owner。</figcaption>
</figure>

#### 长调用链是 harness 的成本

Codex 读起来更绕，跨 crate 多，类型转换多，很多地方第一眼像“为什么不直接调用”。但当目标是多端、多权限、多运行环境、可恢复、可审计的 agent harness，这些中间层就是把内部可变性和外部 contract 分开的成本。

### 3.3 OpenClaw：gateway-first control plane

#### Gateway 先于具体 agent loop

OpenClaw 给了第三个参照系。它的 [`Gateway protocol`][openclaw-gateway-protocol] 明确区分 operator / node role；[`control-ui.md`][openclaw-control-ui] 说明 Control UI 由 Gateway serving 并走 Gateway WebSocket，[`gateway-chat.ts`][openclaw-tui-gateway] 也把 TUI 注册成 Gateway client；ACP runtime 又被抽象成 `ensureSession`、`runTurn`、cancel、close 等接口，甚至可以通过 `acpx` 启动外部命令，见 [`acp runtime types`][openclaw-acp-runtime]。

这说明 OpenClaw 不是先写一个本地 query loop 再慢慢拆，而是先画控制面：operator 怎么控制 node，client 怎么进入 Gateway，sandbox backend 怎么接入，ACP runtime 怎么被统一调用。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-openclaw-gateway-control-plane.png" alt="OpenClaw gateway-first control plane：clients 连接 Gateway，再路由到 ACP runtime、sandbox backend 和 agent work">
  <figcaption>图 5. OpenClaw 的教学重点是“先有控制面”：agent 能力挂在 Gateway、node、sandbox backend 和 ACP runtime 的边界之后。</figcaption>
</figure>

#### 它和 Codex 的差别

Codex 更像一个可被多端驱动的 agent harness，核心是 thread/turn、core Session、tools runtime、context、rollout、trace 的稳定组合。OpenClaw 更 gateway-first，核心是 operator/node/sandbox/runtime 的分布式控制边界。

这两者都比 integrated CLI runtime 更“拆”，但拆的方向不同：Codex 主要为多 surface、恢复和审计拆；OpenClaw 主要为分布式控制、node 路由和外部 runtime 接入拆。

## 四、行业脉络：从补全到可审计执行环境

### 4.1 四个阶段

第一阶段是补全和聊天：模型给建议，人复制、运行、修。工具边界弱，输出主要是文本。

第二阶段是 ReAct：模型可以交错推理和动作，动作结果反馈给下一步。这个阶段把 tool use 放进循环，但还没有完整回答 sandbox、恢复、审计和多端状态。

第三阶段是 ACI：SWE-agent 提醒我们，agent 也是一种新型 end user，接口设计会显著影响它浏览代码、编辑文件和运行测试的表现。

第四阶段是 agent harness：OpenAI Codex、Claude Code、GitHub Copilot cloud agent 等产品都在把 agent 放进可约束、可恢复、可审计的软件执行环境里。[GitHub Copilot cloud agent][github-cloud-agent] 也强调 agent 在 GitHub Actions-powered ephemeral development environment 中研究仓库、改分支、运行测试，并让用户 review diff / PR。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-industry-axis.png" alt="Coding agent 演进轴：从补全聊天到 ReAct、ACI，再到可约束、可恢复、可审计的 agent harness">
  <figcaption>图 6. 行业重心在右移：模型能力仍重要，但执行环境、权限、日志、恢复和 trace 正在变成基础设施。</figcaption>
</figure>

### 4.2 真正新增的是工程能力

当 agent 要独立完成代码任务时，产品会被迫回答一组基础设施问题：

- 能否限制文件系统和网络？
- 能否复现每一步？
- 能否审计 tool call 和 terminal output？
- 能否恢复长会话或 fork 子任务？
- 能否把权限、记忆、上下文、日志和测试证据放进同一生命周期？
- 能否让多个 client 或多个 agent 看到一致的 thread/turn state？

Codex 源码最有价值的地方，就在于它把这些 harness 问题显式工程化了。Claude Code 对照仓库的价值，则在于它展示了一个强集成 CLI runtime 如何快速把 terminal-first 体验做成整体。OpenClaw 的价值，是把 agent 放进 gateway control plane 后，边界会如何重排。

## 五、常见误读：源码对照不等于产品排名

### 5.1 把语言差异当架构差异

TypeScript 和 Rust 不是根因。真正的差异是 owner：Claude Code 对照仓库把应用态集中在 CLI runtime 周围；Codex 把 client、core、tools、context、rollout、trace 拆成 harness；OpenClaw 把 Gateway 放到最前。

### 5.2 把 protocol 类型转换当无意义样板

Codex 的 [`app-server-protocol/v2.rs`][codex-v2-protocol] 有大量 camelCase translation 和错误类型映射。它们不是为了“多写代码”，而是让 core 内部类型和外部 client API 分离。外部协议稳定，内部 runtime 才能继续演进。

### 5.3 把 rollout JSONL 和 rollout trace 混成一件事

[`RolloutRecorder`][codex-rollout-recorder] 的 JSONL 是 session replay / resume 的 durable record；[`rollout-trace`][codex-rollout-trace-readme] 是 opt-in local diagnostic bundle，会把 raw evidence reduce 成 graph。前者偏生产恢复，后者偏诊断解释。把二者混掉，就会误解为什么 Codex 同时需要 `replacement_history`、rollout reconstruction 和 trace reducer。

### 5.4 把 context mechanism 当 prompt 模板

`reference_context_item`、`TurnContextItem`、`ContextManager`、compaction、`replacement_history` 都不是“多拼几段 prompt”。它们分别承担 baseline diff、turn context persistence、model-visible history、history replacement 和 resume/fork hydration。真正要看的不是文本长什么样，而是谁拥有它、生命周期多长、恢复时从哪里重建。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260614v3-agent-architecture-scorecard.png" alt="Agent 架构 scorecard：Integrated CLI、Agent harness、Gateway plane 在主要边界、优势和压力上的对照">
  <figcaption>图 7. 最后的对照表不是排名，而是把 owner、优势、压力和失效边界放在同一张图上。</figcaption>
</figure>

## 六、可迁移规则：判断一个 agent 架构先看 owner

### 6.1 三类架构的边界 scorecard

<div class="wc-responsive-table-wrap">
<table class="wc-responsive-table">
  <thead>
    <tr>
      <th>形态</th>
      <th>主要 owner</th>
      <th>优势</th>
      <th>压力</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Integrated CLI runtime</td>
      <td>应用内 query loop</td>
      <td>产品体验聚合快，terminal-first 接入短</td>
      <td>多端、多环境、多 agent 拆分压力大</td>
    </tr>
    <tr>
      <td>Protocolized agent harness</td>
      <td>CLI / App Server / core / tools / context / rollout</td>
      <td>多端复用、可恢复、可审计</td>
      <td>调用链长，类型转换多，源码阅读成本高</td>
    </tr>
    <tr>
      <td>Gateway-first control plane</td>
      <td>Gateway / node / sandbox / ACP runtime</td>
      <td>分布式控制边界清楚，外部 runtime 易接入</td>
      <td>本地单机体验和低延迟交互要再组合</td>
    </tr>
  </tbody>
</table>
</div>

### 6.2 设计自己的 agent 时，按状态归属下判断

<div class="wc-responsive-table-wrap">
<table class="wc-responsive-table">
  <thead>
    <tr>
      <th>看到的状态</th>
      <th>优先归属</th>
      <th>要保护的 invariant</th>
      <th>源码里应该找什么</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>UI transcript</td>
      <td>client / app runtime</td>
      <td>用户能理解和回看发生过什么</td>
      <td>message store、session view、checkpoint UI</td>
    </tr>
    <tr>
      <td>model-visible history</td>
      <td>context manager / request builder</td>
      <td>模型看到的是受控投影，不是所有本地历史</td>
      <td><code>ContextManager</code>、projection、truncation、compaction</td>
    </tr>
    <tr>
      <td>tool execution</td>
      <td>tool runtime / sandbox / approval</td>
      <td>自主动作被权限和环境约束</td>
      <td>tool registry、router、runtime policy、approval request</td>
    </tr>
    <tr>
      <td>resume baseline</td>
      <td>rollout / replacement history</td>
      <td>长会话恢复时有 canonical history</td>
      <td><code>replacement_history</code>、rollout reconstruction、turn context item</td>
    </tr>
    <tr>
      <td>debug evidence</td>
      <td>trace bundle / reducer</td>
      <td>诊断能追到 raw evidence，但不污染生产恢复路径</td>
      <td><code>trace.jsonl</code>、payload refs、semantic graph reducer</td>
    </tr>
    <tr>
      <td>distributed routing</td>
      <td>Gateway / App Server / thread protocol</td>
      <td>多个 client 或 node 看到一致生命周期</td>
      <td>initialize、thread/start、turn/start、operator/node role</td>
    </tr>
  </tbody>
</table>
</div>

### 6.3 最后的判断顺序

判断一套 coding agent 架构时，可以按这个顺序读源码：

1. 先看官方 contract：它承诺的是 local pair、cloud delegation、IDE sidecar、GitHub review，还是 gateway control plane？
2. 再找 owner：模型可见历史、工具执行、权限、恢复、trace、UI 状态分别归谁？
3. 然后走 execution chain：入口如何进 loop，loop 如何发 tool，tool result 如何回模型，哪些记录进入 durable storage？
4. 最后看 failure boundary：上下文爆掉、tool output 过大、权限拒绝、resume/fork、child agent 失败时，系统靠什么恢复或解释？

Claude Code 对照仓库、Codex 和 OpenClaw 的差异，最终可以压成一句话：当产品主要是“人在终端里和 agent 结对”，integrated runtime 很自然；当产品开始承担“后台任务、远程协作、分支提交、长会话恢复、多 agent 调度、日志审计”，harness 化和 control-plane 化就会越来越重要。

## 参考

- [Claude Code overview][claude-overview]
- [Claude Code: How Claude Code works][claude-how]
- [OpenAI Developers: Codex CLI][codex-cli-docs]
- [OpenAI Developers: Codex web][codex-cloud-docs]
- [OpenAI Developers: Codex IDE extension][codex-ide-docs]
- [OpenAI Developers: Sandbox][codex-sandbox-docs]
- [GitHub Docs: About GitHub Copilot cloud agent][github-cloud-agent]
- [ReAct: Synergizing Reasoning and Acting in Language Models][react-paper]
- [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering][swe-agent-paper]

[claude-overview]: https://code.claude.com/docs/en/overview
[claude-how]: https://code.claude.com/docs/en/how-claude-code-works
[codex-cli-docs]: https://developers.openai.com/codex/cli
[codex-cloud-docs]: https://developers.openai.com/codex/cloud
[codex-ide-docs]: https://developers.openai.com/codex/ide
[codex-sandbox-docs]: https://developers.openai.com/codex/concepts/sandboxing
[github-cloud-agent]: https://docs.github.com/en/copilot/concepts/agents/cloud-agent/about-cloud-agent
[react-paper]: https://arxiv.org/abs/2210.03629
[swe-agent-paper]: https://arxiv.org/abs/2405.15793

[claude-cli]: https://github.com/WineChord/claude-code/blob/5a774a2b62d7949c1d94e0b726281554d7893cfd/src/entrypoints/cli.tsx#L28-L160
[claude-query-engine]: https://github.com/WineChord/claude-code/blob/5a774a2b62d7949c1d94e0b726281554d7893cfd/src/QueryEngine.ts#L130-L180
[claude-query]: https://github.com/WineChord/claude-code/blob/5a774a2b62d7949c1d94e0b726281554d7893cfd/src/query.ts#L181-L220
[claude-query-loop]: https://github.com/WineChord/claude-code/blob/5a774a2b62d7949c1d94e0b726281554d7893cfd/src/query.ts#L241-L307
[claude-tool-schema]: https://github.com/WineChord/claude-code/blob/5a774a2b62d7949c1d94e0b726281554d7893cfd/src/Tool.ts#L15-L21
[claude-tool-permission]: https://github.com/WineChord/claude-code/blob/5a774a2b62d7949c1d94e0b726281554d7893cfd/src/Tool.ts#L123-L148
[claude-tool-type]: https://github.com/WineChord/claude-code/blob/5a774a2b62d7949c1d94e0b726281554d7893cfd/src/Tool.ts#L362-L390

[codex-js-platform]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-cli/bin/codex.js#L15-L85
[codex-cli]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/cli/src/main.rs#L70-L176
[codex-regular-task]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tasks/regular.rs#L40-L86
[codex-tool-spec]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/spec.rs#L71-L338
[codex-tool-router]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/router.rs#L39-L100
[codex-tool-runtime]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/parallel.rs#L27-L143
[codex-tool-registry]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/tools/registry.rs#L215-L247
[codex-v2-protocol]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/v2.rs#L123-L230
[codex-context-manager]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/context_manager/history.rs#L32-L51
[codex-reference-context-item]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/context_manager/history.rs#L40-L50
[codex-turn-context-item]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/protocol.rs#L2827-L2868
[codex-rollout-recorder]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/rollout/src/recorder.rs#L78-L96
[codex-rollout-reconstruction]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/rollout_reconstruction.rs#L86-L128
[codex-rollout-trace-readme]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/rollout-trace/README.md#L1-L20
[codex-rollout-trace-reducer]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/rollout-trace/src/reducer/mod.rs#L43-L86

[openclaw-control-ui]: https://github.com/openclaw/openclaw/blob/5d8ca42c7de8118b15782bad9cbac6240585e13a/docs/web/control-ui.md#L11-L17
[openclaw-gateway-protocol]: https://github.com/openclaw/openclaw/blob/5d8ca42c7de8118b15782bad9cbac6240585e13a/docs/gateway/protocol.md#L135-L164
[openclaw-tui-gateway]: https://github.com/openclaw/openclaw/blob/5d8ca42c7de8118b15782bad9cbac6240585e13a/src/tui/gateway-chat.ts#L153-L167
[openclaw-acp-runtime]: https://github.com/openclaw/openclaw/blob/5d8ca42c7de8118b15782bad9cbac6240585e13a/src/acp/runtime/types.ts#L118-L138
