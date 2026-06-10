---
classes: wide2
title: "Codex 源码剖析：003. 上下文、记忆与压缩"
excerpt: "Codex 不是简单拼 prompt，而是把 base instructions、AGENTS.md、skills、apps、memory、tracked settings diff、compaction 和 rollout 组织成可恢复状态。"
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

前两篇看了主循环和工具系统。这一篇看更隐蔽、但更决定上限的部分：上下文工程。

很多人说 agent 的核心是工具调用，这当然对。但如果上下文组织不好，工具越多越容易出问题：用户规则会丢，项目规范会被覆盖，长会话越跑越满，resume 之后 prompt 形状变了，memory 会把临时误解固化下来，多 Agent 之间也很难解释信息来源。

Codex 当前的做法不是把所有东西一次性拼成一个巨大 prompt，而是拆成几层：

1. `base_instructions`：作为 Responses API 的 `instructions`。
2. 动态 developer/user 上下文：作为 `ResponseItem::Message` 放进 conversation input。
3. `reference_context_item`：作为 tracked settings diff 的 baseline。
4. compaction / rollout：保证长会话可压缩、可恢复。
5. rollout trace：作为本地诊断图谱，不承担生产恢复。

![Codex 上下文状态系统封面图：base instructions、AGENTS.md、skills/apps、memory 进入 ContextManager，再投影为模型视图和 compaction/rollout 恢复路径](/assets/images/posts/2026-04-30-codex-source/context-memory-cover.png)

*图 1. Codex 的上下文不是一段 prompt 字符串，而是一组有来源、生命周期、diff、压缩和恢复语义的状态。*

## 阅读契约

这篇只抓四条线：

1. `base_instructions` 和动态上下文分别进入哪里？
2. AGENTS.md、skills、apps、plugins、memory 为什么不能都当成同一种 prompt 片段？
3. `reference_context_item` 到底覆盖哪些 tracked fields，又没有覆盖哪些字段？
4. compaction、rollout 和 rollout trace 各自承担什么恢复或诊断职责？

读完后，判断一段上下文逻辑时应该先问它处在哪个视图里：配置基线、conversation history、model-visible projection、持久 rollout，还是 opt-in trace bundle。

下图不是 prompt 模板图，而是状态来源图：哪些东西在 session 创建时确定，哪些东西在首个 turn 前注入，哪些东西随着 history、compaction 和 rollout 继续演进。

![Codex 上下文工程总览：base instructions、AGENTS.md、skills、apps、memory、ContextManager、CompactedItem 和 rollout JSONL 的关系](/assets/images/posts/2026-04-30-codex-source/context-memory.png)

*图 2. 左侧是初始上下文来源，中间是 initial context 和 `ContextManager`，右侧是 compaction 与 rollout。`reference_context_item` 是 tracked settings diff 的 baseline，不是所有动态上下文的万能快照。*

## 一、初始上下文不是一整块 system prompt

### 1.1 `base_instructions` 和动态上下文不是一回事

`build_prompt` 里有一个字段叫 `base_instructions`，它会进入模型请求的 instructions。动态上下文则是 history 里的 message item。这个区分很重要。

`base_instructions` 的来源有优先级：config override、rollout 里的 `SessionMeta.base_instructions`、当前模型内置 instructions。这个逻辑在 [`Codex::spawn` 附近][session-spawn-base-instructions]。

为什么不把所有上下文都塞进 base instructions？因为很多上下文会随 turn 变化：

- cwd
- permission profile
- approval policy
- network policy
- current date / timezone
- realtime 状态
- collaboration mode
- personality
- apps / skills / plugins 可用性
- AGENTS.md 层级

如果这些东西全塞进固定 system/developer prompt，resume、fork、model switch、compact 都会很难处理。Codex 的选择是：稳定系统基线放在 `base_instructions`，动态上下文作为可记录、部分可 diff、可恢复的 conversation item。

### 1.2 AGENTS.md 是 user fragment，不是安全边界

Codex 对 AGENTS.md 的处理在 [`agents_md.rs`][agents-md]。文件头注释说明了搜索规则：从 cwd 往上找 project root，默认 root marker 是 `.git`；从 project root 到 cwd 按顺序收集 `AGENTS.md`；不越过 project root。

默认文件名是 `AGENTS.md`，本地 override 是 `AGENTS.override.md`，见 [`DEFAULT_AGENTS_MD_FILENAME` / `LOCAL_AGENTS_MD_FILENAME`][agents-md-filenames]。全局 instructions 则会从 `$CODEX_HOME/AGENTS.override.md` 或 `$CODEX_HOME/AGENTS.md` 读入，逻辑在 [`load_global_instructions`][agents-global]。

真正拼接用户 instructions 的入口是 [`AgentsMdManager::user_instructions`][agents-user-instructions]。它会把 config 里的 `user_instructions` 和项目 AGENTS.md 文档合起来。如果同时存在，中间用 `--- project-doc ---` 分隔。

然后这些内容会被包成 user-role contextual fragment。`UserInstructions` 的 `ROLE` 是 `"user"`，marker 是 `# AGENTS.md instructions for ...`，见 [`context/user_instructions.rs`][context-user-instructions]。

![AGENTS.md 上下文边界图：从 project root 到 cwd 的 AGENTS.md 合并为 UserInstructions，进入 model context，但权限仍由 approval、sandbox 和 filesystem policy 决定](/assets/images/posts/2026-04-30-codex-source/agents-md-user-fragment.png)

*图 3. AGENTS.md 能指导模型行为，但不能替代执行权限检查。把它当成安全边界会读错后面的工具和沙箱代码。*

这个选择很关键：AGENTS.md 会强烈影响模型，但它不是不可变 system prompt。安全和权限边界仍然由 permission profile、approval、sandbox 这些代码路径决定。

### 1.3 首个真实 turn 才完整注入

Codex 并不是 session 一创建就把所有上下文塞进 history。完整动态上下文在首个真实 turn 前注入。

![首个真实 turn 注入上下文：base_instructions 进入 Prompt.instructions，developer/user context items 进入 conversation input，并记录 reference_context_item baseline](/assets/images/posts/2026-04-30-codex-source/initial-context-layers.png)

*图 4. 稳定系统基线和动态上下文走不同通道；后续 turn 尽量发 tracked diff，而不是重建全部上下文。*

`record_context_updates_and_set_reference_context_item` 会判断有没有 baseline：没有 baseline 时调用 `build_initial_context`，完整注入；有 baseline 时只构造 tracked settings update diff。这条逻辑在 [`record_context_updates_and_set_reference_context_item`][record-context-updates]。

`build_initial_context` 会构造 developer sections 和 user sections。developer 侧包括 permission instructions、config developer instructions、memory read prompt、collaboration mode / personality、apps instructions、available skills、available plugins、git commit trailer instructions 等；user 侧包括 AGENTS.md user instructions 和 environment context。这条聚合线在 [`build_initial_context`][build-initial-context]。

这样做的好处是，动态上下文有一个可记录的起点。后续 turn 如果 cwd、权限、模型、realtime 状态变化，不需要总是重新注入一大坨上下文，而是可以基于 `reference_context_item` 生成 tracked settings diff。

## 二、tracked settings diff 只覆盖一部分字段

`ContextManager` 是 thread history 的管理器。它内部保存 `items`、`history_version`、`token_info`，还有一个很关键的字段：`reference_context_item`，见 [`ContextManager`][context-manager]。

注释里说得很明确：`reference_context_item` 是 settings update diff 的 baseline。如果它是 `None`，下一次 turn 会认为没有 baseline，从而完整 reinject context。

settings diff 的构造在 [`context_manager/updates.rs`][context-updates]。这里会分别比较 environment context、permission profile / approval policy、collaboration mode、realtime、personality、model 等字段。最后 developer update 和 contextual user message 分别作为 `ResponseItem::Message` 进入 history。

但这里必须讲准确：[`build_settings_update_items`][context-updates] 只覆盖 tracked context fields，源码注释也说明它还没有覆盖 `build_initial_context` 发出的每一种 model-visible item。因此更准确的说法是：

> Codex 把一部分高频变化的上下文字段做成 baseline + diff，而不是已经实现了所有动态上下文的完整 diff 系统。

![Codex context diff 范围：initial context 很大，后续 tracked settings diff 目前只覆盖一部分字段，源码里也保留待覆盖说明](/assets/images/posts/2026-04-30-codex-source/context-diff-scope.png)

*图 5. 这是本篇最容易误读的地方。tracked settings diff 目前覆盖模型、权限、environment 等字段；它不是对所有 initial context 的完整语义 diff。*

把这段逻辑压成伪代码：

```text
if no reference_context_item:
  items = build_initial_context(turn_context)
  remember reference_context_item
else:
  items = build_settings_update_items(
    previous_reference,
    previous_turn_settings,
    next_turn_context,
  )
  append only non-empty tracked updates
```

所以 `reference_context_item` 的价值不是“永远不再重注入上下文”，而是让 Codex 能区分“这轮真的需要重建上下文”还是“只补几个已追踪字段的变化”。

## 三、动态能力先摘要，按需展开

### 3.1 Skills、Plugins、Apps 不一次性吃完上下文窗口

skills 的可用列表不是直接把每个 `SKILL.md` 全塞进 prompt。`AvailableSkillsInstructions` 是 developer fragment，它只渲染可用 skills 摘要，见 [`AvailableSkillsInstructions`][available-skills-instructions]。

真正显式触发某个 skill 时，Codex 才会读取对应 skill 文件并构造成 turn-scoped injection。这样可以避免“技能系统”自己把上下文窗口吃光。

plugins 又往上抽了一层。`AvailablePluginsInstructions` 说明 plugin 是 skills、MCP servers、apps 的本地 bundle，不是模型直接调用的东西，见 [`available_plugins_instructions.rs`][available-plugins-instructions]。所以 plugin 出现在上下文里，本质上是告诉模型有哪些组合能力可用，以及如何通过 underlying skills / MCP / apps 使用。

apps 则通过 `codex_apps` MCP server 暴露。`AppsInstructions` 只在存在 accessible 且 enabled 的 connector 时注入，见 [`AppsInstructions::from_connectors`][apps-instructions]。它告诉模型：app 可以显式用 `[$app-name](app://connector_id)` 触发，也可以由上下文隐式触发；app 等价于 `codex_apps` MCP 里的工具集合。

这里能看到 Codex 对上下文预算的克制：可用能力先摘要，具体能力按需展开，大工具空间靠 `tool_search` 懒加载，plugin 不直接变工具，而是变成 skills / MCP / apps 的组合入口。

### 3.2 Memory：主 agent 读，隔离 agent 写

memory 是最容易做坏的模块。因为一旦主 agent 在任务过程中可以随意自写长期记忆，它可能把临时误解、失败路径、用户一次性要求都固化下来。

Codex 把 memory 明确拆成 read / write 两个 crate。`memories/README.md` 里写得很清楚：read path 负责 memory developer-instruction injection、memory citation parsing、read telemetry；write path 负责 Phase 1 / Phase 2 prompt rendering、filesystem artifacts、workspace diff 和 extension pruning，见 [`memories/README.md`][memories-readme]。

read path 的效果是：在 initial-context construction 阶段，如果 `MemoryTool` feature 和 `use_memories` 都开启，并且本地 memory summary 存在，Codex 会读取 `~/.codex/memories/memory_summary.md`，截断后渲染 developer instructions，再进入 `build_initial_context`。

write path 的触发条件更保守：root session 启动、非 ephemeral、memory feature enabled、不是 sub-agent、有 state DB。它异步后台运行，先 Phase 1 做 per-thread extraction，再 Phase 2 做 global consolidation。Phase 2 会启动内部 consolidation sub-agent；这个 agent 无网络、无 approvals、只允许写 memory root，并且禁 apps/plugins/collab/memory。

![Memory 读写分离图：root session 在 initial context 中读取 memory summary，后台 thread extraction 和隔离 consolidation sub-agent 才写入 memory root](/assets/images/posts/2026-04-30-codex-source/memory-read-write-split.png)

*图 6. 主 agent 读取长期摘要；长期记忆写入交给受限后台流程，避免当前任务把临时误解直接固化成记忆。*

这条设计的意思是：主 agent 只读长期记忆，长期记忆的写入交给隔离的内部 agent 和受控 workspace diff。它比“让当前 agent 直接记住一切”稳得多。

## 四、压缩、rollout 与 trace 要分清

### 4.1 Compaction 不是删历史，而是安装 checkpoint

长会话迟早会撞上下文窗口。Codex 的自动 compaction 在几个地方触发：turn 前检查 token usage，sampling 后如果达到 auto compact limit 且还需要 follow-up，模型切换到更小 context window 时。触发逻辑可以从 [`run_pre_sampling_compact`][run-pre-compact] 和 `run_turn` 的 token usage 分支看。

本地 compaction 的入口在 [`compact.rs`][compact-rs]。这里有一个很关键的枚举：`InitialContextInjection`。pre-turn / manual compaction 用 `DoNotInject`，替换 history 后清掉 `reference_context_item`，下一次 regular turn 会完整 reinject initial context；mid-turn compaction 用 `BeforeLastUserMessage`，因为模型训练时预期 compaction summary 是 history 最后 item，所以要把 initial context 插到最后真实 user message 前。

真正安装压缩结果时，Codex 会构造 `CompactedItem { message, replacement_history }`，然后调用 `replace_compacted_history`。这样 rollout 里不仅有摘要文本，还有 replacement history checkpoint。

![Compaction 与 rollout 恢复图：old history 经过 compact 生成 replacement_history，写入 rollout JSONL，resume 时反向扫描最新 checkpoint](/assets/images/posts/2026-04-30-codex-source/compaction-rollout-recovery.png)

*图 7. Compaction 不是简单删历史，而是把可恢复的替换历史写进 rollout。checkpoint 越新，resume/fork 越不需要从头 replay。*

这点非常重要。很多系统 compaction 后只留一段 summary，resume 时只能依赖“重播到某个点再重新总结”。Codex 把 replacement history 作为 rollout item 持久化，后面 resume/fork 可以直接从最新 surviving checkpoint 继续。

### 4.2 Rollout 是生产恢复路径

Codex 的生产恢复不是靠 rollout trace，而是靠正常 rollout JSONL。rollout item 包含 `SessionMeta`、`ResponseItem`、`Compacted`、`TurnContext`、`EventMsg` 等类型，见 [`RolloutItem`][rollout-item]。

resume/fork 的重建逻辑在 [`rollout_reconstruction.rs`][rollout-reconstruction]。它不是从头线性 replay 全部历史，而是从后往前扫描 rollout，找最新 surviving `replacement_history` checkpoint，处理 rollback，恢复 previous turn settings 和 `reference_context_item`，再只把 checkpoint 后面的尾部正向 replay。

这比全量 replay 更适合长会话。compaction 越多，checkpoint 越重要。

### 4.3 Rollout Trace 是诊断图谱

`rollout-trace` 是另一条旁路。它的 README 第一段就强调：这是 opt-in diagnostic path，不是 telemetry；只有设置 `CODEX_ROLLOUT_TRACE_ROOT` 才写本地 bundle，且可能包含 prompts、responses、tool inputs/outputs、terminal output 和路径，见 [`rollout-trace/README.md`][rollout-trace-readme]。

它的设计原则是 “observe first, interpret later”。运行时热路径只写 raw events 和 payload references；离线 reducer 再把 bundle 还原成 semantic graph。这个图谱对调试多 Agent、code mode、嵌套工具调用很有价值。但它不是恢复 session 的生产数据源。生产恢复依赖 rollout JSONL 和 `CompactedItem.replacement_history`。

这层区分很工程化：恢复路径要稳定、低风险；诊断路径可以更丰富，但必须 opt-in、本地、best-effort。

## 五、源码阅读规则

| 读到的模块 | 先问什么 | 正确边界 |
| --- | --- | --- |
| `base_instructions` | 它是不是会随 turn 变化？ | 稳定基线进 instructions，动态上下文进 history item。 |
| AGENTS.md | 它是硬安全边界吗？ | 不是。它是 user fragment，权限仍由 policy/sandbox 决定。 |
| `reference_context_item` | 它是完整上下文快照吗？ | 不是。它是 tracked settings diff 的 baseline。 |
| skills/plugins/apps | 是否一次性展开全部内容？ | 不。先摘要，显式触发或搜索后再展开。 |
| memory | 主 agent 能不能随便写？ | 不。read/write 分离，写入走隔离后台流程。 |
| compaction | 是删历史吗？ | 不是。它安装 `replacement_history` checkpoint。 |
| rollout trace | 能用于生产恢复吗？ | 不。生产恢复靠 rollout JSONL。 |

## 小结

Codex 上下文工程不是“把规则拼进 prompt”这么简单。它更像一个状态系统：`base_instructions` 保持系统基线，AGENTS.md、environment、permission、skills、plugins、apps、memory 作为动态 context fragments 注入，`reference_context_item` 记录 baseline，后续 turn 对 tracked settings 发 diff，`ContextManager` 维护 history 和 token usage，compaction 安装 `replacement_history`，rollout JSONL 负责生产恢复，rollout-trace 负责本地诊断图谱。

如果把 agent 当成“一个模型 + 一组工具”，这些东西看起来像附属模块。但从 Codex 源码看，真正的复杂度恰恰在这里：让一个 agent 在多轮、多工具、多端、多权限、多 Agent、多次恢复之后，仍然能保持相对稳定的上下文形状。

## 参考

- [OpenAI: Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/)
- [Claude Code: How Claude remembers your project](https://code.claude.com/docs/en/memory)

[session-spawn-base-instructions]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/mod.rs#L536
[agents-md]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/agents_md.rs#L1-L17
[agents-md-filenames]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/agents_md.rs#L36-L43
[agents-global]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/agents_md.rs#L61-L78
[agents-user-instructions]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/agents_md.rs#L82-L127
[context-user-instructions]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/context/user_instructions.rs#L1-L16
[record-context-updates]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/mod.rs#L2737
[build-initial-context]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/mod.rs#L2509
[context-manager]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/context_manager/history.rs#L32-L51
[context-updates]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/context_manager/updates.rs#L204-L238
[available-skills-instructions]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/context/available_skills_instructions.rs#L23-L30
[available-plugins-instructions]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/context/available_plugins_instructions.rs#L24-L57
[apps-instructions]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/context/apps_instructions.rs#L11-L30
[memories-readme]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/memories/README.md#L1-L157
[run-pre-compact]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L701-L730
[compact-rs]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/compact.rs#L46-L59
[rollout-item]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/protocol.rs#L2791
[rollout-reconstruction]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/rollout_reconstruction.rs#L89-L190
[rollout-trace-readme]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/rollout-trace/README.md#L1-L20
