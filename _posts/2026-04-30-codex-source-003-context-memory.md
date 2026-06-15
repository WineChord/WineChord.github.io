---
classes:
  - wide2
  - codex-source-003-context-memory
title: "Codex 源码剖析：003. 上下文、记忆与压缩"
excerpt: "从 OpenAI Responses API 和 openai/codex 源码解释 Codex 如何把 base instructions、AGENTS.md、skills、apps、memory、tracked settings diff、compaction、rollout 和 trace 组织成可恢复的模型视图。"
last_modified_at: 2026-06-15
locale: zh-CN
toc: true
toc_sticky: true
toc_label: "目录"
toc_levels: 2..4
mathjax: true
canonical_url: "https://www.wineandchord.com/llm/agent/codex-source-003-context-memory/"
header:
  og_image: https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-context-memory-cover.png
  teaser: https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-context-memory-cover.png
  teaser_alt: "Codex 上下文系统总览：base instructions、AGENTS.md、skills、apps、memory、model view、compaction 和 trace bundle"
categories:
  - LLM
  - Agent
tags:
  - Codex
  - Coding Agent
  - Source
---

一个 coding agent 的上下文不是“把聊天记录拼成长 prompt”。如果这样做，长会话里最先坏掉的通常不是模型不会写代码，而是状态边界变得不可恢复：用户规则、项目文档、运行环境、工具能力、权限策略、长期记忆、压缩摘要、工具输出和恢复日志被揉成一段文本以后，runtime 就很难回答三个问题：模型这一轮到底看见了什么，哪些状态可以在下一轮只发 diff，崩溃或 resume 以后应该从哪里接上。

Codex 的源码把这个问题拆成多层状态系统。稳定的模型基线走 Responses API 的 `instructions`；会话材料走 `input` 里的 `ResponseItem::Message`；动态环境和规则先形成 `TurnContextItem` baseline，再在后续 turn 里按 tracked settings diff 追加；旧历史被 compaction 替换时，`CompactedItem.replacement_history` 记录新的可恢复 history；生产恢复走 rollout JSONL 的 segment replay；诊断证据则走 opt-in rollout trace bundle。

本文基于 `openai/codex` 固定源码快照 [`ac4332c0`][codex-snapshot]，复核日期为 2026-06-14。平台契约以 [OpenAI Responses API][responses-overview]、[`responses.create`][responses-create]、[Responses migration guide][responses-migrate]、[OpenAI prompt caching][prompt-caching] 和 [Codex app-server docs][codex-app-server] 为准；源码机制以 pinned GitHub blob 链接为准；服务端如何把 Responses 请求内部序列化成最终 token 序列不在公开源码里，文中只把它当作 provider contract 边界。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-context-memory-cover.png" alt="Codex 上下文系统总览：base instructions、AGENTS.md、skills、apps、memory summary 汇入 ContextManager，再输出 model view、compaction 和 trace bundle">
  <figcaption>图 1. Codex 的 prompt 是一套可追踪、可压缩、可恢复的状态系统；左侧来源、中间 ContextManager、右侧 model view、compaction 与 trace bundle 分别有不同 owner。</figcaption>
</figure>

## 阅读契约

这篇文章只回答一个问题：Codex 如何让模型每一轮看到足够上下文，同时又不把规则、能力、记忆和历史无条件塞进同一段 prompt？读完以后，应该能分清五个视图：provider contract、session baseline、model-visible context、history replacement 和 diagnostic trace。

证据边界也很重要。`ResponseItem::Message`、`TurnContextItem`、`CompactedItem.replacement_history`、`RolloutItem` 这些是源码里能验证的数据结构；prompt caching 的 exact prefix 和静态内容靠前是 OpenAI 官方契约；`instructions`、`tools`、`input` 到模型内部 prompt 的最终排列是 provider 侧行为，客户端只能按公开 API 字段和 usage 反馈设计稳定前缀。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-context-memory-system-overview.png" alt="Codex 上下文工程总览：SessionMeta、TurnContext、AGENTS.md、skills、apps 和 memory 进入 build_initial_context，再产生 developer items、user context items、tracked settings diff、CompactedItem 和 rollout">
  <figcaption>图 2. 上下文工程的核心不是拼得更长，而是把来源、role、可变性、baseline 和恢复路径分开。</figcaption>
</figure>

## 一、Responses API 先把稳定基线和会话材料分开

### 1.1 压力来源：聊天记录、运行环境和工具能力增长速度不同

长会话里有三类东西会同时增长。第一类是用户和 assistant 的真实对话；第二类是工具输出、文件片段、终端日志和图片；第三类是本轮运行环境，例如 cwd、approval policy、sandbox、model、skills、apps、plugins、memory summary、可用动态工具。把它们混在一段 system prompt 里最简单，但失败边界也最明显：任何小变化都会污染稳定前缀，compaction 后也不知道哪些状态应该重新注入。

OpenAI 的 [Responses API][responses-create] 先给 Codex 一个清晰边界：稳定指令走 `instructions`，conversation input 走 `input`，工具定义走 `tools`。Codex 源码里的 [`ResponseInputItem::Message`][response-input-item] 和 [`ResponseItem::Message`][response-item-message] 都是 shape-level 的“消息项”，每一项带 `role`、`content`，可选带 assistant message 的 `phase`。这不是聊天 UI 的唯一形态，而是 runtime 和 provider API 之间的 model-visible item 形态。

下面是形状级示例，字段被简化过，但保留了关键边界：

```json
{
  "instructions": "stable base instructions for this model/session",
  "tools": [
    { "type": "function", "name": "shell", "parameters": { "type": "object" } }
  ],
  "input": [
    {
      "type": "message",
      "role": "developer",
      "content": [
        { "type": "input_text", "text": "<permissions instructions>...</permissions instructions>" }
      ]
    },
    {
      "type": "message",
      "role": "user",
      "content": [
        { "type": "input_text", "text": "# AGENTS.md instructions for ..." }
      ]
    },
    {
      "type": "message",
      "role": "user",
      "content": [
        { "type": "input_text", "text": "<environment_context>...</environment_context>" }
      ]
    },
    {
      "type": "message",
      "role": "user",
      "content": [
        { "type": "input_text", "text": "用户本轮请求" }
      ]
    }
  ]
}
```

这里的不变量是：session-level 基线和 turn-level 材料不能互相冒充。`instructions` 适合承载稳定基线；`input` 适合承载会话和上下文 items；工具 schema 是 provider-visible 的能力表。失败边界是：如果把 cwd、权限、memory summary 或 AGENTS.md 热更新直接覆盖到早期稳定内容，prompt cache 会更容易失效，resume 也更难知道变化是“新事实”还是“旧基线被重写”。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-context-memory-responses-contract.png" alt="Responses API 与 Codex 请求形状对照：instructions、tools 和 input items 分别映射到 Codex 的 base_instructions、tool specs 和 ResponseItem::Message">
  <figcaption>图 3. Provider contract 先规定几个公开表面；Codex 的上下文管理是在这些表面之内维护稳定性和可恢复性。</figcaption>
</figure>

### 1.2 机制：Codex 把模型可见历史当成投影，而不是 UI transcript

`ResponseItem::Message` 里的 `role` 可以是 developer、user、assistant；`content` 可以是 `input_text`、`input_image`、`output_text` 等 [`ContentItem`][response-content-item]。这给 Codex 一个最小协议形态：它可以把权限说明做成 developer message，把 AGENTS.md 和 environment context 做成 user message，把 assistant 输出、tool call output 和后续压缩结果保留在同一条 history 序列里。

但 model-visible history 不是 UI transcript，也不是 rollout JSONL 的全部内容。UI 要保留可读 scrollback；rollout 要保留 `SessionMeta`、`TurnContext`、`EventMsg` 等恢复材料；模型只需要本轮推理所需的投影。[`ContextManager`][context-manager] 的注释把它叫 thread history，但里面还保存 `history_version`、`token_info` 和 `reference_context_item`，后者就是后续 diff 的 baseline。

这个机制保护的不是“token 更少”这一件事，而是 owner 清晰：provider 负责 `instructions` / `tools` / `input` 的 API contract；Codex 负责本轮 `input` items 如何由持久状态和 runtime state 投影出来；rollout 负责 resume 能重建同样的语义路径。失败条件也来自这里：如果一个字段只存在于 UI 而没有进入 `ResponseItem`，模型看不到；如果它只存在于 `ResponseItem` 而没有可靠 rollout baseline，resume 后未必能重建；如果它被塞进 early prefix 并频繁变化，prompt cache 会被破坏。

## 二、Session baseline 决定恢复时不能随当前默认值漂移

### 2.1 压力来源：模型默认指令会升级，但旧会话要可复现

`base_instructions` 的压力来自版本漂移。模型默认 instructions 会随发行版改变，用户也可能在 config 里显式 override。恢复旧会话时，如果 runtime 总是取“当前默认 instructions”，旧会话的行为基线就会悄悄漂移。

Codex 在 [`Codex::spawn`][session-spawn-base-instructions] 里把优先级写死：先用 config override，再用 conversation history 里的 `SessionMeta.base_instructions`，最后才回退到当前 model info 渲染出的默认 instructions。对应的持久结构是 [`SessionMeta`][session-meta]：它保存 session id、cwd、source、model provider、`base_instructions`、`dynamic_tools` 和 memory mode 等 session-level 事实。

形状上可以这样看：

```json
{
  "type": "session_meta",
  "payload": {
    "cwd": "/repo",
    "model_provider": "openai",
    "base_instructions": { "text": "model/session baseline" },
    "dynamic_tools": [
      { "name": "custom_search", "description": "...", "input_schema": { "type": "object" } }
    ],
    "memory_mode": "disabled"
  }
}
```

不变量是：session 开始时决定的模型基线属于 session identity。失败边界是：config override 可以刻意改变恢复基线，但无显式 override 时，旧 rollout 记录的 baseline 应优先于当前默认值；否则 resume 和 fork 会变成“同一聊天历史配新系统基线”，调试时很难解释行为变化。

### 2.2 `RolloutItem` 把消息、上下文、压缩和事件放进同一条恢复日志

Codex 的生产恢复材料不是一条纯聊天 transcript。[`RolloutItem`][rollout-item] 是 tagged enum，里面有 `SessionMeta`、`ResponseItem`、`Compacted`、`TurnContext` 和 `EventMsg`。这解释了为什么 `SessionMeta` 和 `TurnContextItem` 分工不同：前者解释 session 起点，后者解释每个 real user turn 后最新的 durable baseline。

[`TurnContextItem`][turn-context-item] 的注释很直接：它会在每个真实 user turn 计算 model-visible context update 后持久化；mid-turn compaction 重新建立 full context 时也会持久化。它保存 cwd、日期、时区、approval policy、sandbox policy、model、realtime、personality 等 turn-level baseline。这里的关键不是它“也在 rollout 里”，而是它让 resume 能分清：哪些变化已经作为 diff 被模型看过，哪些变化只是当前进程配置。

最小序列如下：

```text
SessionMeta(base_instructions = B0)
TurnStarted(t1)
UserMessage(U1)
ResponseItem::Message(role=developer, content=[permissions, memory, skills])
ResponseItem::Message(role=user, content=[AGENTS.md, environment])
TurnContextItem(t1, cwd=/repo, approval=on-request, model=gpt-5)
TurnComplete(t1)
```

如果后续 turn 没有任何 model-visible diff，Codex 仍然会写入新的 `TurnContextItem`。这是一个很容易被忽略的不变量：没有新 message 不等于没有新 baseline。resume 需要最新 baseline 来判断下一轮是否要 full injection 或只发 diff。

## 三、首个真实 turn 建立完整动态上下文，后续 turn 尽量只发 diff

### 3.1 机制：`build_initial_context` 先收集，再分 role 输出

[`build_initial_context`][build-initial-context] 的输入是 `TurnContext`，输出是若干 `ResponseItem::Message`。它先收集 developer sections：permissions、developer instructions、memory read prompt、collaboration mode、realtime、personality、apps、skills、plugins、commit trailer 等；再收集 contextual user sections：AGENTS.md user instructions 和 environment context；最后用 [`build_developer_update_item` / `build_contextual_user_message`][build-text-message] 分别构造成 role 为 developer 或 user 的 `ResponseItem::Message`。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-context-memory-initial-injection.png" alt="Codex 首个真实 turn 注入：developer bundle、contextual user bundle 和 reference baseline 写入 history items 与 TurnContextItem">
  <figcaption>图 4. 首个真实 turn 同时建立模型可见上下文和后续 diff 的 reference baseline。</figcaption>
</figure>

形状级 before/after 可以压成：

```text
before first real turn:
  reference_context_item = None
  history = []

after build_initial_context + record_context_updates:
  history += Message(role=developer, content=[permissions, memory, skills, apps, plugins])
  history += Message(role=user, content=[AGENTS.md, environment_context])
  rollout += TurnContextItem(current turn baseline)
  reference_context_item = current TurnContextItem
```

这里的压力来源是首轮材料太杂：权限是 developer 级约束，AGENTS.md 是用户/项目指导，environment context 是运行事实，memory summary 是长期偏好摘要，skills/apps/plugins 是能力发现入口。机制是按 role 和来源分包；不变量是每个 fragment 的 owner 可追踪；失败边界是把它们全塞进一段“system prompt”会让安全语义、项目指导和 runtime fact 的优先级变得不可审计。

### 3.2 `record_context_updates_and_set_reference_context_item` 把 diff baseline 写入持久层

后续 turn 的核心分支在 [`record_context_updates_and_set_reference_context_item`][record-context-updates]：

```text
if reference_context_item is None:
  emit build_initial_context(turn_context)
else:
  emit build_settings_update_items(reference_context_item, turn_context)

always:
  persist RolloutItem::TurnContext(turn_context_item)
  set in-memory reference_context_item = turn_context_item
```

[`build_settings_update_items`][context-updates] 当前覆盖 environment、permissions、collaboration mode、realtime、personality 和 model instruction update。源码注释也明确说它还没有覆盖 `build_initial_context` 发出的每一种 model-visible item。这个 gap 不能读成“上下文一定丢失”，它读起来更像一个有意保守的不变量：已 tracked 的字段用 diff，未完整 diff 的字段需要 full injection、replay 或额外持久化事件来保证确定性。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-context-memory-diff-scope.png" alt="tracked settings diff 范围：environment、permissions、collaboration mode、realtime、personality、model 可 diff，apps details、skills body 和部分 initial items 仍需 full injection 或 replay">
  <figcaption>图 5. tracked diff 的价值是降低重复注入；它不是整个上下文系统的身份定义。</figcaption>
</figure>

失败边界也在这个函数产生处：如果 `reference_context_item` 被 compaction、rollback 或历史裁剪清掉，下一轮必须 full reinject；如果字段尚未被 tracked diff 覆盖，不能假设它会在每个 steady-state turn 自动同步；如果一轮没有产生 model-visible diff，仍要把 `TurnContextItem` 写进 rollout，否则 resume 的 baseline 会停在旧 turn。

## 四、AGENTS.md、skills、apps 和动态工具是上下文，不是执行边界

### 4.1 AGENTS.md 的不变量是来源顺序，不是 sandbox 权限

[`agents_md.rs`][agents-md] 说明 Codex 从 cwd 向上找到 project root，再从 project root 到 cwd 依序收集 `AGENTS.md`，默认文件名和 override 文件名由 [`AGENTS.md` / `AGENTS.override.md`][agents-md-filenames] 定义，全局 instructions 由 [`load_global_instructions`][agents-global] 读取。[`AgentsMdManager::user_instructions`][agents-user-instructions] 会把 config instructions 和项目文档用 `--- project-doc ---` 分隔，保留来源感。

压力来源是项目规则需要被模型理解，但不能被误读成安全层。机制是把 AGENTS.md 渲染为 contextual user message，进入 `build_initial_context` 的 user sections。保护的不变量是“规则来源可见、顺序可见、role 可见”。失败边界是：AGENTS.md 可以指导模型行为，却不能阻止工具绕过文件系统；相反，sandbox policy 可以限制 action，却不能告诉模型项目编码风格。把两者混成一个“安全 prompt”会同时误读上下文和权限。

### 4.2 Skills、Plugins、Apps 先暴露 capability metadata，按需再展开

skills、plugins、apps 的压力来源是能力列表可能很长，完整说明和工具 schema 可能更长。Codex 不应该把所有 `SKILL.md`、connector 细节或插件实现一次性塞进上下文窗口。它先渲染摘要：skills 由 [`AvailableSkillsInstructions`][available-skills-instructions]，plugins 由 [`AvailablePluginsInstructions`][available-plugins-instructions]，apps 由 [`AppsInstructions`][apps-instructions] 进入 developer sections；真正需要时再通过 skill 文件、tool search、MCP 或 app connector 展开。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-context-memory-capabilities-lazy.png" alt="skills、plugins、apps 能力摘要：metadata 先进入 instructions，需要时再加载 skill body、tool_search 或 MCP 能力">
  <figcaption>图 6. 能力摘要保护的是上下文预算和能力发现路径；它不是工具执行结果，也不是项目规则。</figcaption>
</figure>

动态工具是同一问题的 app-server 版本。[Codex app-server 文档][codex-app-server] 只说明 `dynamicTools` 是 `thread/start` 的实验字段。源码里，[`DynamicToolSpec`][dynamic-tool-spec] 包含 namespace、name、description、input_schema 和 defer_loading；[`ThreadStartParams.dynamic_tools`][thread-start-dynamic-tools] 接收这些工具；app-server 在 [`thread_start`][app-server-thread-start] 解包 params，并在 [`thread_start_task`][app-server-thread-start-task] 校验、映射为 core dynamic tools 后交给 [`StartThreadOptions.dynamic_tools`][thread-manager-start-options]。恢复行为则来自源码路径：[`Codex::spawn`][session-spawn-dynamic-tools] 在 dynamic tools 为空的 resumed/forked thread 上，先从 [`state DB`][state-db-dynamic-tools] 读，再回退到 rollout-file 里的 tools；恢复出来的工具会进入 [`TurnContext.dynamic_tools`][turn-context-dynamic-tools]，rollout recorder 也会把它们写回 [`SessionMeta.dynamic_tools`][rollout-recorder-dynamic-tools]。

最小恢复序列如下：

```text
thread/start(dynamicTools = [D1])
  -> app-server validates and maps D1
  -> StartThreadOptions.dynamic_tools = [D1]
  -> SessionMeta.dynamic_tools = [D1]
  -> state DB persists D1 for thread id

thread/resume(no dynamicTools supplied)
  -> InitialHistory::Resumed(thread id)
  -> Codex::spawn sees dynamic_tools empty
  -> read state DB tools, fallback to rollout SessionMeta.dynamic_tools
  -> TurnContext.dynamic_tools = restored [D1]
```

这个机制保护的不是“当前进程刚好还有工具列表”，而是 thread identity。失败边界是：如果 resume 只看当前进程暴露的工具，旧 thread 的 tool schema 会漂移；如果只看 rollout 而 state DB 已经 backfill 了更新索引，app-server 的列表/读取路径可能和 core 恢复路径不一致；如果动态工具在长会话中无序变化，OpenAI prompt caching 文档说 tools 必须保持相同才利于 exact prefix 命中，runtime 就需要在能力变化和缓存稳定之间取舍。

## 五、memory 分成读路径和写路径，失败条件发生在不同地方

### 5.1 read path：`memory_summary.md` 只有存在且非空才进 developer prompt

memory 的压力来源是长期偏好有价值，但全量长期记录既贵又危险。Codex 的 read path 不是“打开 memory 就把所有记忆塞进 prompt”。[`build_memory_tool_developer_instructions`][memory-read-prompt] 读取 `memory_summary.md`，按 token limit 截断，trim 后为空则返回 `None`；只有存在内容时，它才渲染成 developer instructions，并在 `build_initial_context` 的 memory section 被加入 developer bundle。

不变量是：模型看见的是 summary，不是 memory 原始数据库。失败条件也在这里产生：文件不存在、读取失败、trim 后为空、或截断后信息不足，都会让本轮没有有效 memory read context；过期 summary 会稳定地被注入，从缓存角度看也可能形成稳定但错误的长期偏好。

### 5.2 write path：异步 pipeline 先过滤，再用受限 consolidation agent 整理

write path 的压力来源完全不同：它要从历史里找长期偏好，但不能把 developer 指令、secret、临时环境和内部工具噪声写进长期记忆。[`start_memories_startup_task`][memory-start-task] 会跳过 ephemeral session、禁用 MemoryTool 的配置和 non-root agent session；如果 state DB 不可用也会跳过；rate limit 不允许时也不继续 phase 1/phase 2。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-context-memory-read-write.png" alt="memory read/write 双通道：read side 从 summary 进入 prompt，startup write side 执行 phase 1 filter 和 phase 2 consolidation">
  <figcaption>图 7. read path 影响当前 prompt；write path 是异步整理长期材料，失败或跳过不会直接改变当前模型视图。</figcaption>
</figure>

phase 1 的 [`sanitize_response_item_for_memories`][memory-phase1-filter] 会过滤 rollout response items、丢弃 developer messages、排除部分 contextual user fragment，并做 secret redaction。phase 2 的 [`consolidation agent`][memory-phase2-agent] 被限制在 memory root，关闭 apps、plugins、memory、collab 等能力，并使用无网络 workspace sandbox。这里的不变量是长期记忆只能来自被过滤后的会话证据，并由能力受限的 worker 写回；失败边界是异步路径天然滞后，下一轮 prompt 不会因为刚刚发生了一个偏好表达就立刻拥有稳定 summary。

## 六、tracked diff 降低重复注入，但 compaction 才会重写历史

### 6.1 tracked diff 的压力来源是稳定前缀和动态事实互相拉扯

OpenAI [prompt caching 文档][prompt-caching] 要求 exact prefix match，并建议静态内容放前、动态内容放后。Codex 的 tracked diff 是对这个契约的 runtime 响应：当 cwd、permissions、realtime、model 或 personality 变化时，不要回头改初始上下文，而是追加 developer/user update item。这样旧 prefix 更容易保持稳定，模型也能看到“状态变化”本身。

但 diff 的成本是 coverage 边界。`ContextManager.reference_context_item` 的注释说得很清楚：当它是 `None` 时，下一轮会把 context state 当成没有 baseline，从而 full reinjection；rollback 也可能在裁掉混合 initial-context developer bundle 时清掉它。换句话说，diff 不是永久省 token 的承诺，而是“只要 baseline 可用且字段被 tracked，就追加变化”。

一个最小序列：

```text
T1: no baseline
  emit full initial context
  rollout += TurnContextItem(cwd=/repo, approval=on-request, model=gpt-5)

T2: same settings
  emit no settings message
  rollout += TurnContextItem(cwd=/repo, approval=on-request, model=gpt-5)

T3: cwd changed to /repo/pkg
  emit Message(role=user, content=[environment update])
  rollout += TurnContextItem(cwd=/repo/pkg, approval=on-request, model=gpt-5)
```

失败条件嵌在机制里：baseline missing 会 full reinject；untracked initial items 不能保证 diff；model switch 要先发 model-specific developer update；如果把“没有 diff item”误读成“没有持久化 turn context”，resume baseline 就会错。

### 6.2 compaction 的不变量是 history replacement，不只是 summary 文本

compaction 的压力来源是上下文窗口，而不是缓存命中。历史太长时，Codex 要把旧 history 替换成更短的可继续对话形状。关键结构是 [`CompactedItem`][compacted-item]：它有 `message` 和可选 `replacement_history`。[`InitialContextInjection`][compact-injection] 定义了两个策略：[`pre-turn/manual compaction`][run-pre-compact] 用 `DoNotInject`，会清掉 `reference_context_item`，下一轮 regular turn 再 full reinject；[`model downshift`][model-downshift-compact] 和 [`post-sampling compact`][post-sampling-compact] 这类 turn 内路径用 `BeforeLastUserMessage`，会把 initial context 插回最后一个真实 user message 前，并把当前 `TurnContextItem` 作为 baseline。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-context-memory-compaction-checkpoint.png" alt="compaction checkpoint：old history 与 recent users 被压缩为 CompactedItem.message，并携带 replacement_history 作为 resume 的恢复基线">
  <figcaption>图 8. compaction 的产物不是普通摘要，而是带 replacement history 语义的 checkpoint。</figcaption>
</figure>

形状级示例：

```json
{
  "type": "compacted",
  "payload": {
    "message": "Summary of the conversation so far...",
    "replacement_history": [
      {
        "type": "message",
        "role": "user",
        "content": [{ "type": "input_text", "text": "recent user request" }]
      },
      {
        "type": "message",
        "role": "assistant",
        "content": [{ "type": "output_text", "text": "Summary of the conversation so far..." }]
      }
    ]
  }
}
```

pre-turn 和 mid-turn 的差异可以这样读：

```text
pre-turn compact:
  new_history = compacted summary + selected recent user messages
  reference_context_item = None
  next regular turn must full-reinject initial context

mid-turn compact:
  new_history = compacted summary + initial context + last real user message
  replacement_history = new_history
  reference_context_item = current TurnContextItem
```

#### replacement_history 才是 checkpoint

[`compact.rs`][compact-install] 里会构造 `new_history`，必要时插入 initial context，然后把 `replacement_history: Some(new_history.clone())` 写进 `CompactedItem` 并调用 `replace_compacted_history`。失败边界由此产生：没有 `replacement_history` 的 legacy rollout 无法直接知道替换后的完整基线；summary 只覆盖旧历史语义，不覆盖权限、cwd、model、skills/apps、memory 等上下文 owner；mid-turn 如果不在 last real user message 前补回 initial context，当前 follow-up prompt 会缺少 session snapshot。

## 七、rollout resume 先逆向找 checkpoint，再正向回放 surviving tail

### 7.1 生产恢复路径不是 trace，而是 rollout JSONL

rollout 的压力来源是进程重启、thread resume、fork、rollback 和 compaction 都可能打断正常内存状态。Codex 不能只读最后一条消息，也不能把 trace graph 当作权威状态；它要从 `RolloutItem` 序列里重建 model history、previous turn settings 和 reference context baseline。

[`reconstruct_history_from_rollout`][rollout-reconstruction] 的注释给出核心算法：先 newest-to-oldest 逆向扫描，直到找到 surviving replacement-history checkpoint 和必要 resume metadata；然后只把 checkpoint 之后的 surviving tail 正向 replay，恢复 exact history semantics。

最小序列：

```text
oldest
  SessionMeta(B0)
  TurnStarted(t1)
  UserMessage(U1)
  ResponseItem(A1)
  TurnContextItem(t1)
  TurnComplete(t1)
  TurnStarted(t2)
  UserMessage(U2)
  CompactedItem(replacement_history = Hc)
  TurnContextItem(t2)
  TurnComplete(t2)
  TurnStarted(t3)
  UserMessage(U3)
  ResponseItem(A3)
newest

resume:
  reverse scan sees t3 segment, then t2 compaction checkpoint
  base_replacement_history = Hc
  reference_context_item = newest surviving TurnContextItem
  forward replay suffix after checkpoint: U3, A3
```

这解释了 `TurnStarted`、`TurnComplete`、`TurnAborted`、`UserMessage`、`TurnContext` 为什么要被归并成 segment：逆向扫描经常先看到 `TurnComplete`，再看到这轮的 `TurnContext` 或 `UserMessage`，直到遇到最老边界 `TurnStarted` 才能 finalize segment。失败边界也嵌在这里：rollback 要跳过 newest N 个 user-turn segments；legacy compaction 没有 `replacement_history` 时只能清掉 `reference_context_item` 并接受临时 out-of-distribution prompt shape；如果把 `EventMsg` 当成 model-visible history，resume 会把诊断/生命周期事件错误地喂给模型。

### 7.2 rollout trace 是诊断图谱，不是 resume 权威状态

rollout trace 的压力来源是生产 transcript 不足以解释“某个工具调用、code-mode runtime 值、terminal operation 或 multi-agent interaction 是怎么来的”。[`rollout-trace/README.md`][rollout-trace-readme] 明确说它不是 telemetry，只在 `CODEX_ROLLOUT_TRACE_ROOT` 设置时写本地 bundle，bundle 可能包含 prompts、responses、tool inputs/outputs、terminal output 和路径。它的设计选择是 observe first, interpret later：热路径写 raw events 和 payload refs，离线 reducer 再生成 semantic graph。

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-context-memory-trace-diagnostic.png" alt="rollout trace 诊断路径：CODEX_ROLLOUT_TRACE_ROOT 启用本地 bundle，raw events 和 payloads 经 replay_bundle 还原 semantic graph，但不作为 resume state">
  <figcaption>图 9. Trace 用来解释运行时证据；生产 resume 仍然走 rollout reconstruction。</figcaption>
</figure>

最小形状：

```text
CODEX_ROLLOUT_TRACE_ROOT unset:
  ThreadTraceContext = disabled
  no trace bundle, no resume effect

CODEX_ROLLOUT_TRACE_ROOT set:
  manifest.json
  trace.jsonl
  payloads/*.json
  codex debug trace-reduce <trace-bundle> -> state.json
```

[`ThreadTraceContext::start_root_or_disabled`][trace-thread-start] 在没有环境变量时返回 disabled；trace startup 失败也不能让 Codex session 不可用。[`replay_bundle`][trace-replay-bundle] 读取 manifest 和 trace event log，输出 reduced `RolloutTrace` graph。这里保护的不变量是诊断路径和恢复路径分离：trace 可以含敏感 raw evidence，可以解释 runtime object，但不能作为恢复 model history 的权威来源。失败边界是把 trace 当 resume 状态会引入隐私、完整性和可用性问题：trace 可能没开，可能写入失败，也可能只为调试保留。

## 八、把机制压成可迁移规则

<div class="wc-responsive-table-wrap">
<table class="wc-responsive-table">
  <thead>
    <tr><th>压力来源</th><th>Codex 机制</th><th>保护的不变量</th><th>失败边界</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>稳定指令会版本漂移</td>
      <td><code>base_instructions</code> 存入 <code>SessionMeta</code>，恢复时优先旧 session baseline</td>
      <td>同一 thread 的模型基线可解释</td>
      <td>显式 config override 可以改变基线；无 override 时不应静默使用新默认值</td>
    </tr>
    <tr>
      <td>运行环境和权限会变化</td>
      <td><code>TurnContextItem</code> + tracked settings diff</td>
      <td>变化被追加为 model-visible update，而不是回写旧 prefix</td>
      <td>diff coverage 不完整时要 full injection 或 replay</td>
    </tr>
    <tr>
      <td>项目规则要指导模型</td>
      <td>AGENTS.md 渲染为 contextual user message</td>
      <td>规则来源和顺序可见</td>
      <td>它不是 sandbox，不能替代 approval 和 filesystem policy</td>
    </tr>
    <tr>
      <td>能力列表过长</td>
      <td>skills/plugins/apps 摘要先入 developer context，细节按需加载</td>
      <td>能力发现和执行证据分离</td>
      <td>摘要不是工具结果；动态工具 resume 还要恢复 thread-start schema</td>
    </tr>
    <tr>
      <td>长期偏好有价值但不能全量注入</td>
      <td>memory read summary + async filtered write pipeline</td>
      <td>模型看到 summary，长期写入经过过滤和受限 worker</td>
      <td>空 summary、rate limit、state DB 不可用或异步滞后都会影响 memory 可见性</td>
    </tr>
    <tr>
      <td>历史超过上下文窗口</td>
      <td><code>CompactedItem.replacement_history</code> 重写 history</td>
      <td>summary 之外仍有可恢复的新 history base</td>
      <td>legacy compaction 无 replacement history 时只能退化恢复</td>
    </tr>
    <tr>
      <td>进程重启、fork、rollback</td>
      <td>rollout reverse segment scan + forward tail replay</td>
      <td>生产恢复不依赖内存状态或诊断 trace</td>
      <td>trace bundle 是诊断证据，不是 resume 权威状态</td>
    </tr>
  </tbody>
</table>
</div>

<figure class="wc-figure">
  <img src="https://cdn.jsdelivr.net/gh/WineChord/typora-images/img/codex-source-20260615v1-context-memory-final-rules.png" alt="Codex 上下文系统最终规则：separate views、pin sources、diff tracked state、compact with replacement history、trace only for diagnosis">
  <figcaption>图 10. 一个成熟 coding agent 的 prompt 不只是输入文本，而是可追踪、可压缩、可恢复的状态系统。</figcaption>
</figure>

最短的读源码规则是三条。

第一，先分 owner。`instructions`、developer/user `ResponseItem::Message`、rollout metadata、trace payload、UI transcript 不一定同形，也不应该强行同形。

第二，追 baseline。`base_instructions` 是 session baseline，`reference_context_item` 是 diff baseline，`TurnContextItem` 是 durable resume baseline。它们名字相近，但解决的问题不同。

第三，看 replacement。compaction 的关键不是“有摘要”，而是这个摘要有没有对应 `replacement_history`，resume 有没有从最新 surviving checkpoint 接上 newer suffix。只要沿着这三条线看，Codex 的上下文、记忆、压缩和 trace 就不再是散落的源码事实，而是一套围绕模型视图不变量组织起来的 runtime。

## 参考

- Codex source snapshot: [`openai/codex@ac4332c0`][codex-snapshot]
- OpenAI API: [Responses overview][responses-overview], [`responses.create`][responses-create], [Responses migration guide][responses-migrate], [prompt caching][prompt-caching]
- Codex docs: [Codex app-server docs][codex-app-server]

[codex-snapshot]: https://github.com/openai/codex/tree/ac4332c05b11e00ae775a24cb762edc05c5b5932
[responses-overview]: https://developers.openai.com/api/reference/responses/overview
[responses-migrate]: https://developers.openai.com/api/docs/guides/migrate-to-responses
[responses-create]: https://developers.openai.com/api/reference/resources/responses/methods/create
[prompt-caching]: https://developers.openai.com/api/docs/guides/prompt-caching
[codex-app-server]: https://developers.openai.com/codex/app-server
[response-input-item]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/models.rs#L659-L668
[response-content-item]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/models.rs#L697-L712
[response-item-message]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/models.rs#L741-L756
[session-spawn-base-instructions]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/mod.rs#L536-L547
[session-spawn-dynamic-tools]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/mod.rs#L549-L574
[session-meta]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/protocol.rs#L2726-L2760
[rollout-item]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/protocol.rs#L2791-L2799
[compacted-item]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/protocol.rs#L2801-L2806
[turn-context-item]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/protocol/src/protocol.rs#L2827-L2845
[build-initial-context]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/mod.rs#L2509-L2717
[build-text-message]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/context_manager/updates.rs#L178-L202
[record-context-updates]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/mod.rs#L2737-L2780
[context-manager]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/context_manager/history.rs#L32-L50
[context-updates]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/context_manager/updates.rs#L204-L238
[agents-md]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/agents_md.rs#L1-L17
[agents-md-filenames]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/agents_md.rs#L36-L43
[agents-global]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/agents_md.rs#L61-L78
[agents-user-instructions]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/agents_md.rs#L80-L127
[available-skills-instructions]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/context/available_skills_instructions.rs#L23-L30
[available-plugins-instructions]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/context/available_plugins_instructions.rs#L24-L57
[apps-instructions]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/context/apps_instructions.rs#L11-L30
[dynamic-tool-spec]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/v2.rs#L668-L679
[thread-start-dynamic-tools]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server-protocol/src/protocol/v2.rs#L3546-L3601
[app-server-thread-start]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/codex_message_processor.rs#L2426-L2455
[app-server-thread-start-task]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/app-server/src/codex_message_processor.rs#L2744-L2784
[thread-manager-start-options]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/thread_manager.rs#L213-L223
[turn-context-dynamic-tools]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn_context.rs#L540-L550
[rollout-recorder-dynamic-tools]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/rollout/src/recorder.rs#L680-L697
[state-db-dynamic-tools]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/rollout/src/state_db.rs#L289-L318
[memory-read-prompt]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/memories/read/src/prompts.rs#L24-L52
[memory-start-task]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/memories/write/src/start.rs#L16-L67
[memory-phase1-filter]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/memories/write/src/phase1.rs#L394-L448
[memory-phase2-agent]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/memories/write/src/phase2.rs#L291-L341
[run-pre-compact]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L701-L730
[model-downshift-compact]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L732-L777
[post-sampling-compact]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/turn.rs#L456-L492
[compact-injection]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/compact.rs#L46-L59
[compact-install]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/compact.rs#L252-L277
[rollout-reconstruction]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/core/src/session/rollout_reconstruction.rs#L87-L300
[rollout-trace-readme]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/rollout-trace/README.md#L1-L20
[trace-thread-start]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/rollout-trace/src/thread.rs#L37-L116
[trace-replay-bundle]: https://github.com/openai/codex/blob/ac4332c05b11e00ae775a24cb762edc05c5b5932/codex-rs/rollout-trace/src/reducer/mod.rs#L43-L86
