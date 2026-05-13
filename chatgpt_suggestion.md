# HorizonRL-Agent 下一版可执行开发指令

## 执行摘要

基于当前母本与最新项目状态，下一版开发不应继续横向扩模块，而应集中修复**“能跑但不好用”**的几个核心体验断点：第一，当前最终产物仍混杂大量调试信息与模板残留，典型症状包括报告日期/作者字段错误、`task_id`/工具细节泄露到用户结果、mock 证据占比过高且不可解释；第二，虽然项目已经实现了 `Writer`、Web Demo、Replanner、L1/L2 Memory 与 Trajectory Logger，但当前用户看到的输出仍偏“执行报告”，不是像聊天产品那样自然、可读、可追溯的最终答案；第三，搜索层虽然已有真实工具与 fallback，但 provenance 不完整，导致“这条结论来自哪个 provider、什么 query、何时抓到、是否 mock”无法在最终答案中被用户读懂。母本要求的主线是“先稳定完成长任务，再谈 RL”，因此当前最合理的下一版指令是：**优先重构 Writer 双模式输出，其次把 Web 入口改造成“普通对话 + 深度研究自动切换”，同时把 Search Provider provenance 做成一等数据，再用 mock/real 分层策略保证本地开发可用、CI 稳定通过。** fileciteturn0file0 fileciteturn0file1 fileciteturn10file0 fileciteturn10file2

从项目现状看，这一修复路线是低风险且高收益的。当前核心系统已经具备 Planner、Worker、Verifier、Replanner、Writer、LangGraph DAG、L1/L2 Memory、Trajectory Logger、Web Demo 和 296 个通过测试，说明“系统能力”不是主问题；真正的问题是**输出层、交互层和证据层还没有被产品化**。项目状态报告已经明确指出当前体验差集中在“太慢、过度拆分、输出形式像学术报告而不是对话、无流式反馈”，而当前生成会话也展示了两个典型失真：一类报告仍是纯 debug/trace 视图，包含 `task_id`、工具调用、mock URL、token=0 与耗时 0.0s；另一类自然语言报告虽然更像成品，但仍出现固定模板字段未替换，直接暴露了 `2023年10月27日` 与 `[您的姓名/代号]` 这种错误元数据。fileciteturn10file0 fileciteturn10file3 fileciteturn10file1

## 现状诊断与问题概述

当前问题不是“没功能”，而是“功能已经够多，但默认运行路径还在偏开发者模式”。项目状态报告和最新版开发路线图都表明，Phase 1 的关键模块已经全部完成，包括 `Writer(模板/LLM)`、`examples/05_web_agent.py`、`logging/trajectory_logger.py`、L1/L2 `HierarchicalMemory`、`Replanner`、`LangGraph` 编排与 296 项测试，因此下一步不该重做核心框架，而应在现有框架上补用户模式与证据模式。fileciteturn10file0 fileciteturn10file2

当前体验差可以归纳为六类具体症状。  
第一，**报告时间/作者错误**：现有自然语言报告仍出现固定模板元数据，典型例子是报告头部写成 `RPT-2023-LLM-Agent-001`、分析师为 `[您的姓名/代号]`、日期为 `2023年10月27日`，与当前会话时间和系统版本完全不一致，这说明 Writer 的 metadata 注入仍被模板残留污染。fileciteturn10file1

第二，**mock 数据占比高且未明确隔离**：当前会话中存在大量 `[Mock] 搜索结果`、`mock-search.local`、`Token=0`、`耗时=0.0s` 的输出，这说明用户模式和 mock 调试模式没有被清晰分流，或者说 mock 数据虽然对调试有价值，但现在被直接暴露给最终用户。fileciteturn10file3

第三，**最终自然语言回答质量差且混入执行细节**：当前系统会输出结构化研究报告，但用户可见结果仍常带有任务 DAG、证据数、工具调用数、task_id 与执行统计，这更像开发调试报告而不是面向用户的最终答案。项目状态报告也已直接描述当前输出“像学术报告不像对话”。fileciteturn10file3 fileciteturn10file0

第四，**Writer 虽然存在模板/LLM 双模式，但 LLM user-facing 路径没有稳定生效**。母本和当前开发计划都表明 `Writer(模板/LLM)` 已实现，且目标是“证据→自然语言报告”；但从当前会话样本看，至少当前默认路径仍大量走模板 fallback，或在 user-facing 渲染阶段没有彻底屏蔽调试字段。是否“完全未调用 LLM”在母本中**未指定**，更准确的判断是：**LLM writer 能力已存在，但用户模式没有被稳定路由到正确的 writer 路径。** fileciteturn10file0 fileciteturn10file2

第五，**Web 前端对话不友好**：项目已有 Web Demo，但当前设计仍偏“提交任务 → 等待报告”而非“先像聊天，再在必要时升级为深度研究流程”。项目状态报告已经把“输出形式像学术报告不像对话”“无流式，中间无反馈”列为体验问题，这意味着 Web 层应增加快慢双路径，而不是只暴露 6-stage pipeline。fileciteturn10file0

第六，**搜索结果不可解释**：当前系统已有 Web Search、Arxiv、Code 工具，也有 DDG/Brave fallback 与 mock 工具，但现有最终答案并没有把 provider、query、抓取时间、raw snippet、score 等 provenance 以用户可理解的形式呈现，导致用户无法判断结果从哪来、可信度如何。当前项目文档中虽然强调每条 `EvidenceItem` 带 `source/source_type`，但更细的 provenance 字段并未被定义，Bocha 支持也尚未进入当前正式结构，因此这部分属于**未完成**。fileciteturn10file2

## 目标与优先级

下一版目标建议明确分成短、中、长三个层级，并继续遵循母本的主线：**先做工程闭环，再做研究增强。** 母本和半年计划都把 Agent 系统、分层记忆、重规划放在 RL 之前，因此当前优先级也应先修交互与输出，而不是开新训练线。fileciteturn0file0 fileciteturn0file1

**短期优先级**是修复 Writer 生成逻辑与双模式输出。这部分包括：修复错误元数据；新增 `UserAnswerWriter` 与 `DebugReportRenderer` 的明确分离；在 mock 下也能生成自然语言 `final_answer.md`；默认同时输出 `debug_report.md` 和 `final_answer.md`；在 user-facing 模式中严禁露出 `task_id`、token、工具 JSON dump；并在报告中可读地显示证据来源和 mock 提示。这个阶段是最直接提升“像 ChatGPT 一样能回答”的关键。fileciteturn10file1 fileciteturn10file3

**中期优先级**是把 Web Demo 重构为真正的对话界面。已有 Web Demo 说明基础页面/交互已经存在，但需要把工作流改成“双路由”：简单问题走快速对话路径，复杂/学术/深度研究问题再触发 Agent pipeline，并在任务完成后在前端显示自然语言结果，同时提供 Markdown 文件下载入口。当前 API 端点命名、Web 框架和下载方式在母本中均**未指定**，因此推荐采用最小可行默认：`FastAPI + 简单 HTML/JS`，并统一使用 `/api/chat`、`/api/report`、`/api/download`。fileciteturn10file0 fileciteturn10file2

**长期优先级**是搜索 provider 可解释性、真实数据接入与评测。项目原始母本要求长链基准、成功率、工具效率、消融与 RL，而当前项目 summary 说明真实联网、L3 经验记忆、消融框架和 LLMVerifier 全管线仍未完善。因此下一版要为未来评测做铺垫：把 provenance 写进 `EvidenceItem` 与轨迹日志，把 mock/real provider 切换策略固化到配置层，把 CI 强制固定为 mock。这样后续做 Bocha/Brave/DDG 对比、检索质量评估、用户可解释性评测时才有数据基础。fileciteturn0file0 fileciteturn10file0 fileciteturn10file2

## 设计方案

### Writer 层重构

当前 Writer 已存在，但新版本必须把“报告生成能力”和“渲染模式”显式拆开。建议在 `src/horizonrl/agent/writer.py` 中重构为三个主类与一个主入口：

- `WriterConfig`
- `DebugReportRenderer`
- `UserAnswerWriter`
- `async write_reports(...) -> tuple[FinalReport, str, str]`

其中 `WriterConfig` 负责运行策略，建议字段至少包括：
- `enable_llm_writer: bool`
- `default_author: str = "HorizonRL-Agent"`
- `include_debug_stats: bool = False`
- `max_evidence_items: int = 8`
- `export_dir: str = "summaries"`
- `user_mode_sections: list[str]`  
以上字段名在母本中**未指定**，这是推荐默认。

`UserAnswerWriter` 的输入建议包含：
- `user_task: UserTask`
- `plan_graph: PlanGraph`
- `step_results: list[StepResult]`
- `evidence_items: list[EvidenceItem]`
- `memory_context: MemoryContext | dict | None`
- `verification_results: list[VerificationResult]`
- `llm_client: LLMClient | None`
- `config: WriterConfig`

输出建议有两层：
1. 内存对象：`FinalReport`
2. 文件路径：`summaries/{session_id}/final_answer.md`

`DebugReportRenderer` 则保留当前开发者模式，输出：
- `summaries/{session_id}/debug_report.md`

建议沿用仓库中已存在的 `summaries/` 目录习惯，而不是新建完全不同的根目录；这一点与当前仓库结构更一致。已有工程文档中已经出现 `summaries/` 与 `trajectories/` 作为 demo 输出目录，因此直接复用可减少改动面。fileciteturn10file4

Writer 路由规则建议是：

- 若 `enable_llm_writer=true` 且存在可用 `llm_client` 且 `offline_mode=false`，则 `UserAnswerWriter` 调用 LLM 合成自然语言答案。
- 否则走模板 fallback。
- 无论哪条路径，最终都必须生成用户可读答案，而不是 dump `StepResult`。
- 若 `evidence_items` 中 `is_mock=true` 占比超过阈值，或 provider 为 `mock`，则在 `final_answer.md` 开头明确插入提示：  
  `当前为 Mock Demo，内容仅用于流程演示，非真实检索结果。`

模板示例建议固定成以下结构：

- 标题
- 核心结论
- 背景与基本原理
- 关键发现
- 方法对比
- 局限与挑战
- 总结
- 参考证据

证据引用格式建议如下：

```text
[证据 1 | provider=bocha | query="..." | 2026-05-13 16:20] 标题
URL: ...
摘要: ...
```

这样既保留 provenance，又能让最终用户读懂。

此外，必须修正 `FinalReport.metadata`，或者在 `schemas/report.py` 中新增 `ReportMetadata`，至少包含：
- `session_id`
- `generated_at`
- `author`
- `mode`
- `used_mock_data`
- `llm_writer_used`

这样可以直接解决“年份写成 2023、作者是占位符”的问题。当前报告出现的固定编号与占位分析师，正说明 metadata 不应从静态模板硬编码。fileciteturn10file1

### Web 前端交互设计

当前已有 `examples/05_web_agent.py`，说明 Web 演示原型已完成，但详细路由、框架与 UX 契约在母本中**未指定**。下一版建议采用最小可行的默认方案：**FastAPI 后端 + 原生 HTML/JS 前端**。使用这个方案的原因不是因为它“最先进”，而是它与项目现有 `asyncio` / Agent pipeline / 文件生成逻辑更容易兼容，且便于做下载接口与同步/异步任务。当前用户明确希望“像自然对话一样”，因此 Web 层的关键不是“更炫”，而是明确快慢路由。fileciteturn10file0

建议把交互拆成两条路径：

**普通对话路径**
- 入口：`POST /api/chat`
- 输入：`message`, `history`, `mode=auto|chat|deep`
- 行为：若 `mode=chat`，或复杂度分类器判定为简单问题，则直接调用 `LLMClient` 返回聊天结果
- 输出：`{"mode":"chat","answer":"..."}`

**深度研究路径**
- 入口：`POST /api/chat`（`mode=auto` 且触发深度研究）或 `POST /api/report`
- 行为：后端创建 `session_id`，异步执行 Agent pipeline
- 轮询：`GET /api/report/{session_id}`
- 下载：`GET /api/download/{session_id}/{kind}`，其中 `kind in {"final","debug"}`
- 输出：完成后返回 `final_answer`, `download_url_final`, `download_url_debug`

触发深度流程的规则建议是：
- 用户显式打开“深度研究”开关，或
- query 长度较长，或
- 命中学术/对比/综述/最新进展/多来源分析等关键词，或
- 分类器 `should_use_agent()` 返回 true  
具体分类器实现方式在母本中**未指定**，推荐先用规则分类器，再预留 LLM classifier。

关于同步/异步，建议：
- `POST /api/chat` 对普通对话同步返回。
- `POST /api/report` 对深度任务立即返回 `session_id` 和 `status=queued`。
- 前端通过轮询或简单 SSE 显示“正在规划 / 正在搜索 / 正在写作 / 已完成”。
- 任务完成后，前端渲染自然语言答案，并自动触发附件下载或展示“下载 Markdown”按钮。  
浏览器自动下载最稳妥的做法是：完成后前端接收下载 URL，再使用 `a.download` 或 `Blob` 触发下载；这一点在母本中也**未指定**，属于推荐默认实现。

### Search Provider 可解释性改进

当前项目已实现 Web Search、Arxiv 与 Code 工具，且已有 DDG/Brave 与 mock 能力，但最终答案中的证据还不够可解释。建议在 `src/horizonrl/schemas/result.py` 中新增：

- `SearchProvenance`
  - `provider`
  - `query`
  - `timestamp`
  - `raw_snippet`
  - `score`
  - `url`
  - `is_mock`

或者在 `EvidenceItem` 内新增可选字段：
- `provider: str | None`
- `query: str | None`
- `retrieved_at: datetime | None`
- `raw_snippet: str | None`
- `score: float | None`
- `is_mock: bool = False`

然后在 `src/horizonrl/tools/manager.py` 中统一做 provenance 记录：所有 `web_search` 返回结果必须经 ToolManager 规范化，由 ToolManager 将 provider/query/timestamp/raw_snippet/score 附加到 `EvidenceItem`。这样 provenance 不依赖具体 provider 自己实现，避免不同工具返回字段不一致。

支持的 provider 建议明确为：
- `bocha`
- `brave`
- `duckduckgo`
- `mock`
- `auto`

其中 `bocha` 当前在母本中**未指定**，但作为国内开发推荐 provider 是合理默认；`brave`、`duckduckgo` 和 `mock` 则与现有规划一致。当前开发计划已明确 web search 存在 DDG/Brave 与 fallback 逻辑，也记录了 DuckDuckGo 在国内环境容易出现可达性问题，因此新增 `bocha` 主要是工程优化，而非研究方向变更。fileciteturn10file0 fileciteturn10file2

最终 `final_answer.md` 的“参考证据”段必须可读地显示 provenance，而不能只显示一堆原始 JSON。建议渲染成：

- 来源平台
- 检索 query
- 抓取时间
- 证据标题
- URL
- 摘要  
如为 mock，则明确标记 `Mock`。

### Mock 与 Real 数据策略

母本和现状都支持“先有可运行系统，再接真实数据”，因此下一版必须把 mock/real 策略固化，而不是让 mock 结果随机流到用户面前。

推荐策略如下：

**CI / 无 key / offline**
- 强制 `search_provider=mock`
- 强制 `enable_llm_writer=false`
- 强制 `offline_mode=true`
- 保证 `pytest`、mock demo、web API smoke test 全部可跑

**本地开发**
- 推荐 `search_provider=bocha`
- 若 `BOCHA_API_KEY` 缺失，则尝试 `brave`
- 若仍失败，fallback 到 `duckduckgo`
- 最后 fallback 到 `mock`

**默认 auto 优先级**
- `bocha` → `brave` → `duckduckgo` → `mock`

推荐 `.env.example` 增加：
- `BOCHA_API_KEY=`
- `BRAVE_API_KEY=`
- `HORIZON_SEARCH_PROVIDER=auto`
- `HORIZON_OFFLINE_MODE=false`
- `ENABLE_LLM_WRITER=true`

这部分中，Bocha 作为默认 provider 在母本中**未指定**；但 mock fallback、真实联网、搜索 provider 分层和 CI 稳定通过，与现有项目方向完全一致。fileciteturn10file0 fileciteturn10file2

## 模块接口与数据流

下表给出本轮需要新增或重构的关键模块接口。当前 repo 中许多模块已完成，因此这里重点列“要动的接口”，不是重画全系统。

| 模块 | 函数/类 | 输入 | 输出 | 关键字段 | 调用频率/延迟要求 | 文件路径 |
|---|---|---|---|---|---|---|
| Writer 配置 | `WriterConfig` | YAML / env / runtime flags | 配置对象 | `enable_llm_writer`, `export_dir`, `default_author`, `include_debug_stats` | 任务启动时读取一次 | `src/horizonrl/agent/writer.py` 或 `src/horizonrl/config/settings.py` |
| 用户答案写作 | `UserAnswerWriter.write_final_answer()` | `UserTask`, `PlanGraph`, `StepResult[]`, `EvidenceItem[]`, `VerificationResult[]`, `MemoryContext`, `LLMClient?` | `FinalReport`, markdown 字符串 | `session_id`, `generated_at`, `author`, `used_mock_data`, `llm_writer_used` | 每个深度任务 1 次；建议 `<20s`，无 LLM fallback `<1s` | `src/horizonrl/agent/writer.py` |
| 调试报告渲染 | `DebugReportRenderer.render_debug_report()` | 全量执行结果、统计 | markdown 字符串 | `task_id`, `tool_calls`, `metrics`, `verification` | 每个深度任务 1 次；建议 `<1s` | `src/horizonrl/agent/writer.py` |
| 报告导出 | `export_reports()` | session_id, 两种 markdown | 文件路径字典 | `summaries/{session_id}/final_answer.md`, `debug_report.md` | 每个任务 1 次 | `src/horizonrl/agent/writer.py` |
| 搜索 provenance | `SearchProvenance` / `EvidenceItem` 扩展 | provider raw result | 结构化 provenance | `provider`, `query`, `timestamp`, `raw_snippet`, `score`, `is_mock`, `url` | 每次 web search 结果都附加 | `src/horizonrl/schemas/result.py` |
| Tool 正规化 | `ToolManager._normalize_search_results()` | raw provider response | `EvidenceItem[]` | provenance + source + source_type | 每次搜索调用；建议 `<100ms` 处理开销 | `src/horizonrl/tools/manager.py` |
| Web provider 路由 | `resolve_search_provider()` | config/env | provider 实例 | `auto`, `bocha`, `brave`, `duckduckgo`, `mock` | 启动时/每次 tool init | `src/horizonrl/tools/web_search.py` |
| Web 聊天入口 | `POST /api/chat` | `message`, `history`, `mode`, `force_deep` | chat answer 或 `session_id` | `mode`, `should_use_agent`, `status` | 高频，chat 路径建议 `<5s` | `src/horizonrl/web/app.py` 或 `examples/05_web_agent.py` |
| 深度任务入口 | `POST /api/report` | `message`, `history`, `download` | `session_id`, `status` | `queued`, `running`, `completed`, `failed` | 中频，立即返回 `<500ms` | 同上 |
| 报告状态查询 | `GET /api/report/{session_id}` | session_id | 状态、答案、下载链接 | `final_answer`, `download_url_final`, `download_url_debug` | 前端轮询，建议 `<200ms` | 同上 |
| 文件下载 | `GET /api/download/{session_id}/{kind}` | session_id, `kind` | markdown 文件 | `final` / `debug` | 任务完成后低频 | 同上 |

## MVP 验收标准与自动化检查

短期 MVP 验收目标不是“更强的 Agent”，而是“让当前 Agent 像产品一样输出”。验收标准建议冻结为以下三条：

- Writer 能在 **mock 环境** 下稳定生成 `final_answer.md` 与 `debug_report.md` 两类文件；`final_answer.md` 必须是自然语言结构化答案，不包含 `task_id`、工具 JSON dump、token/耗时细节；`debug_report.md` 则保留 DAG、验证、统计。当前 session 已经证明两类输出在概念上都存在，但还没有被稳态路由与清洗。fileciteturn10file1 fileciteturn10file3

- Web 页面能先做**普通对话**，只有在学术/综述/复杂问题或显式 deep 模式时才触发 Agent pipeline；任务完成后前端展示自然语言回答，并附带 Markdown 下载链接。当前已有 Web Demo，因此目标不是“从零做 Web”，而是把交互从“提交研究任务”变成“对话 + 升级深研”。fileciteturn10file0 fileciteturn10file2

- CI 必须在 **mock mode** 下通过。当前系统已经有较大测试集和 GitHub CI，因此新改动必须兼容 CI，不得依赖真实 key、真实网络或外部 provider。当前网络 fallback 和 mock 工具已在项目中存在，这为落地提供了基础。fileciteturn10file0 fileciteturn10file2

### 测试用例表

| 测试名 | 目的 | 输入条件 | 期望断言 | 文件 |
|---|---|---|---|---|
| mock 模式 final answer 生成 | 验证无 key 路径 | `offline_mode=true`, `enable_llm_writer=false` | 生成 `final_answer.md` 与 `debug_report.md`；`final_answer.md` 非空 | `tests/test_writer.py` |
| fallback 模板无调试字段泄露 | 验证用户模式清洗 | mock `StepResult[]` + mock evidence | `final_answer.md` 不含 `task_id`, `Token`, `耗时`, `tool_calls` | `tests/test_writer.py` |
| metadata 正确注入 | 修复时间/作者错误 | session_id + default_author | 报告不含 `[您的姓名/代号]`；日期是当前运行生成值而非 2023 固定值 | `tests/test_writer.py` |
| search provenance 写入 evidence | 验证 provider 可解释性 | provider=`mock` 或 `bocha` | 每条 `EvidenceItem` 含 provider/query/timestamp 或 metadata 等价字段 | `tests/test_web_search.py` |
| final answer 引用段包含 provenance | 验证可读引用 | 构造 evidence with provenance | “参考证据”段出现 provider/query/URL/Mock 标记 | `tests/test_writer.py` |
| Web chat 普通对话路径 | 验证非深度问题不触发 Agent | `mode=auto`, 简单问候 | `/api/chat` 返回 `mode=chat` 且无 session_id | `tests/test_web_api.py` |
| Web deep report 路径与下载 | 验证异步报告 + 下载 | `mode=auto`, 学术问题 | 返回 session_id；完成后 `/api/download` 取回 markdown 文件 | `tests/test_web_api.py` |
| CI mock 下通过 | 验证 Actions 稳定 | GitHub env 强制 mock/offline | `pytest` 通过，网络相关测试自动 skip 或 fallback | `.github/workflows/ci.yml` + CI |

## 学习建议与两周行动计划

对你本人来说，这一轮开发需要补的知识不是 RL，而是“产品化 Agent”的几个短期能力。首先是 **FastAPI/Flask 风格的轻后端**，用于路由、异步任务、文件下载；其次是 **前端下载文件与轮询/SSE**，哪怕只用原生 JS，也要会用 `fetch + Blob + a.download`；再其次是 **asyncio 后端并发与后台任务管理**，尤其是不要让 Web 请求直接阻塞 30–60 秒；然后是 **LLM prompt 模板设计**，因为 user-facing writer 的质量高度依赖 prompt 分段与证据格式；最后是 **Mermaid 图嵌入与 Bocha API 文档阅读**，方便你把架构与 provider 策略维护到 README 中。母本已经把你的主线定为“Agent Systems → DeepResearch-Agent → RL → Evaluation”，所以这轮学习仍完全符合长期路线。fileciteturn0file1

### 两周行动计划表

| 时间 | 目标 | 交付物 | 主要风险 | 缓解措施 |
|---|---|---|---|---|
| 第 1 周 | 完成 Writer 双模式、metadata 修复、search provenance schema 与 CI mock 稳定 | `writer.py` 重构、`result.py` 扩展、`test_writer.py`、`test_web_search.py`、`.env.example`、`ci.yml` | 改 schema 破坏现有测试；模板/LLM 双路由混乱 | 先新增可选字段，不破坏旧接口；先通过 mock 路径，再接 LLM writer |
| 第 2 周 | 完成 Web 对话化、深研任务异步执行、Markdown 下载与 README 演示更新 | `web/app.py` 或升级 `05_web_agent.py`、`test_web_api.py`、README 截图与 quickstart | Web 路由阻塞 Agent 任务；下载文件路径不稳定 | 深度任务异步化；统一 `summaries/{session_id}` 导出规范；前端只轮询状态不直连 pipeline |

## 可复制给 Claude Code 的实现 Prompt

```text
现在请基于当前 HorizonRL-Agent 仓库，完成“下一版产品化修复”，重点解决 Writer 用户模式、Web 对话体验、搜索 provenance 以及 mock/real provider 策略。不要重写整个系统，优先在现有模块上增量重构。目标是：让项目在 mock 环境下也能稳定生成自然语言 final_answer，并通过 Web 前端以“普通对话 + 深度研究自动触发”的方式展示给用户，同时保留 debug_report 供开发者分析。

【当前已知前提】
- 核心 Agent 系统已完成：Planner / Worker / Verifier / Replanner / Memory L1/L2 / Trajectory Logger / Writer / LangGraph / Web Demo
- 当前主要问题不是功能缺失，而是用户体验差：输出像报告不是对话；报告 metadata 有模板残留；mock/real 数据混杂；搜索来源不可解释；Web 不够对话化
- CI 需要在无 API Key / 无真实网络情况下通过，因此必须保留 mock 路径并默认用于 CI

━━━━━━━━━━━━━━━━━━
一、必须修改/新增的文件
━━━━━━━━━━━━━━━━━━

1. 修改
- src/horizonrl/agent/writer.py
- src/horizonrl/schemas/result.py
- src/horizonrl/schemas/report.py
- src/horizonrl/tools/manager.py
- src/horizonrl/tools/web_search.py
- src/horizonrl/config/settings.py
- configs/default.yaml
- configs/dev.yaml
- configs/eval.yaml
- .env.example
- examples/04_multi_agent_research.py
- examples/05_web_agent.py 或者将其改为调用新 app
- README.md
- .github/workflows/ci.yml

2. 新增（若当前结构中不存在）
- src/horizonrl/web/app.py
- tests/test_writer.py
- tests/test_web_search.py
- tests/test_web_api.py

━━━━━━━━━━━━━━━━━━
二、Writer 重构要求
━━━━━━━━━━━━━━━━━━

请在 src/horizonrl/agent/writer.py 中新增或重构为以下对象：

1. class WriterConfig
建议字段：
- enable_llm_writer: bool = True
- default_author: str = "HorizonRL-Agent"
- include_debug_stats: bool = False
- export_dir: str = "summaries"
- max_evidence_items: int = 8

2. class DebugReportRenderer
必须保留当前开发者模式能力，负责输出 debug_report.md
输出内容包含：
- execution summary
- task DAG
- tool calls
- evidence list
- verification results
- memory summary
- statistics

3. class UserAnswerWriter
负责输出 final_answer.md
必须支持两条路径：
- enable_llm_writer=true 且 llm_client 可用且非 offline_mode → 走 LLM 合成
- 否则 → 走模板 fallback

4. 入口函数
至少提供：
- async def write_final_answer(...)
- def render_debug_report(...)
- def render_user_answer(...)
- def export_reports(session_id: str, debug_md: str, final_md: str) -> dict[str, str]

建议签名：
async def write_final_answer(
    user_task,
    plan_graph,
    step_results,
    evidence_items,
    memory_context=None,
    verification_results=None,
    llm_client=None,
    config=None,
) -> tuple[object, str]:
    ...
返回：
- FinalReport 对象
- final_answer markdown 字符串

5. metadata 修复（必须做）
不要再出现以下模板残留：
- RPT-2023-LLM-Agent-001
- [您的姓名/代号]
- 固定日期 2023年10月27日
请从运行时注入：
- session_id
- generated_at（当前时间）
- author（默认 HorizonRL-Agent）
- mode（user/debug）
- used_mock_data
- llm_writer_used

如果 schemas/report.py 中没有合适结构，请新增：
- class ReportMetadata
- FinalReport.metadata: ReportMetadata | None

6. final_answer.md 的结构
必须接近真实用户可读答案，而不是 debug 报告。
如果是模板 fallback，也必须输出自然语言结构化内容。

推荐结构：
# 问题标题

## 核心结论
1-2 段总结

## 背景与基本原理
自然语言解释背景

## 关键发现
3-5 点总结

## 方法对比
如有不同方法则比较

## 局限与挑战

## 总结

## 参考证据
- [证据 1 | provider=bocha | 2026-05-13 16:20] 标题
  query: ...
  url: ...
  snippet: ...

7. final_answer 中严格禁止出现
- task_id
- tool_calls JSON
- Token:
- 耗时:
- 原始 StepResult dump
- 原始 mock JSON 数组
- verification 内部细节

8. Mock 提示
如果 evidence 中 provider=mock 或 used_mock_data=true，则在 final_answer 顶部加提示：
“当前为 Mock Demo，内容仅用于流程演示，非真实检索结果。”

━━━━━━━━━━━━━━━━━━
三、Search Provider 与 provenance 改造
━━━━━━━━━━━━━━━━━━

请在 src/horizonrl/schemas/result.py 中扩展 EvidenceItem，或者新增 SearchProvenance 结构。
必须能记录：
- provider
- query
- timestamp
- raw_snippet
- score
- url
- is_mock

允许两种实现方式任选其一：
A. 新增 class SearchProvenance，并在 EvidenceItem 中挂 provenance
B. 直接给 EvidenceItem 增加上述可选字段

请在 src/horizonrl/tools/manager.py 中统一做搜索结果规范化：
- 所有 web_search provider 返回统一 SearchResult / EvidenceItem
- ToolManager 负责把 provider/query/timestamp/raw_snippet/score/url 填进 EvidenceItem
- 不要把 provenance 逻辑散落到多个 writer 或 demo 里

请在 src/horizonrl/tools/web_search.py 中支持以下 provider：
- bocha
- brave
- duckduckgo
- mock
- auto

provider=auto 的优先级：
1. 如果有 BOCHA_API_KEY → bocha
2. 否则如果有 BRAVE_API_KEY → brave
3. 否则尝试 duckduckgo
4. 失败后 fallback 到 mock

如果当前仓库尚未接入 bocha，请新增 provider stub 或最小实现，至少完成：
- provider 注册
- 配置读取
- 无 key 时结构化 fallback
- 测试覆盖
即便真实请求先不跑，也要把 auto 路由和 CI/mock 路由做对。

━━━━━━━━━━━━━━━━━━
四、Mock vs Real 策略
━━━━━━━━━━━━━━━━━━

请在 config/settings.py 与 yaml 配置中增加或整理以下字段：
- search_provider: auto | bocha | brave | duckduckgo | mock
- allow_mock_fallback: bool = True
- offline_mode: bool = False
- enable_llm_writer: bool = True

配置建议：
configs/default.yaml
- search_provider: auto
- allow_mock_fallback: true

configs/dev.yaml
- search_provider: bocha
- allow_mock_fallback: true
- offline_mode: false
- enable_llm_writer: true

configs/eval.yaml
- search_provider: mock
- offline_mode: true
- enable_llm_writer: false

.env.example 增加：
- BOCHA_API_KEY=
- BRAVE_API_KEY=
- HORIZON_SEARCH_PROVIDER=auto
- HORIZON_OFFLINE_MODE=false
- ENABLE_LLM_WRITER=true

━━━━━━━━━━━━━━━━━━
五、examples/04_multi_agent_research.py 改造
━━━━━━━━━━━━━━━━━━

请修改 examples/04_multi_agent_research.py：
- 默认同时生成：
  - summaries/{session_id}/debug_report.md
  - summaries/{session_id}/final_answer.md
- 控制台打印两个路径
- 增加 CLI 参数：
  --mode debug/user/both
  --search-provider auto/bocha/brave/duckduckgo/mock
  --offline
  --enable-llm-writer true/false

要求：
- mode=both 为默认
- mock/offline 下也必须能成功产出 natural final_answer
- 不要因为没有 API Key 而崩溃

━━━━━━━━━━━━━━━━━━
六、Web 前端与 API 设计
━━━━━━━━━━━━━━━━━━

请把当前 Web Demo 改造成“普通对话 + 深度研究自动触发”的最小可用版本。

若当前已有 examples/05_web_agent.py，请优先复用；如果更合适，请新增 src/horizonrl/web/app.py 并让 examples/05_web_agent.py 变成启动脚本。

必须提供这些端点：

1. POST /api/chat
入参：
{
  "message": "...",
  "history": [...],
  "mode": "auto" | "chat" | "deep"
}

行为：
- mode=chat：直接走普通对话 LLM
- mode=deep：直接触发 Agent 深度流程
- mode=auto：调用 should_use_agent() 判断
  - 简单问题 → chat
  - 学术/综述/最新进展/对比分析类问题 → deep

返回：
- 普通对话：
  {"mode":"chat","answer":"..."}
- 深度研究：
  {"mode":"agent","session_id":"...","status":"queued"}

2. POST /api/report
显式触发深度研究任务，立即返回 session_id

3. GET /api/report/{session_id}
返回状态：
- queued / running / completed / failed
如果 completed，则返回：
- final_answer
- download_url_final
- download_url_debug

4. GET /api/download/{session_id}/{kind}
kind in {"final","debug"}
返回 markdown 文件下载

5. should_use_agent() 规则
请先用规则路由，不要求复杂分类模型：
- 如果 query 很短、像寒暄、问候、简单定义 → chat
- 如果 query 包含 “综述 / latest / 最新进展 / 对比 / 调研 / 深度研究 / 论文 / 多来源 / 优缺点” 等关键词 → deep
- 允许后续再接 LLM classifier

6. 前端交互
要求：
- 聊天框支持普通对话
- 触发 deep 后显示“正在规划 / 正在搜索 / 正在写作”
- 任务完成后展示自然语言 final_answer
- 同时显示“下载 Markdown”按钮
- 如果可行，自动触发 final_answer.md 下载；如果浏览器限制，则至少显示明确下载按钮

━━━━━━━━━━━━━━━━━━
七、测试要求
━━━━━━━━━━━━━━━━━━

请新增或修改以下测试：

1. tests/test_writer.py
至少覆盖：
- mock + offline + enable_llm_writer=false 时可生成 final_answer.md
- final_answer 中不包含 task_id / Token / 耗时 / tool_calls
- final_answer 顶部在 mock 数据下有 Mock 提示
- metadata 不再包含 [您的姓名/代号] 或固定 2023 日期
- debug_report 仍包含 task DAG 和 verification 信息
- reference section 中包含 provider/query/url 或等价 provenance 字段

2. tests/test_web_search.py
至少覆盖：
- provider=mock 时不访问网络
- provider=auto 且无 key 时 fallback 到 mock
- provider=bocha 无 key 时返回结构化 fallback 或清晰错误
- normalize 后的 EvidenceItem 带 provenance 字段

3. tests/test_web_api.py
至少覆盖：
- /api/chat 简单问题返回 mode=chat
- /api/chat 学术问题返回 mode=agent + session_id
- /api/report/{session_id} 完成后返回 final_answer 和 download links
- /api/download/{session_id}/final 可下载且文件存在

4. Actions / CI
修改 .github/workflows/ci.yml：
- 强制 HORIZON_SEARCH_PROVIDER=mock
- 强制 HORIZON_OFFLINE_MODE=true
- 强制 ENABLE_LLM_WRITER=false
- pytest 必须在无 API Key 环境下通过
- 所有真实网络或真实 provider 测试默认 skip 或变为 integration tests，不得阻塞 CI

━━━━━━━━━━━━━━━━━━
八、README 更新要求
━━━━━━━━━━━━━━━━━━

请更新 README.md，新增/修改以下内容：

- 当前项目有两种输出：
  - debug_report.md：给开发者看
  - final_answer.md：给最终用户看
- 说明 search providers：
  - 国内推荐 bocha
  - 国际可选 brave / duckduckgo
  - CI 和离线环境默认为 mock
- 展示 Web 界面使用方式
- 补一个 final_answer 示例片段，而不是只展示任务 DAG
- 说明如何配置：
  - BOCHA_API_KEY
  - BRAVE_API_KEY
  - ENABLE_LLM_WRITER
  - offline mode
- 说明如何下载 markdown 结果

━━━━━━━━━━━━━━━━━━
九、本地执行要求
━━━━━━━━━━━━━━━━━━

修改完成后请你自己运行并记录结果：

1. pytest
2. python examples/04_multi_agent_research.py --mode both --search-provider mock --offline
3. 如果 Web 依赖完成，再运行 Web app 的本地 smoke test
4. 如本地有 API Key，可额外测试：
   python examples/04_multi_agent_research.py --mode both --search-provider bocha

━━━━━━━━━━━━━━━━━━
十、输出修改报告
━━━━━━━━━━━━━━━━━━

完成后请输出一份修改报告，必须包含：
- 修改了哪些文件
- 新增了哪些文件
- 哪些功能已完成
- pytest 结果
- 是否通过 mock 路径生成 final_answer/debug_report
- Web API 是否可用
- 还有哪些未解决问题
- 如果某项无法完成，明确说明卡点，不要假装完成

注意：
- 不要删除现有 Debug Report、Trajectory Logger、ToolManager、Replanner、Memory
- 不要把 RL、Benchmark、L3 FAISS 扩展进这一轮任务
- 本轮只做：Writer 用户模式、Web 对话体验、搜索 provenance、mock/real 策略、CI 稳定化
- 优先兼容现有架构，减少破坏性重构
```

