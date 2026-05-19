# Horizon-Agent 开发路线图 v15.0 (Phase 1 80%)

> **更新日期**: 2026-05-19
> **方向**: Horizon-Agent · 溯证智搜 — 多 Agent 协同 · 三层记忆 · 证据溯源
> **测试**: 384 passed, 4 skipped, 0 failed
> **Git**: 待 commit

---

## 一、产品定位

面向开发者和知识工作者的 **AI 深度研究助手**。输入一个研究问题，自动搜索网络和学术论文、
交叉验证结果、撰写结构化报告。支持 CLI / Web / API 三种使用方式，可私有化部署。

对标产品: OpenAI Deep Research、Perplexity、Google Deep Research

## 二、核心功能

1. **层次化记忆** (L1→L2→L3): 工作记忆 → 语义摘要 → 向量检索 (DashScope Embedding + FAISS)
2. **验证器驱动重规划**: 9 规则检查 + 9 种恢复策略，自动诊断和修复
3. **轨迹级日志**: 30 种事件类型, 25+ 事件/次, per-node 追踪
4. **证据溯源**: 报告中每个结论可追溯到搜索来源
5. **Token 级流式**: LLM 写作逐字推送至 Web 前端
6. **5 种任务自动分类**: research / code / comparison / summary / factual_qa
7. **Web 实时交互**: FastAPI + SSE + Token 流式 + 下载

## 三、开发完成状态

### Phase 1: 核心基础设施 ✅
### Phase 2: 产品化 + 评测 ✅
### Day 6-7: 架构收敛 + Web v4 ✅
### Day 8: 工程化转型 + 文档重写 ✅
### Day 9: Web 产品化 + React SPA ✅

```
Day 6 (05/15): 架构收敛    — 共享 Service 层 · Benchmark 重构 · L3 Embedding · 17+ Bug修复
Day 7 (05/16): Web v4      — FastAPI 迁移 · UI 美化 · SSE 防重入 · 下载修复
Day 8 (05/18): 工程化转型   — 论文搜索根治 · 文档清洗 · 产品路线图
Day 9 (05/19): Web 产品化   — SQLite 持久化 · 会话历史 API · React 18 SPA · 384 tests
```

## 四、Demo 列表 (7 个)

| # | 文件 | 功能 | 验收 |
|---|------|------|------|
| 01 | `01_async_demo.py` | asyncio 教程 | ✅ |
| 02 | `02_simple_agent.py` | 最简端到端 | ✅ |
| 03 | `03_llm_demo.py` | LLM 连接测试 | ✅ |
| 04 | `04_multi_agent_research.py` | 旗舰全链路 | ✅ |
| 05 | `05_web_agent.py` | Web 界面 (FastAPI + SSE) | ✅ |
| 06 | `06_ablation_study.py` | 组件分析 | ✅ |
| 07 | `07_benchmark.py` | Benchmark (20 题) | ✅ |

## 五、关键数字

| 维度 | 数值 |
|------|------|
| 源码文件 | 37 (+4 Web路由/sessions/vendor) |
| 测试文件 | 13 |
| Demo | 7 |
| Tests | 414 passed, 4 skipped |
| Task types | 5 |
| Web routes | 7 (FastAPI + OpenAPI docs) |
| 前端框架 | React 18 + htm (零构建) |
| 向量数据库 | FAISS (默认) + ChromaDB (生产推荐) |
| 静态资源总量 | 184KB (全部本地化) |

---

## 六、产品路线图

### Phase 1: 产品化基础 ✅ 100% 完成

目标: 从 Demo 到可用产品

```
✅ SQLite 会话持久化     — 替换内存 SessionManager，重启不丢失 (Day 9)
✅ 会话历史列表           — React 侧栏双 Tab，支持查看/删除 (Day 9)
✅ ChromaDB 向量数据库    — 双后端, 元数据过滤, 自动持久化 (Day 9)
✅ 环境变量完善           — 100+ HORIZON_ 变量, 全部模块可配置 (Day 9)
✅ GitHub Actions CI/CD   — 自动测试 + lint + benchmark smoke (Day 9)
```

### Phase 2: 体验优化 (2-4 周)

目标: 用户友好的交互体验

```
□ 用户 API Key 管理页    — Web 界面配置 LLM Key，加密存储
□ 报告导出 PDF           — weasyprint / pandoc 转换
□ Markdown 渲染增强      — 代码高亮、表格样式、图表支持
□ 研究任务模板           — 论文综述 / 技术对比 / 新闻摘要
□ 暗色/亮色主题切换     — CSS 变量一键切换
□ 多轮对话               — 追问和澄清
```

### Phase 3: 能力扩展 (1-2 月)

目标: 更多数据源、更强能力

```
□ 工具插件机制           — 用户可注册自定义工具
□ 更多数据源             — GitHub、RSS、数据库
□ RAG + Agent 混合模式   — 本地文档上传 + 在线搜索
□ 定时研究任务           — 定期追踪某个话题
□ 多语言支持             — i18n 框架
```

### Phase 4: 部署与发布 (2-4 周)

目标: 一键部署、生产可用

```
□ Docker + docker-compose — 一键启动全部服务
□ Nginx + HTTPS           — 反向代理 + 自动证书
□ 健康检查 + 监控         — /health 端点 + Prometheus
□ 日志轮转 + 备份         — 自动清理 + 备份策略
□ GitHub Release v1.0.0   — 正式发布
```

---

## 七、技术决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 会话存储 | SQLite (Phase 1) → PostgreSQL (Phase 4) | 零配置 → 生产级 |
| 向量数据库 | ChromaDB | Python 原生、零部署、元数据过滤 |
| 前端框架 | 原生 JS + Jinja2 | 零构建步骤，FastAPI 内置 |
| 论文搜索 | OpenAlex 主力 + Semantic Scholar 备选 | 国内可用，无速率限制 |
| RL 训练 | 暂不纳入路线图 | 不需要 GPU，产品不依赖 RL |
| 部署 | Docker Compose | 简单可靠，社区标准 |

---

*本文件 v16.0 — 2026-05-19 Horizon-Agent · Phase 1 完成 · Phase 2 开始*
