# HorizonRL-Agent

<div align="center">

**Long-Horizon Agentic RL System for Stable 20+ Step Complex Task Execution**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-325%20passed-brightgreen.svg)](tests/)
[![Status](https://img.shields.io/badge/Phase-2%20Complete-orange.svg)]()

</div>

---

## Overview

HorizonRL-Agent is a multi-agent LLM orchestration system that enables AI agents to reliably complete complex, long-horizon research tasks (20+ steps). It decomposes a user question into a DAG of subtasks, executes them concurrently with quality verification, and automatically recovers from failures through targeted replanning — all backed by a three-tier hierarchical memory system.

**Key differentiators from vanilla LangGraph/AutoGPT:**

| Challenge | Standard Approach | HorizonRL-Agent |
|-----------|------------------|-----------------|
| Context pollution over long runs | Single flat context window | L1→L2→L3 hierarchical memory with automatic compression |
| Error cascading in multi-step tasks | Full plan rebuild on failure | 9-rule verifier diagnoses errors; replanner patches only failed nodes |
| Tool failures blocking pipeline | Blocking retries with fixed backoff | Circuit breaker + fast-fail on network errors + mock fallback |
| Sequential tool execution | One tool at a time | asyncio.gather concurrent execution per worker |

## Quick Start

### Prerequisites

- Python 3.10+
- No GPU required (RL training phase only)

### Installation

```bash
git clone https://github.com/TheodoreYang6/HorizonRL-Agent.git
cd HorizonRL-Agent
pip install -e .
```

### Run (offline mode — zero API dependencies)

```bash
# End-to-end research pipeline with mock tools
python examples/02_simple_agent.py "Transformer attention mechanism"

# Full multi-agent DAG pipeline
python examples/04_multi_agent_research.py "Latest advances in RL for LLM training"
```

Both commands produce structured research reports with evidence citations.

### Run with LLM (recommended)

```bash
cp .env.example .env
# Add your API key: DEEPSEEK_API_KEY=sk-xxx

python examples/04_multi_agent_research.py --llm "Your research question"
```

Output: `reports/{session_id}/final_answer.md` + `debug_report.md`

### Web Interface

```bash
python examples/05_web_agent.py
# Open http://localhost:8080
```

Three modes: Auto | Chat (instant) | Deep (full agent pipeline with SSE progress streaming)

## Architecture

```
UserTask (natural language)
    │
    ▼
Planner / LLMPlanner ──→ PlanGraph (DAG of TaskSpecs)
    │
    ▼
ResearchOrchestrator (LangGraph StateGraph, 6-node state machine)
    │
    │   plan_task → mark_ready → execute_batch → verify_batch
    │                                        │
    │                     ┌──────────────────┼──────────┐
    │                continue             replan   done/deadlock
    │                     │                  │           │
    │                mark_ready          replan      finalize
    │                (next round)    (local repair)  (Writer)
    │                     │                  │           │
    │                     └──────────────────┘           │
    │                                                    ▼
    │                                                   END
    │
    ├──→ Verifier (rule/hybrid/llm) → VerificationResult
    │       └──→ Replanner → PlanPatch (RETRY/ADD/REMOVE/REORDER)
    │
    ├──→ HierarchicalMemory
    │       L1 (FIFO window) → L2 (semantic summary) → L3 (FAISS retrieval)
    │
    ├──→ TrajectoryLogger (async JSONL, 30 event types)
    │
    └──→ Writer
            ├── UserAnswerWriter → final_answer.md (end-user)
            └── DebugReportRenderer → debug_report.md (developers)
```

## Features

### 1. Verifier-Guided Local Replanning

Instead of rebuilding the entire plan on failure, the system:
- Runs 9 rule-based checks (<0.1ms) on every task result
- Classifies errors into 9 types (EMPTY_RESULT, CODE_ERROR, OFF_TOPIC, HALLUCINATION, etc.)
- Applies targeted patches: RETRY (rewrite query), ADD (supplement task), REMOVE (skip), REORDER

```
ErrorType → PatchType mapping:
  EMPTY_RESULT  → RETRY   |  CODE_ERROR   → RETRY
  TOOL_ERROR    → RETRY   |  OFF_TOPIC    → RETRY
  INCOMPLETE    → ADD     |  HALLUCINATION → RETRY
  FACTUAL_ERROR → RETRY   |  OTHER        → RETRY
```

### 2. Three-Tier Hierarchical Memory

| Tier | Type | Capacity | Behavior |
|------|------|----------|----------|
| L1 | Recent working window | 8K tokens (FIFO) | Auto-compresses to L2 at 80% fullness |
| L2 | Semantic summaries | 50 entries | Template/LLM compression, FIFO eviction |
| L3 | Episodic archive | Unlimited | FAISS n-gram vector search + keyword hybrid retrieval |

L3 uses deterministic MD5-based n-gram hashing (no embedding API required), with optional upgrade to real embeddings via `LLMClient.embed()`.

### 3. Async Multi-Agent DAG Orchestration

- LangGraph StateGraph with conditional routing and deadlock detection
- All task execution and verification use `asyncio.gather` for parallelism
- `asyncio.Semaphore` controls max concurrent workers
- `asyncio.wait` with timeout prevents batch-level stalls (120s cap)
- Circuit breaker prevents repeated calls to failing tools

### 4. Multi-Endpoint Tool Layer

Every tool has multiple fallback paths and concurrent endpoint racing:

| Tool | Strategy |
|------|----------|
| Web Search | 5-backend auto-fallback: Bocha → Brave → DDGS → Wikipedia → Mock. AUTO mode races backends concurrently. |
| Arxiv Search | 3 endpoints raced concurrently (first success wins, 8s cap). Mock paper generation on total failure. |
| Code Execution | Auto-detects natural language input and generates relevant code examples. 15s timeout with safe globals. |

### 5. Trajectory Logging

30 event types logged asynchronously (JSONL) across the full pipeline — from `plan.start` to `session.end`. Provides the data foundation for ablation studies and RL training.

### 6. Dual-Mode Report Generation

- **UserAnswerWriter**: Clean markdown reports with provenance citations, zero debug info
- **DebugReportRenderer**: Full task DAG, verification details, tool call traces, memory stats

## Demos

| # | File | Description | Requires API |
|---|------|-------------|--------------|
| 02 | `02_simple_agent.py` | Minimal end-to-end pipeline | No |
| 03 | `03_llm_demo.py` | LLM connection test + intelligent planning | Yes |
| 04 | `04_multi_agent_research.py` | **Flagship**: 6-stage pipeline, dual report output | Optional |
| 05 | `05_web_agent.py` | Web chat interface (dual routes + SSE + download) | Auto-detect |
| 06 | `06_ablation_study.py` | Ablation experiment framework (5 configs + stress injection) | No |
| 07 | `07_benchmark.py` | Benchmark evaluation (20 questions, 5 categories) | No |

## Project Structure

```
src/horizonrl/
├── schemas/           Data protocol (4 files, 16 data structures, 1061 lines)
│   ├── task.py        TaskSpec · PlanGraph · PlanNode · PlanPatch · UserTask
│   ├── result.py      StepResult · VerificationResult · EvidenceItem · ToolCall
│   ├── event.py       TrajectoryEvent · TrajectorySession · EventType (30 types)
│   └── report.py      FinalReport · ReportSection · CitationMap · ReportMetadata
│
├── config/            Configuration (1 file, 686 lines)
│   └── settings.py    Pydantic V2: LLMConfig · MemoryConfig · AgentRuntimeConfig
│
├── tools/             Tool layer (5 files, ~950 lines)
│   ├── manager.py     ToolManager: timeout/retry/circuit-breaker/stats
│   ├── web_search.py  Multi-backend with concurrent racing (Bocha/Brave/DDGS/Wikipedia/Mock)
│   ├── arxiv_search.py  Multi-endpoint concurrent race + mock fallback
│   ├── code_execution.py  Sandbox with auto code generation from descriptions
│   └── mock.py        Mock tools for offline/CI use
│
├── llm/               LLM client (1 file, 185 lines)
│   └── client.py      chat() + embed() · OpenAI-compatible · DeepSeek verified
│
├── agent/             Agent logic (5 files, ~2000 lines)
│   ├── planner.py     Planner (2 templates) + LLMPlanner (LLM-driven DAG decomposition)
│   ├── worker.py      AgentWorker (async execution + evidence extraction)
│   ├── verifier.py    RuleEngine (9 rules) + Verifier (rule/llm/hybrid)
│   ├── replanner.py   Replanner (9 strategies) + LLMReplanner
│   └── writer.py      UserAnswerWriter + DebugReportRenderer (dual output)
│
├── orchestration/     Orchestration (1 file, ~850 lines)
│   └── dag_workflow.py  ResearchOrchestrator: LangGraph 6-node state machine
│
├── memory/            Memory (1 file, ~750 lines)
│   └── hierarchical_memory.py  L1RecentWindow · L2SemanticSummary · L3EpisodicArchive
│
└── logging/           Logging (1 file, 411 lines)
    └── trajectory_logger.py  Async JSONL · TrajectorySession · 5 analysis tools
```

## Running Tests

```bash
pytest tests/ -v                          # All 325 tests
pytest tests/test_dag_workflow.py -v      # Orchestration layer (28 tests)
pytest tests/test_memory.py -v            # Memory layer (63 tests)
pytest tests/test_replanner.py -v         # Replanner strategies (51 tests)
```

## Configuration

Configuration uses a three-level merge: **code defaults → YAML file → environment variables**.

```bash
# .env file
DEEPSEEK_API_KEY=sk-your-key          # LLM inference
DASHSCOPE_API_KEY=sk-your-key         # Embedding (optional)
BOCHA_API_KEY=sk-your-key             # Web search (optional, China-friendly)

# Environment variable overrides (double-underscore = nesting)
HORIZON_LLM__MODEL=deepseek-chat
HORIZON_AGENT__MAX_STEPS=20
HORIZON_MEMORY__L1_MAX_TOKENS=6000
HORIZON_SEARCH_PROVIDER=auto
```

Config files: `configs/default.yaml` (production) | `configs/dev.yaml` (development) | `configs/eval.yaml` (evaluation)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph StateGraph (6-node DAG, conditional routing) |
| Async | Python asyncio (gather, Semaphore, Queue, wait, wait_for) |
| LLM | OpenAI-compatible SDK (DeepSeek, OpenAI, vLLM, any compatible API) |
| Embedding | MD5 n-gram feature hashing (zero-dependency, deterministic) — upgradeable to Embedding API |
| Vector Search | FAISS IndexFlatL2 + L2 distance threshold + keyword re-ranking |
| Config | Pydantic V2 with three-level merge (code → YAML → env) |
| Logging | Async JSONL with background writer task |
| Testing | pytest + pytest-asyncio (325 tests, 10 test modules) |

## Research

This project addresses the **Long-Horizon Agent Stability** problem. Core research contributions:

1. **Hierarchical Memory**: L1 (working) → L2 (semantic) → L3 (episodic) with deterministic n-gram retrieval
2. **Verifier-Guided Replanning**: 9 diagnostic rules → targeted local patches (not full replan)
3. **Trajectory-Level Logging**: 30 event types for ablation and future RL training
4. **Evidence-Provenance Chain**: Every claim in the final report is traceable to its search source

Target venue: AAAI / IJCAI / ACL Findings 2027

## License

MIT License — see [LICENSE](LICENSE)

## Citation

```bibtex
@misc{horizonrl-agent,
  author = {Qiduo Yang},
  title = {HorizonRL-Agent: Long-Horizon Agentic RL with Hierarchical Memory and Verifier-Guided Replanning},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/TheodoreYang6/HorizonRL-Agent}
}
```

---

*HorizonRL-Agent v0.2.0 — 325 tests · 7 demos · 16K lines · Phase 2 Complete*
