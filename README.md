# ⚡ SynapseForge

> **Transform recurring multi-step agent reasoning loops into deterministic, lightning-fast FastMCP tools — slashing token burn by >65% and compressing 8-step workflows into single-step execution.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Model Context Protocol](https://img.shields.io/badge/MCP-FastMCP-green.svg)](https://github.com/jlowin/fastmcp)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-purple.svg)](https://github.com/langchain-ai/langgraph)
[![Pydantic v2](https://img.shields.io/badge/Validation-Pydantic%20v2-e92063.svg)](https://docs.pydantic.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 💡 Why SynapseForge?

Traditional AI agent memory architectures (like Mem0, Letta, or LangMem) accumulate past experiences as static text snippets and re-inject them into prompts. As agent workflows scale past hundreds of procedures, prompt windows become bloated, response latency spikes, and API token bills skyrocket.

**SynapseForge introduces a paradigm shift: Code-Compiled Memory.** Instead of stuffing repetitive instructions back into the prompt context, SynapseForge monitors live agent execution trajectories, discovers recurring multi-step patterns, compiles them into deterministic Python **FastMCP** tools, and verifies them against historical data before deployment.

---

## 🚀 Core Features & Value Proposition

- 🛠️ **Automated FastMCP Compilation**: Automatically mines episodic traces for recurring 4+ step reasoning chains and synthesizes self-contained, AST-verified Python functions wrapped with Pydantic v2 schemas.
- 📉 **Over 65% Token Reduction**: Replaces costly multi-turn LLM reasoning iterations with immediate 1-step code executions, dropping average steps from ~7.8 down to ~1.4.
- 🔬 **25-Task Local Shadow Replay**: Verifies candidate tools against the last 25 historical execution traces in an isolated local runner, demanding 100% output parity before promotion.
- ⚡ **Two-Speed Smart Router**: Dynamically steers traffic between **Tier 1 (Fast Path)** using pre-compiled FastMCP tools and **Tier 2 (Reasoning Path)** powered by LangGraph.
- 🩺 **Self-Healing Circuit Breaker**: Detects upstream API shifts or unexpected tool errors in real time. If a compiled tool degrades below the health threshold, the router trips fallback to full reasoning and queues background recompilation.
- ⏳ **Bi-Temporal Knowledge Graph**: Retains facts with dual timestamps (`valid_from` and `valid_to`), cleanly expiring superseded operational rules while maintaining strict multi-tenant RBAC privacy.
- 🗜️ **Smart Context Offloading**: Identifies oversized raw payloads and offloads them to a key-value store, keeping active prompt context compact by feeding only structural schema previews.

---

## 🏛️ Architecture Overview

### System Pipeline

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Incoming User Request                           │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │    Two-Speed Smart Router     │
                   └───────┬───────────────┬───────┘
                           │               │
        [Healthy Compiled  │               │ [Novel / Broken /
          Tool Exists]     │               │  Uncompiled]
                           ▼               ▼
           ┌───────────────────────┐   ┌───────────────────────────────┐
           │   Tier 1: Fast Path   │   │   Tier 2: Reasoning Path      │
           │  (1-Step FastMCP Tool)│   │  (LangGraph + Temporal Graph) │
           └───────────┬───────────┘   └───────────────┬───────────────┘
                       │                               │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │   Episodic Trajectory Store   │
                       └───────────────┬───────────────┘
                                       │
                                       ▼ (Periodic Pattern Mining)
                       ┌───────────────────────────────┐
                       │    Trajectory Pattern Miner   │
                       └───────────────┬───────────────┘
                                       │ (Matches 4+ Step Patterns)
                                       ▼
                       ┌───────────────────────────────┐
                       │      FastMCP Tool Compiler    │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │   Local Shadow Replay Runner  │
                       │    (25-Task Verification)     │
                       └───────────────┬───────────────┘
                                       │ [100% Parity Pass]
                                       ▼
                        Promoted to Fast Path Registry
```

### Architectural Subsystems

| Subsystem | Primary Module | Core Functionality |
| :--- | :--- | :--- |
| **Context Management** | `synapseforge.memory.context_manager` | Offloads massive JSON responses out of active prompts into local KV storage. |
| **Episodic Tracing** | `synapseforge.memory.episodic` | Records tool spans, latency, and token consumption per execution trajectory. |
| **Bi-Temporal Memory** | `synapseforge.memory.temporal_graph` | Time-aware fact store with conflict resolution and tenant boundary isolation. |
| **Pattern Mining** | `synapseforge.compiler.pattern_miner` | Clusters successful multi-step execution traces to find optimization candidates. |
| **Tool Compiler** | `synapseforge.compiler.mcp_compiler` | Generates validated FastMCP tool definitions and performs Python AST validation. |
| **Shadow Replay** | `synapseforge.compiler.shadow_replay` | Runs 25-task replay audits ensuring zero behavioral discrepancies. |
| **Smart Routing** | `synapseforge.orchestration.router` | Two-speed orchestrator with health monitoring and self-healing fallback loops. |
| **Observability** | `synapseforge.orchestration.telemetry` | Quantifies step reduction ratios and token compression percentages. |

---

## 💻 Local Quick Start

Set up and explore SynapseForge locally on your workstation in minutes using Python 3.12 and VS Code.

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/synapseforge.git
cd synapseforge
```

### 2. Configure Virtual Environment

Create and activate a fresh local virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install in Editable Mode

Install SynapseForge alongside local development tools:

```bash
pip install -e ".[dev]"
```

### 4. VS Code Integration & Build Tasks

SynapseForge includes pre-configured `.vscode` settings for an optimal developer experience:

- **Open the project in VS Code**:
  ```bash
  code .
  ```
- **Run Build & Verification Tasks**:
  Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and choose **Tasks: Run Build Task**:
  - `Install Editable Package`: Installs local dependencies in development mode.
  - `Lint & Typecheck (Ruff + Mypy)`: Runs instant code linting and strict type checking.

### 5. Launch the Local Server

Start the local FastAPI development server:

```bash
uvicorn synapseforge.main:app --reload --host 127.0.0.1 --port 8000
```

> **Interactive API Docs**: Once launched, navigate to `http://127.0.0.1:8000/docs` to inspect the OpenAPI schema, dispatch `/v1/execute` queries, or invoke the `/v1/compile` background worker.

---

## 🤝 Contribution Guide

We are building the future of autonomous agent memory, and we would love your input! Whether you are designing innovative pattern mining algorithms, refining FastMCP tool synthesizers, or benchmarking token compression ratios:

1. **Fork the Repository**: Create a feature branch for your changes (`git checkout -b feature/smart-miner`).
2. **Follow Coding Standards**: Keep code clean and type-annotated (`ruff check synapseforge` and `mypy synapseforge`).
3. **Submit a Pull Request**: Detail your architectural improvements and benchmark findings.

Let's forge faster, leaner, and smarter AI agents together!
