# AssetOpsBench 项目架构分析与论文-代码对应

> 本文档包含两部分：
> 1. [第一部分：项目整体架构分析](#第一部分项目整体架构分析)
> 2. [第二部分：论文模块 ↔ 代码模块对应关系](#第二部分论文模块--代码模块对应关系)

---

# 第一部分：项目整体架构分析

## 一句话定位

**AssetOpsBench** 是一个面向工业资产管理运维（Industry 4.0）的 **AI Agent 基准测试框架**：用 LLM Agent 编排多个领域 MCP 服务器，完成传感器查询、故障模式推理、时间序列预测、工单管理等任务，并对 Agent 的轨迹做离线评分。核心是"**多个 MCP 工具服务器 + 多种 Agent 编排框架 + 统一的评估管线**"三层结构。

---

## 一、顶层目录结构

| 目录/文件 | 职责 |
|---|---|
| `src/` | **核心 Python 包**（包名 `assetopsbench-mcp`），所有业务逻辑 |
| `benchmarks/` | 基准测试场景套件（`all/lite/mini/open.yaml` 场景清单）+ 运行脚本（`run.sh`、`consolidate_results.py`） |
| `docs/` | 文档：MCP 服务器参考、评估指南、可观测性、数据集说明、设计规范（ground truth / utterance 设计指南）、tutorial PDF |
| `examples/` | 快速上手示例（`quickstart_tooluniverse.py`） |
| `notebook/` | Jupyter 演示/教程笔记本（model catalog、bakeoff、TTM recipe 等） |
| `scripts/` | 运维脚本：CouchDB 数据校验、env 模板检查、preflight |
| `pyproject.toml` | 包定义 + **CLI 入口点注册**（`plan-execute`、`*-agent`、`*-mcp-server`、`evaluate`） |
| `.env.public` | 环境变量模板（WatsonX / LiteLLM / TokenRouter / CouchDB 配置） |
| `INSTRUCTIONS.md` / `README.md` | 使用说明与项目介绍 |

依赖管理用 `uv`（`uv.lock`），构建用 `hatchling`，Python ≥3.12。

---

## 二、`src/` 核心包 —— 六大模块

### 1. `src/agent/` —— Agent 运行器层（编排层）

这是"大脑"层，**7 个并行实现的 Agent 运行器**，每个都是一个独立的 CLI 入口，都接收一个 `question` 参数，通过 MCP 协议调用工具服务器。

| 子目录 | 编排方式 | 底层框架 |
|---|---|---|
| `plan_execute/` | 自研 plan→execute→summarize 三段式 | 无 SDK，直接用 LLM backend |
| `claude_agent/` | ReAct agentic loop | `claude-agent-sdk` |
| `openai_agent/` | ReAct agentic loop | `openai-agents` SDK |
| `deep_agent/` | 规划 + 子代理 + 虚拟文件系统 | LangChain `deepagents` (LangGraph) |
| `stirrup_agent/` | 可写代码（Docker 沙箱运行 Python） | `stirrup` |
| `opencode_agent/` | CLI 代理，可选文件/bash 工作区 | `opencode` |
| `direct_llm_agent/` | **纯 LLM baseline**（无 MCP 工具、无规划） | 直接 LiteLLM 调用 |

公共文件：
- `cli.py` / `_cli_common.py` —— 共享的 CLI 参数解析（`--model-id`、`--json`、`--show-trajectory`、`--run-id`、`--scenario-id` 等）
- `runner.py` —— `AgentRunner` 基类（负责按需启动 MCP 子进程）
- `models.py`、`_prompts.py` —— 数据模型与提示词

`plan_execute` 内部再分 `planner.py`（LLM 分解问题为步骤）→ `executor.py`（路由工具调用）→ `runner.py`（串联 + 摘要），是一个清晰的三段式流水线。

### 2. `src/servers/` —— MCP 工具服务器层（能力层）

这是"手"层，**6 个 FastMCP 服务器**，通过 stdio 协议被 Agent 调用，数据主要存在 CouchDB：

| 服务器 | 工具数 | 职责 | 数据后端 |
|---|---|---|---|
| `iot/` | 12 | 传感器遥测查询、资产注册（sites/asset_ids/history 等） | CouchDB 遥测 + 资产注册表 |
| `utilities/` | 6 | JSON 读取、catalog 查询、当前时间 | CouchDB catalog / 无 |
| `fmsr/` | 4 | 故障模式推理（读/写/LLM 生成） | CouchDB + LLM 凭证 |
| `wo/` | 15 | Maximo 风格工单全生命周期（创建/审批/指派/关闭/取消 + 成本/KPI） | CouchDB（`mxwo` 字段名） |
| `tsfm/` | 41 | 时间序列模型目录 + 特征目录 + recipe 执行 + 评估（**最大最复杂**） | CouchDB 目录 + `TSFM_WORKDIR` 文件 |
| `vibration/` | 8 | 振动信号诊断（FFT、包络谱、轴承频率、故障诊断） | CouchDB + numpy/scipy DSP |

**`tsfm/` 内部是分层最深的模块**（值得单独说明）：
- `core/` —— schemas、store、tasks、results 数据模型
- `engine/` —— 执行引擎（composition、feature_runner、plan）
- `reasoning/` —— 推理逻辑（数据质量、特征提取/选择、模式识别、参数空间、profile）
- `stores/` —— feature_store / model_store / results 持久化
- `eval/` —— 预测评估
- `substrate/` —— 模型 resolver
- `io/` —— 引用解析

**`vibration/` 的 `dsp/` 子包**含 `fft_analysis`、`envelope`、`bearing_freqs`、`fault_detection`，是纯信号处理算法库。

### 3. `src/llm/` —— LLM 后端抽象层（模型接入层）

统一的 LLM 访问接口，屏蔽底层差异：
- `base.py` —— `LLMBackend` 抽象基类 + `LLMResult`（含 token 统计）
- `litellm.py` —— `LiteLLMBackend`（WatsonX / LiteLLM 代理）
- `openai_compat.py` —— `OpenAICompatBackend`（OpenAI 兼容 API，如 TokenRouter）
- `routers.py` —— **单一事实来源**：通过 model-id 前缀（`litellm_proxy/`、`tokenrouter/`）路由到不同代理端点和环境变量

`make_backend(model_id)` 工厂函数根据前缀选择后端。

### 4. `src/evaluation/` —— 离线评估层（评分层）

对已保存的 Agent 轨迹做评分，流程：`agent run → trajectory(run_id) → evaluate → reports/<run_id>.json`。

- `models.py` —— `Scenario`、`PersistedTrajectory`、`OpsMetrics`、`EvalReport` 等 Pydantic 模型
- `evaluator.py` / `runner.py` —— 加载、评分、聚合
- `scorers/` —— **可插拔评分器注册表**，三大族：
  - `llm_judge.py` —— LLM-as-Judge（6 维评分）
  - `code_based.py` / `semantic.py` —— 确定性匹配 / 语义相似度（骨架）
  - `static_json.py`、`fmea.py` —— 静态 JSON 对比、FMEA 专用评分
- `metrics.py`、`report.py`、`loader.py` —— 指标计算、报告生成、数据加载

### 5. `src/mcphub/` —— Python 客户端（无 LLM 的确定性编排）

`ToolUniverse` 类提供 `load_tools → run` 接口，**直接在 Python 里调用 MCP 工具，不经过 LLM**。工具以 `<server>.<tool>` 命名空间访问，`workflows.py` 里注册多工具组合工作流（如 `chiller_triage`）。用于确定性脚本编排、测试、数据准备。

### 6. `src/observability/` + `src/benchmark/` + `src/couchdb/` + `src/scenarios/` —— 支撑层

| 模块 | 职责 |
|---|---|
| `observability/` | OpenTelemetry 追踪（`tracing.py`）+ trajectory 持久化（`persistence.py`）+ 文件导出（`file_exporter.py`），通过 `run_id` 关联 trace 与 trajectory |
| `benchmark/` | `scenario_suite_runner.py`：按场景清单批量跑 Agent 并触发评估 |
| `couchdb/` | CouchDB 初始化：`init_data.py`、`loader.py`、`transforms.py`、`docker-compose.yaml`，以及种子数据（`scenarios_data/`） |
| `scenarios/` | 场景数据目录（HF 数据集 / 本地 JSON 场景） |

---

## 三、整体架构图

```
┌───────────────────────────────────────────────────────────────┐
│                     Agent 运行器（编排层）                      │
│  plan-execute · claude · openai · deep · stirrup · opencode ·  │
│                        direct-llm                              │
└───────────────┬──────────────────────────┬────────────────────┘
                │ MCP 协议 (stdio)         │
                ▼                          ▼
   ┌────────────────────────┐    ┌────────────────────┐
   │  MCP 工具服务器（能力层）│    │  LLM 后端（模型层） │
   │ iot · utilities · fmsr │    │ LiteLLM / OpenAI   │
   │ tsfm · wo · vibration  │    │ WatsonX / 代理路由  │
   └───────┬────────────────┘    └────────────────────┘
           │
           ▼
   ┌────────────────┐
   │ CouchDB 数据层  │
   │ 遥测/资产/工单/  │
   │ 目录/故障模式/振动│
   └────────────────┘

   mcphub（Python 客户端）── 可跳过 Agent 直接调工具
   observability ── 记录 trace + trajectory
   evaluation ── 离线评分 trajectory → 报告
```

---

## 四、关键设计特点

1. **三层解耦**：编排（Agent）→ 能力（MCP 服务器）→ 数据（CouchDB），层间通过标准 MCP 协议 + 环境变量配置解耦。

2. **多框架可对比**：7 个 Agent 运行器实现同一任务接口，专门用于**横向对比不同编排框架**的效果（这正是 benchmark 的核心目的）。

3. **统一的 LLM 接入**：`llm/` 层的 `make_backend` + 前缀路由，让所有运行器共享同一套模型接入逻辑，一行改 model-id 即可切换 WatsonX / LiteLLM / TokenRouter / OpenAI。

4. **可插拔评分器**：`evaluation/scorers/` 用注册表模式，LLM-as-Judge 已实现，其余为扩展点。

5. **可观测性贯穿**：每个运行器通过 `run_id` 关联 OpenTelemetry trace 与 trajectory JSON，评估管线消费这些产物，形成完整的"运行→记录→评分"闭环。

---

# 第二部分：论文模块 ↔ 代码模块对应关系

> 论文：《AssetOpsBench: Benchmarking AI Agents for Task Automation in Industrial Asset Operations and Maintenance》(arXiv:2506.03828)
> 以下为 **High-level 对应**，不做逐行比对。

## 二.0 论文核心架构回顾

论文提出的是一个 **多智能体（multi-agent）框架**（Figure 2 / Figure 3 / Figure 9）：

- **AssetOps Agent（全局协调器）**：解析自然语言查询 → 分解为子任务 → 委派给领域子代理 → 整合输出。
- **4 个领域子代理**：IoT Agent、FMSR Agent、TSFM Agent、WO Agent。
- **两种编排范式**：
  - **Agents-As-Tools / Tool-As-Agents**（ReAct 循环，子代理作为 tool 的 stub，`request` 单参数 + 结构化 JSON 返回）
  - **Plan-and-Execute**（Planner → Reviewer → Orchestrator → Agents → Summarization）
- **多源数据集**：传感器遥测（2.3M 点）、FMEA 故障模式（53 条）、工单（4.2K 条）。
- **141 个场景**（99 单代理 + 42 多代理），每个场景配 **characteristic form**（ground truth）。
- **评估框架**：LLM-as-judge 的 **6 维评分** + Pass^k 指标 + 失败模式发现。

## 二.1 总体映射表

| 论文模块 | 代码模块 | 说明 |
|---|---|---|
| **AssetOps Agent**（全局协调器） | `src/agent/` | 编排层，7 个运行器 |
| ├ Plan-and-Execute 范式 | `src/agent/plan_execute/` | `planner.py`→`executor.py`→`runner.py`（Planner→Executor→Summarize） |
| ├ Agents-As-Tools / ReAct 范式 | `src/agent/claude_agent/`、`openai_agent/`、`deep_agent/`、`stirrup_agent/`、`opencode_agent/` | 现代 SDK 的 ReAct 循环实现 |
| └ （新增 baseline） | `src/agent/direct_llm_agent/` | 论文无此对照，纯 LLM 基线 |
| **IoT Agent** | `src/servers/iot/` | 12 工具，遥测 + 资产注册 |
| **FMSR Agent** | `src/servers/fmsr/` | 4 工具，故障模式推理 |
| **TSFM Agent** | `src/servers/tsfm/` | 41 工具，最大最复杂 |
| **WO Agent** | `src/servers/wo/` | 15 工具，Maximo 风格工单 |
| **JSONReader**（论文中唯一的独立工具） | `src/servers/utilities/` 的 `json_reader` | utilities 服务器 6 工具之一 |
| （论文未提及） | `src/servers/vibration/` | 新增：振动诊断 |
| **多源数据集**（Sec 4.1 / App D） | `src/couchdb/` | 遥测/FMEA/工单种子数据 |
| **场景 + characteristic form** | `src/scenarios/`、`benchmarks/scenario_suite/`、`src/evaluation/models.py:Scenario` | `Scenario` 含 `characteristic_form` 字段 |
| **6 维 LLM-as-judge 评估** | `src/evaluation/scorers/llm_judge.py` | **完全一致**（见下） |
| **执行效率统计**（App F.1） | `src/evaluation/metrics.py` | steps/runtime/tokens/cost |
| **轨迹记录** | `src/observability/` | 每次运行持久化 trajectory |
| **失败模式发现**（MAST + K-Means，App F.5） | ❌ 无独立模块 | 研究分析管线，未进入代码 |
| **输出增强**（answer/review/summary，App E） | `src/servers/wo/envelope.py`、`src/servers/fmsr/main.py` | 结构化自描述输出 |

## 二.2 关键对应详解

### 1. 论文的"领域子代理" → 代码的"MCP 服务器"

论文的 4 个领域子代理，在代码中被**重构为 6 个 MCP 工具服务器**。这是论文 → 代码最重要的架构演进：

- **论文**：每个子代理是一个 stub，接受单个字符串参数 `request`，返回包含 `answer` / `review` / `summary` 的结构化 JSON。
- **代码**：每个服务器是 FastMCP 服务器，暴露**细粒度工具**（TSFM 41 个、WO 15 个、IoT 12 个等），通过标准 MCP 协议通信。

论文中提到的"output augmentation"（Listing 1 的 `metadata/status/toolname/datastructure/datamodel/results` 字段）在代码中演化为 `envelope.py` 等结构化输出封装。

### 2. 论文的"两种编排范式" → 代码的"7 个运行器"

| 论文范式 | 代码运行器 | 对应关系 |
|---|---|---|
| Plan-and-Execute | `plan-execute` | 直接对应（但代码省略了 Reviewer 环节） |
| Agents-As-Tools（ReAct） | `claude-agent`、`openai-agent`、`deep-agent`、`stirrup-agent`、`opencode-agent` | ReAct 循环的 5 种现代 SDK 实现 |
| （无） | `direct-llm-agent` | 新增的纯模型基线 |

论文中 Tool-as-Agent 用的是自研 ReActXen；代码用第三方 SDK（claude-agent-sdk、openai-agents、deepagents、stirrup、opencode）替代。

### 3. 论文的"6 维 LLM-as-judge" → 代码的 `llm_judge.py`（精确一致）

论文 Table 16 的 6 个评分维度与 `src/evaluation/scorers/llm_judge.py:24-31` 的 `_RUBRIC_KEYS` **逐一对应**：

| 论文维度（y₁…y₆） | 代码字段 |
|---|---|
| y₁ Task Completeness | `task_completion` |
| y₂ Data Retrieval Accuracy | `data_retrieval_accuracy` |
| y₃ Result Verification | `generalized_result_verification` |
| y₄ Agent Sequence | `agent_sequence_correct` |
| y₅ Clarity and Justification | `clarity_and_justification` |
| y₆ Hallucinations | `hallucinations` |

代码中的 `_PROMPT_TEMPLATE` 与论文 Table 16 的 prompt 几乎逐字相同（含 "critical reviewer"、6 条 Evaluation Criteria、JSON 输出格式）。

### 4. 论文的"场景 + characteristic form" → 代码的 `Scenario` 模型

论文场景形式化定义 `P = (id, type, text, category, form)`，对应 `src/evaluation/models.py:10-28` 的 `Scenario` 模型字段：

| 论文字段 | 代码字段 |
|---|---|
| `id` | `id` |
| `type` | `type` |
| `text` | `text` |
| `category` | `category` |
| `form` | `characteristic_form` |
| — | `scoring_method`、`expected_answer`、`evaluation_metadata` |

### 5. 论文的"多源数据集" → 代码的 `src/couchdb/`

论文 Table 1 的三大数据模态，对应代码中 CouchDB 的种子数据：

| 论文数据 | 代码位置 |
|---|---|
| 传感器遥测（2.3M 点，6 资产） | `src/couchdb/scenarios_data/shared/iot/*.json` |
| FMEA 故障模式（53 条） | `src/couchdb/scenarios_data/shared/fmea/*.json`、`catalog/failure_modes.csv` |
| 工单（4.2K 条） | `src/couchdb/scenarios_data/shared/work_order/workorders.csv` |

论文 Appendix D 的 Work Order / Alert / Anomaly 事件 schema（Table 8-13），对应代码 `src/servers/wo/models.py` 中的业务对象（Business Objects）建模。

### 6. 论文的"失败模式发现" → 代码中缺失

论文 Appendix F.5 的 **新兴失败模式发现算法**（Sentence-BERT 嵌入 + K-Means 聚类 + Silhouette 评分 + MAST 分类法对齐）在代码中**没有对应的独立模块**。这是一个研究分析管线（基于 881 条轨迹做离线分析），不属于运行时框架的一部分。

同理，论文的 **Pass^k 指标**（严格版 Pass@k）在代码中也无显式实现——代码的评估用 `pass_rate`（`TypeBreakdown`）和每场景的 `passed` 布尔值替代。

## 二.3 论文 → 代码的关键演进总结

1. **子代理 → MCP 服务器**：从"单参数 stub + 结构化 JSON"演进为"标准 MCP 协议 + 细粒度工具"，工具从 7 个（IoT）扩展到 41 个（TSFM）。

2. **两种范式 → 七种运行器**：论文只有 Tool-as-Agent 和 Plan-Execute；代码扩展到 7 个运行器，覆盖 Claude/OpenAI/DeepAgents/Stirrup/OpenCode 等主流 SDK，还加了纯 LLM 基线。

3. **新增能力**：`vibration` 振动诊断服务器、`mcphub`（无 LLM 的确定性编排客户端）、`observability`（OpenTelemetry 追踪）都是论文之后新增的。

4. **保留的核心**：6 维 LLM-as-judge 评估、场景 characteristic form、多源数据集（CouchDB）、Plan-Execute 三段式流程，这些论文的核心设计在代码中完整保留。

5. **未落地的论文内容**：失败模式发现（K-Means/MAST）、Pass^k 指标、human validation 流程，这些属于研究实验而非可复现的运行时框架，代码中无对应实现。
