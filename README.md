# CaseSentinel（案件哨兵）

CaseSentinel（案件哨兵）是一个面向刑侦领域的多智能体研究平台，目标是复现并扩展
传统刑侦推理流程中的“证据整理 → 假说构建 → 行动规划 → 结果评估”闭环。
本仓库聚焦于以下三类产物：

1. **数据管线**：将裁判文书原文加工为易于推理的结构化数据、叙事文本和微调样本。
2. **知识底座**：构建向量数据库 + 关系图谱，为后续的检索增强生成 (RAG) 提供支撑。
3. **多智能体规范**：定义中心思想板 (Blackboard) 模型与跨 Agent 可调用的工具协议。

## 目录结构

```
├── data.json                # 原始裁判文书样本
├── project.md               # 高层项目规划
├── reference.md             # 相关研究参考
├── src/
│   ├── blackboard/
│   │   └── blackboard_template.md
│   ├── agents/
│   │   ├── analyst.py
│   │   ├── base.py
│   │   ├── forecaster.py
│   │   ├── framework.py
│   │   ├── prompts.py
│   │   ├── strategist.py
│   │   └── __init__.py
│   ├── common/
│   │   ├── llm.py
│   │   └── __init__.py
│   ├── data/
│   │   ├── cleaner.py
│   │   ├── fine_tuning_generator.py
│   │   ├── narrative_generator.py
│   │   └── __init__.py
│   ├── knowledge/
│   │   └── importer.py
│   └── runtime/
│       ├── session.py
│       └── __init__.py
├── tests/
│   ├── test_agents.py       # 多智能体 orchestrator 迭代测试
│   ├── test_cleaner.py      # 数据清洗管线的基础单元测试
│   └── test_runtime.py      # Phase 3 集成会话测试
└── outputs/
    └── (空目录, 用于存放处理结果)
```

## 快速开始

### 1. 安装依赖

建议使用 Python 3.10+ 并创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

> 如使用 `pip install -e .[dev]` 报错，可改用 `pip install -r requirements.txt`
> （后续将根据实验需求提供）。

### 2. 数据清洗

```bash
python -m src.data.cleaner data.json outputs/cleaned_cases.jsonl
```

- 输出为 JSON Lines，每行一个 `CaseRecord`。
- 建议搭配 `CASESENTINEL_MOCK_LLM=1` 环境变量，在离线情况下演示后续步骤。
- 支持 `--mode llm` 直接调用大模型完成抽取；`--mode hybrid` 会在 LLM 输出缺失字段时回退到规则结果。

### 3. 叙事与微调样本生成

```bash
# 离线演示（启用 Mock）
CASESENTINEL_MOCK_LLM=1 python -m src.data.narrative_generator_demo --limit 1

# 真实 API 调用（以 Qwen 为例）
unset CASESENTINEL_MOCK_LLM
export QWEN_API_KEY="<your-key>"
python -m src.data.narrative_generator_demo outputs/cleaned_cases.jsonl --output outputs/narratives.jsonl --log INFO
```

- 如果使用 OpenAI 兼容模型，请改为设置 `OPENAI_API_KEY`（可选 `OPENAI_MODEL_NAME`）。
- 如果重新充值后需从中断处继续，可添加 `--resume`，脚本会跳过已写入 `outputs/narratives.jsonl` 的案件。
- 默认会把调用失败的案件写入 `outputs/narratives_failed.jsonl`，可配合 `--skip-failures` 跳过这些案件或自定义 `--failed-log` 路径。
- 可以通过 `--markdown-dir outputs/narratives_md` 额外导出 Markdown 版本。
- 演示脚本会按顺序读取 `outputs/cleaned_cases.jsonl`，每生成一个叙事就立即落盘。

### 4. 知识库导入（可选 / 实验性）

> ⚠️ 当前多智能体协作流程尚未直接依赖知识库，本步骤主要用于后续的 RAG / 图谱推理实验。如果只想演示 Phase 2/3，可跳过此节。

```bash
python -m src.knowledge.importer data.json \
    --cleaned-jsonl outputs/cleaned_cases_llm.jsonl \
    --persist ./knowledge_store \
    --graph outputs/case_graph.json \
    --mode llm
```

该命令会：
- 生成一个可持久化的向量数据库目录 `knowledge_store/`
- 输出包含案件、人物、证据关系的图谱 `outputs/case_graph.json`
- 默认使用 `--mode llm` 基于 LLM 清洗结果构建知识库；如需改用规则或混合策略，可分别指定 `--mode rule` 或 `--mode hybrid`
- 若已有 JSONL 清洗产物，可通过 `--cleaned-jsonl` 直接复用

### 5. 多智能体协作（Phase 2）

```bash
export CASESENTINEL_MOCK_LLM=1  # 离线模式，真实部署请改用实际模型
python - <<'PY'
from src.agents import AgentOrchestrator
from src.blackboard.board import Blackboard

board = Blackboard()
orchestrator = AgentOrchestrator(board)
orchestrator.run(iterations=1)
print(board.snapshot())
PY
```

上述脚本会：
- 加载中心思想板模板；
- 依次运行分析师 → 战略家 → 预测员；
- 将输出写回思想板的 "侦查行动池"、"风险与应对策略" 等板块。

### 6. 协作推理会话（Phase 3）

```bash
export CASESENTINEL_MOCK_LLM=1
python -m src.runtime.session outputs/cleaned_cases.jsonl --iterations 2
```

- 默认从 `outputs/cleaned_cases.jsonl` 中读取第一个案件，初始化中心思想板。
- 每轮协作结束后，会把黑板快照写入 `outputs/sessions/<case_id>_iter*.md`，并生成 JSON 日志与 `*_summary.json` 便于追踪 Agent 输出。
- 通过 `--case-id` 或 `--case-index` 可切换案件；`--no-snapshots` 可关闭持久化。
- 如先运行“知识库导入”步骤生成 `knowledge_store/` 和 `outputs/case_graph.json`，多智能体会在每轮迭代时自动注入 RAG 检索结果与图谱关联要点；若缺少上述资产，则会自动降级为纯黑板推理。

### 7. 可视化界面

```bash
export CASESENTINEL_MOCK_LLM=1
python -m src.runtime.session outputs/cleaned_cases.jsonl --iterations 2
python -m src.visualization.dashboard --session-dir outputs/sessions --port 8000
```

- 第一步生成最新的会话数据；第二步启动基于 FastAPI 的面板，默认地址为 <http://127.0.0.1:8000/>。
- 页面提供案件列表、每轮黑板快照以及各 Agent 输出详情，适合复盘多轮协作过程。
- 在网页顶部填写 cleaned_cases.jsonl 路径、case_id / index、迭代次数等参数后，即可直接触发新的会话运行；“任务状态”面板将实时刷新执行进度与输出位置。
- 若暂无任何文书，可选择“全新案件（空白模板）”模式，仅填写基本线索（甚至留空）即可生成占位 `CaseRecord` 并启动协作，后续再在互动界面逐步补充事实、证据等信息。
- 同一页面新增「互动模式」：填写参数即可创建逐步执行的会话，侦查人员可在每个 Agent 输出后进行就地编辑或回滚，再推进下一步。点击会话卡片会打开 `/interactive/<session_id>` 页面，可直接在各 Agent 卡片内修改输出与对应黑板板块，同时提供“执行下一步”“撤销上一步”等操作，黑板与历史记录实时同步。

### 8. 测试

```bash

pytest
```

## 开发路线图

详见 `project.md`，当前阶段聚焦于：

- [x] 数据管线与黑板模板（Phase 1）
- [x] Agent 框架与协作回路（Phase 2）
- [x] 系统集成与协作推理会话（Phase 3）
- [ ] 可视化界面
- [ ] 系统化评估与论文撰写

## 环境变量

> 所有变量均可写入根目录下的 `.env` 文件，程序会在运行时自动加载。

| 变量名 | 说明 |
| ------ | ---- |
| `CASESENTINEL_MOCK_LLM` | 设为任意非空值可启用离线模拟叙事输出 |
| `CASESENTINEL_LLM_PROVIDER` | 指定底层模型提供方，支持 `openai`（默认）或 `qwen` |
| `CASESENTINEL_LLM_MODEL` | 全局覆盖默认的模型名称（优先级高于各 provider 默认值） |
| `OPENAI_API_KEY` | 若调用 OpenAI 兼容接口生成叙事，需要提供 |
| `OPENAI_MODEL_NAME` | （可选）覆盖默认的 OpenAI 模型名 |
| `QWEN_API_KEY` | 使用 Qwen DashScope OpenAI 兼容接口时必填 |
| `QWEN_API_BASE` | Qwen OpenAI 兼容接口地址，默认 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `QWEN_MODEL_NAME` | （可选）覆盖默认的 Qwen 模型名（默认 `qwen-plus`） |

## 许可

本项目处于研究阶段，具体许可协议将在首版论文投稿前公布。
