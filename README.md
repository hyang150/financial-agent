Finance Agent: Multi-Tool RAG & Fine-tuned LLM 🚀📖 项目简介Finance Agent 是一个垂直领域的智能金融分析助手，旨在解决通用大模型在金融财报分析中存在的“幻觉”和“计算能力弱”的问题。本项目构建了一个能够根据用户意图自动选择工具（搜索引擎、Python 代码执行器、向量数据库）的 AI Agent，并结合 RAG (检索增强生成) 和 LoRA Fine-tuning (微调) 技术，在 SEC 财报问答场景中实现高质量的自主推理。✨ 核心特性🧠 智能路由 (Intent Routing): 自动判断用户意图，决定是查文档 (RAG)、联网搜索 (Search) 还是写代码计算 (Python REPL)。📚 高级 RAG 管道: 实现了混合检索 (Hybrid Search) 和 Cross-Encoder 重排序 (Reranking)，精准定位财报中的关键数据。🔧 工具调用微调 (Tool-Use Fine-tuning): 基于 Qwen2.5/LLaMA3 使用 QLoRA 进行微调，显著提升模型调用工具的准确率和参数格式的正确性。📊 量化评估: 集成 RAGAS 框架，对检索精度 (Recall/Precision) 和生成质量 (Faithfulness) 进行自动化评分。🏗️ 技术架构graph TD
    User[用户查询] --> Router{Query Router<br>意图分类}
    
    Router -->|需外部知识| RAG[RAG Pipeline]
    Router -->|需计算/搜索| Agent[Tool Agent]
    Router -->|混合任务| Hybrid[RAG + Tool]

    subgraph RAG Pipeline
    Doc[PDF/财报] --> Split[Chunking] --> Emb[Embedding Model] --> VecDB[(FAISS/Chroma)]
    VecDB --> Retrieve[Top-K Document Chunks]
    Retrieve --> Rerank[Cross-Encoder Reranking]
    end

    subgraph Tool Agent
    AgentLogic[ReAct Logic] --> Tools
    Tools --> Calc[Calculator]
    Tools --> Web[Web Search]
    Tools --> Code[Python Interpreter]
    end

    Rerank --> Context
    Tools --> Context

    Context --> LLM[Fine-tuned LLM<br>(Qwen2.5-7B + LoRA)]
    LLM --> Response[最终回答 + 引用来源]
🛠️ 环境与安装本项目使用 uv 进行极速依赖管理。前置要求Python >= 3.10NVIDIA GPU (推荐 16GB+ VRAM 用于微调，推理 8GB+ 即可)Hugging Face Token (用于下载模型)Tavily API Key (用于联网搜索)快速开始克隆项目git clone [https://github.com/your-username/finance-agent.git](https://github.com/your-username/finance-agent.git)
cd finance_agent
配置环境变量复制 .env.example (需自行创建) 为 .env 并填入密钥：# .env 文件内容示例
HF_TOKEN=hf_xxxxxx
TAVILY_API_KEY=tvly-xxxxxx
OPENAI_API_KEY=sk-xxxxxx  # 可选，仅用于 RAGAS 评估裁判
安装依赖 (使用 uv)# 自动创建虚拟环境并同步所有依赖
uv sync
环境检查uv run setup_check.py
📂 项目结构finance_agent/
├── config/              # 配置文件 (模型参数, RAG参数)
├── data/                # 数据存放 (PDFs, VectorDB)
├── notebooks/           # 实验性 Jupyter Notebooks
├── src/
│   ├── ingestion.py     # 数据获取与向量库构建 (ETL)
│   ├── rag_chain.py     # 检索与 Rerank 逻辑
│   ├── tools.py         # Agent 工具定义 (Search, Calc)
│   ├── agent.py         # Agent 核心编排 (LangGraph/LangChain)
│   └── train.py         # LoRA 微调脚本
├── pyproject.toml       # 依赖管理配置
├── README.md            # 项目文档
└── setup_check.py       # 环境自检脚本
📅 开发路线图 (Roadmap)我们目前处于 Phase 1 阶段。[ ] Phase 1: RAG Pipeline (Data Ingestion)[ ] 实现 SEC 10-K/10-Q 文档自动下载器 (src/ingestion.py)[ ] 文档分块与清洗 (Unstructured / LlamaParse)[ ] 向量数据库构建 (FAISS/Chroma + BGE Embeddings)[ ] 基础检索测试[ ] Phase 2: Agent Construction[ ] 定义工具集: Tavily Search, Python REPL[ ] 实现 ReAct Agent 逻辑[ ] 接入 RAG 作为 Agent 的一个 Tool[ ] Phase 3: Fine-tuning (LoRA)[ ] 构造 Tool-Use 指令数据集[ ] QLoRA 微调 Qwen2.5-7B[ ] 评估微调后模型在 Function Calling 上的准确率[ ] Phase 4: Evaluation & Demo[ ] RAGAS 自动化评估[ ] Streamlit/Gradio 演示界面🤝 协作指南 (Gemini CLI)本 README 将作为 Gemini CLI 的上下文锚点。在后续对话中，你可以直接引用本文件中的任务阶段（如 "开始 Phase 1 的文档下载功能"），无需重复背景信息。Let's build something amazing! 🚀