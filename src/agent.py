"""
金融分析 Agent 模块。

基于 LangGraph 的 `create_react_agent` 编排 HuggingFace 聊天模型与多工具（RAG、搜索、计算、REPL），
提供 `query` / `chat` / `batch_query` 等对外接口。
"""
import os
from typing import Optional, Dict, Any, List

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv

from src.tools import create_tools
from src.rag_chain import create_rag_chain
from config.model_config import ModelConfig, AgentConfig, RAGConfig, EmbeddingConfig

load_dotenv()

SYSTEM_PROMPT = """You are a highly capable financial analysis assistant. 
You have access to a suite of tools to answer questions about companies, market events, and financial reports.

IMPORTANT GUIDELINES:
1. For specific financial data, SEC filings, or historical reports, ALWAYS use 'rag_search' first.
2. For current news, recent market events, or real-time information, use 'web_search'.
3. For calculations (e.g., margins, growth rates), use 'calculator' for simple math, and 'python_repl' for complex operations.
4. Always provide specific numbers, metrics, and cite your sources when possible.
5. If you cannot find the necessary information after using the tools, state clearly that you don't have enough data to answer accurately.
"""


class FinanceAgent:
    """
    金融 Agent：封装 LLM、RAG、工具列表与 LangGraph 执行图。
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        agent_config: Optional[AgentConfig] = None,
        rag_config: Optional[RAGConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        vector_db_path: str = "data/vector_db"
    ):
        """
        依次初始化语言模型、RAG、工具与 ReAct 图。

        参数:
            model_config: HuggingFace 因果语言模型与生成参数。
            agent_config: 交互与 REPL 安全等 Agent 行为配置。
            rag_config: 检索与重排配置。
            embedding_config: 向量嵌入配置。
            vector_db_path: Chroma 持久化目录；不存在时 RAG 会失败并降级为无 rag_search。
        """
        self.model_config = model_config or ModelConfig()
        self.agent_config = agent_config or AgentConfig()
        self.rag_config = rag_config or RAGConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.vector_db_path = vector_db_path

        print("🚀 Initializing Finance Agent with LangGraph...")
        self._init_llm()
        self._init_rag()
        self._init_tools()
        self._init_agent()

        print("✅ Agent initialized successfully!")

    def _init_llm(self):
        """加载 Tokenizer 与因果 LM，构建 text-generation pipeline 并包装为 ChatHuggingFace。"""
        print(f"📦 Loading LLM: {self.model_config.model_name}...")

        device = self.model_config.device if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("⚠️ CUDA not available, using CPU (this will be slow)")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.model_config.max_new_tokens,
            temperature=self.model_config.temperature,
            top_p=self.model_config.top_p,
            do_sample=self.model_config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        base_llm = HuggingFacePipeline(pipeline=pipe)
        self.chat_model = ChatHuggingFace(llm=base_llm)
        print(f"✅ LLM loaded and wrapped as ChatModel on {device}")

    def _init_rag(self):
        """创建 RAG 链；失败时打印警告并将 rag_chain 置为 None。"""
        print("📚 Initializing RAG system...")

        try:
            self.rag_chain = create_rag_chain(
                vector_db_path=self.vector_db_path,
                embedding_config=self.embedding_config,
                rag_config=self.rag_config
            )
            print("✅ RAG system ready")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize RAG system: {e}")
            print("   RAG tool will not be available. Run ingestion.py first.")
            self.rag_chain = None

    def _init_tools(self):
        """根据当前 rag_chain 与 agent_config 调用 create_tools 构建工具列表。"""
        print("🔧 Setting up tools...")

        self.tools = create_tools(
            rag_chain=self.rag_chain,
            allow_dangerous_code=self.agent_config.allow_dangerous_code
        )

        print(f"✅ {len(self.tools)} tools ready: {[t.name for t in self.tools]}")

    def _init_agent(self):
        """使用 LangGraph 预置 ReAct 图绑定 chat_model、tools 与系统提示。"""
        print("🤖 Creating LangGraph ReAct agent...")

        self.agent_executor = create_react_agent(
            model=self.chat_model,
            tools=self.tools,
            state_modifier=SYSTEM_PROMPT
        )

        print("✅ Graph Agent ready to use")

    def query(self, question: str) -> Dict[str, Any]:
        """
        对用户问题执行一轮图调用，返回最终回复与工具调用摘要字符串列表。

        参数:
            question: 用户自然语言问题。

        返回:
            字典包含 question、answer、intermediate_steps（每项为描述字符串）、success。
        """
        print(f"\n💬 Question: {question}")
        print("🤔 Agent is thinking...\n")

        try:
            inputs = {"messages": [HumanMessage(content=question)]}

            result = self.agent_executor.invoke(inputs)

            messages = result["messages"]
            final_answer = messages[-1].content

            intermediate_steps = [
                f"Used Tool: {m.name} (Output: {m.content[:100]}...)"
                for m in messages if m.type == "tool"
            ]

            return {
                "question": question,
                "answer": final_answer,
                "intermediate_steps": intermediate_steps,
                "success": True
            }

        except Exception as e:
            print(f"❌ Error during execution: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "intermediate_steps": [],
                "success": False
            }

    def chat(self):
        """在终端循环读取用户输入，调用 query 并依 verbose 配置打印中间步骤。"""
        print("\n" + "="*60)
        print("💼 Finance Agent (LangGraph) - Interactive Mode")
        print("="*60)
        print("\nAvailable commands:")
        print("  - Type your question to get an answer")
        print("  - Type 'quit' or 'exit' to end the session")
        print("  - Type 'tools' to see available tools")
        print("\n" + "="*60 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == 'tools':
                    print("\n🔧 Available Tools:")
                    for i, tool in enumerate(self.tools, 1):
                        print(f"\n{i}. {tool.name}")
                        print(f"   {tool.description}")
                    print()
                    continue

                if not user_input:
                    continue

                result = self.query(user_input)

                print(f"\n🤖 Agent: {result['answer']}\n")

                if self.agent_config.verbose and result['intermediate_steps']:
                    print("\n📝 Reasoning Steps (Tools Used):")
                    for i, step in enumerate(result['intermediate_steps'], 1):
                        print(f"  Step {i}: {step}")
                    print()

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        顺序对多个问题调用 query，用于脚本或测试。

        参数:
            questions: 问题字符串列表。

        返回:
            与 `query` 返回结构相同的字典列表。
        """
        results = []

        print(f"\n📋 Processing {len(questions)} questions...\n")

        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}]")
            result = self.query(question)
            results.append(result)
            print()

        return results


def create_agent(
    model_config: Optional[ModelConfig] = None,
    agent_config: Optional[AgentConfig] = None,
    rag_config: Optional[RAGConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None,
    vector_db_path: str = "data/vector_db"
) -> FinanceAgent:
    """
    工厂函数：构造 `FinanceAgent` 实例。

    参数:
        model_config / agent_config / rag_config / embedding_config: 各子模块配置。
        vector_db_path: 向量库路径。

    返回:
        初始化完成的 FinanceAgent。
    """
    return FinanceAgent(
        model_config=model_config,
        agent_config=agent_config,
        rag_config=rag_config,
        embedding_config=embedding_config,
        vector_db_path=vector_db_path
    )


if __name__ == "__main__":
    print("🧪 Testing Finance Agent (LangGraph)...\n")

    try:
        agent = create_agent()

        test_questions = [
            "What was Apple's total revenue in their latest annual report?",
            "Calculate the compound annual growth rate if revenue grew from 100M to 150M over 3 years",
        ]

        results = agent.batch_query(test_questions)

        print("\n" + "="*60)
        print("📊 Results Summary")
        print("="*60)

        for i, result in enumerate(results, 1):
            print(f"\nQ{i}: {result['question']}")
            print(f"A{i}: {result['answer'][:200]}...")
            print(f"Success: {result['success']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Tips:")
        print("  1. Make sure you've run ingestion.py first to build the vector DB.")
        print("  2. Check that your .env file has HF_TOKEN set.")
        print("  3. Ensure you have enough GPU memory, or let it fallback to CPU.")
