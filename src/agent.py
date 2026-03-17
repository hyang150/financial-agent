"""
Finance Agent: LangGraph ReAct Agent with Tool Orchestration
"""
import os
from typing import Optional, Dict, Any, List

# --- LangGraph & Modern LangChain Imports ---
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline
# --------------------------------------------

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv

from src.tools import create_tools
from src.rag_chain import create_rag_chain
from config.model_config import ModelConfig, AgentConfig, RAGConfig, EmbeddingConfig

# Load environment variables
load_dotenv()

# In LangGraph, the System Prompt acts as a state modifier.
# We no longer need complex Thought/Action parsing templates.
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
    Main Finance Agent using LangGraph for robust state-based reasoning
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
        Initialize the LangGraph-based Finance Agent

        Args:
            model_config: LLM model configuration
            agent_config: Agent behavior configuration
            rag_config: RAG pipeline configuration
            embedding_config: Embedding model configuration
            vector_db_path: Path to vector database
        """
        self.model_config = model_config or ModelConfig()
        self.agent_config = agent_config or AgentConfig()
        self.rag_config = rag_config or RAGConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.vector_db_path = vector_db_path

        # Initialize components
        print("🚀 Initializing Finance Agent with LangGraph...")
        self._init_llm()
        self._init_rag()
        self._init_tools()
        self._init_agent()

        print("✅ Agent initialized successfully!")

    def _init_llm(self):
        """Initialize the Language Model and wrap it for Tool Calling"""
        print(f"📦 Loading LLM: {self.model_config.model_name}...")

        # Check if model should use GPU
        device = self.model_config.device if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("⚠️ CUDA not available, using CPU (this will be slow)")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=True
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Create base pipeline
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

        # Wrap in LangChain Pipeline
        base_llm = HuggingFacePipeline(pipeline=pipe)
        
        # KEY CHANGE: Wrap the base LLM into a ChatModel to support native Tool Calling capabilities required by LangGraph
        self.chat_model = ChatHuggingFace(llm=base_llm)
        print(f"✅ LLM loaded and wrapped as ChatModel on {device}")

    def _init_rag(self):
        """Initialize RAG chain"""
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
        """Initialize agent tools"""
        print("🔧 Setting up tools...")

        self.tools = create_tools(
            rag_chain=self.rag_chain,
            allow_dangerous_code=self.agent_config.allow_dangerous_code
        )

        print(f"✅ {len(self.tools)} tools ready: {[t.name for t in self.tools]}")

    def _init_agent(self):
        """Initialize LangGraph ReAct Agent"""
        print("🤖 Creating LangGraph ReAct agent...")

        # KEY CHANGE: One-line graph agent creation, replacing the heavy AgentExecutor
        self.agent_executor = create_react_agent(
            model=self.chat_model,
            tools=self.tools,
            state_modifier=SYSTEM_PROMPT
        )

        print("✅ Graph Agent ready to use")

    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the agent with a question using Graph invocation

        Args:
            question: User question

        Returns:
            Dictionary with answer and metadata
        """
        print(f"\n💬 Question: {question}")
        print("🤔 Agent is thinking...\n")

        try:
            # KEY CHANGE: LangGraph inputs/outputs are standard Message arrays
            inputs = {"messages": [HumanMessage(content=question)]}
            
            # Invoke the graph
            result = self.agent_executor.invoke(inputs)
            
            # Extract final answer from the last message
            messages = result["messages"]
            final_answer = messages[-1].content
            
            # Collect tool call records for intermediate steps
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
        """Interactive chat mode"""
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
                # Get user input
                user_input = input("You: ").strip()

                # Handle commands
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

                # Query the agent
                result = self.query(user_input)

                # Display answer
                print(f"\n🤖 Agent: {result['answer']}\n")

                # Optionally show intermediate steps in verbose mode
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
        Process multiple questions in batch

        Args:
            questions: List of questions

        Returns:
            List of results
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
    Factory function to create a LangGraph Finance Agent

    Args:
        model_config: Model configuration
        agent_config: Agent configuration
        rag_config: RAG configuration
        embedding_config: Embedding configuration
        vector_db_path: Vector database path

    Returns:
        Configured Finance Agent
    """
    return FinanceAgent(
        model_config=model_config,
        agent_config=agent_config,
        rag_config=rag_config,
        embedding_config=embedding_config,
        vector_db_path=vector_db_path
    )


if __name__ == "__main__":
    # Test the agent
    print("🧪 Testing Finance Agent (LangGraph)...\n")

    try:
        # Create agent with default config
        agent = create_agent()

        # Test queries
        test_questions = [
            "What was Apple's total revenue in their latest annual report?",
            "Calculate the compound annual growth rate if revenue grew from 100M to 150M over 3 years",
        ]

        # Run batch query
        results = agent.batch_query(test_questions)

        # Display results
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