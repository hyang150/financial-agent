#!/usr/bin/env python3
"""
Finance Agent - Main Entry Point
Interactive CLI for financial analysis with RAG and Tool-Use capabilities
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from src.agent import create_agent
from src.rag_chain import create_rag_chain
from config.model_config import ModelConfig, AgentConfig, RAGConfig, EmbeddingConfig


def print_banner():
    """Print application banner"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║           🏦 FINANCE AGENT - AI Assistant 🤖             ║
║                                                           ║
║   Multi-Tool RAG System for Financial Report Analysis    ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_prerequisites():
    """Check if basic setup is complete"""
    issues = []

    # Check .env file
    if not Path(".env").exists():
        issues.append("❌ .env file not found. Copy .env.example to .env and configure it.")

    # Check HF token
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token or hf_token.startswith("hf_xxx"):
        issues.append("❌ HF_TOKEN not configured in .env file")

    # Check vector database
    if not Path("data/vector_db").exists():
        issues.append("⚠️  Vector database not found. Run 'python src/ingestion.py' first.")

    if issues:
        print("⚠️  Setup Issues Detected:\n")
        for issue in issues:
            print(f"   {issue}")
        print("\n💡 Run 'python setup_check.py' for detailed diagnostics\n")
        return False

    return True


def run_interactive_mode(args):
    """Run agent in interactive chat mode"""
    print("\n🚀 Initializing agent (this may take a minute)...\n")

    try:
        # Configure agent
        model_config = ModelConfig(
            model_name=args.model,
            device=args.device,
            temperature=args.temperature,
        )

        agent_config = AgentConfig(
            verbose=args.verbose,
            allow_dangerous_code=args.allow_code,
        )

        # Create agent
        agent = create_agent(
            model_config=model_config,
            agent_config=agent_config,
            vector_db_path=args.db_path
        )

        # Start interactive chat
        agent.chat()

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Run 'python setup_check.py' to verify setup")
        print("   2. Make sure you have run 'python src/ingestion.py'")
        print("   3. Check that your .env file has valid API keys")
        sys.exit(1)


def run_single_query(args):
    """Run a single query and exit"""
    print("\n🚀 Initializing agent...\n")

    try:
        # Configure agent
        model_config = ModelConfig(
            model_name=args.model,
            device=args.device,
            temperature=args.temperature,
        )

        agent_config = AgentConfig(
            verbose=args.verbose,
            allow_dangerous_code=args.allow_code,
        )

        # Create agent
        agent = create_agent(
            model_config=model_config,
            agent_config=agent_config,
            vector_db_path=args.db_path
        )

        # Execute query
        result = agent.query(args.query)

        # Display result
        print(f"\n{'='*60}")
        print(f"Question: {result['question']}")
        print(f"{'='*60}")
        print(f"\nAnswer:\n{result['answer']}\n")

        if args.verbose and result['intermediate_steps']:
            print(f"\n{'='*60}")
            print("Reasoning Steps:")
            print(f"{'='*60}")
            for i, (action, observation) in enumerate(result['intermediate_steps'], 1):
                print(f"\nStep {i}:")
                print(f"  Tool: {action.tool}")
                print(f"  Input: {action.tool_input}")
                print(f"  Output: {str(observation)[:200]}...")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def run_rag_only(args):
    """Run RAG search only (no agent)"""
    print("\n🔍 Running RAG search...\n")

    try:
        # Create RAG chain
        rag = create_rag_chain(
            vector_db_path=args.db_path,
            embedding_config=EmbeddingConfig(),
            rag_config=RAGConfig()
        )

        # Execute search
        result = rag.get_context(args.query)

        # Display results
        print(f"Query: {args.query}")
        print(f"\n{'='*60}")
        print(f"Retrieved {result['num_documents']} documents")
        print(f"{'='*60}\n")

        for source in result['sources']:
            print(f"[{source['rank']}] {source['source']}")
            if source['relevance_score'] != 'N/A':
                print(f"    Relevance Score: {source['relevance_score']:.4f}")
            print(f"    Preview: {source['content_preview']}\n")

        if args.verbose:
            print(f"\n{'='*60}")
            print("Full Context:")
            print(f"{'='*60}\n")
            print(result['context'])

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you've run 'python src/ingestion.py' first")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Finance Agent - AI-powered financial analysis assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main.py

  # Single query
  python main.py --query "What was Apple's revenue in Q3 2023?"

  # RAG search only
  python main.py --rag-only --query "risk factors"

  # Use different model
  python main.py --model Qwen/Qwen2.5-14B-Instruct

  # Enable verbose mode
  python main.py --verbose

For more information, see README.md
        """
    )

    # Mode selection
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to execute (non-interactive mode)"
    )

    parser.add_argument(
        "--rag-only",
        action="store_true",
        help="Use RAG search only, without agent reasoning"
    )

    # Model configuration
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-7B-Instruct)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)"
    )

    # Agent configuration
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with reasoning steps"
    )

    parser.add_argument(
        "--allow-code",
        action="store_true",
        help="Allow execution of arbitrary Python code (use with caution)"
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="data/vector_db",
        help="Path to vector database (default: data/vector_db)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Print banner
    print_banner()

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Route to appropriate mode
    if args.rag_only:
        if not args.query:
            print("❌ Error: --rag-only requires --query")
            sys.exit(1)
        run_rag_only(args)

    elif args.query:
        run_single_query(args)

    else:
        run_interactive_mode(args)


if __name__ == "__main__":
    main()
