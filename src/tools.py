"""
Agent Tools: Search, Calculator, Python REPL, and RAG
"""
import os
import re
from typing import Optional, Type, Dict, Any
from langchain.tools import BaseTool, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class CalculatorInput(BaseModel):
    """Input schema for Calculator tool"""
    expression: str = Field(description="Mathematical expression to evaluate, e.g., '2 + 2' or '100 * 1.05'")


class Calculator(BaseTool):
    """
    Calculator tool for basic arithmetic operations.
    Safer alternative to Python REPL for simple calculations.
    """
    name: str = "calculator"
    description: str = """
    Useful for performing mathematical calculations and arithmetic operations.
    Input should be a valid mathematical expression like '2 + 2', '100 * 1.05', or '(1000 - 500) / 2'.
    Supports +, -, *, /, **, (), and basic math functions.
    """
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """Execute the calculation"""
        try:
            # Security: only allow safe mathematical operations
            # Remove any dangerous characters/keywords
            dangerous_patterns = ['import', 'exec', 'eval', '__', 'open', 'file', 'os', 'sys']
            for pattern in dangerous_patterns:
                if pattern in expression.lower():
                    return f"Error: Expression contains forbidden keyword: {pattern}"

            # Evaluate safely with limited builtins
            allowed_names = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow, 'len': len
            }

            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Result: {result}"

        except Exception as e:
            return f"Calculation error: {str(e)}"

    async def _arun(self, expression: str) -> str:
        """Async version"""
        return self._run(expression)


class PythonREPLInput(BaseModel):
    """Input schema for Python REPL tool"""
    code: str = Field(description="Valid Python code to execute. Can include multiple lines and import statements.")


class SafePythonREPL(BaseTool):
    """
    Python REPL tool for executing code.
    Useful for complex calculations, data processing, and analysis.
    """
    name: str = "python_repl"
    description: str = """
    A Python shell for executing Python code. Use this for:
    - Complex calculations and data analysis
    - Working with numbers, lists, and dictionaries
    - Data transformations and processing
    - Financial computations (NPV, IRR, growth rates, etc.)

    Input should be valid Python code. The tool will execute it and return the output.
    You can use print() to display results.

    Example inputs:
    - "print(sum([100, 200, 300]))"
    - "revenue = [1000, 1200, 1500]\ngrowth = [(revenue[i]/revenue[i-1] - 1)*100 for i in range(1, len(revenue))]\nprint(f'Growth rates: {growth}')"
    """
    args_schema: Type[BaseModel] = PythonREPLInput
    python_repl: PythonREPL = Field(default_factory=PythonREPL)
    allow_dangerous: bool = False

    def __init__(self, allow_dangerous: bool = False, **kwargs):
        """Initialize with safety flag"""
        super().__init__(**kwargs)
        self.allow_dangerous = allow_dangerous
        self.python_repl = PythonREPL()

    def _run(self, code: str) -> str:
        """Execute Python code"""
        if not self.allow_dangerous:
            # Check for dangerous operations
            dangerous_patterns = [
                r'\bos\.',
                r'\bsys\.',
                r'\bsubprocess\.',
                r'\beval\(',
                r'\bexec\(',
                r'\b__import__\(',
                r'\bopen\(',
                r'\bfile\(',
            ]

            for pattern in dangerous_patterns:
                if re.search(pattern, code):
                    return f"Error: Code contains potentially dangerous operation: {pattern}"

        try:
            result = self.python_repl.run(code)
            return result if result else "Code executed successfully (no output)"
        except Exception as e:
            return f"Execution error: {str(e)}"

    async def _arun(self, code: str) -> str:
        """Async version"""
        return self._run(code)


class WebSearchInput(BaseModel):
    """Input schema for Web Search tool"""
    query: str = Field(description="Search query to find information on the web")


class WebSearchTool(BaseTool):
    """
    Web search tool using Tavily API.
    Useful for finding up-to-date information not in the knowledge base.
    """
    name: str = "web_search"
    description: str = """
    Search the web for current information, news, and data not available in the financial reports.
    Use this when you need:
    - Recent news or events
    - Current stock prices or market data
    - Information about events after the report dates
    - General knowledge not in SEC filings

    Input should be a clear search query.
    Returns a summary of search results with sources.
    """
    args_schema: Type[BaseModel] = WebSearchInput
    tavily_search: Optional[TavilySearchResults] = None

    def __init__(self, **kwargs):
        """Initialize with Tavily API key"""
        super().__init__(**kwargs)

        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            print("⚠️ Warning: TAVILY_API_KEY not found. Web search will not work.")
            self.tavily_search = None
        else:
            self.tavily_search = TavilySearchResults(
                api_key=tavily_api_key,
                max_results=5
            )

    def _run(self, query: str) -> str:
        """Execute web search"""
        if not self.tavily_search:
            return "Error: Web search is not configured. Please set TAVILY_API_KEY in .env file."

        try:
            results = self.tavily_search.invoke(query)

            if not results:
                return "No results found."

            # Format results
            formatted_results = ["Web Search Results:\n"]
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                content = result.get('content', 'No content')
                url = result.get('url', 'No URL')

                formatted_results.append(f"[{i}] {title}")
                formatted_results.append(f"    {content}")
                formatted_results.append(f"    Source: {url}\n")

            return '\n'.join(formatted_results)

        except Exception as e:
            return f"Search error: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)


class RAGToolInput(BaseModel):
    """Input schema for RAG tool"""
    query: str = Field(description="Question to search in the financial reports database")


class RAGTool(BaseTool):
    """
    RAG tool for searching financial reports in the vector database.
    This tool has access to SEC 10-K and 10-Q filings.
    """
    name: str = "rag_search"
    description: str = """
    Search through SEC financial reports (10-K annual reports and 10-Q quarterly reports) in the database.
    Use this tool when you need information from company financial statements such as:
    - Revenue, profit, or other financial metrics
    - Business operations and strategy
    - Risk factors
    - Management discussion and analysis
    - Historical financial performance

    Input should be a specific question about the financial reports.
    Returns relevant excerpts from the reports with source attribution.
    """
    args_schema: Type[BaseModel] = RAGToolInput
    rag_chain: Any = None  # Will be injected when creating tools

    def _run(self, query: str) -> str:
        """Execute RAG search"""
        if not self.rag_chain:
            return "Error: RAG system is not initialized. Please run ingestion.py first."

        try:
            # Get context from RAG chain
            result = self.rag_chain.get_context(query)

            # Format response
            response = [f"Found {result['num_documents']} relevant document(s):\n"]

            for source in result['sources']:
                response.append(f"[{source['rank']}] {source['source']}")
                if source['relevance_score'] != 'N/A':
                    response.append(f"    Relevance: {source['relevance_score']:.4f}")

            response.append(f"\nContext:\n{result['context']}")

            return '\n'.join(response)

        except Exception as e:
            return f"RAG search error: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)


def create_tools(
    rag_chain: Any = None,
    allow_dangerous_code: bool = False
) -> list:
    """
    Create all agent tools

    Args:
        rag_chain: RAG chain instance for document search
        allow_dangerous_code: Whether to allow potentially dangerous code execution

    Returns:
        List of configured tools
    """
    tools = []

    # Always include calculator
    tools.append(Calculator())

    # Python REPL (with safety flag)
    tools.append(SafePythonREPL(allow_dangerous=allow_dangerous_code))

    # Web search (if API key is available)
    web_search = WebSearchTool()
    if web_search.tavily_search:
        tools.append(web_search)
    else:
        print("ℹ️ Web search tool skipped (no API key)")

    # RAG tool (if rag_chain is provided)
    if rag_chain:
        rag_tool = RAGTool()
        rag_tool.rag_chain = rag_chain
        tools.append(rag_tool)
    else:
        print("ℹ️ RAG tool skipped (no RAG chain provided)")

    return tools


def get_tool_descriptions(tools: list) -> str:
    """
    Get formatted descriptions of all tools

    Args:
        tools: List of tools

    Returns:
        Formatted string describing all tools
    """
    descriptions = ["Available Tools:\n"]

    for i, tool in enumerate(tools, 1):
        descriptions.append(f"{i}. {tool.name}: {tool.description}\n")

    return '\n'.join(descriptions)


if __name__ == "__main__":
    # Test tools
    print("🧪 Testing Agent Tools...\n")

    # Test Calculator
    print("1️⃣ Testing Calculator:")
    calc = Calculator()
    print(f"   2 + 2 = {calc._run('2 + 2')}")
    print(f"   (100 * 1.05) ** 2 = {calc._run('(100 * 1.05) ** 2')}\n")

    # Test Python REPL
    print("2️⃣ Testing Python REPL:")
    repl = SafePythonREPL(allow_dangerous=False)
    code = """
revenue = [1000, 1200, 1500, 1800]
growth = [(revenue[i]/revenue[i-1] - 1) * 100 for i in range(1, len(revenue))]
print(f"Revenue: {revenue}")
print(f"Growth rates (%): {[f'{g:.1f}' for g in growth]}")
    """
    print(f"   {repl._run(code)}\n")

    # Test Web Search
    print("3️⃣ Testing Web Search:")
    search = WebSearchTool()
    if search.tavily_search:
        result = search._run("latest Apple stock price")
        print(f"   {result[:200]}...\n")
    else:
        print("   Skipped (no API key)\n")

    print("✅ Tool testing complete!")
