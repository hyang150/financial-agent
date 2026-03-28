"""
Agent 工具定义模块。

提供计算器、Python REPL、Tavily 联网搜索、RAG 检索等 LangChain Tool，
供 LangGraph ReAct Agent 绑定调用；`create_tools` 按依赖与配置组装工具列表。
"""
import os
import re
from typing import Optional, Type, Dict, Any
from langchain_core.tools import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class CalculatorInput(BaseModel):
    """计算器工具的入参模式：数学表达式字符串。"""
    expression: str = Field(description="Mathematical expression to evaluate, e.g., '2 + 2' or '100 * 1.05'")


class Calculator(BaseTool):
    """
    简易计算器工具：在安全受限环境下对算术表达式求值。
    """
    name: str = "calculator"
    description: str = """
    Useful for performing mathematical calculations and arithmetic operations.
    Input should be a valid mathematical expression like '2 + 2', '100 * 1.05', or '(1000 - 500) / 2'.
    Supports +, -, *, /, **, (), and basic math functions.
    """
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """同步执行：校验关键字后在受限命名空间中 eval 表达式。"""
        try:
            dangerous_patterns = ['import', 'exec', 'eval', '__', 'open', 'file', 'os', 'sys']
            for pattern in dangerous_patterns:
                if pattern in expression.lower():
                    return f"Error: Expression contains forbidden keyword: {pattern}"

            allowed_names = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow, 'len': len
            }

            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Result: {result}"

        except Exception as e:
            return f"Calculation error: {str(e)}"

    async def _arun(self, expression: str) -> str:
        """异步入口：委托给 `_run`。"""
        return self._run(expression)


class PythonREPLInput(BaseModel):
    """Python REPL 工具的入参：待执行的代码字符串。"""
    code: str = Field(description="Valid Python code to execute. Can include multiple lines and import statements.")


class SafePythonREPL(BaseTool):
    """
    基于 LangChain Experimental 的 Python REPL；可选正则拦截危险 API。
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
        """初始化 REPL 实例与是否允许危险代码开关。"""
        super().__init__(**kwargs)
        self.allow_dangerous = allow_dangerous
        self.python_repl = PythonREPL()

    def _run(self, code: str) -> str:
        """同步执行代码；未开启 allow_dangerous 时用正则拦截部分危险写法。"""
        if not self.allow_dangerous:
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
        """异步入口：委托给 `_run`。"""
        return self._run(code)


class WebSearchInput(BaseModel):
    """联网搜索工具的入参：搜索查询字符串。"""
    query: str = Field(description="Search query to find information on the web")


class WebSearchTool(BaseTool):
    """
    使用 Tavily API 的网页搜索工具；未配置 API Key 时 `_run` 返回错误说明。
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
        """读取环境变量中的 TAVILY_API_KEY，成功则构造 TavilySearchResults。"""
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
        """调用 Tavily 并格式化为带标题、摘要、URL 的文本。"""
        if not self.tavily_search:
            return "Error: Web search is not configured. Please set TAVILY_API_KEY in .env file."

        try:
            results = self.tavily_search.invoke(query)

            if not results:
                return "No results found."

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
        """异步入口：委托给 `_run`。"""
        return self._run(query)


class RAGToolInput(BaseModel):
    """RAG 检索工具的入参：针对财报知识库的自然语言问题。"""
    query: str = Field(description="Question to search in the financial reports database")


class RAGTool(BaseTool):
    """
    封装 AdvancedRAGChain 的检索能力；需在创建后注入 `rag_chain` 实例。
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
    rag_chain: Any = None

    def _run(self, query: str) -> str:
        """调用已注入的 RAG 链 `get_context`，拼接来源与上下文文本返回。"""
        if not self.rag_chain:
            return "Error: RAG system is not initialized. Please run ingestion.py first."

        try:
            result = self.rag_chain.get_context(query)

            response = [f"Found {result['num_documents']} relevant document(s):\n"]

            for source in result['sources']:
                response.append(f"[{source['rank']}] {source['source']}")
                rel = source['relevance_score']
                if rel != 'N/A':
                    response.append(f"    Relevance: {float(rel):.4f}")

            response.append(f"\nContext:\n{result['context']}")

            return '\n'.join(response)

        except Exception as e:
            return f"RAG search error: {str(e)}"

    async def _arun(self, query: str) -> str:
        """异步入口：委托给 `_run`。"""
        return self._run(query)


def create_tools(
    rag_chain: Any = None,
    allow_dangerous_code: bool = False
) -> list:
    """
    组装并返回当前 Agent 可用的工具列表。

    参数:
        rag_chain: 已初始化的 RAG 链；为 None 时不注册 rag_search。
        allow_dangerous_code: 是否放宽 Python REPL 的危险模式检测。

    返回:
        LangChain BaseTool 实例列表。
    """
    tools = []

    tools.append(Calculator())

    tools.append(SafePythonREPL(allow_dangerous=allow_dangerous_code))

    web_search = WebSearchTool()
    if web_search.tavily_search:
        tools.append(web_search)
    else:
        print("ℹ️ Web search tool skipped (no API key)")

    if rag_chain:
        rag_tool = RAGTool()
        rag_tool.rag_chain = rag_chain
        tools.append(rag_tool)
    else:
        print("ℹ️ RAG tool skipped (no RAG chain provided)")

    return tools


def get_tool_descriptions(tools: list) -> str:
    """
    将工具名称与 description 格式化为一段可读文本（用于调试或展示）。

    参数:
        tools: 工具对象列表。

    返回:
        多行字符串。
    """
    descriptions = ["Available Tools:\n"]

    for i, tool in enumerate(tools, 1):
        descriptions.append(f"{i}. {tool.name}: {tool.description}\n")

    return '\n'.join(descriptions)


if __name__ == "__main__":
    print("🧪 Testing Agent Tools...\n")

    print("1️⃣ Testing Calculator:")
    calc = Calculator()
    print(f"   2 + 2 = {calc._run('2 + 2')}")
    print(f"   (100 * 1.05) ** 2 = {calc._run('(100 * 1.05) ** 2')}\n")

    print("2️⃣ Testing Python REPL:")
    repl = SafePythonREPL(allow_dangerous=False)
    code = """
revenue = [1000, 1200, 1500, 1800]
growth = [(revenue[i]/revenue[i-1] - 1) * 100 for i in range(1, len(revenue))]
print(f"Revenue: {revenue}")
print(f"Growth rates (%): {[f'{g:.1f}' for g in growth]}")
    """
    print(f"   {repl._run(code)}\n")

    print("3️⃣ Testing Web Search:")
    search = WebSearchTool()
    if search.tavily_search:
        result = search._run("latest Apple stock price")
        print(f"   {result[:200]}...\n")
    else:
        print("   Skipped (no API key)\n")

    print("✅ Tool testing complete!")
