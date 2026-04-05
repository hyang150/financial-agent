"""
数据摄取（ETL）模块。

从 SEC EDGAR 下载 10-K/10-Q 财报 HTML，经分块与 BGE 嵌入后写入 Chroma 向量库。
可作为脚本直接运行：`python src/ingestion.py`（支持 CLI，见 `main()`）。
"""
import argparse
import os

from dotenv import load_dotenv
from sec_edgar_downloader import Downloader
from langchain_community.document_loaders import BSHTMLLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_DIR = "data"
SEC_DIR = os.path.join(DATA_DIR, "sec_filings")
DB_DIR = os.path.join(DATA_DIR, "vector_db")

USER_EMAIL = os.getenv("SEC_EMAIL", "student@university.edu")
USER_NAME = "FinanceAgentProject"


def download_reports(tickers, limit_10k: int = 5, limit_10q: int = 5):
    """
    从 SEC EDGAR 批量下载指定股票代码的 10-K（年报）与 10-Q（季报）。

    参数:
        tickers: 股票代码字符串列表，例如 ["AAPL", "MSFT"]。
        limit_10k: 每只股票下载的 10-K 份数上限。
        limit_10q: 每只股票下载的 10-Q 份数上限。
    """
    print(f"🚀 初始化下载器... (User: {USER_NAME} <{USER_EMAIL}>)")
    dl = Downloader(USER_NAME, USER_EMAIL, SEC_DIR)

    for ticker in tickers:
        print(f"📥 正在下载 {ticker} 的财报... (10-K≤{limit_10k}, 10-Q≤{limit_10q})")
        dl.get("10-K", ticker, limit=limit_10k, download_details=True)
        dl.get("10-Q", ticker, limit=limit_10q, download_details=True)

    print("✅ 所有文件下载完成！")


def ingest_data():
    """
    扫描本地财报 HTML/HTM，切分文本块，用 BGE-small 嵌入并持久化到 Chroma。

    若目录下无可用文档则打印提示并返回，不抛异常。
    """
    search_path = os.path.join(SEC_DIR, "sec-edgar-filings")
    if not os.path.exists(search_path):
        search_path = SEC_DIR

    print(f"🔍 正在搜索目录: {search_path}")

    print("📂 开始加载财报文件...")

    docs = []
    for ext in ["**/*.html", "**/*.htm"]:
        loader = DirectoryLoader(
            search_path,
            glob=ext,
            loader_cls=BSHTMLLoader,
            show_progress=True,
            use_multithreading=True
        )
        loaded_docs = loader.load()
        docs.extend(loaded_docs)
        print(f"   - 匹配 {ext}: 找到 {len(loaded_docs)} 个文档")

    print(f"📄 总计加载了 {len(docs)} 个有效文档")

    if len(docs) == 0:
        print("⚠️ 错误: 未找到任何 HTML/HTM 文档。")
        print("💡 请检查 data/sec_filings 目录下是否有文件。有些旧财报可能是 .txt 格式。")
        return

    print("✂️ 正在切分文本...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"🧩 生成了 {len(splits)} 个文本块 (Chunks)")

    print("💾 正在生成向量并存入 ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )
    print(f"✅ 向量数据库构建成功！存储位置: {DB_DIR}")


def _parse_args():
    p = argparse.ArgumentParser(
        description="下载 SEC 财报并切分、嵌入写入 Chroma（默认股票可在下方修改）"
    )
    p.add_argument(
        "--tickers",
        "-t",
        nargs="+",
        default=["AAPL", "MSFT", "TSLA"],
        help="要下载的股票代码列表，例如: -t AAPL NVDA",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="跳过下载，仅扫描本地 data/sec_filings 做切分与入库",
    )
    p.add_argument(
        "--limit-10k",
        type=int,
        default=5,
        help="每只股票 10-K 下载上限（默认 5）",
    )
    p.add_argument(
        "--limit-10q",
        type=int,
        default=5,
        help="每只股票 10-Q 下载上限（默认 5）",
    )
    return p.parse_args()


if __name__ == "__main__":
    cli = _parse_args()
    if not cli.no_download:
        download_reports(
            cli.tickers,
            limit_10k=cli.limit_10k,
            limit_10q=cli.limit_10q,
        )
    ingest_data()
