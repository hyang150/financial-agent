import os
from sec_edgar_downloader import Downloader
from langchain_community.document_loaders import BSHTMLLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil

# --- 配置部分 ---
DATA_DIR = "data"
SEC_DIR = os.path.join(DATA_DIR, "sec_filings")
DB_DIR = os.path.join(DATA_DIR, "vector_db")

# SEC 要求下载时提供 User-Agent (格式: "Name email@domain.com")
USER_EMAIL = os.getenv("SEC_EMAIL", "student@university.edu") 
USER_NAME = "FinanceAgentProject"

def download_reports(tickers):
    """
    从 SEC EDGAR 下载 10-K (年报) 和 10-Q (季报)
    """
    print(f"🚀 初始化下载器... (User: {USER_NAME} <{USER_EMAIL}>)")
    # 注意：v5.x 版本的 sec-edgar-downloader 会在 SEC_DIR 下创建 'sec-edgar-filings' 文件夹
    dl = Downloader(USER_NAME, USER_EMAIL, SEC_DIR)
    
    for ticker in tickers:
        print(f"📥 正在下载 {ticker} 的财报...")
        # 限制数量以加快演示速度 (1份年报，1份季报)
        # 实际项目中可以将 limit 设为 5 或更多
        dl.get("10-K", ticker, limit=5, download_details=True)
        dl.get("10-Q", ticker, limit=5, download_details=True)
    
    print("✅ 所有文件下载完成！")

def ingest_data():
    """
    读取 HTML/HTM -> 清洗 -> 切块 -> 向量化 -> 存入 ChromaDB
    """
    # 检查下载路径是否存在
    # sec-edgar-downloader 通常会生成这个子目录
    search_path = os.path.join(SEC_DIR, "sec-edgar-filings")
    if not os.path.exists(search_path):
        search_path = SEC_DIR
    
    print(f"🔍 正在搜索目录: {search_path}")

    # 1. 加载数据 - 优化 glob 以包含 .htm 和 .html
    print("📂 开始加载财报文件...")
    
    # 尝试加载多种后缀
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

    # 2. 文本切块 (Chunking)
    print("✂️ 正在切分文本...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150, # 略微增加重叠度以保持上下文
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"🧩 生成了 {len(splits)} 个文本块 (Chunks)")

    # 3. 向量化与存储
    print("💾 正在生成向量并存入 ChromaDB...")
    
    # 使用 BGE-Small 模型
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 清理旧的数据库以防冲突 (可选)
    # if os.path.exists(DB_DIR):
    #     shutil.rmtree(DB_DIR)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )
    print(f"✅ 向量数据库构建成功！存储位置: {DB_DIR}")

if __name__ == "__main__":
    target_tickers = ["AAPL", "MSFT", "TSLA"]
    
    # 如果已经下载过，可以将 RUN_DOWNLOAD 设为 False 以节省时间
    RUN_DOWNLOAD = True 
    
    if RUN_DOWNLOAD:
        download_reports(target_tickers)
    
    ingest_data()