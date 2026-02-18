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
# 请务必修改这里，或者在 .env 中设置 SEC_EMAIL
USER_EMAIL = os.getenv("SEC_EMAIL", "student@university.edu") 
USER_NAME = "FinanceAgentProject"

def download_reports(tickers):
    """
    从 SEC EDGAR 下载 10-K (年报) 和 10-Q (季报)
    """
    print(f"🚀 初始化下载器... (User: {USER_NAME} <{USER_EMAIL}>)")
    dl = Downloader(USER_NAME, USER_EMAIL, SEC_DIR)
    
    for ticker in tickers:
        print(f"📥 正在下载 {ticker} 的财报...")
        # 限制数量以加快演示速度 (1份年报，1份季报)
        # 实际项目中可以将 limit 设为 5 或更多
        dl.get("10-K", ticker, limit=1)
        dl.get("10-Q", ticker, limit=1)
    
    print("✅ 所有文件下载完成！")

def ingest_data():
    """
    读取 HTML -> 清洗 -> 切块 -> 向量化 -> 存入 ChromaDB
    """
    # 1. 加载数据
    print("📂 开始加载 HTML 文件...")
    # SEC 下载的是 HTML 格式，我们用 BSHTMLLoader 提取纯文本
    loader = DirectoryLoader(
        SEC_DIR,
        glob="**/*.html",
        loader_cls=BSHTMLLoader,
        show_progress=True,
        use_multithreading=True
    )
    docs = loader.load()
    print(f"📄 加载了 {len(docs)} 个文档")

    if len(docs) == 0:
        print("⚠️ 未找到文档，请先运行下载步骤！")
        return

    # 2. 文本切块 (Chunking)
    print("✂️ 正在切分文本...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # 每个块约 1000 字符
        chunk_overlap=100,     # 重叠 100 字符，防止上下文丢失
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"🧩 生成了 {len(splits)} 个文本块 (Chunks)")

    # 3. 向量化与存储 (Embedding & Storage)
    print("💾 正在生成向量并存入 ChromaDB (这可能需要几分钟)...")
    
    # 使用 BGE-Small 模型，效果好且速度快
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}, # 如果有 GPU 改为 'cuda'
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 创建并持久化向量库
    if os.path.exists(DB_DIR):
        print("   (检测到已有数据库，正在追加数据...)")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )
    print(f"✅ 向量数据库构建成功！存储位置: {DB_DIR}")

if __name__ == "__main__":
    # 目标公司：Apple, Microsoft, Tesla
    target_tickers = ["AAPL", "MSFT", "TSLA"]
    
    # --- 步骤开关 ---
    # 第一次运行时，请确保 download=True
    # 之后如果只想重新生成数据库，可以设为 False
    RUN_DOWNLOAD = True
    
    if RUN_DOWNLOAD:
        download_reports(target_tickers)
    
    ingest_data()