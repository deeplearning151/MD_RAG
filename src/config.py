"""
项目配置模块
-----------
集中管理所有配置项，支持环境变量覆盖。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------- 加载 .env 文件 ----------
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------- 项目根目录 ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- 数据目录 ----------
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# ---------- 数据库 ----------
DB_PATH = DATA_DIR / "md_rag.db"

# ---------- CSV 配置 ----------
CSV_ENCODING = "utf-8"
CSV_DELIMITER = ","

# ---------- LLM 配置 ----------
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ---------- Embedding 配置 ----------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

# ---------- RAG 配置 ----------
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "20"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ---------- Streamlit 页面配置 ----------
PAGE_TITLE = "参会者智能问答系统"
PAGE_ICON = "🤖"
PAGE_LAYOUT = "wide"
