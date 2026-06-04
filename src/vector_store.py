"""
向量索引模块
------------
使用 sentence-transformers 将 SQLite 数据库中的参会者数据向量化，
并存入本地 Chroma 向量数据库，支持语义检索。
"""

import sqlite3
from pathlib import Path
from typing import List, Dict


# ==================== 模型加载 ====================

_embedding_model = None


def vector_store_exists(persist_dir: str) -> bool:
    """
    判断 Chroma 向量库是否已构建并有数据。

    Args:
        persist_dir: Chroma 持久化目录路径。

    Returns:
        True 如果向量库目录存在、collection "md_attendees" 存在且至少有 1 条 document。
    """
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        return False

    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name="md_attendees")
        return collection.count() > 0
    except Exception:
        return False


def get_vector_store_count(persist_dir: str) -> int:
    """
    返回 Chroma collection "md_attendees" 中的 document 数量。

    Args:
        persist_dir: Chroma 持久化目录路径。

    Returns:
        document 数量，如果不存在或读取失败返回 0。
    """
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name="md_attendees")
        return collection.count()
    except Exception:
        return 0


def get_embedding_model():
    """
    加载本地 embedding 模型（单例缓存）。

    默认模型：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

    Returns:
        SentenceTransformer 模型对象。

    Raises:
        RuntimeError: 模型加载失败时抛出。
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    try:
        from sentence_transformers import SentenceTransformer
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        _embedding_model = SentenceTransformer(model_name)
        return _embedding_model
    except Exception as e:
        raise RuntimeError(
            f"Embedding 模型加载失败: {e}\n"
            f"请确认已安装 sentence-transformers: pip install sentence-transformers"
        )


# ==================== Document 构建 ====================

def build_documents_from_db(db_path: str) -> List[Dict]:
    """
    从 SQLite 数据库 attendees 表读取所有参会者，转换为 Document 列表。

    每个 Document 格式：
    {
        "id": "attendee_<id>",
        "content": "姓名：...\n公司：...\n...",
        "metadata": { "id": ..., "name": ..., ... }
    }

    Args:
        db_path: SQLite 数据库文件路径。

    Returns:
        Document 字典列表。如果数据库无数据，返回空列表。
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")

    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute("SELECT * FROM attendees ORDER BY id")
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    documents = []
    for row in rows:
        row_dict = dict(row)

        # 安全获取字段值
        name = str(row_dict.get("name") or "")
        company = str(row_dict.get("company") or "")
        job_title = str(row_dict.get("job_title") or "")
        city = str(row_dict.get("city") or "")
        country = str(row_dict.get("country") or "")
        industry = str(row_dict.get("industry") or "")
        interest_direction = str(row_dict.get("interest_direction") or "")
        session = str(row_dict.get("session") or "")
        lead_stage = str(row_dict.get("lead_stage") or "")
        lead_score = row_dict.get("lead_score", 0)
        email = str(row_dict.get("email") or "")
        notes = str(row_dict.get("notes") or "")

        doc_id = f"attendee_{row_dict['id']}"

        # 构建语义内容（适合向量检索）
        content = (
            f"姓名：{name}\n"
            f"公司：{company}\n"
            f"职位：{job_title}\n"
            f"城市：{city}\n"
            f"国家：{country}\n"
            f"行业：{industry}\n"
            f"兴趣方向：{interest_direction}\n"
            f"参会场次：{session}\n"
            f"Lead Stage：{lead_stage}\n"
            f"Lead Score：{lead_score}\n"
            f"邮箱：{email}\n"
            f"备注：{notes}"
        )

        metadata = {
            "id": row_dict["id"],
            "name": name,
            "company": company,
            "job_title": job_title,
            "city": city,
            "country": country,
            "industry": industry,
            "interest_direction": interest_direction,
            "lead_stage": lead_stage,
            "lead_score": lead_score,
            "email": email,
        }

        documents.append({
            "id": doc_id,
            "content": content,
            "metadata": metadata,
        })

    return documents


# ==================== 向量库构建 ====================

def build_vector_store(db_path: str, persist_dir: str) -> int:
    """
    基于数据库内容构建 Chroma 向量库并持久化。

    流程：
    1. 从数据库构建 Document 列表
    2. 使用 embedding 模型计算向量
    3. 创建/清空 Chroma collection "md_attendees"
    4. 写入 documents、ids、metadatas、embeddings

    Args:
        db_path: SQLite 数据库文件路径。
        persist_dir: Chroma 持久化目录。

    Returns:
        写入的 document 数量。

    Raises:
        ValueError: 数据库无数据时抛出。
        RuntimeError: Chroma 初始化失败时抛出。
    """
    # 构建 documents
    documents = build_documents_from_db(db_path)
    if not documents:
        raise ValueError("数据库中没有参会者数据，无法构建向量索引。请先上传 CSV 并写入数据库。")

    # 加载 embedding 模型
    model = get_embedding_model()

    # 提取 texts 和 ids
    texts = [doc["content"] for doc in documents]
    ids = [doc["id"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    # 计算 embeddings
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    # 初始化 Chroma
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        raise RuntimeError("chromadb 未安装，请执行: pip install chromadb")

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(persist_path),
        settings=Settings(anonymized_telemetry=False),
    )

    # 如果 collection 已存在则删除重建
    collection_name = "md_attendees"
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # collection 不存在，忽略

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "MD_RAG 参会者向量索引"},
    )

    # 批量写入
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        collection.add(
            ids=ids[i:batch_end],
            documents=texts[i:batch_end],
            metadatas=metadatas[i:batch_end],
            embeddings=embeddings[i:batch_end],
        )

    return len(documents)


# ==================== 向量检索 ====================

def search_vector_store(
    query: str,
    persist_dir: str,
    top_k: int = 20,
) -> List[Dict]:
    """
    从 Chroma 向量库中检索与 query 最相关的参会者。

    Args:
        query: 检索查询文本。
        persist_dir: Chroma 持久化目录。
        top_k: 返回的最大结果数，默认 20。

    Returns:
        结果列表，每个元素为 dict:
        {
            "content": str,
            "metadata": dict,
            "distance": float,
        }
        如果向量库不存在或无结果，返回空列表。
    """
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"向量库目录不存在: {persist_dir}\n请先点击「构建向量索引」。"
        )

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        raise RuntimeError("chromadb 未安装，请执行: pip install chromadb")

    client = chromadb.PersistentClient(
        path=str(persist_path),
        settings=Settings(anonymized_telemetry=False),
    )

    collection_name = "md_attendees"
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        raise FileNotFoundError(
            f"向量 collection '{collection_name}' 不存在。\n请先点击「构建向量索引」。"
        )

    # 用 embedding 模型编码 query
    model = get_embedding_model()
    query_embedding = model.encode([query], show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"] or not results["ids"][0]:
        return []

    formatted = []
    ids_list = results["ids"][0]
    docs_list = results["documents"][0] if results["documents"] else []
    metas_list = results["metadatas"][0] if results["metadatas"] else []
    dists_list = results["distances"][0] if results["distances"] else []

    for i in range(len(ids_list)):
        item = {
            "content": docs_list[i] if i < len(docs_list) else "",
            "metadata": metas_list[i] if i < len(metas_list) else {},
            "distance": dists_list[i] if i < len(dists_list) else 0.0,
        }
        formatted.append(item)

    return formatted
