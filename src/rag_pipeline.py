"""
RAG Pipeline 模块
-----------------
负责检索增强生成的主流程：向量检索 -> 构建上下文 -> 调用 LLM 生成回答。
"""

from typing import List, Dict

from .config import TOP_K_RETRIEVAL
from .vector_store import search_vector_store
from .prompts import build_rag_prompt
from .llm_client import call_llm


def answer_with_rag(
    question: str,
    persist_dir: str,
    top_k: int = TOP_K_RETRIEVAL,
) -> Dict:
    """
    完整的 RAG 问答流程：
    1. 向量检索获取相关上下文
    2. 构建 RAG Prompt
    3. 调用 LLM 生成回答

    Args:
        question: 用户自然语言问题。
        persist_dir: Chroma 向量库持久化目录。
        top_k: 检索返回的最大文档数，默认 20。

    Returns:
        {
            "answer": str,       # LLM 生成的回答
            "contexts": list[dict]  # 检索到的原始上下文（含 metadata、content、distance）
        }
    """
    # Step 1: 向量检索
    try:
        contexts = search_vector_store(question, persist_dir, top_k=top_k)
    except FileNotFoundError as e:
        return {
            "answer": f"向量库未找到：{e}\n请先构建向量索引。",
            "contexts": [],
        }
    except Exception as e:
        return {
            "answer": f"向量检索失败: {e}",
            "contexts": [],
        }

    # Step 2: 无结果处理
    if not contexts:
        return {
            "answer": "根据当前上传的 md.csv 数据，未找到足够依据。",
            "contexts": [],
        }

    # Step 3: 构建 Prompt
    prompt = build_rag_prompt(question, contexts)

    # Step 4: 调用 LLM
    answer = call_llm(prompt)

    return {
        "answer": answer,
        "contexts": contexts,
    }
