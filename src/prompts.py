"""
提示词模块
----------
集中管理 RAG 系统使用的所有提示词模板。
"""

from typing import List, Dict


def build_context_text(contexts: List[Dict]) -> str:
    """
    将检索到的上下文列表格式化为编号文本。

    Args:
        contexts: search_vector_store 返回的结果列表。

    Returns:
        格式化的上下文字符串。
    """
    if not contexts:
        return "（无检索结果）"

    parts = []
    for i, ctx in enumerate(contexts, 1):
        meta = ctx.get("metadata", {})
        content = ctx.get("content", "")

        parts.append(
            f"[{i}]\n"
            f"姓名：{meta.get('name', '')}\n"
            f"公司：{meta.get('company', '')}\n"
            f"职位：{meta.get('job_title', '')}\n"
            f"城市：{meta.get('city', '')}\n"
            f"国家：{meta.get('country', '')}\n"
            f"行业：{meta.get('industry', '')}\n"
            f"兴趣方向：{meta.get('interest_direction', '')}\n"
            f"Lead Stage：{meta.get('lead_stage', '')}\n"
            f"Lead Score：{meta.get('lead_score', '')}\n"
            f"备注：{meta.get('notes', content.split('备注：')[-1] if '备注：' in content else '')}\n"
        )

    return "\n".join(parts)


def build_rag_prompt(question: str, contexts: List[Dict]) -> str:
    """
    构建 RAG 回答的完整 Prompt。

    Args:
        question: 用户自然语言问题。
        contexts: 向量检索返回的上下文列表。

    Returns:
        完整的 prompt 字符串，可直接传给 call_llm。
    """
    context_text = build_context_text(contexts)

    prompt = f"""你是一个参会者销售线索分析助手。
你只能根据给定的参会者数据回答问题。
不要编造不存在的参会者、公司、职位、城市或备注。
如果上下文中没有足够信息，请回答：
"根据当前上传的 md.csv 数据，未找到足够依据。"

回答要求：
1. 如果问题是名单筛选类，请用 Markdown 表格回答；
2. 表格字段优先包括：姓名、公司、职位、城市、Lead Score、Lead Stage、依据；
3. 如果问题需要解释，请说明筛选依据；
4. 如果涉及销售线索，请给出推荐理由；
5. 不要输出无关背景知识。

## 参会者数据

{context_text}

## 用户问题

{question}

## 你的回答"""
    return prompt


# ---- 以下为历史遗留的旧模板（保留兼容） ----

SYSTEM_PROMPT = """你是一个专业的参会者信息查询助手。你的任务是根据提供的参会者数据，准确回答用户的问题。

## 你的能力
- 查询特定参会者的详细信息（姓名、公司、职位、行业等）
- 统计参会者数据（按行业、城市、Session 等维度）
- 筛选符合特定条件的参会者（如高意向、特定行业等）
- 回答关于参会者兴趣方向和会议场次的问题

## 回答规则
1. 始终基于提供的参会者数据回答，不要编造信息。
2. 如果数据不足以回答问题，请明确告知用户。
3. 回答时列出相关的参会者姓名和关键信息。
4. 保持回答简洁、结构清晰。
5. 使用中文回答。
"""

RAG_QA_PROMPT = """请根据以下参会者数据，回答用户的问题。

## 参会者数据
{context}

## 用户问题
{query}

## 回答要求
- 基于以上数据回答，不要编造。
- 如果数据不足以回答问题，请说明。
- 回答简洁、清晰，使用中文。
"""

SUMMARY_PROMPT = """请对以下参会者数据进行简要总结。

## 数据概览
- 总人数: {total}
- 行业分布: {industries}
- 城市分布: {cities}
- Session 分布: {sessions}

请用 2-3 句话概括这批参会者的整体特征。
"""

ERROR_FILE_NOT_FOUND = "❌ 文件未找到，请先上传 CSV 文件。"
ERROR_CSV_PARSE = "❌ CSV 文件解析失败，请检查文件格式。"
ERROR_NO_DATA = "⚠️ 数据库中暂无数据，请先上传 CSV 文件。"
ERROR_DB_INIT = "❌ 数据库初始化失败，请检查数据目录权限。"
