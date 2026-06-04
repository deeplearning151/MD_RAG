"""
自然语言问题路由模块
-------------------
根据用户输入的自然语言问题，判断属于 6 个基础题中的哪一个，
返回对应的路由标识符，供上层调用对应的 SQL 查询函数。
"""


def route_question(question: str) -> str:
    """
    根据问题内容路由到对应的查询类别。

    Args:
        question: 用户输入的自然语言问题字符串。

    Returns:
        路由标识符，可选值：
        - "shanghai_high_score"  : 上海地区 Lead Score > 90
        - "companies_3plus"      : 公司参会人数 >= 3
        - "rag_kb_interested"    : RAG / 企业知识库兴趣
        - "top5_industries"      : Top 5 行业统计
        - "healthcare_key_roles" : 医疗健康关键职位
        - "non_china_attendees"  : 中国以外参会者统计
        - "rag"                  : 其他（进入 RAG 流程）
    """
    q = question.strip()

    # ----- 规则 1: 上海地区 Lead Score > 90 -----
    _shanghai_kw = ["上海", "Shanghai"]
    _score_kw = ["Lead Score", "高意向", "大于90", "大于 90", ">90", "> 90"]
    if _contains_any(q, _shanghai_kw) and _contains_any(q, _score_kw):
        return "shanghai_high_score"

    # ----- 规则 2: 公司参会人数 >= 3 -----
    _company_kw = ["公司"]
    _count_kw = ["3位", "3 位", "三位", "及以上", "参会人数", "3人", "3 人", "≥3", ">=3"]
    if _contains_any(q, _company_kw) and _contains_any(q, _count_kw):
        return "companies_3plus"

    # ----- 规则 3: RAG / 企业知识库兴趣 -----
    _rag_kw = [
        "RAG", "企业知识库", "知识库", "向量检索", "Embedding",
        "文档问答", "检索增强", "知识管理",
    ]
    if _contains_any(q, _rag_kw):
        return "rag_kb_interested"

    # ----- 规则 4: Top 5 行业统计 -----
    _industry_kw = ["Industry", "行业统计", "前5", "前 5", "行业分布", "行业排名"]
    if _contains_any(q, _industry_kw):
        return "top5_industries"

    # 额外：单纯问"行业"但没有其他限定词时也命中（避免遗漏）
    if "行业" in q and not _contains_any(q, _shanghai_kw + _company_kw + _rag_kw + ["医疗", "健康"]):
        return "top5_industries"

    # ----- 规则 5: 医疗健康关键职位 -----
    _health_kw = ["医疗", "健康", "Healthcare", "Medical", "生命科学", "生物医药", "医药"]
    _role_kw = ["CEO", "创始人", "CTO", "算法负责人", "产品负责人", "关键职位", "Head of AI"]
    if _contains_any(q, _health_kw) and _contains_any(q, _role_kw):
        return "healthcare_key_roles"

    # ----- 规则 6: 中国以外参会者统计 -----
    _non_china_kw = ["中国以外", "国外", "海外", "非中国", "按国家统计", "国家分布"]
    if _contains_any(q, _non_china_kw):
        return "non_china_attendees"

    # ----- 规则 7: 智能客服 / Agent 工具调用销售线索 (加分题) -----
    _agent_kw = ["智能客服", "客服", "Agent", "工具调用", "Function Calling", "销售线索"]
    if _contains_any(q, _agent_kw):
        return "agent_customer_service_leads"

    # ----- 规则 8: Top 10 销售跟进名单 (加分题) -----
    _followup_kw = ["Top 10", "销售跟进", "最值得跟进", "跟进名单", "优先跟进", "Top10"]
    if _contains_any(q, _followup_kw):
        return "top10_sales_followup"

    # ----- 规则 9: LLMOps / AI Infra / 向量数据库公司聚合 (加分题) -----
    _llmops_kw = ["LLMOps", "AI Infra", "AI基础设施", "向量数据库", "Vector DB", "Milvus", "Embedding"]
    if _contains_any(q, _llmops_kw):
        return "llmops_aiinfra_vector_companies"

    # ----- 默认: RAG 流程 -----
    return "rag"


def _contains_any(text: str, keywords: list) -> bool:
    """
    判断 text 中是否包含 keywords 中的任意一个（大小写不敏感）。

    Args:
        text: 待检测的文本。
        keywords: 关键词列表。

    Returns:
        True 如果包含任意关键词。
    """
    text_lower = text.lower()
    for kw in keywords:
        if kw.lower() in text_lower:
            return True
    return False
