"""
销售线索分析模块
----------------
基于 SQLite 数据库的规则筛选、评分排序和结果展示，
为 3 个加分题提供稳定、可解释的分析结果。
"""

import sqlite3
from collections import Counter
from typing import Dict, Iterable, List

import pandas as pd


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _contains(text: str, keyword: str) -> bool:
    return keyword.lower() in str(text or "").lower()


def _matched_keywords(texts: Iterable[str], keywords: Iterable[str]) -> List[str]:
    matched = []
    combined = " ".join(str(text or "") for text in texts)
    for keyword in keywords:
        if _contains(combined, keyword):
            matched.append(keyword)
    return sorted(set(matched))


def _join_unique(values: Iterable) -> str:
    cleaned = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text and text.lower() != "nan" and text not in seen:
            cleaned.append(text)
            seen.add(text)
    return "、".join(cleaned)


def _top_people_text(df: pd.DataFrame, name_col: str = "name", limit: int = 3) -> str:
    if df.empty:
        return "暂无"
    people = []
    for _, row in df.head(limit).iterrows():
        name = str(row.get(name_col, "") or "").strip() or "未知姓名"
        company = str(row.get("company", "") or "").strip() or "未知公司"
        people.append(f"{name}（{company}）")
    return "、".join(people) if people else "暂无"


def _top_companies_text(df: pd.DataFrame, limit: int = 5) -> str:
    if df.empty or "company" not in df.columns:
        return "暂无"
    counts = (
        df["company"]
        .fillna("")
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .value_counts()
        .head(limit)
    )
    return "、".join(counts.index.tolist()) if not counts.empty else "暂无"


def _keyword_counts(
    df: pd.DataFrame,
    keywords: Iterable[str],
    columns: Iterable[str],
    limit: int = 8,
) -> List[str]:
    counter: Counter[str] = Counter()
    for _, row in df.iterrows():
        texts = [str(row.get(column, "") or "") for column in columns]
        for keyword in _matched_keywords(texts, keywords):
            counter[keyword] += 1
    return [keyword for keyword, _ in counter.most_common(limit)]


def _as_result(summary: str, data: pd.DataFrame) -> Dict[str, object]:
    return {"summary": summary, "data": data}


# ==================== 1. 智能客服 / Agent 工具调用销售线索 ====================

def analyze_agent_customer_service_leads(db_path: str) -> Dict[str, object]:
    """
    筛选适合作为"智能客服 / Agent 工具调用"方向销售线索的参会者。

    Returns:
        {"summary": str, "data": DataFrame}
    """
    keywords = [
        "智能客服", "客服", "Agent", "工具调用", "Function Calling",
        "函数调用", "自动化", "工作流", "企业知识库", "RAG",
        "LLM应用", "AI客服", "客户服务", "对话机器人", "Chatbot",
        "CRM", "会议纪要", "自动跟进", "知识库问答",
    ]
    scenario_keywords = [
        "CRM", "会议纪要", "自动跟进", "客服自动化", "知识库问答",
        "工具调用", "工作流", "Function Calling", "RAG", "企业知识库",
    ]
    title_keywords = [
        "CEO", "Founder", "创始人", "CTO", "CIO", "VP", "Director",
        "Head", "负责人", "产品", "算法", "技术", "工程",
    ]

    output_columns = [
        "name", "company", "job_title", "city", "industry", "lead_score",
        "lead_stage", "interest_direction", "notes", "lead_reason",
        "analysis_score",
    ]

    conn = _get_conn(db_path)
    try:
        like_clauses = []
        params = []
        for keyword in keywords:
            like_clauses.append(
                "(interest_direction LIKE ? OR notes LIKE ? OR job_title LIKE ?)"
            )
            params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])

        df = pd.read_sql_query(
            f"""
            SELECT name, company, job_title, city, industry, lead_score,
                   lead_stage, interest_direction, notes
            FROM attendees
            WHERE {" OR ".join(like_clauses)}
            """,
            conn,
            params=params,
        )
    finally:
        conn.close()

    if df.empty:
        empty_df = pd.DataFrame(columns=output_columns)
        return _as_result(
            "本次未筛选出智能客服 / Agent 工具调用方向的潜在线索。筛选条件基于 interest_direction 和 notes 中的智能客服、Agent、工具调用、Function Calling、自动化、工作流、企业知识库、RAG 等关键词，并结合 Lead Score、Lead Stage 和职位影响力进行评分。",
            empty_df,
        )

    scores = []
    reasons = []

    for _, row in df.iterrows():
        score = 0
        reason_parts = []

        interest = str(row.get("interest_direction", "") or "")
        notes = str(row.get("notes", "") or "")
        job_title = str(row.get("job_title", "") or "")
        lead_stage = str(row.get("lead_stage", "") or "")
        lead_score = _to_float(row.get("lead_score", 0))

        interest_hits = _matched_keywords([interest], keywords)
        note_hits = _matched_keywords([notes], keywords)
        scenario_hits = _matched_keywords([notes], scenario_keywords)

        if interest_hits:
            score += 30 + min(len(interest_hits), 4) * 3
            reason_parts.append(
                f"兴趣方向命中 {', '.join(interest_hits[:5])}"
            )

        if note_hits:
            score += 18 + min(len(note_hits), 4) * 4
            reason_parts.append(
                f"备注提到 {', '.join(note_hits[:5])}，符合 Agent 工具调用或客服自动化场景"
            )

        if scenario_hits:
            score += 10
            reason_parts.append(
                f"备注存在明确落地场景：{', '.join(scenario_hits[:5])}"
            )

        if lead_score >= 90:
            score += 25
            reason_parts.append(
                f"Lead Score >= 90（{lead_score:.0f}），意向强度高"
            )
        elif lead_score >= 80:
            score += 15
            reason_parts.append(
                f"Lead Score >= 80（{lead_score:.0f}），具备较高跟进价值"
            )

        stage_upper = lead_stage.upper()
        if any(token in stage_upper or token in lead_stage for token in ["HOT", "SQL", "OPPORTUNITY", "高意向"]):
            score += 20
            reason_parts.append(f"Lead Stage 为 {lead_stage}，优先跟进价值较高")
        elif "意向" in lead_stage or "WARM" in stage_upper:
            score += 10
            reason_parts.append(f"Lead Stage 为 {lead_stage}，已有跟进信号")

        title_hits = _matched_keywords([job_title], title_keywords)
        if title_hits:
            score += 15
            reason_parts.append(
                f"职位为 {job_title}，具备决策、技术或产品影响力"
            )

        if not reason_parts:
            reason_parts.append(
                "命中智能客服 / Agent 相关筛选条件，适合作为销售线索进一步确认需求"
            )
            score = max(score, 1)

        scores.append(score)
        reasons.append("；".join(reason_parts))

    df["analysis_score"] = scores
    df["lead_reason"] = reasons
    df = df[df["analysis_score"] > 0].sort_values(
        ["analysis_score", "lead_score"], ascending=[False, False]
    )
    df = df[output_columns].reset_index(drop=True)

    top3 = _top_people_text(df, limit=3)
    main_keywords = _keyword_counts(
        df,
        keywords,
        ["interest_direction", "notes"],
        limit=8,
    )
    main_keyword_text = "、".join(main_keywords) if main_keywords else "智能客服、Agent、工具调用"
    top_companies = _top_companies_text(df.head(20), limit=5)
    summary = (
        "本次筛选主要基于 interest_direction 和 notes 中是否包含智能客服、Agent、工具调用、Function Calling、自动化、工作流、企业知识库、RAG 等关键词，"
        "并结合 Lead Score、Lead Stage 和职位影响力进行评分。"
        f"本次共筛选出 {len(df)} 位潜在线索，最高分 Top 3 为 {top3}。"
        f"主要命中关键词包括 {main_keyword_text}，高优先级线索主要集中在 {top_companies} 等公司。"
        "这些参会者适合销售跟进，是因为他们同时具备明确的客服自动化 / Agent 工具调用 / 知识库问答场景信号，部分对象还具有较高 Lead Score、较强 Lead Stage 或决策层 / 技术负责人 / 产品负责人影响力。"
    )

    return _as_result(summary, df)


# ==================== 2. Top 10 销售跟进名单 ====================

def generate_top10_sales_followup(db_path: str) -> Dict[str, object]:
    """
    生成 Top 10 销售跟进名单，综合考虑 Lead Stage、Lead Score、职位和备注。

    Returns:
        {"summary": str, "data": DataFrame}
    """
    output_columns = [
        "rank", "name", "company", "job_title", "city", "country", "industry",
        "lead_score", "lead_stage", "followup_score", "followup_reason",
        "notes",
    ]

    conn = _get_conn(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT name, company, job_title, city, country, industry,
                   lead_score, lead_stage, interest_direction, notes
            FROM attendees
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        empty_df = pd.DataFrame(columns=output_columns)
        return _as_result(
            "当前数据库中没有参会者数据，无法生成 Top 10 销售跟进名单。",
            empty_df,
        )

    scores = []
    reasons = []

    opportunity_keywords = {
        "RAG/企业知识库": ["RAG", "企业知识库", "知识库", "检索增强"],
        "Agent/智能客服": ["Agent", "智能客服", "工具调用", "Function Calling", "自动化"],
        "LLMOps/AI Infra": ["LLMOps", "AI Infra", "模型部署", "模型监控", "推理加速"],
        "向量数据库": ["向量数据库", "Vector DB", "Milvus", "FAISS", "pgvector", "Embedding"],
    }

    for _, row in df.iterrows():
        score = 0.0
        reason_parts = []

        lead_score = _to_float(row.get("lead_score", 0))
        lead_stage = str(row.get("lead_stage", "") or "")
        job_title = str(row.get("job_title", "") or "")
        interest = str(row.get("interest_direction", "") or "")
        notes = str(row.get("notes", "") or "")

        score += lead_score * 0.5
        if lead_score >= 90:
            reason_parts.append(f"Lead Score 高（{lead_score:.0f}），整体意向强")
        elif lead_score >= 80:
            reason_parts.append(f"Lead Score 较高（{lead_score:.0f}），具备跟进价值")

        stage_upper = lead_stage.upper()
        stage_bonus = 0
        if "HOT" in stage_upper or "高意向" in lead_stage:
            stage_bonus = 30
        elif any(token in stage_upper for token in ["SQL", "OPPORTUNITY"]):
            stage_bonus = 25
        elif "WARM" in stage_upper or "中意向" in lead_stage:
            stage_bonus = 15
        elif "意向" in lead_stage:
            stage_bonus = 10
        score += stage_bonus
        if stage_bonus:
            reason_parts.append(f"Lead Stage 为 {lead_stage}，跟进阶段优先级较高")

        title_upper = job_title.upper()
        title_bonus = 0
        if any(token in title_upper or token in job_title for token in ["CEO", "FOUNDER", "创始人", "CO-FOUNDER"]):
            title_bonus = 25
            reason_parts.append("职位为 CEO / Founder / 创始人，具备采购或业务决策影响力")
        elif any(token in title_upper or token in job_title for token in ["CTO", "CIO", "技术负责人", "算法负责人", "AI负责人", "HEAD OF AI"]):
            title_bonus = 20
            reason_parts.append("职位为 CTO / AI / 技术负责人，具备技术决策影响力")
        elif any(token in title_upper or token in job_title for token in ["VP", "DIRECTOR", "HEAD", "负责人", "总监"]):
            title_bonus = 15
            reason_parts.append("职位为 VP / Director / Head / 负责人，具备团队影响力")
        elif any(token in title_upper or token in job_title for token in ["PRODUCT", "产品"]):
            title_bonus = 12
            reason_parts.append("职位涉及产品方向，适合验证落地场景与预算需求")
        score += title_bonus

        matched_labels = []
        for label, keywords in opportunity_keywords.items():
            if _matched_keywords([interest, notes], keywords):
                matched_labels.append(label)
        if matched_labels:
            bonus = 15 * len(matched_labels)
            score += bonus
            reason_parts.append(
                f"备注或兴趣方向出现 {', '.join(matched_labels)} 等明确需求信号"
            )

        if not reason_parts:
            reason_parts.append("综合 Lead Score、Lead Stage、职位和备注内容进入候选排序")

        scores.append(round(score, 1))
        reasons.append("；".join(reason_parts))

    df["followup_score"] = scores
    df["followup_reason"] = reasons
    df = df.sort_values(
        ["followup_score", "lead_score"], ascending=[False, False]
    ).head(10)
    df = df.reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df = df[output_columns]

    top3 = _top_people_text(df, limit=3)
    direction_counter: Counter[str] = Counter()
    for _, row in df.iterrows():
        for label, keywords in opportunity_keywords.items():
            if _matched_keywords([row.get("notes", ""), row.get("interest_direction", "")], keywords):
                direction_counter[label] += 1
    directions = "、".join([label for label, _ in direction_counter.most_common(4)])
    if not directions:
        directions = "RAG、Agent、企业知识库、LLMOps、向量数据库"

    summary = (
        "Top 10 销售跟进名单综合考虑 Lead Score、Lead Stage、职位影响力和备注中的真实需求信号。"
        "Lead Score 用于衡量整体意向强度，Lead Stage 用于判断跟进阶段，职位用于判断采购决策权或技术影响力，备注和兴趣方向用于识别 RAG、Agent、企业知识库、LLMOps、向量数据库等潜在采购场景。"
        f"本次 Top 3 推荐对象为 {top3}。"
        f"建议优先跟进这些对象，是因为他们在高意向 / 高分数 / 决策或技术影响力 / 明确业务需求信号中至少命中多项。"
        f"主要业务机会方向集中在 {directions}，适合从场景验证、PoC 方案、知识库检索、Agent 工作流或 AI Infra 工具链切入。"
    )

    return _as_result(summary, df)


# ==================== 3. LLMOps / AI Infra / 向量数据库公司聚合 ====================

def analyze_llmops_aiinfra_vector_companies(db_path: str) -> Dict[str, object]:
    """
    找出兴趣方向涉及 LLMOps、AI Infra 或向量数据库的参会者，
    并按公司聚合总结。

    Returns:
        {"summary": str, "data": DataFrame}
    """
    keywords = [
        "LLMOps", "AI Infra", "AI基础设施", "向量数据库", "Vector DB",
        "Milvus", "Faiss", "FAISS", "Chroma", "Embedding", "pgvector",
        "检索增强", "RAG", "模型部署", "模型监控", "推理加速",
        "模型服务", "MLOps", "模型网关", "日志评测", "成本监控",
    ]
    output_columns = [
        "company", "attendee_count", "people", "job_titles", "main_interests",
        "avg_lead_score", "evidence", "company_summary",
    ]

    conn = _get_conn(db_path)
    try:
        like_clauses = []
        params = []
        for keyword in keywords:
            like_clauses.append("(interest_direction LIKE ? OR notes LIKE ?)")
            params.extend([f"%{keyword}%", f"%{keyword}%"])

        df = pd.read_sql_query(
            f"""
            SELECT name, company, job_title, interest_direction, notes, lead_score
            FROM attendees
            WHERE {" OR ".join(like_clauses)}
            """,
            conn,
            params=params,
        )
    finally:
        conn.close()

    if df.empty:
        empty_df = pd.DataFrame(columns=output_columns)
        return _as_result(
            "本次未发现兴趣方向涉及 LLMOps、AI Infra 或向量数据库的公司级线索。",
            empty_df,
        )

    company_results = []
    for company, group in df.groupby("company", dropna=False):
        company_name = str(company or "").strip()
        if not company_name:
            continue

        attendee_count = len(group)
        people = _join_unique(group["name"])
        job_titles = _join_unique(group["job_title"])
        main_interests = _join_unique(group["interest_direction"])
        lead_scores = pd.to_numeric(group["lead_score"], errors="coerce").fillna(0)
        avg_lead_score = round(float(lead_scores.mean()), 1)

        interest_hits = _matched_keywords(group["interest_direction"].tolist(), keywords)
        note_hits = _matched_keywords(group["notes"].tolist(), keywords)
        evidence_parts = []
        if interest_hits:
            evidence_parts.append(f"Interest Direction 命中 {', '.join(interest_hits[:8])}")
        if note_hits:
            evidence_parts.append(f"Notes 提到 {', '.join(note_hits[:8])}")
        if not evidence_parts:
            evidence_parts.append("参会者兴趣方向或备注命中 AI Infra / LLMOps 相关关键词")
        evidence = "；".join(evidence_parts)

        role_signal = ""
        if any(
            _matched_keywords([title], ["CTO", "CIO", "VP", "Head", "负责人", "产品", "技术", "工程"])
            for title in group["job_title"]
        ):
            role_signal = "，其中包含技术负责人、产品负责人或管理层角色"

        interest_focus = "、".join((interest_hits + note_hits)[:4]) or "LLMOps / AI Infra / 向量数据库"
        company_summary = (
            f"该公司有 {attendee_count} 位参会者关注 {interest_focus}{role_signal}，"
            f"平均 Lead Score 为 {avg_lead_score}，适合作为 AI Infra / 向量数据库 / LLMOps 方向的公司级销售线索。"
        )

        company_results.append(
            {
                "company": company_name,
                "attendee_count": attendee_count,
                "people": people,
                "job_titles": job_titles,
                "main_interests": main_interests,
                "avg_lead_score": avg_lead_score,
                "evidence": evidence,
                "company_summary": company_summary,
            }
        )

    result_df = pd.DataFrame(company_results, columns=output_columns)
    if result_df.empty:
        return _as_result(
            "本次未发现可聚合的公司级 LLMOps / AI Infra / 向量数据库线索。",
            result_df,
        )

    result_df = result_df.sort_values(
        ["attendee_count", "avg_lead_score"], ascending=[False, False]
    ).reset_index(drop=True)

    top5 = "、".join(result_df["company"].head(5).tolist())
    main_keywords = _keyword_counts(
        df,
        keywords,
        ["interest_direction", "notes"],
        limit=8,
    )
    main_keyword_text = "、".join(main_keywords) if main_keywords else "LLMOps、AI Infra、向量数据库"
    summary = (
        "本次分析按公司聚合了所有兴趣方向涉及 LLMOps、AI Infra、向量数据库、Embedding、Milvus、Faiss、Chroma、模型部署、模型监控、推理加速等关键词的参会者。"
        f"共涉及 {len(result_df)} 家公司，Top 5 公司为 {top5}。"
        f"主要兴趣方向和证据信号集中在 {main_keyword_text}。"
        "优先关注参会人数较多、平均 Lead Score 较高、且职位覆盖技术负责人或产品负责人的公司。"
        "这类公司可能正在建设企业级 AI 基础设施或知识库检索系统，适合从向量数据库、RAG 平台、模型服务和 LLMOps 工具链方向切入。"
    )

    return _as_result(summary, result_df)
