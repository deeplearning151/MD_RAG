"""
SQL 查询模块
------------
提供 6 个预定义的基础题查询函数，每个函数查询 SQLite 数据库并返回 pandas DataFrame。
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import List


def _get_conn(db_path: str) -> sqlite3.Connection:
    """内部辅助：获取数据库连接。"""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _check_db(db_path: str) -> None:
    """检查数据库是否存在。"""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"数据库文件不存在: {db_path}"
                                f"\n请先上传 CSV 并点击'写入数据库'。")


# ==================== 1. 上海地区 Lead Score > 90 ====================

def query_shanghai_high_score(db_path: str) -> pd.DataFrame:
    """
    找出上海地区 Lead Score 大于 90 的参会者，按 Lead Score 从高到低排序。

    Returns columns: name, company, job_title, city, lead_score, lead_stage,
                     interest_direction, notes
    """
    _check_db(db_path)
    conn = _get_conn(db_path)
    try:
        df = pd.read_sql_query("""
            SELECT name, company, job_title, city, lead_score, lead_stage,
                   interest_direction, notes
            FROM attendees
            WHERE (city LIKE '%上海%' OR city LIKE '%Shanghai%')
              AND lead_score > 90
            ORDER BY lead_score DESC
        """, conn)
        return df
    finally:
        conn.close()


# ==================== 2. 公司参会人数 >= 3 ====================

def query_companies_with_3plus_attendees(db_path: str) -> pd.DataFrame:
    """
    找出有 3 位及以上参会者的公司，列出公司名称、人数和主要兴趣方向。

    先用 SQL 按 company 分组统计人数，再用 pandas 计算每个公司的
    main_interest_direction（出现频率最高的 interest_direction）。

    Returns columns: company, attendee_count, main_interest_direction
    """
    _check_db(db_path)
    conn = _get_conn(db_path)
    try:
        # 先获取满足 >= 3 人的公司列表
        df_groups = pd.read_sql_query("""
            SELECT company, COUNT(*) AS attendee_count
            FROM attendees
            WHERE company IS NOT NULL AND company != ''
            GROUP BY company
            HAVING attendee_count >= 3
            ORDER BY attendee_count DESC
        """, conn)

        if df_groups.empty:
            return pd.DataFrame(columns=["company", "attendee_count", "main_interest_direction"])

        # 获取这些公司所有参会者的 interest_direction
        companies = df_groups["company"].tolist()
        placeholders = ",".join(["?" for _ in companies])
        df_interests = pd.read_sql_query(
            f"""
            SELECT company, interest_direction
            FROM attendees
            WHERE company IN ({placeholders})
              AND interest_direction IS NOT NULL
              AND interest_direction != ''
            """,
            conn,
            params=companies,
        )

        # 用 pandas 计算每个公司出现频率最高的 interest_direction
        if not df_interests.empty:
            interest_counts = (
                df_interests.groupby(["company", "interest_direction"])
                .size()
                .reset_index(name="count")
            )
            idx = interest_counts.groupby("company")["count"].idxmax()
            main_interests = interest_counts.loc[idx, ["company", "interest_direction"]]
            main_interests = main_interests.rename(
                columns={"interest_direction": "main_interest_direction"}
            )
            df_result = df_groups.merge(main_interests, on="company", how="left")
        else:
            df_result = df_groups.copy()
            df_result["main_interest_direction"] = ""

        df_result["main_interest_direction"] = df_result["main_interest_direction"].fillna("")
        return df_result
    finally:
        conn.close()


# ==================== 3. RAG / 企业知识库兴趣 ====================

def query_rag_or_kb_interested(db_path: str) -> pd.DataFrame:
    """
    找出所有对 RAG 或企业知识库感兴趣的参会者。

    在 interest_direction 和 notes 字段中检索关键词：
    RAG、检索增强、企业知识库、知识库、知识管理、向量检索、文档问答、
    Embedding、向量数据库

    Returns columns: name, company, job_title, city, interest_direction, notes, evidence
    """
    _check_db(db_path)
    conn = _get_conn(db_path)
    try:
        keywords = [
            "RAG", "检索增强", "企业知识库", "知识库", "知识管理",
            "向量检索", "文档问答", "Embedding", "向量数据库",
        ]

        like_clauses = []
        params = []
        for kw in keywords:
            like_clauses.append(
                "(interest_direction LIKE ? OR notes LIKE ?)"
            )
            params.extend([f"%{kw}%", f"%{kw}%"])

        where_clause = " OR ".join(like_clauses)

        df = pd.read_sql_query(
            f"""
            SELECT name, company, job_title, city, interest_direction, notes
            FROM attendees
            WHERE {where_clause}
            """,
            conn,
            params=params,
        )

        if df.empty:
            return pd.DataFrame(
                columns=["name", "company", "job_title", "city",
                         "interest_direction", "notes", "evidence"]
            )

        # 构造 evidence 字段
        evidence_list = []
        for _, row in df.iterrows():
            interest = str(row.get("interest_direction", ""))
            note = str(row.get("notes", ""))
            matched = []

            for kw in keywords:
                if kw.lower() in interest.lower():
                    matched.append(f"Interest Direction 命中「{kw}」")
                if kw.lower() in note.lower():
                    matched.append(f"Notes 命中「{kw}」")

            evidence_list.append("；".join(matched) if matched else "关键词匹配")

        df["evidence"] = evidence_list
        df = df[df["evidence"] != ""]
        return df
    finally:
        conn.close()


# ==================== 4. Top 5 行业统计 ====================

def query_top5_industries(db_path: str) -> pd.DataFrame:
    """
    按 industry 统计参会人数，列出人数最多的前 5 个行业。

    Returns columns: industry, attendee_count
    """
    _check_db(db_path)
    conn = _get_conn(db_path)
    try:
        df = pd.read_sql_query("""
            SELECT industry, COUNT(*) AS attendee_count
            FROM attendees
            WHERE industry IS NOT NULL AND industry != ''
            GROUP BY industry
            ORDER BY attendee_count DESC
            LIMIT 5
        """, conn)
        return df
    finally:
        conn.close()


# ==================== 5. 医疗健康关键职位 ====================

def query_healthcare_key_roles(db_path: str) -> pd.DataFrame:
    """
    找出医疗健康行业中职位包含关键角色的人。

    Industry 关键词：医疗、健康、Healthcare、Medical、生命科学、生物医药、医药
    Job Title 关键词：CEO、创始人、Founder、CTO、算法负责人、产品负责人、
                     Head of AI、Product、产品、算法

    Returns columns: name, company, job_title, city, country, industry,
                     lead_score, lead_stage, notes
    """
    _check_db(db_path)
    conn = _get_conn(db_path)
    try:
        industry_kw = [
            "医疗", "健康", "Healthcare", "Medical",
            "生命科学", "生物医药", "医药",
        ]
        job_kw = [
            "CEO", "创始人", "Founder", "CTO", "算法负责人",
            "产品负责人", "Head of AI", "Product", "产品", "算法",
        ]

        ind_clauses = []
        ind_params = []
        for kw in industry_kw:
            ind_clauses.append("industry LIKE ?")
            ind_params.append(f"%{kw}%")

        job_clauses = []
        job_params = []
        for kw in job_kw:
            job_clauses.append("job_title LIKE ?")
            job_params.append(f"%{kw}%")

        where_clause = (
            "(" + " OR ".join(ind_clauses) + ")"
            " AND "
            "(" + " OR ".join(job_clauses) + ")"
        )
        all_params = ind_params + job_params

        df = pd.read_sql_query(
            f"""
            SELECT name, company, job_title, city, country, industry,
                   lead_score, lead_stage, notes
            FROM attendees
            WHERE {where_clause}
            ORDER BY lead_score DESC
            """,
            conn,
            params=all_params,
        )
        return df
    finally:
        conn.close()


# ==================== 6. 中国以外参会者统计 ====================

def query_non_china_attendees_by_country(db_path: str) -> pd.DataFrame:
    """
    找出来自中国以外地区的参会者，并按国家统计人数。

    排除 Country：中国、China、CN、PRC、中华人民共和国、Mainland China

    Returns columns: country, attendee_count
    """
    _check_db(db_path)
    conn = _get_conn(db_path)
    try:
        exclude = ["中国", "China", "CN", "PRC", "中华人民共和国", "Mainland China"]
        placeholders = ",".join(["?" for _ in exclude])

        df = pd.read_sql_query(
            f"""
            SELECT country, COUNT(*) AS attendee_count
            FROM attendees
            WHERE country IS NOT NULL
              AND country != ''
              AND country NOT IN ({placeholders})
            GROUP BY country
            ORDER BY attendee_count DESC
            """,
            conn,
            params=exclude,
        )
        return df
    finally:
        conn.close()

