"""
MD_RAG - 参会者智能问答系统
============================

基于 CSV 结构化数据的 SQL + RAG 混合检索问答系统。

启动方式:
    streamlit run app.py
"""

import html
import sys
from pathlib import Path

import streamlit as st

# 确保 src 在 Python 路径中
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATA_DIR, DB_PATH, LLM_API_KEY, PAGE_ICON, PAGE_LAYOUT, PAGE_TITLE
from src.csv_loader import CSVLoadError, load_csv
from src.db import init_db, save_attendees
from src.lead_analysis import (
    analyze_agent_customer_service_leads,
    analyze_llmops_aiinfra_vector_companies,
    generate_top10_sales_followup,
)
from src.query_router import route_question
from src.rag_pipeline import answer_with_rag
from src.sql_queries import (
    query_companies_with_3plus_attendees,
    query_healthcare_key_roles,
    query_non_china_attendees_by_country,
    query_rag_or_kb_interested,
    query_shanghai_high_score,
    query_top5_industries,
)
from src.ui_helpers import convert_df_to_csv
from src.vector_store import build_vector_store


st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="collapsed",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background: #eef3f9;
        height: 100vh;
        overflow: hidden;
    }

    #MainMenu,
    header[data-testid="stHeader"],
    .stDeployButton,
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    section[data-testid="stSidebar"],
    footer {
        display: none !important;
    }

    .main .block-container {
        max-width: none !important;
        padding: 0 !important;
        height: 100vh;
        overflow: hidden;
    }

    .st-key-chat_shell {
        width: min(1280px, 96vw);
        height: 90vh;
        max-height: 90vh;
        margin: 20px auto;
        background: #ffffff;
        border-radius: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    .st-key-chat_shell > div[data-testid="stVerticalBlock"] {
        height: 100%;
        max-height: 100%;
        display: flex;
        flex-direction: column;
        gap: 0;
        overflow: hidden;
    }

    .st-key-chat_shell > div[data-testid="stVerticalBlock"] > div {
        min-height: 0;
    }

    .st-key-chat_shell > div[data-testid="stVerticalBlock"] > div:has(.st-key-chat_header) {
        flex: 0 0 80px;
        height: 80px;
        min-height: 0;
        overflow: hidden;
    }

    .st-key-chat_shell > div[data-testid="stVerticalBlock"] > div:has(.st-key-data_manager) {
        flex: 0 0 auto;
        max-height: 220px;
        min-height: 0;
        overflow: hidden;
    }

    .st-key-chat_shell > div[data-testid="stVerticalBlock"] > div:has(.st-key-chat_input_area) {
        flex: 0 0 72px;
        height: 72px;
        min-height: 0;
        overflow: hidden;
    }

    .st-key-chat_shell > div[data-testid="stVerticalBlock"] > div:nth-child(1) {
        flex: 0 0 80px;
        min-height: 0;
        overflow: hidden;
    }

    .st-key-chat_shell > div[data-testid="stVerticalBlock"] > div:nth-child(2) {
        flex: 0 0 auto;
        max-height: 220px;
        min-height: 0;
        overflow: hidden;
    }

    .st-key-chat_shell > div[data-testid="stVerticalBlock"] > div:nth-child(3) {
        flex: 1 1 auto;
        min-height: 0;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        padding: 24px 28px;
        box-sizing: border-box;
    }

    .st-key-chat_shell > div[data-testid="stVerticalBlock"] > div:nth-child(4) {
        flex: 0 0 72px;
        min-height: 0;
        overflow: hidden;
    }

    .st-key-chat_header {
        flex: 0 0 80px;
        height: 80px;
        position: relative;
        background: #ffffff;
        overflow: hidden;
    }

    .st-key-chat_header > div[data-testid="stVerticalBlock"] {
        height: 80px;
        gap: 0;
        position: relative;
        overflow: hidden;
    }

    .st-key-chat_header .stMarkdown {
        margin: 0 !important;
    }

    .st-key-chat_header div[data-testid="stMarkdownContainer"],
    .st-key-chat_header div[data-testid="stMarkdownContainer"] p {
        margin: 0 !important;
    }

    .chat-header {
        height: 80px;
        display: grid;
        grid-template-columns: minmax(380px, 1fr) auto 110px;
        align-items: center;
        gap: 24px;
        padding: 0 28px;
        background: #ffffff;
        border-bottom: 1px solid #e5e7eb;
        box-sizing: border-box;
    }

    .header-left {
        display: flex;
        align-items: center;
        gap: 14px;
        min-width: 0;
    }

    .app-icon {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        font-size: 20px;
        line-height: 1;
    }

    .title-box {
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-width: 0;
    }

    .app-title {
        font-size: 20px;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
        margin: 0;
        padding: 0;
        border: none;
        letter-spacing: 0;
        white-space: nowrap;
    }

    .app-subtitle {
        font-size: 13px;
        color: #64748b;
        line-height: 1.2;
        margin-top: 6px;
        padding: 0;
        border: none;
        letter-spacing: 0;
        white-space: nowrap;
    }

    .header-badges {
        display: flex;
        align-items: center;
        gap: 10px;
        white-space: nowrap;
    }

    .header-badges span {
        display: inline-flex;
        align-items: center;
        height: 28px;
        padding: 0 12px;
        border-radius: 999px;
        background: #ecfdf5;
        color: #047857;
        font-size: 12px;
        font-weight: 600;
        border: none;
        letter-spacing: 0;
        white-space: nowrap;
    }

    .header-badges span.status-ok {
        background: #ecfdf5;
        color: #047857;
    }

    .header-badges span.status-warn {
        background: #fffbeb;
        color: #b45309;
    }

    .header-action {
        display: flex;
        align-items: center;
        justify-content: flex-end;
    }

    .st-key-chat_header .st-key-clear_btn {
        position: absolute;
        top: 22px;
        right: 28px;
        z-index: 2;
    }

    .st-key-data_manager {
        flex: 0 0 auto;
        max-height: 220px;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 8px 28px;
        border-bottom: 1px solid #eef2f7;
        background: #fff;
    }

    .st-key-data_manager > div[data-testid="stVerticalBlock"] {
        max-height: 204px;
        overflow-y: auto;
        overflow-x: hidden;
    }

    .st-key-data_manager div[data-testid="stExpander"] {
        border-radius: 8px;
        border-color: #d1d5db;
        background: #fff;
    }

    .st-key-chat_input_area {
        flex: 0 0 72px;
        height: 72px;
        padding: 12px 24px;
        border-top: 1px solid #e5e7eb;
        background: #fff;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .st-key-chat_input_area > div[data-testid="stVerticalBlock"] {
        width: 100%;
        gap: 0;
    }

    .st-key-chat_input_area div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }

    .welcome-card {
        max-width: 760px;
        background: #f3f6fb;
        color: #111827;
        border-radius: 14px;
        padding: 16px 18px;
        margin: 0 8px 10px;
        line-height: 1.7;
        font-size: 14px;
    }

    .msg-row {
        display: flex;
        width: auto;
        margin: 0 8px 10px;
        align-items: flex-start;
    }

    .msg-row.assistant {
        justify-content: flex-start;
        margin-left: 8px;
        margin-right: 8px;
    }

    .msg-row.user {
        justify-content: flex-end;
        margin-left: 8px;
        margin-right: 8px;
    }

    .msg-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 2px;
        font-size: 14px;
    }

    .msg-row.assistant .msg-avatar {
        background: #6478ea;
        color: #fff;
        margin-right: 10px;
    }

    .msg-row.user .msg-avatar {
        background: #9aa3af;
        color: #111827;
        margin-left: 10px;
        order: 1;
    }

    .msg-bubble {
        max-width: min(78%, 820px);
        padding: 12px 16px;
        border-radius: 14px;
        line-height: 1.65;
        font-size: 14px;
        color: #111827;
        word-break: break-word;
        letter-spacing: 0;
    }

    .msg-row.assistant .msg-bubble {
        background: #f3f4f6;
        border-bottom-left-radius: 4px;
    }

    .msg-row.user .msg-bubble {
        color: #fff;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-bottom-right-radius: 4px;
    }

    .st-key-result_block {
        margin: 12px 20px 20px;
        max-width: calc(100% - 40px);
    }

    .st-key-chat_shell > div[data-testid="stVerticalBlock"] > div:nth-child(3) [data-testid="stExpander"] {
        margin: 12px 20px 20px;
    }

    .context-card {
        background: #f8fbff;
        border-left: 3px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 10px 12px;
        margin: 8px 0;
        font-size: 13px;
        line-height: 1.55;
        color: #1f2937;
    }

    .rag-context-list {
        max-height: 260px;
        overflow-y: auto;
        overflow-x: hidden;
        padding-right: 4px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0 !important;
    }

    .st-key-send_btn button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #fff !important;
        border: 0 !important;
        min-height: 40px;
    }

    .st-key-clear_btn button {
        background: #f3f4f6 !important;
        color: #374151 !important;
        border: 1px solid #e5e7eb !important;
        width: 86px !important;
        height: 36px !important;
        min-height: 36px;
        border-radius: 10px !important;
        padding: 0 !important;
        font-size: 13px !important;
        white-space: nowrap;
    }

    .stTextInput input {
        border-radius: 10px !important;
        min-height: 40px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }

    @media (max-width: 760px) {
        .st-key-chat_shell {
            width: 96vw;
            height: 90vh;
            max-height: 90vh;
            margin: 8px auto;
        }

        .st-key-chat_header {
            flex-basis: 80px;
            height: 80px;
            min-height: 80px;
        }

        .chat-header {
            grid-template-columns: minmax(220px, 1fr) auto 90px;
            gap: 12px;
            padding: 0 16px;
        }

        .app-title,
        .app-subtitle {
            white-space: normal;
        }

        .app-title {
            font-size: 18px;
        }

        .app-subtitle {
            font-size: 12px;
        }

        .header-badges {
            gap: 6px;
        }

        .header-badges span {
            padding: 0 8px;
            font-size: 11px;
        }

        .st-key-chat_header .st-key-clear_btn {
            right: 16px;
        }

        .st-key-data_manager,
        .st-key-chat_input_area {
            padding-left: 16px;
            padding-right: 16px;
        }

        .st-key-chat_input_area {
            flex-basis: 72px;
            height: 72px;
        }

        .msg-bubble {
            max-width: 86%;
        }

        .st-key-result_block {
            margin: 12px 12px 18px;
            max-width: calc(100% - 24px);
        }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


WELCOME = """你好，我是参会者智能问答助手。你可以直接提问，我会根据当前已导入的参会者数据进行查询和分析。

你可以试试这些基础题：
1. 找出上海地区 Lead Score 大于90的高意向参会者。
2. 哪些公司有3位及以上参会者？
3. 找出所有对RAG或企业知识库感兴趣的参会者。
4. 按Industry统计参会人数，列出人数最多的前5个行业。
5. 找出医疗健康行业中职位包含CEO、创始人、CTO、算法负责人或产品负责人的人。
6. 找出来自中国以外地区的参会者，并按国家统计人数。"""

QUERY_ROUTES = {
    "shanghai_high_score": query_shanghai_high_score,
    "companies_3plus": query_companies_with_3plus_attendees,
    "rag_kb_interested": query_rag_or_kb_interested,
    "top5_industries": query_top5_industries,
    "healthcare_key_roles": query_healthcare_key_roles,
    "non_china_attendees": query_non_china_attendees_by_country,
}

LEAD_ANALYSIS_ROUTES = {
    "agent_customer_service_leads": analyze_agent_customer_service_leads,
    "top10_sales_followup": generate_top10_sales_followup,
    "llmops_aiinfra_vector_companies": analyze_llmops_aiinfra_vector_companies,
}

RESULT_FILENAMES = {
    "shanghai_high_score": "shanghai_high_score.csv",
    "companies_3plus": "companies_3plus.csv",
    "rag_kb_interested": "rag_kb_interested.csv",
    "top5_industries": "top5_industries.csv",
    "healthcare_key_roles": "healthcare_key_roles.csv",
    "non_china_attendees": "non_china_attendees.csv",
    "agent_customer_service_leads": "agent_customer_service_leads.csv",
    "top10_sales_followup": "top10_sales_followup.csv",
    "llmops_aiinfra_vector_companies": "llmops_aiinfra_vector_companies.csv",
}

LEAD_ANALYSIS_TABLE_TITLES = {
    "agent_customer_service_leads": "查看智能客服 / Agent 销售线索明细",
    "top10_sales_followup": "查看 Top 10 销售跟进名单",
    "llmops_aiinfra_vector_companies": "查看 LLMOps / AI Infra / 向量数据库公司聚合结果",
}


def init_session_state() -> None:
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "df" not in st.session_state:
        st.session_state.df = None
    if "attendee_count" not in st.session_state:
        st.session_state.attendee_count = 0
    if "db_loaded" not in st.session_state:
        from src.db import database_exists

        st.session_state.db_loaded = database_exists(str(DB_PATH))
    if "db_count" not in st.session_state:
        from src.db import get_table_count

        st.session_state.db_count = (
            get_table_count(str(DB_PATH)) if st.session_state.db_loaded else 0
        )
    if "vector_index_built" not in st.session_state:
        from src.vector_store import vector_store_exists

        st.session_state.vector_index_built = vector_store_exists(
            str(DATA_DIR / "vector_store")
        )
    if "vector_count" not in st.session_state:
        from src.vector_store import get_vector_store_count

        st.session_state.vector_count = (
            get_vector_store_count(str(DATA_DIR / "vector_store"))
            if st.session_state.vector_index_built
            else 0
        )

    for key in (
        "current_question",
        "current_answer",
        "current_result_df",
        "current_result_filename",
        "current_contexts",
    ):
        if key not in st.session_state:
            st.session_state[key] = None

    if "input_version" not in st.session_state:
        st.session_state.input_version = 0
    if "scroll_to_bottom" not in st.session_state:
        st.session_state.scroll_to_bottom = False
    if "chat_history" not in st.session_state or not st.session_state.chat_history:
        st.session_state.chat_history = [build_assistant_message(WELCOME)]

    # 旧版本的消息状态不再参与渲染。
    st.session_state.pop("chat_messages", None)
    st.session_state.pop("display_turns", None)


def reset_current_round() -> None:
    st.session_state["current_question"] = None
    st.session_state["current_answer"] = None
    st.session_state["current_result_df"] = None
    st.session_state["current_result_filename"] = None
    st.session_state["current_contexts"] = None
    st.session_state.input_version += 1


def reset_conversation() -> None:
    reset_current_round()
    st.session_state.chat_history = [build_assistant_message(WELCOME)]
    st.session_state.scroll_to_bottom = False


def build_assistant_message(
    content: str,
    answer_type: str = "text",
    result_df=None,
    filename: str | None = None,
    contexts=None,
    table_title: str | None = None,
) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "answer_type": answer_type,
        "result_df": result_df,
        "filename": filename,
        "contexts": contexts,
        "table_title": table_title,
    }


def trim_chat_history(max_messages: int = 11) -> None:
    history = st.session_state.chat_history
    if len(history) <= max_messages:
        return

    welcome = history[:1]
    recent_messages = history[1:]
    while len(recent_messages) > max_messages - 1:
        recent_messages = recent_messages[2:]
    st.session_state.chat_history = welcome + recent_messages


def render_message(role: str, content: str) -> None:
    escaped = html.escape(content or "").replace("\n", "<br>")
    is_user = role == "user"
    icon = "👤" if is_user else "🤖"
    st.markdown(
        f"""
        <div class="msg-row {'user' if is_user else 'assistant'}">
            <div class="msg-avatar">{icon}</div>
            <div class="msg-bubble">{escaped}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    st.markdown(
        f'<div class="welcome-card">{html.escape(WELCOME).replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )


def render_message_artifacts(message: dict, message_index: int) -> None:
    result_df = message.get("result_df")
    if result_df is not None:
        filename = message.get("filename") or "result.csv"
        turn_number = max(1, message_index // 2)
        table_title = message.get("table_title") or "查看查询结果表格"
        table_height = 300
        with st.expander(f"{table_title}（第 {turn_number} 轮）", expanded=True):
            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True,
                height=table_height,
            )
            st.download_button(
                f"下载 CSV ({filename})",
                convert_df_to_csv(result_df),
                filename,
                "text/csv",
                key=f"download_{message_index}",
            )

    contexts = message.get("contexts")
    if contexts:
        turn_number = max(1, message_index // 2)
        with st.expander(f"查看检索依据（第 {turn_number} 轮）", expanded=False):
            context_cards = []
            for rank, context in enumerate(contexts[:5], 1):
                metadata = context.get("metadata", {})
                distance = context.get("distance", 0.0)
                context_cards.append(
                    f"""
                    <div class="context-card">
                        <strong>#{rank}</strong>
                        {html.escape(str(metadata.get('name', '?')))}
                        · {html.escape(str(metadata.get('company', '?')))}
                        · {html.escape(str(metadata.get('job_title', '?')))}
                        <br>
                        兴趣：{html.escape(str(metadata.get('interest_direction', '?')))}
                        | Score：{html.escape(str(metadata.get('lead_score', '?')))}
                        | 距离：{distance:.4f}
                    </div>
                    """
                )
            st.markdown(
                f'<div class="rag-context-list">{"".join(context_cards)}</div>',
                unsafe_allow_html=True,
            )


def process_question(question: str) -> dict:
    st.session_state["current_answer"] = None
    st.session_state["current_result_df"] = None
    st.session_state["current_result_filename"] = None
    st.session_state["current_contexts"] = None
    st.session_state["current_question"] = question

    route = route_question(question)

    if route == "rag":
        if not st.session_state.vector_index_built:
            answer = "请先构建向量索引，再使用 RAG 问答。"
            st.session_state["current_answer"] = answer
            return build_assistant_message(answer)

        try:
            rag_result = answer_with_rag(
                question, str(DATA_DIR / "vector_store"), top_k=20
            )
            answer = rag_result.get("answer", "")
            contexts = rag_result.get("contexts", [])
            st.session_state["current_answer"] = answer
            st.session_state["current_contexts"] = rag_result.get("contexts", [])
            return build_assistant_message(
                answer,
                answer_type="rag",
                contexts=contexts,
            )
        except Exception as exc:
            answer = f"RAG 问答失败: {exc}"
            st.session_state["current_answer"] = answer
            return build_assistant_message(answer)

    if not st.session_state.db_loaded:
        answer = "当前还没有可查询的参会者数据，请先上传 CSV 并写入数据库。"
        st.session_state["current_answer"] = answer
        return build_assistant_message(answer)

    if route in QUERY_ROUTES:
        query_func = QUERY_ROUTES[route]
        answer_prefix = "已为你完成查询"
    elif route in LEAD_ANALYSIS_ROUTES:
        query_func = LEAD_ANALYSIS_ROUTES[route]
    else:
        answer = "暂时无法识别这个问题的查询类型。"
        st.session_state["current_answer"] = answer
        return build_assistant_message(answer)

    try:
        if route in LEAD_ANALYSIS_ROUTES:
            result = query_func(str(DB_PATH))
            result_df = result.get("data")
            summary = result.get("summary") or "已完成销售线索分析。"
            if result_df is None or result_df.empty:
                st.session_state["current_answer"] = summary
                return build_assistant_message(summary, answer_type="lead_analysis")

            filename = RESULT_FILENAMES.get(route, "result.csv")
            st.session_state["current_answer"] = summary
            st.session_state["current_result_df"] = result_df
            st.session_state["current_result_filename"] = filename
            return build_assistant_message(
                summary,
                answer_type="lead_analysis",
                result_df=result_df,
                filename=filename,
                table_title=LEAD_ANALYSIS_TABLE_TITLES.get(route),
            )

        result_df = query_func(str(DB_PATH))
        if result_df.empty:
            answer = "未查询到符合条件的数据。"
            st.session_state["current_answer"] = answer
            return build_assistant_message(answer)

        answer = f"{answer_prefix}，共找到 {len(result_df)} 条结果。"
        filename = RESULT_FILENAMES.get(route, "result.csv")
        st.session_state["current_answer"] = answer
        st.session_state["current_result_df"] = result_df
        st.session_state["current_result_filename"] = filename
        return build_assistant_message(
            answer,
            answer_type="sql",
            result_df=result_df,
            filename=filename,
        )
    except Exception as exc:
        answer = f"查询执行失败: {exc}"
        st.session_state["current_answer"] = answer
        return build_assistant_message(answer)


def render_data_manager() -> None:
    with st.expander("🛠️ 数据管理", expanded=False):
        uploaded_file = st.file_uploader(
            "上传 CSV 文件", type=["csv"], label_visibility="collapsed"
        )
        import_mode = st.radio(
            "导入模式",
            ["追加到现有数据", "覆盖现有数据"],
            index=0,
            horizontal=True,
        )

        if uploaded_file is not None:
            df = None
            try:
                is_new_file = (
                    st.session_state.get("_uploaded_name") != uploaded_file.name
                )
                if is_new_file:
                    df = load_csv(uploaded_file)
                    if df.empty:
                        st.error("CSV 文件为空。")
                    else:
                        st.session_state["df"] = df
                        st.session_state["_uploaded_name"] = uploaded_file.name
                        st.session_state.data_loaded = True
                        st.session_state.attendee_count = len(df)
                else:
                    df = st.session_state.get("df")
                    if df is not None:
                        st.session_state.data_loaded = True

                if df is not None and not df.empty:
                    st.success(f"CSV 解析成功：{len(df)} 条记录")
                    write_col, status_col = st.columns([1, 2], vertical_alignment="center")
                    with write_col:
                        if st.button("写入数据库", type="secondary"):
                            with st.spinner("正在写入 SQLite..."):
                                mode = "append" if "追加" in import_mode else "replace"
                                count = save_attendees(df, str(DB_PATH), mode=mode)
                                st.session_state.db_loaded = True
                                st.session_state.db_count = count
                                st.session_state.vector_index_built = False
                                st.session_state.vector_count = 0
                                reset_conversation()
                                st.success(f"写入成功，共 {count} 条")
                                st.rerun()
                    with status_col:
                        if st.session_state.db_loaded:
                            st.caption(f"数据库：{st.session_state.db_count} 条")

                    if st.session_state.db_loaded:
                        index_col, index_status_col = st.columns(
                            [1, 2], vertical_alignment="center"
                        )
                        with index_col:
                            if st.button("构建向量索引", type="secondary"):
                                with st.spinner("正在构建向量索引..."):
                                    vector_count = build_vector_store(
                                        str(DB_PATH), str(DATA_DIR / "vector_store")
                                    )
                                    st.session_state.vector_index_built = True
                                    st.session_state.vector_count = vector_count
                                    st.success(f"向量索引成功：{vector_count} 条")
                                    st.rerun()
                        with index_status_col:
                            if st.session_state.vector_index_built:
                                st.caption(f"索引：{st.session_state.vector_count} 条")
            except CSVLoadError as exc:
                st.error(f"字段校验失败\n\n{exc.message}")
            except Exception as exc:
                st.error(str(exc))


init_db(str(DB_PATH))
init_session_state()


with st.container(key="chat_shell"):
    with st.container(key="chat_header"):
        db_ok = st.session_state.db_loaded
        vec_ok = st.session_state.vector_index_built
        api_ok = bool(LLM_API_KEY)
        st.markdown(
            f"""
            <div class="chat-header">
                <div class="header-left">
                    <div class="app-icon">☻</div>
                    <div class="title-box">
                        <div class="app-title">参会者智能问答系统</div>
                        <div class="app-subtitle">基于参会者数据的 SQL + RAG 混合问答助手</div>
                    </div>
                </div>
                <div class="header-badges">
                    <span class="{'status-ok' if db_ok else 'status-warn'}">
                        数据：{'已加载 ' + str(st.session_state.db_count) + ' 条' if db_ok else '未加载'}
                    </span>
                    <span class="{'status-ok' if vec_ok else 'status-warn'}">
                        索引：{'已构建' if vec_ok else '未构建'}
                    </span>
                    <span class="{'status-ok' if api_ok else 'status-warn'}">
                        API：{'已连接' if api_ok else '未配置'}
                    </span>
                </div>
                <div class="header-action"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("清空", key="clear_btn", width="content"):
            reset_conversation()
            st.rerun()

    with st.container(key="data_manager"):
        render_data_manager()

    chat_area = st.container(height=560, border=False)
    with chat_area:
        for index, message in enumerate(st.session_state.chat_history):
            if index == 0 and message.get("role") == "assistant":
                render_welcome()
                continue

            render_message(message.get("role", "assistant"), message.get("content", ""))
            if message.get("role") == "assistant":
                render_message_artifacts(message, index)
        st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)

    if st.session_state.get("scroll_to_bottom"):
        st.components.v1.html(
            """
            <script>
            setTimeout(() => {
                const target = window.parent.document.getElementById("chat-bottom");
                if (!target) {
                    return;
                }

                let scrollParent = target.parentElement;
                while (scrollParent && scrollParent !== window.parent.document.body) {
                    const style = window.parent.getComputedStyle(scrollParent);
                    const canScroll = /(auto|scroll|overlay)/.test(style.overflowY);
                    if (canScroll && scrollParent.scrollHeight > scrollParent.clientHeight) {
                        scrollParent.scrollTo({
                            top: scrollParent.scrollHeight,
                            behavior: "smooth"
                        });
                        return;
                    }
                    scrollParent = scrollParent.parentElement;
                }

                target.scrollIntoView({ behavior: "smooth", block: "end" });
            }, 100);
            </script>
            """,
            height=0,
        )
        st.session_state.scroll_to_bottom = False

    with st.container(key="chat_input_area"):
        input_key = f"question_input_{st.session_state.input_version}"
        input_col, send_col = st.columns([6, 1], vertical_alignment="center")
        with input_col:
            user_input = st.text_input(
                "输入问题",
                placeholder="请输入问题，例如：哪些公司有3位及以上参会者？",
                label_visibility="collapsed",
                key=input_key,
            )
        with send_col:
            send_clicked = st.button("发送", key="send_btn", width="stretch")

if send_clicked and user_input.strip():
    question = user_input.strip()
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.spinner("正在分析问题..."):
        assistant_message = process_question(question)
        st.session_state.chat_history.append(assistant_message)
        trim_chat_history()
    st.session_state.input_version += 1
    st.session_state.scroll_to_bottom = True
    st.rerun()
