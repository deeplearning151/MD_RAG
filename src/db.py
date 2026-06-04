"""
数据库模块
----------
负责 SQLite 数据库的创建、连接和基本 CRUD 操作。
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

from .config import DB_PATH


# ==================== 列名映射：CSV 原始字段 -> 数据库字段 ====================
COLUMN_MAPPING = {
    "Name": "name",
    "Job Title": "job_title",
    "Company": "company",
    "City": "city",
    "Country": "country",
    "Industry": "industry",
    "Interest Direction": "interest_direction",
    "Session": "session",
    "Lead Stage": "lead_stage",
    "Lead Score": "lead_score",
    "Email": "email",
    "Notes": "notes",
}


def _get_conn(db_path: str) -> sqlite3.Connection:
    """
    获取 SQLite 数据库连接（内部辅助函数）。

    Args:
        db_path: 数据库文件路径字符串。

    Returns:
        sqlite3.Connection 对象。
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: str) -> None:
    """
    初始化 SQLite 数据库。
    如果 attendees 表不存在则创建。

    Args:
        db_path: 数据库文件路径，默认建议为 data/md_rag.db。
    """
    conn = _get_conn(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                job_title TEXT,
                company TEXT,
                city TEXT,
                country TEXT,
                industry TEXT,
                interest_direction TEXT,
                session TEXT,
                lead_stage TEXT,
                lead_score REAL,
                email TEXT,
                notes TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def database_exists(db_path: str) -> bool:
    """
    判断 SQLite 数据库是否已构建并有数据。

    Args:
        db_path: 数据库文件路径。

    Returns:
        True 如果数据库文件存在、attendees 表存在且至少有 1 条记录。
    """
    path = Path(db_path)
    if not path.exists():
        return False

    conn = _get_conn(db_path)
    try:
        # 检查 attendees 表是否存在
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='attendees'"
        )
        if not cursor.fetchone():
            return False
        # 检查是否有数据
        count_cursor = conn.execute("SELECT COUNT(*) FROM attendees")
        return count_cursor.fetchone()[0] > 0
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def save_attendees(df: pd.DataFrame, db_path: str, mode: str = "append") -> int:
    """
    将 DataFrame 写入 attendees 表。

    Args:
        df: 包含参会者数据的 DataFrame。
        db_path: 数据库文件路径。
        mode: "append" 追加写入, "replace" 清空旧数据后写入。

    Returns:
        写入后数据库中的总记录数。
    """
    conn = _get_conn(db_path)
    try:
        # 确保表存在
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT, job_title TEXT, company TEXT, city TEXT, country TEXT,
                industry TEXT, interest_direction TEXT, session TEXT,
                lead_stage TEXT, lead_score REAL, email TEXT, notes TEXT
            )
        """)

        # 根据模式处理旧数据
        if mode == "replace":
            conn.execute("DELETE FROM attendees")
            conn.commit()

        # 列名映射
        available_csv_cols = [c for c in COLUMN_MAPPING if c in df.columns]
        db_cols = [COLUMN_MAPPING[c] for c in available_csv_cols]

        if not available_csv_cols:
            raise ValueError("DataFrame 不包含任何可映射的字段，请确认 CSV 格式正确。")

        placeholders = ", ".join(["?" for _ in db_cols])
        columns_str = ", ".join(db_cols)
        sql = f"INSERT INTO attendees ({columns_str}) VALUES ({placeholders})"

        rows = []
        for _, row in df.iterrows():
            rows.append(tuple(row[c] for c in available_csv_cols))

        conn.executemany(sql, rows)
        conn.commit()

        # 返回当前总记录数
        return get_table_count(db_path)
    finally:
        conn.close()


def get_table_count(db_path: str) -> int:
    """
    返回 attendees 表中的记录数量。

    Args:
        db_path: 数据库文件路径。

    Returns:
        记录数。如果表不存在，返回 0。
    """
    path = Path(db_path)
    if not path.exists():
        return 0

    conn = _get_conn(db_path)
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM attendees")
        return cursor.fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def get_all_attendees(db_path: str, limit: int = 20) -> List[Dict]:
    """
    返回数据库中前 limit 条记录，用于页面预览。

    Args:
        db_path: 数据库文件路径。
        limit: 返回的最大记录数，默认 20。

    Returns:
        字典列表，每个字典代表一条参会者记录。
    """
    conn = _get_conn(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM attendees ORDER BY id LIMIT ?", (limit,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

