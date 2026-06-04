"""
CSV 加载模块
------------
负责解析上传的 md.csv 参会者名单，返回结构化数据。
"""

import io
import pandas as pd
from pathlib import Path
from typing import Optional, Union


# ==================== 标准字段定义 ====================
EXPECTED_COLUMNS = [
    "Name",
    "Job Title",
    "Company",
    "City",
    "Country",
    "Industry",
    "Interest Direction",
    "Session",
    "Lead Stage",
    "Lead Score",
    "Email",
    "Notes",
]


class CSVLoadError(Exception):
    """CSV 加载/校验异常，携带用户友好的错误信息。"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def load_csv(
    source: Union[str, Path, io.BytesIO],
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """
    加载 md.csv 参会者名单，执行字段校验，返回干净的 DataFrame。

    功能：
    - 支持文件路径、文件类对象（BytesIO）等多种输入；
    - 自动尝试 UTF-8-SIG、UTF-8 编码；
    - 校验是否包含全部标准字段，缺失时抛出 CSVLoadError；
    - 将 Lead Score 列转换为数值类型；
    - 缺失值填充为空字符串。

    Args:
        source: CSV 来源，可以是文件路径 (str/Path) 或 BytesIO 对象。
        encoding: 强制指定编码，为 None 时自动检测。

    Returns:
        pandas DataFrame，包含经过清洗的参会者数据。

    Raises:
        CSVLoadError: 字段缺失、编码错误或其他解析问题时抛出。
        FileNotFoundError: 文件路径不存在时抛出。
    """
    # ---- 1. 读取原始 CSV ----
    encodings_to_try = (
        [encoding] if encoding else ["utf-8-sig", "utf-8", "gbk"]
    )

    raw_df = None
    last_error = None

    for enc in encodings_to_try:
        try:
            if isinstance(source, (str, Path)):
                path = Path(source)
                if not path.exists():
                    raise FileNotFoundError(f"文件不存在: {source}")
                raw_df = pd.read_csv(
                    path,
                    encoding=enc,
                    dtype=str,
                    keep_default_na=False,
                )
            else:
                # 文件类对象（如 Streamlit UploadedFile）
                source.seek(0)
                raw_df = pd.read_csv(
                    source,
                    encoding=enc,
                    dtype=str,
                    keep_default_na=False,
                )
            break  # 读取成功，跳出编码尝试循环
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
        except Exception as e:
            raise CSVLoadError(f"CSV 文件读取失败: {e}") from e

    if raw_df is None:
        raise CSVLoadError(
            f"无法解析 CSV 文件编码，已尝试: {', '.join(encodings_to_try)}。"
            f"请确认文件编码为 UTF-8。最后错误: {last_error}"
        )

    # ---- 2. 清理列名（去除引号、首尾空格） ----
    raw_df.columns = [str(col).strip().strip('"').strip() for col in raw_df.columns]

    # ---- 3. 字段校验 ----
    existing_cols = set(raw_df.columns)
    expected_set = set(EXPECTED_COLUMNS)
    missing = expected_set - existing_cols

    if missing:
        missing_sorted = sorted(missing)
        raise CSVLoadError(
            f"CSV 文件缺少必需字段 ({len(missing)} 个):\n"
            + "\n".join(f"  • {col}" for col in missing_sorted)
            + f"\n\n文件实际包含字段: {list(raw_df.columns)}"
        )

    # 仅保留标准字段（忽略多余列）
    df = raw_df[EXPECTED_COLUMNS].copy()

    # ---- 4. Lead Score 数值转换 ----
    if "Lead Score" in df.columns:
        df["Lead Score"] = pd.to_numeric(df["Lead Score"], errors="coerce").fillna(0).astype(int)

    # ---- 5. 缺失值填充为空字符串 ----
    # 对文本列进行填充
    text_cols = [c for c in df.columns if c != "Lead Score"]
    df[text_cols] = df[text_cols].fillna("").replace("nan", "")

    # ---- 6. 去除完全空行 ----
    df = df.dropna(how="all")
    df = df[~(df.astype(str).eq("").all(axis=1))]

    return df


def get_summary(df: pd.DataFrame) -> dict:
    """
    获取数据摘要信息。

    Args:
        df: 已加载的 DataFrame。

    Returns:
        包含行数、列数、列名等信息的字典。
    """
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "missing_values": df.isnull().sum().to_dict() if df is not None else {},
    }
