"""
UI 辅助工具模块
---------------
提供 Streamlit 页面通用的辅助函数，不包含业务逻辑。
"""

import pandas as pd


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """
    将 DataFrame 转换为 UTF-8-SIG 编码的 CSV 字节流，
    用于 st.download_button。

    Args:
        df: pandas DataFrame。

    Returns:
        UTF-8-SIG 编码的 CSV 字节数据。
    """
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
