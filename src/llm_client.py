"""
LLM 客户端模块
--------------
负责与大语言模型 API 的通信（OpenAI 兼容接口）。
支持 DeepSeek、Qwen、OpenAI 等兼容 API。
"""

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


def call_llm(prompt: str) -> str:
    """
    调用大语言模型 API，使用 OpenAI 兼容格式。

    Args:
        prompt: 完整的提示词（包含系统指令和用户消息）。

    Returns:
        模型生成的回答文本。如果未配置 API Key 或调用失败，返回错误提示。

    Note:
        - temperature = 0.2（保证回答相对确定）
        - max_tokens = 1200
        - 从 .env 文件读取 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL
    """
    # 检查 API Key 是否配置
    if not LLM_API_KEY:
        return (
            "⚠️ 未配置 LLM_API_KEY，请在项目根目录创建 .env 文件并配置大模型 API Key。\n\n"
            "示例 .env 内容：\n"
            "LLM_API_KEY=your_api_key\n"
            "LLM_BASE_URL=https://api.deepseek.com\n"
            "LLM_MODEL=deepseek-chat"
        )

    try:
        from openai import OpenAI
    except ImportError:
        return "⚠️ 未安装 openai 库，请执行: pip install openai"

    try:
        client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
        )

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
        )

        return response.choices[0].message.content or "（模型返回空内容）"

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return "❌ API Key 无效或未授权，请检查 .env 中的 LLM_API_KEY。"
        elif "404" in error_msg:
            return f"❌ API 端点或模型不存在，请检查 LLM_BASE_URL 和 LLM_MODEL 配置。当前: {LLM_BASE_URL} / {LLM_MODEL}"
        elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            return f"❌ API 连接超时，请检查网络连接或 LLM_BASE_URL。当前: {LLM_BASE_URL}"
        else:
            return f"❌ LLM 调用失败: {error_msg}"
