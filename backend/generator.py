import json
from typing import List, Dict, Generator, Optional
from openai import OpenAI
from backend.config import settings
from backend.memory import session_manager


class RAGGenerator:
    """RAG 回答生成器，负责构建 Prompt、调用 DeepSeek API、流式返回"""

    def __init__(self):
        """初始化 OpenAI 客户端（指向 DeepSeek API）"""
        self.client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1"
        )
        self.model = "deepseek-chat"

    def build_prompt(
            self,
            query: str,
            context_docs: List[str],
            history: List[Dict[str, str]]
    ) -> str:
        """
        构建完整的 Prompt（包含上下文、历史、问题）

        参数:
            query: 用户当前问题
            context_docs: 检索到的相关文档列表（Top-5）
            history: 对话历史列表 [{"role": "user", "content": "..."}, ...]

        返回:
            str: 完整的 Prompt 字符串
        """
        # 1. 格式化上下文（带引用编号）
        if context_docs:
            context_text = "\n\n".join([
                f"[引用{i + 1}] {doc}" for i, doc in enumerate(context_docs)
            ])
        else:
            context_text = "（未找到相关参考资料）"

        # 2. 格式化对话历史（只取最近 5 轮，避免过度冗长）
        recent_history = history[-10:] if len(history) > 10 else history
        history_text = ""
        if recent_history:
            history_lines = []
            for msg in recent_history:
                role = "用户" if msg["role"] == "user" else "助手"
                history_lines.append(f"{role}: {msg['content']}")
            history_text = "\n".join(history_lines)
        else:
            history_text = "（无历史对话）"

        # 3. 构建最终 Prompt（使用清晰的指令结构）
        prompt = f"""
你是一个专业、准确、友好的智能问答助手。请严格遵循以下规则：

## 参考资料
{context_text}

## 对话历史
{history_text}

## 当前用户问题
{query}

## 回答要求
1. **优先使用参考资料**中的信息回答，如果参考资料不足，诚实说明“未找到相关信息”。
2. **必须标注引用来源**，在引用内容后加上 [引用X]（X 为编号）。
3. **回答要简洁、清晰、有条理**，不要过度展开无关内容。
4. **结合对话历史**，保持上下文连贯性。
5. **如果问题不明确**，可以追问用户澄清。

现在请回答用户的问题：
"""
        return prompt

    def generate_stream(
            self,
            query: str,
            context_docs: List[str],
            session_id: str
    ) -> Generator[str, None, None]:
        """
        流式生成回答，并自动保存对话历史到 Redis

        参数:
            query: 用户当前问题
            context_docs: 检索到的文档列表
            session_id: 会话 ID

        返回:
            Generator: 流式输出的文本片段
        """
        # 1. 获取对话历史
        session = session_manager.get_session(session_id)
        history = session.get("history", [])

        # 2. 构建 Prompt
        prompt = self.build_prompt(query, context_docs, history)

        # 3. 调用 DeepSeek API（流式）
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的AI助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 低温度让回答更确定、更准确
                max_tokens=2048,  # 限制最大输出长度
                stream=True
            )

            # 4. 收集完整回答（用于保存历史）
            full_answer = ""

            # 5. 逐块返回
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_answer += content
                    yield content  # 实时推送给前端

            # 6. 流结束后，保存对话历史到 Redis
            if full_answer.strip():
                session_manager.update_session(session_id, [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": full_answer}
                ])
                print(f"💾 会话 {session_id} 已保存（新增 2 条消息）")

        except Exception as e:
            error_msg = f"❌ 生成回答时出错: {e}"
            print(error_msg)
            yield error_msg


# 创建全局单例
generator = RAGGenerator()

# ========== 测试代码 ==========
if __name__ == "__main__":
    print("🧪 开始测试 RAG 生成器（需要有效的 DeepSeek API Key）...")

    # 模拟输入
    test_query = "桂林电子科技大学有哪些王牌专业？"
    test_docs = [
        "桂林电子科技大学（简称桂电）是广西重点建设的高校，其计算机科学与技术、电子信息工程、通信工程是国家级特色专业。",
        "桂电在桂林市金鸡路1号，拥有多个国家级实验教学示范中心。",
        "桂林山水甲天下，桂电校园环境优美。"
    ]
    test_session = "test_user_001"

    # 清空旧会话（方便重复测试）
    session_manager.clear_session(test_session)

    print(f"\n📝 用户问题: {test_query}")
    print("🤖 回答生成中...\n")

    # 流式输出
    full = ""
    for chunk in generator.generate_stream(test_query, test_docs, test_session):
        print(chunk, end="")
        full += chunk

    print("\n\n✅ 生成完成！")
    print(f"📊 回答总字数: {len(full)}")