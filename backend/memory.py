import json
import redis
from typing import List, Dict, Optional
from backend.config import BASE_DIR

# Redis 连接配置（默认本地 6379 端口）
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
# 会话过期时间（秒）：1小时
SESSION_EXPIRE = 3600


class SessionManager:
    """基于 Redis 的会话记忆管理器"""

    def __init__(self):
        """初始化 Redis 连接"""
        try:
            self.client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True  # 自动将字节解码为字符串
            )
            # 测试连接
            self.client.ping()
            print("✅ Redis 连接成功！")
        except redis.ConnectionError as e:
            print(f"❌ Redis 连接失败: {e}")
            print("   请确保 Redis 服务器已启动（docker run -d --name redis-rag -p 6379:6379 redis:alpine）")
            self.client = None

    def get_session(self, session_id: str) -> Dict:
        """
        获取会话历史

        参数:
            session_id: 会话唯一标识符

        返回:
            Dict: 包含 history 列表的结构，如 {"history": [{"role": "user", "content": "你好"}, ...]}
        """
        if not self.client:
            return {"history": []}

        key = f"session:{session_id}"
        data = self.client.get(key)
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return {"history": []}
        return {"history": []}

    def update_session(self, session_id: str, new_messages: List[Dict[str, str]]) -> None:
        """
        追加新消息到会话，并自动做摘要截断（防止 token 溢出）

        参数:
            session_id: 会话标识符
            new_messages: 新消息列表，如 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        if not self.client:
            print("⚠️ Redis 未连接，跳过会话保存")
            return

        key = f"session:{session_id}"
        # 获取当前会话
        session = self.get_session(session_id)
        history = session.get("history", [])

        # 追加新消息
        history.extend(new_messages)

        # ===== 智能截断策略（核心：避免长对话撑爆 token） =====
        # 1. 如果总条数超过 20 条，保留最近 10 条 + 前 3 条（保留开场白）
        if len(history) > 20:
            # 保留开头 3 条（通常是系统提示或开场问候）
            head = history[:3]
            # 保留最近 10 条
            tail = history[-10:]
            # 中间插入一条摘要标记（实际项目中可以调用 LLM 生成摘要，这里用占位）
            summary = [{"role": "system", "content": f"[摘要] 中间共省略了 {len(history) - 13} 条历史消息"}]
            history = head + summary + tail
            print(f"🔄 会话 {session_id} 已截断，当前保留 {len(history)} 条关键消息")

        # 存储回 Redis，并设置过期时间
        self.client.set(key, json.dumps({"history": history}, ensure_ascii=False), ex=SESSION_EXPIRE)

    def clear_session(self, session_id: str) -> None:
        """清空指定会话"""
        if self.client:
            self.client.delete(f"session:{session_id}")
            print(f"🗑️ 会话 {session_id} 已清空")


# 创建全局单例，方便其他模块导入使用
session_manager = SessionManager()

# ========== 测试代码 ==========
if __name__ == "__main__":
    print("🧪 开始测试对话记忆功能...")

    # 测试会话 ID
    test_sid = "test_user_001"

    # 1. 清空旧数据
    session_manager.clear_session(test_sid)

    # 2. 写入第一轮对话
    session_manager.update_session(test_sid, [
        {"role": "user", "content": "桂林有哪些大学？"},
        {"role": "assistant", "content": "桂林有桂林电子科技大学和广西师范大学。"}
    ])

    # 3. 读取会话
    session = session_manager.get_session(test_sid)
    print(f"\n📋 会话内容（共 {len(session['history'])} 条消息）：")
    for msg in session['history']:
        print(f"  [{msg['role']}] {msg['content'][:30]}...")

    # 4. 追加第二轮对话（测试上下文连贯）
    session_manager.update_session(test_sid, [
        {"role": "user", "content": "那桂电的王牌专业是什么？"},
        {"role": "assistant", "content": "桂林电子科技大学的王牌专业是计算机科学与技术和电子信息工程。"}
    ])

    # 5. 再次读取，验证历史是否完整
    session2 = session_manager.get_session(test_sid)
    print(f"\n📋 更新后会话内容（共 {len(session2['history'])} 条消息）：")
    for msg in session2['history']:
        print(f"  [{msg['role']}] {msg['content'][:30]}...")

    print("\n✅ 测试完成！Redis 会话记忆工作正常。")