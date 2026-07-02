import os
from pathlib import Path
from dotenv import load_dotenv

# 获取项目根目录（因为 config.py 在 backend 文件夹里，所以 parent.parent 就是根目录）
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载根目录下的 .env 文件
load_dotenv(BASE_DIR / '.env')


class Settings:
    """统一管理项目配置"""

    # DeepSeek API 密钥（从环境变量读取）
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")

    # 后面我们还会在这里加 Redis 地址、Chroma 路径等配置


# 创建一个全局配置对象，方便其他文件导入使用
settings = Settings()

# 简单的启动校验（如果没填 Key，运行时会打印警告）
if not settings.DEEPSEEK_API_KEY:
    print("⚠️ 警告：未检测到 DEEPSEEK_API_KEY，请在 .env 文件中填写你的密钥！")