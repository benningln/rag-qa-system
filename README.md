# 📚 RAG 知识库问答系统

基于检索增强生成（RAG）技术的智能问答系统，支持上传 PDF 文档，通过自然语言提问获取带引用来源的精准回答。

# 功能特点

- 📄 PDF 文档解析：支持上传 PDF 文件，自动提取文本内容（含 OCR 图片识别）
- 🧩 智能文本分块：采用 chunk + overlap 策略，保证语义完整性
- 🔍 向量检索：基于 BGE-Large 中文 Embedding 模型 + Chroma 向量数据库
- 🎯 精排重排：使用 BGE-Reranker 对召回结果进行二次精排，提升准确率
- 💬 多轮对话：基于 Redis 存储会话历史，支持上下文连贯追问
- 🤖 AI 生成回答：调用 DeepSeek API，流式生成带引用标记的回答
- 🌐 Web 界面：基于 Next.js 构建，支持 SSE 流式实时推送

# 技术栈

| 层级 | 技术 |
| :--- | :--- |
| 前端 | Next.js + TypeScript + Tailwind CSS |
| 后端 | FastAPI + Python 3.10 |
| 向量库 | Chroma + BGE-Large-zh + BGE-Reranker |
| 大模型 | DeepSeek API (兼容 OpenAI 协议) |
| 会话记忆 | Redis |
| 文档解析 | PyMuPDF + PaddleOCR |

# 项目结构

rag-qa-system/
├── backend/ # 后端代码
│ ├── main.py # FastAPI 主服务
│ ├── parser.py # PDF 解析器
│ ├── chunker.py # 文本分块器
│ ├── vector_store.py # 向量检索模块
│ ├── reranker.py # 精排重排模块
│ ├── memory.py # Redis 会话记忆
│ ├── generator.py # RAG 生成器
│ └── config.py # 配置管理
├── frontend/ # 前端代码（Next.js）
│ ├── src/app/ # 页面组件
│ └── src/lib/ # API 客户端
├── data/ # 向量数据库（本地）
├── local_models/ # 本地模型文件（需自行下载）
├── requirements.txt # Python 依赖
└── .env.example # 环境变量模板

# 后端配置
创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
安装依赖
pip install -r requirements.txt
配置环境变量
cp .env.example .env
编辑 .env，填入你的 DeepSeek API Key

# 下载模型
使用 modelscope 下载（推荐）
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-large-zh-v1.5', cache_dir='./local_models', local_dir='./local_models/bge-large-zh-v1.5')"
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-reranker-large', cache_dir='./local_models', local_dir='./local_models/bge-reranker-large')"

# 启动 Redis
Docker 方式（推荐）
docker run -d --name redis-rag -p 6379:6379 redis:alpine
或 Windows 直接运行 （没有Docker）
redis-server.exe

# 启动后端
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 启动前端
cd frontend
npm install
npm run dev
