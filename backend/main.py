import os
import shutil
import uuid
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from backend.config import BASE_DIR
from backend.parser import parse_pdf
from backend.chunker import chunk_text
from backend.vector_store import add_chunks_to_chroma, search_chroma
from backend.reranker import rerank
from backend.memory import session_manager
from backend.generator import generator

# 创建 FastAPI 应用
app = FastAPI(
    title="RAG 知识库问答系统",
    description="支持 PDF 文档上传、向量检索、RAG 生成回答的智能问答系统",
    version="1.0.0"
)

# 允许跨域（方便 Next.js 前端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议替换为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 接口 1：上传文档 ====================
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    上传 PDF 文档，自动解析、分块、向量化并存入 Chroma 数据库
    """
    # 1. 校验文件类型
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    # 2. 保存临时文件
    temp_dir = BASE_DIR / "data" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = temp_dir / file.filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    # 3. 解析 PDF
    try:
        df = parse_pdf(str(file_path))
        if df.empty:
            raise HTTPException(status_code=400, detail="PDF 文件未解析出任何文本内容")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 解析失败: {str(e)}")

    # 4. 分块
    try:
        chunks_df = chunk_text(df)
        if chunks_df.empty:
            raise HTTPException(status_code=400, detail="分块后未生成任何有效文本块")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文本分块失败: {str(e)}")

    # 5. 向量化并入库
    try:
        count = add_chunks_to_chroma(chunks_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"向量入库失败: {str(e)}")

    # 6. 清理临时文件
    try:
        os.remove(file_path)
    except:
        pass  # 删除失败也不影响主流程

    return {
        "status": "success",
        "message": f"文档 {file.filename} 处理完成",
        "pages": len(df),
        "chunks": count
    }


# ==================== 接口 2：问答（SSE 流式） ====================
@app.get("/query")
async def query_stream(
        q: str = Query(..., description="用户问题"),
        session_id: Optional[str] = Query(None, description="会话ID，用于多轮对话")
):
    """
    流式回答接口（SSE）。自动进行检索 → 重排 → 生成 → 返回
    """
    # 1. 如果未提供 session_id，自动生成一个新的
    if not session_id:
        session_id = str(uuid.uuid4())

    # 2. 向量检索（召回 Top-20）
    try:
        search_result = search_chroma(q, top_k=20)
        documents = search_result.get('documents', [[]])[0]
        if not documents:
            # 没有检索到任何文档，直接让生成器返回"未找到"提示
            documents = ["未找到与问题相关的参考资料"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"向量检索失败: {str(e)}")

    # 3. 重排（精排 Top-5）
    try:
        if len(documents) > 5:
            reranked_docs, scores = rerank(q, documents, top_k=5)
        else:
            reranked_docs = documents
            scores = [1.0] * len(documents)
    except Exception as e:
        # 如果重排失败（比如模型没下载完），降级使用 Top-5 原始结果
        print(f"⚠️ 重排失败，降级使用原始结果: {e}")
        reranked_docs = documents[:5]

    # 4. 定义 SSE 事件生成器
    async def event_generator():
        # 先发送 session_id（让前端知道当前会话）
        yield {
            "event": "session",
            "data": session_id
        }

        # 再发送检索到的引用文档（方便前端展示引用来源）
        yield {
            "event": "references",
            "data": str(reranked_docs)  # 简化处理，实际可用 JSON
        }

        # 流式生成回答
        full_answer = ""
        try:
            for chunk in generator.generate_stream(q, reranked_docs, session_id):
                full_answer += chunk
                yield {
                    "event": "chunk",
                    "data": chunk
                }
        except Exception as e:
            yield {
                "event": "error",
                "data": f"生成回答失败: {str(e)}"
            }
            return

        # 全部生成完毕，发送结束信号
        yield {
            "event": "done",
            "data": "回答完成"
        }

    # 5. 返回 SSE 流式响应
    return EventSourceResponse(event_generator())


# ==================== 接口 3：清空会话（可选） ====================
@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """清空指定会话的历史记录"""
    try:
        session_manager.clear_session(session_id)
        return {"status": "success", "message": f"会话 {session_id} 已清空"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空会话失败: {str(e)}")


# ==================== 接口 4：健康检查 ====================
@app.get("/health")
async def health_check():
    """检查服务是否正常运行"""
    return {
        "status": "ok",
        "service": "RAG QA System",
        "redis": "connected" if session_manager.client else "disconnected"
    }


# ==================== 启动命令（方便直接运行） ====================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )