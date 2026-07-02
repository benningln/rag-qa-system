import os
import pandas as pd
import chromadb
from backend.config import BASE_DIR

# ========== 使用 transformers 手动加载 BGE 模型 ==========
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# 模型路径
MODEL_PATH = os.path.join(BASE_DIR, 'local_models', 'bge-large-zh-v1.5')
MODEL_PATH = os.path.abspath(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 找不到模型文件: {MODEL_PATH}")

print(f"🔄 从本地加载 BGE 模型（使用 transformers）: {MODEL_PATH}")

# 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.eval()  # 切换到推理模式

print(f"✅ BGE 模型加载成功！")


def get_embeddings(texts):
    """
    使用 BGE 模型生成向量（手动实现 pooling）
    """
    if isinstance(texts, str):
        texts = [texts]

    # 1. 编码文本
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,  # BGE 最大支持 512 token
        return_tensors='pt'
    )

    # 2. 通过模型获取向量
    with torch.no_grad():
        outputs = model(**encoded)
        # 使用 CLS token 的向量（BGE 使用 CLS 池化）
        embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]

    # 3. 归一化（L2 归一化）
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # 4. 转为 numpy 列表
    return embeddings.cpu().numpy().tolist()


# ========== Chroma 配置（不变） ==========
PERSIST_DIR = str(BASE_DIR / 'data' / 'chroma_db')

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _client is None:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=PERSIST_DIR)
        _collection = _client.get_or_create_collection(
            name="rag_docs",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Chroma 已连接，当前记录数: {_collection.count()}")
    return _collection


def add_chunks_to_chroma(chunks_df: pd.DataFrame) -> int:
    if chunks_df.empty:
        return 0
    collection = get_collection()
    ids = chunks_df['chunk_id'].tolist()
    documents = chunks_df['text'].tolist()
    metadatas = [{"page": int(row['page'])} for _, row in chunks_df.iterrows()]
    print(f"🔄 正在生成 {len(documents)} 个文本块的向量（本地 BGE 模型）...")
    embeddings = get_embeddings(documents)
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"✅ 入库 {len(ids)} 个块")
    return len(ids)


def search_chroma(query: str, top_k: int = 20):
    collection = get_collection()
    print(f"🔄 生成查询向量...")
    query_emb = get_embeddings([query])[0]
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    print(f"🔍 召回了 {len(results['documents'][0])} 个文档")
    return results