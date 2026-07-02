import os
from typing import List, Tuple
from FlagEmbedding import FlagReranker
from backend.config import BASE_DIR

# ========== 加载本地 Reranker 模型 ==========
MODEL_PATH = os.path.join(BASE_DIR, 'local_models', 'bge-reranker-large')
MODEL_PATH = os.path.abspath(MODEL_PATH).replace('\\', '/')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 找不到 Reranker 模型: {MODEL_PATH}")

print(f"🔄 从本地加载 Reranker 模型: {MODEL_PATH}")
_reranker = FlagReranker(MODEL_PATH, use_fp16=True)
print(f"✅ Reranker 模型加载成功！")


def rerank(query: str, documents: List[str], top_k: int = 5) -> Tuple[List[str], List[float]]:
    if not documents:
        return [], []
    if len(documents) <= top_k:
        return documents, [1.0] * len(documents)

    pairs = [[query, doc] for doc in documents]
    print(f"🔄 正在对 {len(pairs)} 个候选文档进行精排打分...")
    scores = _reranker.compute_score(pairs, normalize=True)

    if isinstance(scores, float):
        scores = [scores]

    sorted_pairs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in sorted_pairs[:top_k]]
    top_scores = [score for _, score in sorted_pairs[:top_k]]

    print(f"✅ 重排完成，已选出最相关的 {len(top_docs)} 个文档片段")
    return top_docs, top_scores