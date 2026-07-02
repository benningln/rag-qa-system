import pandas as pd
import re


def chunk_text(df: pd.DataFrame, chunk_size: int = 500, overlap: int = 100) -> pd.DataFrame:
    """
    将解析好的 PDF 文本按页进行分块（chunk），并保留重叠部分（overlap）。

    参数:
        df: parse_pdf() 返回的 DataFrame，必须包含 'page' 和 'text' 列
        chunk_size: 每个块的最大字数（建议 300~800，中文按字符数算）
        overlap: 相邻块之间的重叠字数（建议 chunk_size 的 20%~30%）

    返回:
        pd.DataFrame: 包含四列 ['chunk_id', 'page', 'text', 'start_idx']
    """
    all_chunks = []

    for _, row in df.iterrows():
        page_num = row['page']
        text = row['text']

        # 先用正则按句子边界粗略分割（避免把句子切断）
        # 按中文句号、问号、感叹号、换行分割
        sentences = re.split(r'(?<=[。！？\n])', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 构建滑动窗口
        current_chunk = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            # 如果单句超过 chunk_size，强行截断（避免无限大块）
            if sent_len > chunk_size:
                # 按 chunk_size 强行切
                for i in range(0, sent_len, chunk_size - overlap):
                    piece = sent[i:i + chunk_size]
                    if piece:
                        chunk_id = f"page_{page_num}_chunk_{len(all_chunks)}"
                        all_chunks.append({
                            "chunk_id": chunk_id,
                            "page": page_num,
                            "text": piece,
                            "start_idx": i
                        })
                continue

            # 正常累积句子
            if current_len + sent_len <= chunk_size:
                current_chunk.append(sent)
                current_len += sent_len
            else:
                # 当前块已满，保存
                if current_chunk:
                    chunk_text = "".join(current_chunk).strip()
                    if chunk_text:
                        chunk_id = f"page_{page_num}_chunk_{len(all_chunks)}"
                        all_chunks.append({
                            "chunk_id": chunk_id,
                            "page": page_num,
                            "text": chunk_text,
                            "start_idx": 0  # 简化处理，不精确追踪起始位置
                        })

                # 重叠部分：从当前块尾部取 overlap 个字符作为下一块的种子
                overlap_text = ""
                if current_chunk:
                    # 把当前块所有句子连起来，取最后 overlap 个字符
                    full_current = "".join(current_chunk)
                    if len(full_current) >= overlap:
                        overlap_text = full_current[-overlap:]
                    else:
                        overlap_text = full_current

                # 重置当前块，用重叠部分 + 当前句子初始化
                current_chunk = [overlap_text] if overlap_text else []
                current_chunk.append(sent)
                current_len = len(overlap_text) + sent_len

        # 处理页面最后剩余的内容
        if current_chunk:
            chunk_text = "".join(current_chunk).strip()
            if chunk_text:
                chunk_id = f"page_{page_num}_chunk_{len(all_chunks)}"
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_num,
                    "text": chunk_text,
                    "start_idx": 0
                })

    # 返回 DataFrame
    result_df = pd.DataFrame(all_chunks)

    # 过滤掉异常短的块（少于 10 个字符的碎片无意义）
    result_df = result_df[result_df['text'].str.len() >= 10]

    return result_df


# ============ 测试代码 ============
if __name__ == "__main__":
    # 模拟一个简单测试
    test_data = pd.DataFrame([
        {"page": 1, "text": "这是第一页的内容。包含多个句子。这里是另一个句子。"},
        {"page": 2, "text": "第二页内容。这里讨论RAG系统的核心原理。需要分块处理。"}
    ])

    chunks = chunk_text(test_data, chunk_size=30, overlap=10)
    print(f"✅ 生成了 {len(chunks)} 个文本块")
    print("\n预览：")
    print(chunks[['chunk_id', 'page', 'text']].head())