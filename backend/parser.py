import io
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
from paddleocr import PaddleOCR

# 初始化 OCR 模型（第一次运行时会自动下载模型文件，约 10MB，请保持网络畅通）
# use_angle_cls=True 开启方向分类，lang='ch' 支持中英文混合
ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)


def parse_pdf(file_path: str) -> pd.DataFrame:
    """
    解析 PDF 文件，提取每页文本。
    如果某页提取的文本少于 10 个字符，则认为该页是纯图片型，
    自动调用 PaddleOCR 进行文字识别补全。

    参数:
        file_path: PDF 文件的路径

    返回:
        pd.DataFrame: 包含两列 ['page', 'text']
    """
    # 打开 PDF 文档
    doc = fitz.open(file_path)
    all_pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # 先尝试直接提取文本
        text = page.get_text().strip()

        # 如果文本太短，说明可能是扫描图片页
        if len(text) < 10:
            # 将当前页面渲染为图片（分辨率 300 DPI 保证 OCR 清晰度）
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            # 调用 PaddleOCR 识别
            result = ocr.ocr(img, cls=True)
            if result and result[0]:
                # 提取所有识别到的文字，并过滤置信度低于 0.5 的结果
                texts = []
                for line in result[0]:
                    # line 的结构: [bbox, (text, confidence)]
                    if line[1][1] > 0.5:
                        texts.append(line[1][0])
                text = "\n".join(texts)
            else:
                text = ""  # OCR 也没识别到，留空

        all_pages.append({
            "page": page_num + 1,
            "text": text.strip()
        })

    doc.close()

    # 用 pandas 清洗数据：去除多余换行、空格，过滤掉空白页
    df = pd.DataFrame(all_pages)
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
    df['text'] = df['text'].str.strip()
    df = df[df['text'] != '']  # 过滤完全空白的页

    return df


# ============ 测试代码（只在你直接运行这个文件时执行） ============
if __name__ == "__main__":
    # 这是一个简单的测试，方便你验证解析功能
    test_pdf = input("请输入要测试的 PDF 文件路径: ").strip()
    try:
        df = parse_pdf(test_pdf)
        print(f"\n✅ 成功解析 {len(df)} 页文本！")
        print("\n前两页预览：")
        print(df.head(2))
    except Exception as e:
        print(f"\n❌ 解析失败: {e}")