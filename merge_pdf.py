import PyPDF2
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import RectangleObject

def merge_pdfs_horizontally(pdf1_path, pdf2_path, output_path):
    """
    将两个PDF文件横向拼接在一起
    
    Args:
        pdf1_path: 第一个PDF文件路径（左侧）
        pdf2_path: 第二个PDF文件路径（右侧）
        output_path: 输出PDF文件路径
    """
    
    # 读取两个PDF文件
    pdf1 = PdfReader(pdf1_path)
    pdf2 = PdfReader(pdf2_path)
    
    # 创建PDF写入器
    pdf_writer = PdfWriter()
    
    # 获取两个PDF的页数
    pages1 = len(pdf1.pages)
    pages2 = len(pdf2.pages)
    
    # 确保两个PDF页数相同，如果不同则取较小值
    min_pages = min(pages1, pages2)
    if pages1 != pages2:
        print(f"警告: 两个PDF页数不同 (PDF1: {pages1}页, PDF2: {pages2}页)")
        print(f"将只处理前 {min_pages} 页")
    
    # 逐页拼接
    for i in range(min_pages):
        # 获取两个PDF的当前页
        page1 = pdf1.pages[i]
        page2 = pdf2.pages[i]
        
        # 获取页面尺寸
        width1 = float(page1.mediabox.width)
        height1 = float(page1.mediabox.height)
        width2 = float(page2.mediabox.width)
        height2 = float(page2.mediabox.height)
        
        # 计算新页面的尺寸（宽度相加，高度取最大值）
        new_width = width1 + width2
        new_height = max(height1, height2)
        
        # 创建新页面
        new_page = page1.__class__(None)
        new_page.mediabox = RectangleObject([0, 0, new_width, new_height])
        
        # 合并页面
        new_page.merge_page(page1)
        
        # 将第二个页面平移到右侧
        translation = (width1, 0)
        new_page.merge_translated_page(page2, translation[0], translation[1])
        
        # 将新页面添加到写入器
        pdf_writer.add_page(new_page)
    
    # 保存合并后的PDF
    with open(output_path, 'wb') as output_file:
        pdf_writer.write(output_file)
    
    print(f"PDF合并完成！输出文件: {output_path}")

def merge_pdfs_horizontally_with_pymupdf(pdf1_path, pdf2_path, output_path):
    """
    使用PyMuPDF（fitz）库将两个PDF横向拼接在一起
    这个方法可能提供更好的效果
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("请先安装PyMuPDF: pip install PyMuPDF")
        return
    
    # 打开两个PDF文件
    doc1 = fitz.open(pdf1_path)
    doc2 = fitz.open(pdf2_path)
    
    # 创建新PDF文档
    result_doc = fitz.open()
    
    # 获取页数
    pages1 = len(doc1)
    pages2 = len(doc2)
    min_pages = min(pages1, pages2)
    
    if pages1 != pages2:
        print(f"警告: 两个PDF页数不同 (PDF1: {pages1}页, PDF2: {pages2}页)")
        print(f"将只处理前 {min_pages} 页")
    
    # 逐页处理
    for i in range(min_pages):
        # 获取页面
        page1 = doc1[i]
        page2 = doc2[i]
        
        # 获取页面尺寸
        rect1 = page1.rect
        rect2 = page2.rect
        
        # 计算新页面尺寸
        new_width = rect1.width + rect2.width
        new_height = max(rect1.height, rect2.height)
        
        # 创建新页面
        new_page = result_doc.new_page(width=new_width, height=new_height)
        
        # 将第一个页面复制到新页面（左侧）
        new_page.show_pdf_page(
            fitz.Rect(0, 0, rect1.width, rect1.height),
            doc1,
            i
        )
        
        # 将第二个页面复制到新页面（右侧）
        new_page.show_pdf_page(
            fitz.Rect(rect1.width, 0, rect1.width + rect2.width, rect2.height),
            doc2,
            i
        )
    
    # 保存结果
    result_doc.save(output_path)
    result_doc.close()
    doc1.close()
    doc2.close()
    
    print(f"PDF合并完成！输出文件: {output_path}")

if __name__ == "__main__":
    # 使用示例
    pdf1_path = "/Users/mc03002/Downloads/figure_2.pdf"   # 左侧PDF文件
    pdf2_path = "/Users/mc03002/Downloads/figure_6.pdf"  # 右侧PDF文件
    output_path = "/Users/mc03002/Downloads/figure_26.pdf"  # 输出PDF文件
    
    # 方法1: 使用PyPDF2（基础功能）
    try:
        merge_pdfs_horizontally(pdf1_path, pdf2_path, output_path)
    except Exception as e:
        print(f"PyPDF2方法失败: {e}")
        print("尝试使用PyMuPDF方法...")
        merge_pdfs_horizontally_with_pymupdf(pdf1_path, pdf2_path, output_path)