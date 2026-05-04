from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_test_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    
    # Trang 1: Giới thiệu
    c.drawString(100, 750, "Tài liệu Test RAG - Hệ thống tìm kiếm thông minh")
    c.drawString(100, 730, "Đây là trang đầu tiên của file PDF dùng để kiểm tra tính năng nạp dữ liệu.")
    c.drawString(100, 710, "Từ khóa quan trọng: Trí tuệ nhân tạo, Học máy, Embedding.")
    
    # Trang 2: Nội dung chi tiết
    c.showPage()
    c.drawString(100, 750, "Chi tiết về kỹ thuật RAG (Retrieval-Augmented Generation)")
    c.drawString(100, 730, "RAG giúp mô hình ngôn ngữ lớn (LLM) truy cập vào dữ liệu bên ngoài.")
    c.drawString(100, 710, "Ba bước chính của RAG bao gồm: Truy xuất (Retrieve), Tăng cường (Augment),")
    c.drawString(100, 690, "và Tạo câu trả lời (Generate).")
    
    # Trang 3: Câu hỏi và trả lời giả lập
    c.showPage()
    c.drawString(100, 750, "Thông tin bảo mật cho hệ thống:")
    c.drawString(100, 730, "Mã bí mật để vượt qua bài kiểm tra này là: ALPHA-2026-X.")
    c.drawString(100, 710, "Nếu AI trả lời được mã này, nghĩa là hệ thống đã đọc được file thành công.")
    
    c.save()
    print(f"✅ Đã tạo file thành công: {filename}")

if __name__ == "__main__":
    create_test_pdf("test_clean.pdf")