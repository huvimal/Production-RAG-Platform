import chromadb
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import os

# --- Cấu hình hệ thống ---
CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Khởi tạo các thành phần AI
embed_model = SentenceTransformer(EMBED_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

def extract_pdf_text(pdf_path: str) -> str:
    """Sử dụng PyMuPDF để trích xuất văn bản từ PDF."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            page_text = page.get_text("text")
            if page_text.strip():
                text += page_text + "\n\n"
        doc.close()
    except Exception as e:
        print(f"❌ Lỗi đọc PDF: {e}")
    return text

def chunk_text_by_line(text: str) -> list:
    """Chia nhỏ văn bản dựa trên dòng để đảm bảo file ngắn vẫn ra nhiều chunks."""
    # Lấy các dòng có độ dài ý nghĩa
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 10]
    chunks = []
    # Gom 3 dòng thành 1 đoạn (chunk) để tăng mật độ thông tin
    for i in range(0, len(lines), 3):
        chunk = " ".join(lines[i:i+3])
        chunks.append(chunk)
    return chunks

def get_all_chunks(col_name: str) -> list:
    """Lấy tất cả chunks phục vụ cho thuật toán BM25 (Hybrid Search)."""
    try:
        col = chroma_client.get_collection(col_name)
        result = col.get()
        return result["documents"]
    except Exception as e:
        print(f"❌ Lỗi khi lấy chunks: {e}")
        return []

def ingest_pdf(pdf_path: str):
    """Quy trình nạp dữ liệu vào ChromaDB."""
    pdf_name = os.path.basename(pdf_path)
    col_name = pdf_name.replace(".pdf", "").replace(" ", "_")[:40]
    
    print(f"📄 Đang xử lý: {pdf_name}")
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text_by_line(text)
    
    if not chunks:
        print("⚠️ Không tìm thấy nội dung văn bản hợp lệ!")
        return col_name
        
    print(f"✂️  Đã chia văn bản thành {len(chunks)} đoạn nhỏ.")
    
    # Lấy hoặc tạo mới collection
    col = chroma_client.get_or_create_collection(name=col_name)
    
    # Dọn dẹp dữ liệu cũ của chính file này nếu đã tồn tại
    if col.count() > 0:
        print("🧹 Đang dọn dẹp dữ liệu cũ trong collection...")
        col.delete(where={"source": pdf_name})

    # Chuyển đổi sang Vector và lưu trữ
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings = embed_model.encode(batch).tolist()
        col.upsert(
            ids=[f"{pdf_name}_{i+j}" for j in range(len(batch))],
            documents=batch,
            embeddings=embeddings,
            metadatas=[{"source": pdf_name} for _ in range(len(batch))]
        )
    
    print(f"✅ Hoàn tất! Collection '{col_name}' hiện có {col.count()} chunks.")
    # TRẢ VỀ DUY NHẤT 1 GIÁ TRỊ ĐỂ ĐỒNG BỘ VỚI UI
    return col_name

if __name__ == "__main__":
    # Test nhanh với file PDF có sẵn
    target_file = "test_clean.pdf" 
    
    if os.path.exists(target_file):
        name = ingest_pdf(target_file)
        print(f"\n🚀 SẴN SÀNG! Tên collection là: {name}")
    else:
        print(f"❌ Không tìm thấy file '{target_file}' để chạy thử.")