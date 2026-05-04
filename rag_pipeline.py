# Phần 1: Setup
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from ingest import get_all_chunks
import numpy as np

load_dotenv()

CHROMA_PATH = "./chroma_db"
EMBED_MODEL  = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

embed_model  = SentenceTransformer(EMBED_MODEL)
reranker     = CrossEncoder(RERANK_MODEL)
llm          = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
chroma       = chromadb.PersistentClient(path=CHROMA_PATH)

RAG_PROMPT = """Trả lời câu hỏi dựa trên context sau.
Trích dẫn thông tin cụ thể từ tài liệu. Nếu không có thông tin, hãy nói rõ.

Context:
{context}

Câu hỏi: {question}
Trả lời (bằng tiếng Việt):"""
# Phần 2: 3 Retrievers
def simple_retrieve(query: str, col_name: str, k: int = 4) -> list:
    """Strategy 1: Vector search đơn giản."""
    col = chroma.get_collection(col_name)
    q_emb = embed_model.encode([query]).tolist()
    results = col.query(query_embeddings=q_emb, n_results=k)
    return results["documents"][0]

def hybrid_retrieve(query: str, col_name: str, k: int = 4) -> list:
    """Strategy 2: BM25 + Vector search kết hợp (Reciprocal Rank Fusion)."""
    # Vector search
    col = chroma.get_collection(col_name)
    q_emb = embed_model.encode([query]).tolist()
    vec_results = col.query(query_embeddings=q_emb, n_results=k*2)
    vec_docs = vec_results["documents"][0]

    # BM25 search
    all_chunks = get_all_chunks(col_name)
    tokenized = [doc.lower().split() for doc in all_chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:k*2]
    bm25_docs = [all_chunks[i] for i in top_bm25_idx]

    # Reciprocal Rank Fusion (RRF) — kết hợp 2 ranking
    scores = {}
    for rank, doc in enumerate(vec_docs):
        scores[doc] = scores.get(doc, 0) + 1.0 / (rank + 60)
    for rank, doc in enumerate(bm25_docs):
        scores[doc] = scores.get(doc, 0) + 1.0 / (rank + 60)

    # Sắp xếp theo RRF score, lấy top k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:k]]

def hybrid_rerank_retrieve(query: str, col_name: str,
                            initial_k: int = 20, final_k: int = 4) -> list:
    """Strategy 3: Hybrid search + CrossEncoder reranking."""
    # Bước 1: Hybrid retrieve nhiều doc hơn cần
    candidates = hybrid_retrieve(query, col_name, k=initial_k)

    # Bước 2: CrossEncoder rerank
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)

    # Bước 3: Lấy top final_k sau rerank
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:final_k]]
# Phần 3: RAG Answer + Benchmark
STRATEGIES = {
    "simple":         simple_retrieve,
    "hybrid":         hybrid_retrieve,
    "hybrid_rerank":  hybrid_rerank_retrieve,
}

def ask(question: str, col_name: str, strategy: str = "hybrid_rerank") -> dict:
    try:
        retrieve_fn = STRATEGIES.get(strategy, hybrid_rerank_retrieve)
        docs = retrieve_fn(question, col_name)
        
        if not docs:
            return {"answer": "⚠️ Không tìm thấy dữ liệu liên quan.", "sources": [], "strategy": strategy}

        context = "\n\n".join(docs)
        prompt = RAG_PROMPT.format(context=context, question=question)
        
        # Thử gọi LLM
        response = llm.invoke(prompt)
        
        return {
            "answer": response.content,
            "sources": docs,
            "strategy": strategy,
        }
    except Exception as e:
        # Nếu có lỗi, nó sẽ hiện chữ đỏ ở phần Trả lời thay vì trắng tinh
        return {
            "answer": f"❌ LỖI HỆ THỐNG: {str(e)}",
            "sources": docs,
            "strategy": strategy,
        }
    
def benchmark(question: str, col_name: str) -> dict:
    """So sánh 3 strategies với cùng 1 câu hỏi."""
    results = {}
    for name, fn in STRATEGIES.items():
        docs = fn(question, col_name)
        context = "".join(docs)
        prompt = RAG_PROMPT.format(context=context, question=question)
        answer = llm.invoke(prompt).content
        results[name] = {"answer": answer, "sources": docs}
    return results

if __name__ == "__main__":
    # Test nhanh — cần có PDF đã ingest
    col_name = "test_clean"  # thay bằng tên collection thật
    question = "Mã bí mật để vượt qua bài kiểm tra này là gì"

    print("=== Simple RAG ===")
    r = ask(question, col_name, strategy="simple")
    print(r["answer"][:200])

    print("=== Hybrid + Rerank ===")
    r = ask(question, col_name, strategy="hybrid_rerank")
    print(r["answer"][:200])
