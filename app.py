import gradio as gr
from ingest import ingest_pdf
from rag_pipeline import ask, benchmark
import os

# Biến lưu trữ tên collection hiện tại
current_collection = {"name": None}

# ── Xử lý upload PDF ─────────────────────────────────────
def handle_upload(pdf_file):
    if pdf_file is None:
        return "⚠️ Chưa chọn file!", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
    try:
        # SỬA DÒNG NÀY: Chỉ nhận 1 giá trị là col_name
        col_name = ingest_pdf(pdf_file.name) 
        
        # Cập nhật tên collection vào biến global
        current_collection["name"] = col_name
        
        # LƯU Ý: Không dùng biến 'col.count()' ở đây nữa vì biến 'col' không tồn tại
        status = f"✅ Đã index thành công file vào collection: {col_name}"
        
        # Kích hoạt lại các nút bấm trên giao diện
        return status, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
    except Exception as e:
        return f"❌ Lỗi: {e}", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

# ── Xử lý câu hỏi ────────────────────────────────────────
def handle_question(question: str, strategy: str):
    col_name = current_collection["name"]
    if not col_name:
        return "⚠️ Hãy upload PDF trước!", ""
    if not question.strip():
        return "⚠️ Nhập câu hỏi!", ""
    try:
        result = ask(question, col_name, strategy=strategy)
        # Format lại nguồn để hiển thị Markdown đẹp hơn
        sources_text = "\n\n---\n\n".join(
            [f"**Nguồn {i+1}:**\n{s[:500]}..." for i, s in enumerate(result["sources"])]
        )
        return result["answer"], sources_text
    except Exception as e:
        return f"❌ Lỗi: {e}", ""

# ── Benchmark 3 strategies ────────────────────────────────
def handle_benchmark(question: str):
    col_name = current_collection["name"]
    if not col_name:
        return "⚠️ Hãy upload PDF trước!"
    if not question.strip():
        return "⚠️ Nhập câu hỏi!"
    try:
        results = benchmark(question, col_name)
        output = f"# Kết quả Benchmark cho: {question}\n\n"
        labels = {
            "simple":        "🔵 Simple Vector RAG",
            "hybrid":        "🟡 Hybrid Search (BM25 + Vector)",
            "hybrid_rerank": "🟢 Hybrid + CrossEncoder Reranking"
        }
        for name, label in labels.items():
            # Thêm xuống dòng để Markdown nhận diện đúng thẻ tiêu đề
            output += f"## {label}\n{results[name]['answer']}\n\n---\n\n"
        return output
    except Exception as e:
        return f"❌ Lỗi: {e}"

# ── Gradio UI ─────────────────────────────────────────────
with gr.Blocks(title="Production RAG Platform", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📚 Production RAG Platform
    **LangChain + ChromaDB + Hybrid Search + CrossEncoder Reranking**
    """)

    with gr.Tab("📄 Q&A trên PDF"):
        with gr.Row():
            pdf_upload = gr.File(label="1. Upload PDF", file_types=[".pdf"])
            upload_status = gr.Textbox(label="Trạng thái hệ thống", interactive=False)

        strategy_choice = gr.Radio(
            choices=["simple", "hybrid", "hybrid_rerank"],
            value="hybrid_rerank",
            label="2. Chọn chiến lược Retrieval",
            info="hybrid_rerank cho kết quả chính xác nhất",
            interactive=False
        )

        question_input = gr.Textbox(
            placeholder="Nhập câu hỏi về nội dung PDF tại đây...",
            label="3. Đặt câu hỏi", 
            interactive=False
        )
        ask_btn = gr.Button("🔍 Gửi câu hỏi", variant="primary", interactive=False)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 🤖 Trả lời")
                answer_output = gr.Markdown()
            with gr.Column(scale=1):
                gr.Markdown("### 📍 Nguồn trích dẫn")
                sources_output = gr.Markdown()

        # Sự kiện khi upload file
        pdf_upload.change(
            fn=handle_upload, 
            inputs=[pdf_upload],
            outputs=[upload_status, strategy_choice, question_input, ask_btn]
        )

        # Sự kiện khi bấm nút hoặc Enter
        ask_btn.click(
            fn=handle_question,
            inputs=[question_input, strategy_choice],
            outputs=[answer_output, sources_output]
        )
        question_input.submit(
            fn=handle_question,
            inputs=[question_input, strategy_choice],
            outputs=[answer_output, sources_output]
        )

    with gr.Tab("📊 So sánh Strategies (Benchmark)"):
        gr.Markdown("Chạy đồng thời 3 chiến lược để so sánh chất lượng câu trả lời.")
        bench_question = gr.Textbox(
            placeholder="Nhập câu hỏi cần so sánh...",
            label="Câu hỏi Benchmark"
        )
        bench_btn = gr.Button("⚡ Chạy Benchmark", variant="primary")
        bench_output = gr.Markdown()

        bench_btn.click(
            fn=handle_benchmark, 
            inputs=[bench_question],
            outputs=[bench_output]
        )

if __name__ == "__main__":
    # Đảm bảo server chạy đúng port bạn mong muốn
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)