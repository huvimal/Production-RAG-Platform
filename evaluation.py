from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset
from rag_pipeline import simple_retrieve, hybrid_retrieve, hybrid_rerank_retrieve
from ingest import ingest_pdf
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# Setup RAGAS
ragas_llm = LangchainLLMWrapper(ChatGroq(model="qwen/qwen3-32b", temperature=0))
ragas_emb = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

def build_eval_dataset(questions: list, ground_truths: list,
                        answers: list, contexts: list) -> Dataset:
    """Tạo dataset cho RAGAS evaluate."""
    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

def run_ragas(dataset: Dataset) -> dict:
    """Chạy RAGAS evaluation, trả về dict scores."""
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_emb,
    )
    df = results.to_pandas()
    return {
        "faithfulness":      df["faithfulness"].mean(),
        "answer_relevancy":  df["answer_relevancy"].mean(),
        "context_precision": df["context_precision"].mean(),
    }

def benchmark_all(col_name: str, test_cases: list) -> pd.DataFrame:
    """
    Benchmark cả 3 strategies.
    test_cases = [{"question": "...", "ground_truth": "...", "answer": "..."}, ...]
    """
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    strategies = {
        "Simple RAG":         simple_retrieve,
        "Hybrid Search":      hybrid_retrieve,
        "Hybrid + Reranking": hybrid_rerank_retrieve,
    }

    RAG_PROMPT = """Answer based on context:
{context}

Question: {question}"""
    rows = []

    for strategy_name, retrieve_fn in strategies.items():
        print(f"⏳ Đánh giá: {strategy_name}...")
        questions, answers, contexts, ground_truths = [], [], [], []

        for tc in test_cases:
            docs = retrieve_fn(tc["question"], col_name)
            context = "".join(docs)
            answer = llm.invoke(
                RAG_PROMPT.format(context=context, question=tc["question"])
            ).content

            questions.append(tc["question"])
            answers.append(answer)
            contexts.append(docs)
            ground_truths.append(tc["ground_truth"])

        dataset = build_eval_dataset(questions, ground_truths, answers, contexts)
        scores = run_ragas(dataset)
        scores["Strategy"] = strategy_name
        rows.append(scores)
        print(f"  ✅ Faithfulness: {scores['faithfulness']:.3f} | "
              f"Relevancy: {scores['answer_relevancy']:.3f} | "
              f"Precision: {scores['context_precision']:.3f}")

    df = pd.DataFrame(rows).set_index("Strategy")
    return df

if __name__ == "__main__":
    # Thay bằng PDF và câu hỏi thực tế của bạn
    PDF_PATH = "test_clean.pdf"
    col_name = ingest_pdf(PDF_PATH)

    TEST_CASES = [
        {
            "question": "Mã bí mật để vượt qua bài kiểm tra này là gì?",
            "ground_truth": "AI phải trả lời chính xác là ALPHA-2026-X."
        },
        {
            "question": "RAG là gì và nó giúp ích gì cho mô hình ngôn ngữ lớn (LLM)?",
            "ground_truth": "AI sẽ trích dẫn thông tin về việc RAG giúp LLM truy cập vào dữ liệu bên ngoài để tăng tính chính xác."
        },
    ]

    print("" + "="*55)
    print("RAGAS BENCHMARK — 3 Retrieval Strategies")
    print("="*55)

    results_df = benchmark_all(col_name, TEST_CASES)

    print("📊 KẾT QUẢ CUỐI CÙNG:")
    print(results_df.to_string())

    # Lưu kết quả ra CSV để ghi vào README
    results_df.to_csv("benchmark_results.csv")
    print("✅ Đã lưu: benchmark_results.csv")