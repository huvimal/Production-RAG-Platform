🚀 Production-Grade RAG Platform

Advanced Retrieval • Evaluation • Benchmarking

A production-oriented Retrieval-Augmented Generation (RAG) platform designed to handle complex PDF documents with high accuracy by combining multi-stage retrieval strategies and a built-in evaluation + benchmarking system.

This project focuses not only on building a RAG system, but on measuring, comparing, and improving it systematically.

🔗 Live Demo

https://huvimal-production-rag-platform.hf.space

🎯 Problem Statement

Traditional RAG pipelines often fail in real-world scenarios due to:

❌ Weak keyword matching (misses exact terms)
❌ Pure vector search (semantic drift)
❌ No evaluation → “looks correct but is wrong”
❌ No benchmarking → cannot improve systematically

👉 This project addresses all of the above.

🧠 System Architecture
User Query
    ↓
Query Processing
    ↓
Retrieval Layer
    ├── Vector Search (Dense Retrieval)
    ├── BM25 (Sparse Retrieval)
    └── Hybrid Fusion (RRF)
    ↓
Reranking Layer (Cross-Encoder)
    ↓
Context Selection
    ↓
LLM Generation (Llama 3.3 via Groq)
    ↓
Answer + Citations
    ↓
Evaluation (RAGAS)

⚙️ Core Components
1. Multi-Stage Retrieval
🔹 Dense Retrieval (Vector Search)
Embedding-based semantic search
Handles paraphrasing & intent
🔹 Sparse Retrieval (BM25)
Strong keyword matching
Critical for exact technical terms
🔹 Hybrid Retrieval (RRF Fusion)
Combines both signals using Reciprocal Rank Fusion
Improves recall without sacrificing precision

3. Reranking Layer
Model: Cross-Encoder
Purpose:
Re-score retrieved documents
Push most relevant chunks to top
Result:
Significant boost in precision & answer quality

5. Generation Layer
LLM: Llama 3.3 (via Groq API)
Features:
Context-aware answering
Source-grounded responses
Reduced hallucination via strict context injection

7. Evaluation System (RAGAS)

Automated evaluation pipeline using:

Faithfulness
→ Is the answer grounded in retrieved context?
Answer Relevancy
→ Does it answer the question correctly?
Context Precision
→ Is retrieved context actually useful?

📊 Benchmark Results

Strategy	Faithfulness	Relevancy	Context Precision
Simple RAG	   0.850	      0.820	          0.780
Hybrid Search	0.910	      0.880	          0.850
Hybrid + Reranking	0.965	  0.940	          0.925

🔍 Key Insight
Hybrid > Vector-only
Reranking = biggest performance boost
Evaluation is mandatory, not optional

🧩 Tech Stack
Layer	Technology
LLM	Llama 3.3 (Groq API)
Embeddings	Sentence Transformers
Vector DB	ChromaDB
Retrieval	BM25 + Dense + RRF
Reranking	Cross-Encoder
Evaluation	RAGAS
Backend	Python
UI	Gradio
📁 Project Structure
.
├── app.py              # Gradio UI
├── ingest.py           # PDF processing + chunking + indexing
├── rag_pipeline.py     # Retrieval + Reranking + Generation
├── ragas_eval.py       # Evaluation + Benchmarking
├── benchmark_results.csv
├── requirements.txt
└── .env

⚡ Quick Start
1. Install dependencies
pip install -r requirements.txt
2. Setup environment

Create .env file:
GROQ_API_KEY=your_api_key_here

3. Run application
python app.py

🧪 Evaluation Workflow
Define test queries
Run pipeline (all strategies)
Collect outputs
Evaluate with RAGAS
Export results → benchmark_results.csv

👉 Enables data-driven improvement, not guesswork.

⚖️ Design Trade-offs
Decision	Trade-off
Hybrid Search	↑ Accuracy, ↑ Complexity
Reranking	↑ Precision, ↑ Latency
Chunk Size	↑ Context quality vs ↓ recall
Evaluation (RAGAS)	↑ Reliability, ↑ compute cost

🔒 Production Considerations
Caching embeddings to reduce cost
Batch retrieval for scalability
Monitoring hallucination rate
Logging queries for continuous improvement
Versioning datasets + benchmarks

🚧 Future Improvements
 Query rewriting (improve retrieval)
 Multi-hop reasoning
 Streaming responses
 Docker + deployment pipeline
 Vector DB scaling (FAISS / Milvus)
 Fine-tuned reranker
 
👤 Author
Lê Mai Vĩnh Hưng

🎯 Focus: AI Engineering • Data Engineering • LLM Systems
