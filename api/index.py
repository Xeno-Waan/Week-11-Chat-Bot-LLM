import os
import sys
from dotenv import load_dotenv

# Muat file .env secara lokal
load_dotenv()

import wandb
from flask import Flask, request, jsonify
from flask_cors import CORS

# Konfigurasi Weights & Biases agar tidak memblokir serverless Vercel jika API Key tidak di-set
if not os.environ.get("WANDB_API_KEY"):
    os.environ["WANDB_MODE"] = "disabled"

# Tambahkan direktori utama ke sys.path agar bisa mengimpor rag_core secara lokal maupun di Vercel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from api.rag_core import (
        llm, rag_chain, generation_chain, embeddings, retriever, 
        format_docs, Chroma, index_uploaded_pdf
    )
except Exception as e:
    err_msg_1 = str(e)
    try:
        from rag_core import (
            llm, rag_chain, generation_chain, embeddings, retriever, 
            format_docs, Chroma, index_uploaded_pdf
        )
    except Exception as e2:
        err_msg_2 = str(e2)
        # Fallback/Mock jika terjadi masalah impor atau error inisialisasi RAG
        class MockLLM:
            def invoke(self, text):
                class Content:
                    content = f"Ini jawaban baseline mock (Detail error: {err_msg_2}) untuk: '{text}'"
                return Content()
        class MockRAGChain:
            def invoke(self, text):
                return f"Ini jawaban RAG mock (Detail error: {err_msg_1}) untuk: '{text}'"
        
        llm = MockLLM()
        rag_chain = MockRAGChain()
        generation_chain = MockRAGChain()
        embeddings = None
        retriever = None
        def format_docs(docs): return str(docs)
        Chroma = None
        def index_uploaded_pdf(file_path): return 0, 0

app = Flask(__name__)
# Aktifkan CORS agar frontend HTML statis bisa memanggil API dari domain/lokal berbeda
CORS(app)

@app.route("/", methods=["GET"])
@app.route("/index.html", methods=["GET"])
def index():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(root_dir, "index.html")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error loading index.html: {str(e)}", 404

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "message": "RAG Flask Backend is up and running."
    })

@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        if "pdf" not in request.files:
            return jsonify({"error": "Tidak ada file PDF yang dikirim"}), 400
            
        file = request.files["pdf"]
        if file.filename == "":
            return jsonify({"error": "Nama file kosong"}), 400
            
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "File harus berupa dokumen PDF"}), 400
            
        # Simpan sementara ke folder temp OS
        temp_dir = "/tmp" if os.environ.get("VERCEL") else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        
        # Jalankan proses indexing PDF ke ChromaDB
        page_count, chunk_count = index_uploaded_pdf(file_path)
        
        # Bersihkan berkas temp PDF setelah berhasil diindeks
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Gagal menghapus berkas temp PDF: {str(e)}")
            
        return jsonify({
            "status": "success",
            "filename": file.filename,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "message": f"Dokumen '{file.filename}' berhasil diindeks ke ChromaDB! ({page_count} halaman, {chunk_count} potongan)"
        })
        
    except Exception as e:
        return jsonify({"error": f"Gagal mengunggah/mengindeks PDF: {str(e)}"}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        question = data.get("question", "")
        
        if not question.strip():
            return jsonify({"error": "Pertanyaan tidak boleh kosong"}), 400
        
        # Tentukan persist directory
        persist_dir = "/tmp/chroma_db" if os.environ.get("VERCEL") else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
        
        # 1. Tentukan retriever yang aktif
        current_retriever = retriever
        is_custom_db = False
        
        if Chroma is not None and os.path.exists(persist_dir) and os.listdir(persist_dir):
            try:
                db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
                current_retriever = db.as_retriever(search_kwargs={"k": 3})
                is_custom_db = True
            except Exception as e:
                print(f"Gagal memuat ChromaDB dari disk: {str(e)}")
        
        # 2. Panggil RAG (Retrieval + Generation)
        retrieved_docs = []
        citations = []
        try:
            if current_retriever is not None:
                # Ambil dokumen pendukung (Retrieval)
                retrieved_docs = current_retriever.invoke(question)
                
                # Buat sitasi sumber unik
                seen_citations = set()
                for doc in retrieved_docs:
                    source_path = doc.metadata.get("source", "dokumen")
                    source_name = os.path.basename(source_path)
                    page = doc.metadata.get("page", 0) + 1
                    citation_key = f"{source_name}_p{page}"
                    if citation_key not in seen_citations:
                        citations.append({
                            "source": source_name,
                            "page": page
                        })
                        seen_citations.add(citation_key)
            
            # Format dokumen menjadi konteks teks
            context_text = format_docs(retrieved_docs)
            
            # Jalankan LLM dengan konteks (Generation)
            rag_answer = generation_chain.invoke({"context": context_text, "input": question})
        except Exception as e:
            rag_answer = f"❌ Gagal memanggil RAG: {str(e)}"
            citations = []
            
        # 3. Panggil Baseline LLM (tanpa konteks dokumen)
        try:
            response_baseline = llm.invoke(question)
            baseline_answer = getattr(response_baseline, 'content', str(response_baseline))
        except Exception as e:
            baseline_answer = f"❌ Gagal memanggil Baseline LLM: {str(e)}"
            
        # 4. Kirim log eksperimen ke Weights & Biases
        try:
            wandb.init(project="week11-rag-chatbot", reinit=True)
            wandb.log({
                "question": question,
                "rag_answer": rag_answer,
                "baseline_answer": baseline_answer,
                "rag_length": len(rag_answer),
                "baseline_length": len(baseline_answer),
                "is_custom_db": is_custom_db,
                "citation_count": len(citations)
            })
            wandb.finish()
        except Exception as we:
            print(f"Weights & Biases logging error: {str(we)}")

        return jsonify({
            "rag_answer": rag_answer,
            "baseline_answer": baseline_answer,
            "citations": citations
        })
        
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# Untuk testing lokal: python api/index.py
if __name__ == "__main__":
    app.run(port=5000, debug=True)
