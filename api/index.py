import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

# Tambahkan direktori utama ke sys.path agar bisa mengimpor rag_core secara lokal maupun di Vercel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from api.rag_core import llm, rag_chain
except ImportError:
    try:
        from rag_core import llm, rag_chain
    except ImportError:
        # Fallback/Mock jika terjadi masalah impor
        class MockLLM:
            def invoke(self, text):
                class Content:
                    content = f"Ini jawaban baseline mock untuk: '{text}'"
                return Content()
        class MockRAGChain:
            def invoke(self, text):
                return f"Ini jawaban RAG mock untuk: '{text}'"
        llm = MockLLM()
        rag_chain = MockRAGChain()

app = Flask(__name__)
# Aktifkan CORS agar frontend HTML statis bisa memanggil API dari domain/lokal berbeda
CORS(app)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "message": "RAG Flask Backend is up and running."
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        question = data.get("question", "")
        
        if not question.strip():
            return jsonify({"error": "Pertanyaan tidak boleh kosong"}), 400
        
        # Panggil RAG Chain
        try:
            rag_answer = rag_chain.invoke(question)
        except Exception as e:
            rag_answer = f"❌ Gagal memanggil RAG Chain: {str(e)}"
            
        # Panggil Baseline LLM
        try:
            response_baseline = llm.invoke(question)
            baseline_answer = getattr(response_baseline, 'content', str(response_baseline))
        except Exception as e:
            baseline_answer = f"❌ Gagal memanggil Baseline LLM: {str(e)}"
            
        return jsonify({
            "rag_answer": rag_answer,
            "baseline_answer": baseline_answer
        })
        
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# Untuk testing lokal: python api/index.py
if __name__ == "__main__":
    app.run(port=5000, debug=True)
