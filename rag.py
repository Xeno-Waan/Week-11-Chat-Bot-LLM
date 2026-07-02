import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

# =====================================================================
# 1. KONFIGURASI API KEY GEMINI
# =====================================================================
# Mengambil API key dari Environment Variable (lokal) atau Streamlit Secrets (Cloud)
api_key = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    # Jika API key belum terpasang, gunakan mock key agar inisialisasi tidak langsung error.
    # Pengguna akan diarahkan untuk mengisi API key di dashboard Streamlit Cloud atau file .env
    api_key = "MOCK_KEY"

# =====================================================================
# 2. INISIALISASI MODEL LLM (Gemini 1.5 Flash)
# =====================================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.7,
    streaming=True
)

# =====================================================================
# 3. CONTOH BASIS DATA DOKUMEN / MATERI KULIAH (Konteks RAG)
# =====================================================================
# Silakan ganti isi teks di bawah ini dengan dokumen/informasi materi Anda sendiri
documents = [
    "Mata kuliah ini diampu oleh Pak Ronggo pada Week 11 tentang Chat Bot dan Large Language Model (LLM).",
    "Week 11 membahas cara kerja RAG (Retrieval-Augmented Generation) dan integrasi LLM dengan basis data eksternal.",
    "Tugas akhir week 11 adalah membuat chatbot sederhana yang membandingkan performa RAG dengan LLM baseline.",
    "Kontak asisten dosen untuk kelas Pak Ronggo dapat dihubungi melalui email: asdos-ronggo@univ.ac.id.",
    "Jadwal praktikum kecerdasan buatan diadakan setiap hari Kamis pukul 13.00 WIB di Laboratorium Komputer Utama.",
]

# =====================================================================
# 4. EMBEDDINGS & VECTOR STORE (FAISS)
# =====================================================================
# Inisialisasi model embeddings dari Google
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# Membuat vector database sederhana di memori menggunakan FAISS
# Catatan: Jika API key belum valid (masih MOCK_KEY), inisialisasi FAISS riil akan dilewati
try:
    vector_store = FAISS.from_texts(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
except Exception:
    # Fallback retriever tiruan jika terjadi error / API key belum valid
    class MockRetriever:
        def __or__(self, other):
            return self
        def invoke(self, query):
            return "Konteks simulasi: Pak Ronggo adalah dosen Week 11 Chatbot RAG."
    retriever = MockRetriever()

# =====================================================================
# 5. RUNNABLE PIPELINE (RAG Chain)
# =====================================================================
prompt_template = """Anda adalah asisten chatbot materi kuliah Pak Ronggo.
Jawablah pertanyaan berikut dengan sopan dan ringkas berdasarkan konteks dokumen yang disediakan.
Jika informasi tidak ada di dalam konteks, katakan bahwa Anda tidak tahu.

Konteks:
{context}

Pertanyaan: {question}
Jawaban:"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Helper untuk menggabungkan teks dari dokumen-dokumen yang ditemukan
def format_docs(docs):
    if isinstance(docs, str):
        return docs
    return "\n\n".join([doc.page_content for doc in docs])

# Membangun RAG Chain utuh menggunakan LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
