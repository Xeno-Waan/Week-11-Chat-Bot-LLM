import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Untuk memuat PDF jika dipasang
try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    PyPDFLoader = None

# =====================================================================
# 1. KONFIGURASI API KEY GEMINI
# =====================================================================
api_key = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not api_key:
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
# 3. LOADER DOKUMEN DINAMIS (Membaca folder 'documents')
# =====================================================================
docs_dir = os.path.join(os.path.dirname(__file__), "documents")
os.makedirs(docs_dir, exist_ok=True)

loaded_docs = []

# Teks default jika folder documents kosong
fallback_texts = [
    "Mata kuliah ini diampu oleh Pak Ronggo pada Week 11 tentang Chat Bot dan Large Language Model (LLM).",
    "Week 11 membahas cara kerja RAG (Retrieval-Augmented Generation) dan integrasi LLM dengan basis data eksternal.",
    "Tugas akhir week 11 adalah membuat chatbot sederhana yang membandingkan performa RAG dengan LLM baseline.",
    "Kontak asisten dosen untuk kelas Pak Ronggo dapat dihubungi melalui email: asdos-ronggo@univ.ac.id.",
    "Jadwal praktikum kecerdasan buatan diadakan setiap hari Kamis pukul 13.00 WIB di Laboratorium Komputer Utama.",
]

# Mencari berkas dokumen pendukung
files = os.listdir(docs_dir)
has_valid_files = False

for file in files:
    file_path = os.path.join(docs_dir, file)
    if os.path.isfile(file_path):
        # 1. Baca berkas TXT / MD
        if file.endswith((".txt", ".md")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        # Buat objek dokumen tiruan
                        class SimpleDocument:
                            def __init__(self, page_content, metadata):
                                self.page_content = page_content
                                self.metadata = metadata
                        loaded_docs.append(SimpleDocument(content, {"source": file}))
                        has_valid_files = True
            except Exception as e:
                st.sidebar.warning(f"Gagal membaca berkas {file}: {str(e)}")
        
        # 2. Baca berkas PDF
        elif file.endswith(".pdf") and PyPDFLoader is not None:
            try:
                loader = PyPDFLoader(file_path)
                loaded_docs.extend(loader.load())
                has_valid_files = True
            except Exception as e:
                st.sidebar.warning(f"Gagal membaca PDF {file}: {str(e)}")

# Jika tidak ada berkas dokumen, gunakan teks fallback bawaan
if not has_valid_files:
    class FallbackDocument:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata
    loaded_docs = [FallbackDocument(text, {"source": "fallback"}) for text in fallback_texts]

# Memotong teks dokumen menjadi potongan kecil (chunks) agar pencarian lebih akurat
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
doc_chunks = text_splitter.split_documents(loaded_docs)

# =====================================================================
# 4. EMBEDDINGS & VECTOR STORE (FAISS)
# =====================================================================
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

try:
    vector_store = FAISS.from_documents(doc_chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
except Exception:
    # Fallback retriever tiruan jika API Key belum valid
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

def format_docs(docs):
    if isinstance(docs, str):
        return docs
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
