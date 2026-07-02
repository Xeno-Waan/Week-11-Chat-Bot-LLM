import streamlit as st
import os
import time

# =====================================================================
# 1. IMPOR / INISIALISASI LLM DAN RAG_CHAIN
# =====================================================================
try:
    from rag import llm, rag_chain
except ImportError:
    # Fallback / Mock Object jika file rag.py tidak dapat diimpor
    class MockLLM:
        def invoke(self, text):
            class Content:
                content = (
                    f"Ini adalah jawaban **baseline** dari LLM tanpa menggunakan pencarian RAG "
                    f"untuk pertanyaan Anda: *\"{text}\"*.\n\n"
                    f"Jawaban ini murni berasal dari data latih internal model bahasa tanpa "
                    f"konteks dokumen eksternal tambahan."
                )
            return Content()

    class MockRAGChain:
        def stream(self, text):
            response = (
                f"Ini adalah jawaban dari **RAG Chain** untuk pertanyaan Anda: *\"{text}\"*.\n\n"
                f"Jawaban ini telah diperkaya dengan informasi relevan yang diambil dari "
                f"basis pengetahuan dokumen internal (Retrieval-Augmented Generation), sehingga "
                f"lebih akurat, spesifik, dan meminimalkan halusinasi."
            )
            for word in response.split(" "):
                yield word + " "
                time.sleep(0.04)
        def invoke(self, text):
            return "".join(self.stream(text))

    llm = MockLLM()
    rag_chain = MockRAGChain()

# =====================================================================
# 2. KONFIGURASI HALAMAN & TEMA STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="RAG Chatbot Explorer",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk tampilan premium ala Google Gemini (Modern & Clean)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Font global */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Utama */
    .chatbot-header {
        text-align: center;
        margin-bottom: 30px;
        padding-top: 20px;
    }
    
    .chatbot-title {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 50%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 5px;
    }
    
    .chatbot-subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
    }

    /* Kustomisasi Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: #f59e0b !important;
    }

    .streamlit-expanderContent {
        border-left: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 0 0 8px 8px !important;
        padding: 15px !important;
        background-color: rgba(0, 0, 0, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 3. SIDEBAR CONTROLS
# =====================================================================
with st.sidebar:
    st.markdown("### 🛠️ Pengaturan Chatbot")
    st.write("Gunakan menu ini untuk mengontrol sesi obrolan Anda.")
    st.markdown("---")
    
    # Tombol Reset Chat History
    if st.button("🗑️ Hapus Percakapan", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Halo! Saya asisten chatbot RAG untuk materi kuliah Pak Ronggo. Ada yang bisa saya bantu terkait materi Week 11?",
                "baseline": None
            }
        ]
        st.rerun()

    st.markdown("---")
    st.info(
        "💡 **RAG Chain** akan mencari konteks dari dokumen internal sebelum menjawab. "
        "Jawaban **Baseline LLM** dapat dibuka melalui tombol opsi di bawah setiap respons bot."
    )

# =====================================================================
# 4. TAMPILAN HEADER UTAMA
# =====================================================================
st.markdown("""
<div class="chatbot-header">
    <div class="chatbot-title">🤖 RAG Chatbot Assistant</div>
    <div class="chatbot-subtitle">Asisten AI Cerdas berbasis Retrieval-Augmented Generation — Week 11</div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# 5. INITIALISASI STATE PESAN (CHAT HISTORY)
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Halo! Saya asisten chatbot RAG untuk materi kuliah Pak Ronggo. Ada yang bisa saya bantu terkait materi Week 11?",
            "baseline": None
        }
    ]

# =====================================================================
# 6. MENAMPILKAN RIWAYAT PESAN (CHAT BUBBLE STYLE)
# =====================================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Tampilkan perbandingan baseline jika pesan tersebut dikirim oleh asisten dan memiliki data baseline
        if message["role"] == "assistant" and message.get("baseline"):
            with st.expander("🔍 Lihat Perbandingan: Baseline LLM (Tanpa RAG)"):
                st.markdown(message["baseline"])

# =====================================================================
# 7. PROSES INPUT PENGGUNA & GENERASI RESPON
# =====================================================================
if user_input := st.chat_input("Ketik pertanyaan Anda di sini..."):
    # Tampilkan pesan user
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Simpan ke riwayat pesan
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Buat respon dari asisten
    with st.chat_message("assistant"):
        # Penampung respons RAG (dengan efek streaming)
        rag_placeholder = st.empty()
        
        # Generator stream untuk RAG
        def get_rag_stream():
            try:
                # Cek apakah objek rag_chain memiliki metode stream (LangChain LCEL)
                if hasattr(rag_chain, "stream"):
                    for chunk in rag_chain.stream(user_input):
                        yield chunk
                else:
                    # Fallback jika hanya ada invoke
                    response_text = rag_chain.invoke(user_input)
                    # Simulasikan ketikan agar estetik
                    for word in response_text.split(" "):
                        yield word + " "
                        time.sleep(0.04)
            except Exception as e:
                yield f"❌ Gagal memanggil RAG Chain: {str(e)}"
        
        # Jalankan streaming respons ke layar
        response_rag_text = st.write_stream(get_rag_stream())
        
        # Panggil baseline LLM secara paralel/berurutan di latar belakang
        with st.spinner("Mengambil jawaban perbandingan baseline..."):
            try:
                response_baseline = llm.invoke(user_input)
                response_baseline_text = getattr(response_baseline, 'content', str(response_baseline))
            except Exception as e:
                response_baseline_text = f"❌ Gagal memanggil Baseline LLM: {str(e)}"
        
        # Tampilkan expander perbandingan baseline
        with st.expander("🔍 Lihat Perbandingan: Baseline LLM (Tanpa RAG)"):
            st.markdown(response_baseline_text)
            
    # Simpan respon asisten (RAG + Baseline) ke riwayat pesan
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_rag_text,
        "baseline": response_baseline_text
    })
