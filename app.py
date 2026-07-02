import streamlit as st

# ==========================================
# IMPORT / INISIALISASI LLM DAN RAG_CHAIN
# ==========================================
# Silakan sesuaikan baris import di bawah ini dengan nama file/modul 
# tempat Anda mendefinisikan dan menginisialisasi `llm` dan `rag_chain`.
# Contoh: jika diinisialisasi di `rag_setup.py`, gunakan `from rag_setup import llm, rag_chain`.
try:
    # Ganti 'rag' dengan nama file python Anda jika nanti Anda mengubahnya
    from rag import llm, rag_chain
except ImportError:
    # Fallback / Mock Object agar aplikasi Streamlit tetap dapat dijalankan dan diuji 
    # secara independen sebelum diintegrasikan dengan modul RAG asli Anda.
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
        def invoke(self, text):
            return (
                f"Ini adalah jawaban dari **RAG Chain** untuk pertanyaan Anda: *\"{text}\"*.\n\n"
                f"Jawaban ini telah diperkaya dengan informasi relevan yang diambil dari "
                f"basis pengetahuan dokumen internal (Retrieval-Augmented Generation), sehingga "
                f"lebih akurat, spesifik, dan meminimalkan halusinasi."
            )

    llm = MockLLM()
    rag_chain = MockRAGChain()

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="RAG vs Baseline LLM Comparison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan premium modern (Glassmorphism & Elegant Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Ganti font global */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Judul Utama */
    .main-title {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    
    .sub-title {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 25px;
    }

    /* Box Pertanyaan */
    .question-box {
        background-color: rgba(59, 130, 246, 0.08);
        border-left: 4px solid #3b82f6;
        border-radius: 0px 12px 12px 0px;
        padding: 16px 20px;
        margin-top: 15px;
        margin-bottom: 25px;
    }
    
    .question-title {
        font-weight: 700;
        color: #60a5fa;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 5px;
    }

    .question-text {
        font-size: 1.1rem;
        color: #f3f4f6;
    }

    /* Desain Card untuk Jawaban */
    .answer-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 24px;
        height: 100%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .answer-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: #4b5563;
    }
    
    /* Header di dalam Card */
    .card-header-rag {
        color: #10b981;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #374151;
        padding-bottom: 10px;
    }

    .card-header-baseline {
        color: #f59e0b;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #374151;
        padding-bottom: 10px;
    }
    
    /* Konten Teks di dalam Card */
    .card-body {
        color: #e5e7eb;
        line-height: 1.6;
        font-size: 0.975rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR INFORMASI
# ==========================================
with st.sidebar:
    st.markdown("### 🛠️ Pengaturan Modul")
    st.info(
        "Aplikasi ini membandingkan jawaban dari LLM murni (Baseline) dengan "
        "LLM yang dilengkapi data eksternal (RAG Chain)."
    )
    st.markdown("---")
    st.markdown("#### 💡 Petunjuk Integrasi:")
    st.write(
        "Untuk menghubungkan ke `llm` dan `rag_chain` riil Anda, ubah bagian import di baris paling atas berkas `app.py`:"
    )
    st.code("from nama_file_anda import llm, rag_chain", language="python")

# ==========================================
# TAMPILAN UTAMA (HEADER)
# ==========================================
st.markdown('<div class="main-title">🤖 RAG vs Baseline LLM Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Bandingkan performa jawaban LLM berbasis Retrieval-Augmented Generation dengan Model Standar secara real-time.</div>', unsafe_allow_html=True)

# ==========================================
# AREA INPUT PENGGUNA
# ==========================================
pertanyaan_pengguna = st.text_area(
    "Masukkan Pertanyaan Anda:",
    placeholder="Contoh: Apa saja topik utama yang dibahas dalam dokumen materi week 11?",
    height=120
)

# Tombol Jawab / Kirim
col_btn, _ = st.columns([1, 4])
with col_btn:
    tombol_kirim = st.button("🚀 Jawab Pertanyaan", use_container_width=True)

# ==========================================
# LOGIKA SAAT TOMBOL DITEKAN
# ==========================================
if tombol_kirim:
    if pertanyaan_pengguna.strip() == "":
        st.warning("⚠️ Mohon masukkan pertanyaan terlebih dahulu sebelum mengirim.")
    else:
        # 4a. Tampilkan pertanyaan pengguna
        st.markdown(
            f"""
            <div class="question-box">
                <div class="question-title">Pertanyaan Anda:</div>
                <div class="question-text">"{pertanyaan_pengguna}"</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Grid kolom untuk menampilkan perbandingan secara side-by-side
        col_rag, col_baseline = st.columns(2)
        
        # Jalankan pemanggilan model dengan spinner indikator proses
        with st.spinner("Sedang memproses jawaban..."):
            try:
                # 4b. Panggil rag_chain.invoke
                jawaban_rag = rag_chain.invoke(pertanyaan_pengguna)
            except Exception as e:
                jawaban_rag = f"❌ Gagal memanggil RAG Chain: {str(e)}"
                
            try:
                # 4c. Panggil llm.invoke
                response_baseline = llm.invoke(pertanyaan_pengguna)
                # Mendapatkan isi teks dari content
                jawaban_baseline = getattr(response_baseline, 'content', str(response_baseline))
            except Exception as e:
                jawaban_baseline = f"❌ Gagal memanggil Baseline LLM: {str(e)}"
        
        # Tampilkan Hasil Jawaban RAG Chain
        with col_rag:
            st.markdown(
                f"""
                <div class="answer-card">
                    <div class="card-header-rag">
                        ✨ RAG Chain Response
                    </div>
                    <div class="card-body">
                """, 
                unsafe_allow_html=True
            )
            # Menggunakan st.markdown agar formatting markdown dari jawaban teraplikasikan secara rapi
            st.markdown(jawaban_rag)
            st.markdown("</div></div>", unsafe_allow_html=True)
            
        # Tampilkan Hasil Jawaban Baseline LLM
        with col_baseline:
            st.markdown(
                f"""
                <div class="answer-card">
                    <div class="card-header-baseline">
                        🤖 Baseline LLM (No RAG)
                    </div>
                    <div class="card-body">
                """, 
                unsafe_allow_html=True
            )
            # Menggunakan st.markdown agar formatting markdown dari jawaban teraplikasikan secara rapi
            st.markdown(jawaban_baseline)
            st.markdown("</div></div>", unsafe_allow_html=True)
