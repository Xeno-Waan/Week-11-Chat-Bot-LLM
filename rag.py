import os
import streamlit as st

# Inject Streamlit Secrets ke Environment Variables sebelum memuat rag_core
api_key = None
try:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.environ.get("GOOGLE_API_KEY")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# Import modul RAG utama dari folder api
from api.rag_core import llm, rag_chain
