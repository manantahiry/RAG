# app.py — Interface Streamlit pour ton RAG avec Mistral
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import OllamaLLM

# --- Configuration page ---
st.set_page_config(page_title="🧠 Chatbot RAG Mistral", layout="wide")
st.title("🧠 Chatbot RAG avec Ollama (Mistral)")
st.markdown("Pose une question sur ton PDF ou ton fichier texte.")

# --- Upload fichiers ---
uploaded_files = st.file_uploader("📄 Upload un fichier (.pdf ou .txt)", type=["pdf", "txt"], accept_multiple_files=True)
question = st.text_input("❓ Pose ta question :", "Qu’est-ce que le machine learning ?")

# --- Bouton d’analyse ---
if st.button("🔍 Interroger le document"):
    if not uploaded_files:
        st.warning("Merci de charger un fichier PDF ou TXT.")
        st.stop()

    # --- Charger les documents ---
    documents = []
    for f in uploaded_files:
        if f.name.endswith(".pdf"):
            with open("temp.pdf", "wb") as tmp:
                tmp.write(f.read())
            loader = PyPDFLoader("temp.pdf")
        else:
            with open("temp.txt", "wb") as tmp:
                tmp.write(f.read())
            loader = TextLoader("temp.txt")
        documents += loader.load()

    # --- Chunking ---
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # --- Embedding ---
    with st.spinner("🔢 Vectorisation en cours..."):
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(chunks, embedding)

    # --- LLM via Ollama ---
    llm = OllamaLLM(model="mistral")

    # --- RAG chain ---
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    # --- Réponse ---
    with st.spinner("💬 Génération de la réponse..."):
        result = chain.invoke(question)
        st.success("✅ Réponse générée :")
        st.write(result)
