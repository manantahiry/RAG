# api_rag.py — API automatisée RAG Oracle
import os
import re
import json
import uuid
import cx_Oracle
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_text_splitters import CharacterTextSplitter
from schema_loader import load_schema_from_file
from openpyxl import Workbook
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware




# --- Initialisation
load_dotenv()

# --- Config
oracle_user = os.getenv("ORACLE_USER")
oracle_password = os.getenv("ORACLE_PASSWORD")
oracle_dsn = os.getenv("ORACLE_DSN")
schema_description = load_schema_from_file("schema_description.txt")
use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

llm = Ollama(model="mistral") if use_ollama else ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"
)

# --- Embeddings + Vectorstore (RAG)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts([schema_description], embeddings)
retriever = vectorstore.as_retriever()

# --- App
app = FastAPI()

# CORS configuration
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   

# --- Models
class QuestionRequest(BaseModel):
    question: str

class FullAnswer(BaseModel):
    question: str
    generated_sql: str
    result: list
    rag_sql: str
    
def slugify(text):
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s]+", "_", text)


@app.post("/ask_question", response_model=FullAnswer)
def ask_question(data: QuestionRequest):
    question = data.question

    # 1. Prompt SQL classique
    sql_prompt = f"""
    En te basant uniquement sur le schéma suivant :
    {schema_description}
    Génère une requête Oracle SQL SELECT valide pour répondre à cette question :
    {question}
    """
    try:
        response = llm.invoke(sql_prompt)
        generated_sql = response.content.strip().rstrip(";")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur génération SQL : {e}")

    # 2. Exécution Oracle
    try:
        conn = cx_Oracle.connect(oracle_user, oracle_password, oracle_dsn)
        cursor = conn.cursor()
        cursor.execute(generated_sql)
        columns = [col[0] for col in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Oracle : {e}")

     # 3. Génération Excel
     
    try:
        slug = slugify(question)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{slug}_{timestamp}.xlsx"
        os.makedirs("files", exist_ok=True)
        filepath = os.path.join("files", filename)
        wb = Workbook()
        ws = wb.active
        ws.title = "Résultats"
        ws.append(["Question", "SQL Généré"])
        ws.append([question, generated_sql])
        ws.append([])
        ws.append(columns)
        for row in rows:
            ws.append([row[col] for col in columns])
            
        wb.save(filepath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur génération Excel : {e}")

    # 4. RAG : générer requête enrichie
    try:
        rag_prompt = f"""
        Pour répondre à la question suivante, indique la table Oracle et les colonnes les plus pertinentes à utiliser.
        Question : {question}
        Donne la requête SQL SELECT à utiliser, sans explication ni commentaire.
        """
        rag_sql = llm.invoke(rag_prompt).content.strip().rstrip(";")
    except Exception as e:
        rag_sql = ""

     # 5. Retour JSON avec lien vers fichier Excel
    download_url = f"http://127.0.0.1:8000/files/{filename}"

    return JSONResponse(content={
        "question": question,
        "generated_sql": generated_sql,
        "result": rows,
        "rag_sql": rag_sql,
        "download_link": download_url,
        "message": "Réponse générée avec succès. Cliquez sur le lien pour télécharger le fichier Excel."
    })
