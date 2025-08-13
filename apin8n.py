import os
import re
import cx_Oracle
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from schema_loader import load_schema_from_file
from openpyxl import Workbook
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# --- Initialisation
load_dotenv()

# --- Configuration
oracle_user = os.getenv("ORACLE_USER")
oracle_password = os.getenv("ORACLE_PASSWORD")
oracle_dsn = os.getenv("ORACLE_DSN")
use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
schema_description = load_schema_from_file("schema_description.txt")

llm = Ollama(model="mistral") if use_ollama else ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"
)

# --- Embeddings + Vectorstore (RAG)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts([schema_description], embeddings)
retriever = vectorstore.as_retriever()

# --- FastAPI App
app = FastAPI()

# --- CORS (sécuriser selon besoin)
app.add_middleware(
    CORSMiddleware,
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

# --- Utils
def slugify(text):
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s]+", "_", text)

# --- Endpoint
@app.post("/ask_question", response_model=FullAnswer)
def ask_question(data: QuestionRequest):
    question = data.question

    # Étape 1 : Génération de la requête SQL
    sql_prompt = f"""
    En te basant uniquement sur le schéma suivant :
    {schema_description}
    Génère une requête Oracle SQL SELECT valide pour répondre à cette question :
    {question}
    """
    try:
        response = llm.invoke(sql_prompt)
        generated_sql = response.content.strip().rstrip(";")
        if not generated_sql:
            raise ValueError("La requête SQL générée est vide")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur génération SQL : {e}")

    # Étape 2 : Exécution SQL Oracle
    try:
        conn = cx_Oracle.connect(oracle_user, oracle_password, oracle_dsn)
        cursor = conn.cursor()
        cursor.execute(generated_sql)
        columns = [col[0] for col in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        conn.close()
    except cx_Oracle.DatabaseError as e:
        raise HTTPException(status_code=500, detail=f"Erreur exécution Oracle : {e}")

        # Étape 3 : Génération du fichier Excel (format tableau vertical clair)
    try:
        slug = slugify(question)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{slug}_{timestamp}.xlsx"
        os.makedirs("files", exist_ok=True)
        filepath = os.path.join("files", filename)

        wb = Workbook()
        ws = wb.active
        ws.title = "Résultats"

        # Titre : question + SQL généré
        ws.append(["Question", question])
        ws.append(["SQL Généré", generated_sql])
        ws.append([])

        # Résultats : table claire
        if rows:
            headers = list(rows[0].keys())
            ws.append(headers)
            for row in rows:
                ws.append([row[h] for h in headers])
        else:
            ws.append(["Aucun résultat"])

        wb.save(filepath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Excel : {e}")

    # Étape 4 : Génération requête RAG
    try:
        rag_prompt = f"""
        Pour répondre à la question suivante, indique la table Oracle et les colonnes les plus pertinentes à utiliser.
        Question : {question}
        Donne uniquement la requête SQL SELECT.
        """
        rag_sql = llm.invoke(rag_prompt).content.strip().rstrip(";")
        if not rag_sql:
            rag_sql = "Aucune requête RAG générée."
    except Exception as e:
        rag_sql = "Erreur génération requête RAG."

    # Étape 5 : Réponse JSON
    download_url = f"http://127.0.0.1:8000/files/{filename}"
    return JSONResponse(content={
        "question": question,
        "generated_sql": generated_sql,
        "result": rows,
        "rag_sql": rag_sql,
        "download_link": download_url,
        "message": "Réponse générée avec succès. Cliquez sur le lien pour télécharger le fichier Excel."
    })
