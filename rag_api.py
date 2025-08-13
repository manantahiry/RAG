from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import cx_Oracle
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import Ollama
from schema_loader import load_schema_from_file

# Charger les variables d'environnement
load_dotenv()
app = FastAPI()

# Config Oracle & LLM
ORACLE_USER = os.getenv("ORACLE_USER")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD")
ORACLE_DSN = os.getenv("ORACLE_DSN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"

# Choix LLM
LLM = Ollama(model="mistral") if USE_OLLAMA else ChatOpenAI(openai_api_key=OPENAI_API_KEY)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        return {"error": "Question vide."}

    # --- Classification SQL ou RAG
    decision_prompt = f"""
    Si cette question peut être répondue par une requête SQL simple sur Oracle (compte, somme...), réponds SQL.
    Sinon, réponds RAG. Pas d'explication.
    Question : {question}
    """
    decision = LLM.invoke(decision_prompt).strip().upper()

    if decision == "SQL":
        
        schema_description = load_schema_from_file("schema_description.txt")
        sql_prompt = f"""
        en tenant basant uniquement sur le schéma suivant :
        {schema_description}
        et en respectant la syntaxe SQL d'Oracle,
        réponds à la question suivante.
        Génère une requête Oracle SQL SELECT valide pour repondre à cette question :
        {question}
        """
        generated_sql = LLM.invoke(sql_prompt).strip()
        try:
            conn = cx_Oracle.connect(ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN)
            cursor = conn.cursor()
            cursor.execute(generated_sql)
            columns = [col[0] for col in cursor.description]
            data = [dict(zip(columns, row)) for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return {
                "mode": "SQL",
                "query": generated_sql,
                "result": data
            }
        except Exception as e:
            return {"mode": "SQL", "error": str(e)}

    elif decision == "RAG":
        try:
            conn = cx_Oracle.connect(ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN)
            cursor = conn.cursor()
            cursor.execute("SELECT OBJET || ' - ' || DESCRIPTION FROM COURRIER WHERE OBJET IS NOT NULL")
            textes = [row[0] for row in cursor.fetchall()]
            schema_description = load_schema_from_file("schema_description.txt")
            textes.append(schema_description)
            conn.close()
        except Exception as e:
            return {"mode": "RAG", "error": str(e)}

        docs = [{"page_content": t} for t in textes]
        chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(chunks, embedding)
        qa_chain = RetrievalQA.from_chain_type(llm=LLM, retriever=vectordb.as_retriever())
        answer = qa_chain.run(question)

        return {
            "mode": "RAG",
            "answer": answer
        }

    else:
        return {"error": "Type de question non reconnu."}
