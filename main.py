# main.py — Assistant hybride Oracle (SQL) + RAG
import os
import cx_Oracle
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import Ollama
from schema_loader import load_schema_from_file

# --- Charger les variables d'environnement
load_dotenv()

# --- Config Oracle
oracle_user = os.getenv("ORACLE_USER")
oracle_password = os.getenv("ORACLE_PASSWORD")
oracle_dsn = os.getenv("ORACLE_DSN")

#lire le schéma depuis le fichier
schema_description = load_schema_from_file("schema_description.txt")

# --- choix du LLM
use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
if use_ollama:
    llm = Ollama(model="mistral")
else:
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

# --- Pose de question
print("🧠 Pose ta question (ex : 'Quels sont les courriers en retard ?') :")
question = input(" 👉").strip()

#generer une requete SQL Oracle à partire du schéma
sql_prompt = f"""
en te basant uniquement sur le schéma suivant :

{schema_description}

Génère une requête Oracle SQL SELECT valide pour répondre à cette question :
{question}
"""
response = llm.invoke(sql_prompt)
generated_sql = response.content.strip()
if generated_sql.endswith(";"):
    generated_sql = generated_sql[:-1]
    
print(f"\n📝 Requête SQL générée :\n{generated_sql}")


# --- Exécuter Oracle
try:
    conn = cx_Oracle.connect(oracle_user, oracle_password, oracle_dsn)
    cursor = conn.cursor()
    cursor.execute(generated_sql)
    columns = [col[0] for col in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    print("\n📊 Résultat SQL :")
    for row in rows:
        print(row)

except Exception as e:
    print(f"❌ Erreur lors de l'exécution SQL :{e}")
