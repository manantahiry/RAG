# app.py — Assistant Oracle + RAG avec validation et historique
import os
import cx_Oracle
import streamlit as st
import datetime
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from schema_loader import load_schema_from_file

# --- Initialisation
load_dotenv()
st.set_page_config(page_title="Assistant Administratif", page_icon="🧠")

st.title("🧠 Assistant Administratif")
st.markdown("Pose ta question en langage naturel, je te répondrai à partir des données Oracle et du schéma de la base de données.")

# --- Charger description du schéma
schema_description = load_schema_from_file("schema_description.txt")

# --- Configuration Oracle
oracle_user = os.getenv("ORACLE_USER")
oracle_password = os.getenv("ORACLE_PASSWORD")
oracle_dsn = os.getenv("ORACLE_DSN")

# --- Choix du LLM
use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
if use_ollama:
    llm = Ollama(model="mistral")
else:
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo"
    )

# --- Fonctions utiles
def enregistrer_validation(conn, question, requete_sql, reponse, statut, validateur="utilisateur_streamlit", commentaires=None):
    cursor = conn.cursor()
    sql_insert = """
    INSERT INTO IASA.VALIDATIONS_RAG
    (QUESTION, REQUETE_SQL, REPONSE, STATUT_VALIDATION, DATE_VALIDATION, VALIDATEUR, COMMENTAIRES)
    VALUES (:question, :requete_sql, :reponse, :statut, :date_val, :validateur, :commentaires)
    """
    cursor.execute(sql_insert, {
        "question": question,
        "requete_sql": requete_sql,
        "reponse": json.dumps(reponse, ensure_ascii=False),  # JSON propre
        "statut": statut,
        "date_val": datetime.datetime.now(),
        "validateur": validateur,
        "commentaires": commentaires
    })
    conn.commit()
    cursor.close()

def chercher_reponse_validee(conn, fragment_sql):
    sql = """
    SELECT REPONSE
    FROM VALIDATIONS_RAG
    WHERE DBMS_LOB.INSTR(REQUETE_SQL, :fragment_sql) > 0
    AND STATUT_VALIDATION = 'VALIDEE'
    """
    cursor = conn.cursor()
    cursor.execute(sql, {"fragment_sql": fragment_sql})
    result = cursor.fetchone()
    cursor.close()
    if result:
        return result[0]
    return None

@st.cache_resource()
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Zone de saisie
question = st.text_input("💬 Pose ta question en langage naturel :")

# Initialiser historique
if "history" not in st.session_state:
    st.session_state.history = []

if question:
    try:
        conn = cx_Oracle.connect(oracle_user, oracle_password, oracle_dsn)
    except Exception as e:
        st.error(f"❌ Erreur connexion Oracle :\n```\n{e}\n```")
        conn = None

    generated_sql = None
    rows = None

    if conn:
        # Vérifier si une réponse validée existe
        reponse_validee = chercher_reponse_validee(conn, question)
        if reponse_validee:
            st.success("💾 Réponse validée trouvée en base.")
            df = pd.DataFrame([json.loads(row) for row in reponse_validee], columns=reponse_validee)
            st.dataframe(df)
            rows = reponse_validee
        else:
            # --- Génération SQL
            with st.spinner("🧠 Génération de la requête SQL..."):
                sql_prompt = f"""
En te basant uniquement sur le schéma suivant :

{schema_description}

Génère une requête Oracle SQL SELECT valide pour répondre à cette question :
{question}
"""
                response = llm.invoke(sql_prompt)
                generated_sql = response.content.strip().rstrip(";")

            # --- Exécution SQL
            try:
                with st.spinner("📡 Exécution sur Oracle..."):
                    cursor = conn.cursor()
                    cursor.execute(generated_sql)
                    columns = [col[0] for col in cursor.description]
                    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    cursor.close()

                st.success("✅ Résultat récupéré avec succès")
                df = pd.DataFrame(rows)
                st.dataframe(df)
            except Exception as e:
                st.error(f"❌ Erreur exécution SQL :\n```\n{e}\n```")

        # --- Boutons de validation / invalidation
        if rows:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Valider la réponse"):
                    try:
                        enregistrer_validation(conn, question, generated_sql, rows, "VALIDEE")
                        st.success("Réponse validée et enregistrée ✅")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Erreur enregistrement : {e}")
            with col2:
                if st.button("❌ Invalider la réponse"):
                    try:
                        enregistrer_validation(conn, question, generated_sql, rows, "INVALIDE")
                        st.success("Réponse invalidée et enregistrée ❌")
                    except Exception as e:
                        st.error(f"Erreur enregistrement : {e}")

        conn.close()

        # --- Sauvegarde dans l'historique local
        st.session_state.history.append({"question": question, "reponse": rows})

# --- Afficher historique
if st.session_state.history:
    st.markdown("### 🕑 Historique des questions / réponses")
    for i, entry in enumerate(reversed(st.session_state.history)):
        st.markdown(f"**Q{i+1}:** {entry['question']}")
        st.markdown(f"**R{i+1}:** {entry['reponse']}")
        st.markdown("---")
    if st.button("Réinitialiser l'historique"):
        st.session_state.history = []

# --- RAG (chargé une seule fois)
embeddings = get_embeddings()
vectorstore = FAISS.from_texts([schema_description], embeddings)
st.success("🔍 RAG activé, prêt à répondre aux questions contextuelles !")
rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)
