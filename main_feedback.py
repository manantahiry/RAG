# streamlit_app.py — Assistant SQL + Feedback utilisateur
import os
import json
import cx_Oracle
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from schema_loader import load_schema_from_file

# Charger les variables d'environnement
load_dotenv()

# Connexion Oracle
oracle_user = os.getenv("ORACLE_USER")
oracle_password = os.getenv("ORACLE_PASSWORD")
oracle_dsn = os.getenv("ORACLE_DSN")

# Charger le schéma
schema_description = load_schema_from_file("schema_description.txt")

# Choix LLM
use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
llm = Ollama(model="mistral") if use_ollama else ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"
)

st.title("🧠 Assistant SQL Oracle (LLM + Feedback)")

# Pose ta question
question = st.text_input("Pose ta question :", placeholder="Quels sont les courriers en retard ?")

if st.button("Générer la requête SQL") and question:
    # Prompt SQL
    sql_prompt = f"""
    En te basant uniquement sur le schéma suivant :

    {schema_description}

    Génère une requête Oracle SQL SELECT valide pour répondre à cette question :
    {question}
    """
    with st.spinner("💭 Génération SQL..."):
        response = llm.invoke(sql_prompt)
        generated_sql = response.content.strip().rstrip(";")

    st.code(generated_sql, language="sql")

    # Exécution SQL
    try:
        conn = cx_Oracle.connect(oracle_user, oracle_password, oracle_dsn)
        cursor = conn.cursor()
        cursor.execute(generated_sql)
        columns = [col[0] for col in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        conn.close()

        st.success("✅ Requête exécutée avec succès !")
        st.dataframe(rows)

        # Feedback utilisateur
        feedback = st.radio("Es-tu satisfait de cette réponse ?", ["Oui", "Non", "Corriger"])
        correction = ""
        if feedback == "Corriger":
            correction = st.text_area("Propose ta version corrigée :", height=100)

        if st.button("Enregistrer le feedback"):
            entry = {
                "question": question,
                "generated_sql": generated_sql,
                "feedback": feedback.lower(),
                "correction": correction if correction else None
            }
            feedback_file = "feedback_log.json"
            if os.path.exists(feedback_file):
                with open(feedback_file, "r") as f:
                    feedback_data = json.load(f)
            else:
                feedback_data = []

            feedback_data.append(entry)
            with open(feedback_file, "w") as f:
                json.dump(feedback_data, f, indent=2)

            st.success("📝 Feedback enregistré !")

    except Exception as e:
        st.error(f"❌ Erreur Oracle : {e}")
        if st.button("Enregistrer cette erreur"):
            error_entry = {
                "question": question,
                "generated_sql": generated_sql,
                "error": str(e)
            }
            error_file = "erreurs_connues.json"
            if os.path.exists(error_file):
                with open(error_file, "r") as f:
                    error_log = json.load(f)
            else:
                error_log = []

            error_log.append(error_entry)
            wi
