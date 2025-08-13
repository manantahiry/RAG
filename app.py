# app.py — Assistant Oracle + RAG avec Streamlit
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
from sentence_transformers import SentenceTransformer
# --- Initialisation
load_dotenv()
st.set_page_config(page_title="Assistant Administratif", page_icon="🧠")

st.title("🧠 Assistant Administratif")
st.markdown("Pose ta question en langage naturel, je te repondre à partir des données Oracle et du schéma de la base de données.")

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

# --- Zone de saisie
question = st.text_input("💬 Pose ta question en langage naturel :")

if question:
    # --- Génération SQL
    with st.spinner("🧠 Génération de la requête SQL..."):
        sql_prompt = f"""
En te basant uniquement sur le schéma suivant :

{schema_description}

Génère une requête Oracle SQL SELECT valide pour répondre à cette question :
{question}
"""
        response = llm.invoke(sql_prompt)
        generated_sql = response.content.strip()
        if generated_sql.endswith(";"):
            generated_sql = generated_sql[:-1]

    

    # --- Exécution Oracle
    try:
        with st.spinner("📡 Exécution de la requête SQL sur Oracle..."):
            conn = cx_Oracle.connect(oracle_user, oracle_password, oracle_dsn)
            cursor = conn.cursor()
            cursor.execute(generated_sql)
            columns = [col[0] for col in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            cursor.close()
            conn.close()

        st.success("✅ Résultat récupéré avec succès")
        st.dataframe(rows)

    except Exception as e:
        st.error(f"❌ Erreur lors de l'exécution SQL :\n```\n{e}\n```")
        
    # --- reprendre la conversation
    if st.button("Reprendre la conversation"):
        st.rerun()
    # --- validation de reponse
    if st.button("Valider la reponse"):
        st.balloons()
    
    # --- génération de RAG
    @st.cache_resource()
    def get_embeddings():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    embeddings = get_embeddings()

    vectorstore = FAISS.from_texts([schema_description], embeddings)
    st.success("🔍 RAG activé, prêt à répondre aux questions contextuelles !")
    
    rag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    rag_prompt = f"""Pour/respondre à la question suivante, indique la table Oracle et les colonnes les plus pertinentes à utiliser.
    Question : {question}
    Donne la requête SQL SELECT à utiliser, sans explication ni commentaire."""
    rag_sql = llm.invoke(rag_prompt).content.strip()
    if rag_sql.endswith(";"):
        rag_sql = rag_sql[:-1]
    
    
    
# Initialiser la liste historique dans la session
if "history" not in st.session_state:
    st.session_state.history = []

if question:
    # Ici tu appelles ton LLM pour générer la réponse (exemple simplifié)
    reponse = f"Réponse générée pour : {question}"  # Remplace par ton vrai appel à llm.invoke()

    # Sauvegarde dans l'historique
    st.session_state.history.append({"question": question, "reponse": reponse})

# Afficher l'historique (questions + réponses)
if st.session_state.history:
    st.markdown("### 🕑 Historique des questions / réponses")
    for i,(question, reponse) in enumerate(reversed(st.session_state.history)):
        st.markdown(f"**Q{i+1}:** {question}")
        st.markdown(f"**R{i+1}:** {reponse}")
        st.markdown("---")

    # Bouton de réinitialisation
    if st.button("Réinitialiser l'historique"):
        st.session_state.history = [] 