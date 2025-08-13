import os
import cx_Oracle
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

load_dotenv()

# Connexion Oracle
oracle_user = os.getenv("ORACLE_USER")
oracle_password = os.getenv("ORACLE_PASSWORD")
oracle_dsn = os.getenv("ORACLE_DSN")
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Choisir LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")  # ou Ollama(model="mistral")

# --- Étape 1 : Pose de la question
print("\n🧠 Pose ta question :")
question = input("> ")

# --- Étape 2 : Déterminer si la question est SQL ou sémantique
decision_prompt = f"""
Tu es un assistant intelligent. Si la question ci-dessous peut être répondue par une requête SQL simple sur des tables Oracle (ex: nombre, somme, moyenne, filtre), réponds juste par 'SQL'. Sinon, réponds 'RAG'.

Question : {question}
Réponds uniquement par SQL ou RAG.
"""

decision = llm.predict(decision_prompt).strip().upper()
print(f"🔍 Type détecté : {decision}")

if decision == "SQL":
    # Générer la requête SQL
    sql_prompt = f"""
    Génère uniquement une requête SQL Oracle SÉLECT valide (sans explication)
    pour répondre à la question suivante, en te basant sur les tables comme COURRIER, AGENT, etc.

    Question : {question}
    """
    generated_sql = llm.predict(sql_prompt).strip()
    print(f"\n📝 Requête SQL générée :\n{generated_sql}")

    # Exécution Oracle
    try:
        conn = cx_Oracle.connect(oracle_user, oracle_password, oracle_dsn)
        cursor = conn.cursor()
        cursor.execute(generated_sql)
        rows = cursor.fetchall()
        conn.close()

        print("\n📊 Résultat SQL :")
        for row in rows:
            print(" - ", row)

    except Exception as e:
        print(f"❌ Erreur lors de l'exécution SQL : {e}")

elif decision == "RAG":
    # Charger les textes depuis Oracle
    try:
        conn = cx_Oracle.connect(oracle_user, oracle_password, oracle_dsn)
        cursor = conn.cursor()
        cursor.execute("SELECT OBJET || ' - ' || DESCRIPTION FROM COURRIER WHERE OBJET IS NOT NULL")
        textes = [row[0] for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        print(f"❌ Erreur Oracle : {e}")
        exit()

    # Embedding + FAISS
    docs = [{"page_content": text} for text in textes]
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embedding)
    retriever = vectordb.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa_chain.run(question)

    print("\n📘 Réponse RAG :")
    print(answer)

else:
    print("❌ Type de question indéterminé.")
