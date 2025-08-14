import os
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter

# Charger la clé API
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Étape 1 : Charger le fichier Excel
# -----------------------------
df = pd.read_excel("./data/Situation suivie Août (2) (1).xlsx")

# Nettoyage des noms de colonnes
df.columns = df.columns.str.strip().str.lower()

# Afficher les colonnes pour vérification
print(df.columns)

# Conversion des dates
df["date début de traitement"] = pd.to_datetime(df["date début de traitement"], errors='coerce', dayfirst=True)
df["date fin de traitement"] = pd.to_datetime(df["date fin de traitement"], errors='coerce', dayfirst=True)

# Calcul de la durée en jours
df["durée (jours)"] = (df["date fin de traitement"] - df["date début de traitement"]).dt.days

# Garder uniquement les colonnes nécessaires
df = df[["Type de Dossier", "Date Début de Traitement", "Date Fin de Traitement", "Durée (Jours)"]]

# Conversion en texte pour LangChain
text = df.to_string(index=False)

# Vérifier la taille totale du texte
if len(text.split()) > 15000:  # Limite à 15 000 tokens
    raise ValueError("Le texte est trop volumineux pour être traité.")

# -----------------------------
# Étape 2 : Split en chunks
# -----------------------------
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
docs = [Document(page_content=chunk) for chunk in chunks]

# -----------------------------
# Étape 3 : Création des embeddings et index FAISS
# -----------------------------
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectordb = FAISS.from_documents(docs, embedding)

# -----------------------------
# Étape 4 : Construire le RAG
# -----------------------------
retriever = vectordb.as_retriever(search_kwargs={"k": 2})  # Réduit à 2 documents
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4-32k", temperature=0, openai_api_key=openai_api_key),
    retriever=retriever,
    return_source_documents=False
)

# -----------------------------
# Étape 5 : Boucle de questions
# -----------------------------
print("Tape 'exit' pour quitter.")
while True:
    query = input("Question : ")
    if query.lower() == "exit":
        break
    result = qa_chain.run(query)
    print("\nRéponse :\n", result, "\n")
