# LangChain RAG avec Ollama et Mistral
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import OllamaLLM  # Assure-toi que ce module est installé

# --- 1. Charger les documents PDF + TXT
pdf_loader = PyPDFLoader("docs/exemple.pdf")
txt_loader = TextLoader("docs/exemple.txt")
documents = pdf_loader.load() + txt_loader.load()

# --- 2. Découper les documents en morceaux (chunks)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# --- 3. Générer les embeddings avec HuggingFace
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 4. Indexer les chunks avec FAISS
vectordb = FAISS.from_documents(docs, embedding)

# --- 5. Utiliser le modèle MISTRAL via Ollama
llm = OllamaLLM(model="mistral")  # ✅ ici on utilise mistral en faisant "ollama pull mistral"

# --- 6. Créer la chaîne RAG
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

# --- 7. Poser la question
print("\n❓ Pose ta question (ex : Qu’est-ce que le machine learning ?) :")
query = input("> ")
response = qa_chain.invoke(query)

# --- 8. Afficher la réponse
print("\n📘 Réponse générée par Mistral via Ollama :")
print(response)
