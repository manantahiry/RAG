import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

# Charger les variables d'environnement
load_dotenv()

st.title("Analyse Excel avec PandasAI et OpenAI")

# -----------------------------
# Étape 1 : Charger le fichier Excel
# -----------------------------
# Chemin vers le fichier existant
file_path = "./data/Situation suivie Août (2) (1).xlsx"

# Charger le fichier Excel
df = pd.read_excel(file_path)

# Afficher un aperçu des données
st.write("Aperçu des données :")


# Initialiser le LLM avec la clé OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("La variable d'environnement OPENAI_API_KEY n'est pas définie !")
else:
        llm = OpenAI(api_token=api_key)
        sdf = SmartDataframe(df, config={"llm": llm, "save_logs": False})

        # Entrée utilisateur pour la question
        question = st.text_input("Posez votre question (en langage naturel) :")
        if st.button("Obtenir la réponse"):
            if question.strip():
                try:
                    reponse = sdf.chat(question)
                    st.write("Réponse :", reponse)
                except Exception as e:
                    st.error(f"Erreur lors de la génération de la réponse : {e}")
