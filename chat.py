import os
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

# -----------------------------
# Étape 1 : Charger le fichier Excel
# -----------------------------
df = pd.read_excel("./data/Situation suivie Août (2) (1).xlsx")

# LLM (clé OpenAI dans la variable d’environnement OPENAI_API_KEY)
llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"])

# SmartDataframe qui sait répondre aux questions en langage naturel
sdf = SmartDataframe(df, config={"llm": llm})

# Boucle interactive : question -> réponse
while True:
    question = input("Posez votre question (ou tapez 'quit' pour quitter) : ")
    if question.lower() in ["quit", "exit"]:
        break
    reponse = sdf.chat(question)
    print("Réponse :", reponse, "\n")
