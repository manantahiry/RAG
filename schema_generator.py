import os
import cx_Oracle
from dotenv import load_dotenv

load_dotenv()
oracle_user = os.getenv("ORACLE_USER")
oracle_password = os.getenv("ORACLE_PASSWORD")
oracle_dsn = os.getenv("ORACLE_DSN")

# Connexion Oracle (adapte selon ton environnement)
dsn = cx_Oracle.makedsn("localhost", 1521, service_name="xe")
conn = cx_Oracle.connect(oracle_user, oracle_password, oracle_dsn)
cursor = conn.cursor()
# Récupérer la liste des tables
cursor.execute("""
    SELECT table_name FROM user_tables
""")
tables = [row[0] for row in cursor.fetchall()]

schema_description = ""

for table in tables:
    schema_description += f"Table {table}:\n"

    # Colonnes
    cursor.execute(f"""
        SELECT column_name FROM user_tab_columns
        WHERE table_name = '{table}'
    """)
    columns = cursor.fetchall()
    for col in columns:
        schema_description += f"- {col[0]}\n"
    schema_description += "\n"

# Relations (clés étrangères)
cursor.execute("""
SELECT
    a.table_name AS child_table,
    a.column_name AS child_column,
    c_pk.table_name AS parent_table,
    b.column_name AS parent_column
FROM
    user_cons_columns a
    JOIN user_constraints c ON a.constraint_name = c.constraint_name
    JOIN user_constraints c_pk ON c.r_constraint_name = c_pk.constraint_name
    JOIN user_cons_columns b ON c_pk.constraint_name = b.constraint_name AND a.position = b.position
WHERE
    c.constraint_type = 'R'
""")

relations = cursor.fetchall()
schema_description += "Relations:\n"
for rel in relations:
    schema_description += f"{rel[0]}.{rel[1]} = {rel[2]}.{rel[3]}\n"

# Écrire dans un fichier texte
with open("schema_description.txt", "w") as f:
    f.write(schema_description)

print("✅ Description du schéma générée avec succès.")
