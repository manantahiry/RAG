import os
import cx_Oracle
from dotenv import load_dotenv

load_dotenv()

oracle_user = os.getenv("ORACLE_USER")
oracle_password = os.getenv("ORACLE_PASSWORD")
oracle_dsn = os.getenv("ORACLE_DSN")

try:
    conn = cx_Oracle.connect(oracle_user, oracle_password, oracle_dsn)
    cursor = conn.cursor()

    cursor.execute("SELECT table_name FROM user_tables ORDER BY table_name")
    tables = cursor.fetchall()

    print("\n📦 Tables disponibles dans le schéma Oracle :")
    for (table,) in tables:
        print(f" - {table}")

    conn.close()
except Exception as e:
    print(f"❌ Erreur Oracle : {e}")
