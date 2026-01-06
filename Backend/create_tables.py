import psycopg2
import os
from dotenv import load_dotenv

# load .env from parent directory
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

conn = None

try:
    conn = psycopg2.connect(DB_URL)
    print("Database connection established.")

    cur = conn.cursor()

    # USERS TABLE (Primary Key)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL
        );
    """)

    # CHAT HISTORY / MESSAGES TABLE (Foreign Key)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            prompt TEXT,
            answer TEXT
        );
    """)

    conn.commit()
    cur.close()
    print("Tables created successfully.")

except(Exception , psycopg2.DatabaseError) as error:
    print(f"Error: {error}")

finally:
    if conn:
        conn.close()
        print("Database connection closed.")
