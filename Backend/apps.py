import os
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# -------------------- ENV & DB --------------------

try:
    import psycopg2
except ImportError:
    psycopg2 = None
    print("Install psycopg2-binary")

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

def get_db_conn():
    if psycopg2 is None:
        raise RuntimeError("psycopg2 not installed")
    return psycopg2.connect(DB_URL)

# -------------------- FASTAPI APP --------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------- REQUEST MODELS --------------------

class UserRequest(BaseModel):
    username: str


class HistoryRequest(BaseModel):
    user_id: int


class QueryRequest(BaseModel):
    query: str
    user_id: Optional[int] = 1

# -------------------- RAG SETUP (LAZY) --------------------

rag_enabled = False
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

def init_rag():
    global rag_enabled, db, rag_chain, HumanMessage, AIMessage

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_classic.chains import create_retrieval_chain
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.messages import HumanMessage, AIMessage
    except Exception as e:
        print("RAG deps missing:", e)
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use context to answer in max 3 sentences.\nContext: {context}\nChat history: {chat_history}"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    rag_enabled = True
    print("RAG initialized")

# -------------------- AUTH / USER --------------------

@app.post("/get_or_create_user")
def get_or_create_user(req: UserRequest):
    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute("SELECT id FROM users WHERE username = %s", (req.username,))
    row = cur.fetchone()

    if row:
        user_id = row[0]
    else:
        cur.execute(
            "INSERT INTO users (username) VALUES (%s) RETURNING id",
            (req.username,)
        )
        conn.commit()
        user_id = cur.fetchone()[0]

    cur.close()
    conn.close()

    return {"user_id": user_id, "username": req.username}

# -------------------- CHAT HISTORY --------------------

@app.post("/get_history")
def get_history(req: HistoryRequest):
    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT prompt, answer FROM chat_history WHERE user_id = %s ORDER BY id ASC",
        (req.user_id,)
    )
    rows = cur.fetchall()

    cur.close()
    conn.close()

    history = []
    for p, a in rows:
        history.append({"role": "human", "content": p})
        history.append({"role": "ai", "content": a})

    return {"history": history}

# -------------------- QUERY RAG --------------------

@app.post("/query")
def query_rag(req: QueryRequest):
    if not rag_enabled:
        init_rag()
        if not rag_enabled:
            return JSONResponse(
                {"error": "RAG not initialized"},
                status_code=503
            )

    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT prompt, answer FROM chat_history WHERE user_id = %s ORDER BY id ASC",
        (req.user_id,)
    )
    rows = cur.fetchall()

    chat_history = []
    for p, a in rows:
        chat_history.append(HumanMessage(content=p))
        chat_history.append(AIMessage(content=a))

    response = rag_chain.invoke({
        "input": req.query,
        "chat_history": chat_history
    })

    answer = response.get("answer", "I don't know")

    cur.execute(
        "INSERT INTO chat_history (user_id, prompt, answer) VALUES (%s, %s, %s)",
        (req.user_id, req.query, answer)
    )
    conn.commit()

    cur.close()
    conn.close()

    return {"answer": answer}

# -------------------- FILE UPLOAD --------------------

@app.post("/upload")
def upload_file(file: UploadFile, username: Optional[str] = Form(None)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    saved_path = os.path.join(upload_dir, f"{ts}_{file.filename}")

    with open(saved_path, "wb") as f:
        f.write(file.file.read())

    if not username:
        username = f"anonymous_{ts}"

    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    row = cur.fetchone()

    if row:
        user_id = row[0]
    else:
        cur.execute(
            "INSERT INTO users (username) VALUES (%s) RETURNING id",
            (username,)
        )
        user_id = cur.fetchone()[0]
        conn.commit()

    cur.execute(
        "INSERT INTO files (user_id, filename, filepath, uploaded_at) VALUES (%s, %s, %s, %s)",
        (user_id, file.filename, saved_path, datetime.utcnow())
    )
    conn.commit()

    cur.close()
    conn.close()

    return {
        "status": "uploaded",
        "user_id": user_id,
        "filename": file.filename
    }

# -------------------- ROOT --------------------

@app.get("/")
def root():
    return {"message": "FastAPI running. Visit /docs"}
