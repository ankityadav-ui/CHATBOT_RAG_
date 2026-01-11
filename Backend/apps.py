import os
try:
    import psycopg2
except ImportError:
    psycopg2 = None
    print("Missing dependency 'psycopg2'. Install with: pip install psycopg2-binary or pip install -r requirements.txt")
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form
from typing import Optional
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

# Load .env file from the parent directory
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
# If GOOGLE_API_KEY is unset or empty in the environment/.env, ensure it's removed
_gkey = os.getenv("GOOGLE_API_KEY")
if _gkey:
    os.environ["GOOGLE_API_KEY"] = _gkey
else:
    os.environ.pop("GOOGLE_API_KEY", None)


# Lazy RAG initialization: keeps imports optional so app can import without all deps
rag_enabled = False
# Use an absolute path for the FAISS index so the app can load it regardless of working directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

try:
    from file_service import process_uploaded_file
except Exception:
    process_uploaded_file = None
    file_service_import_error = None
    try:
        # capture import error text if available
        raise
    except Exception as _e:
        file_service_import_error = str(_e)

def init_rag():
    global rag_enabled, embeddings, db, llm, retriever, SYSTEM_PROMPT, prompt, qa_chain, rag_chain, HumanMessage, AIMessage
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Replace your current lines with these:
        from langchain_classic.chains import create_retrieval_chain
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.messages import HumanMessage, AIMessage
    except Exception as e:
        print("RAG initialization skipped; missing RAG dependencies:", e)
        rag_enabled = False
        return

    try:
        print("Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        print("Loading FAISS index...")
        db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

        retriever = db.as_retriever(search_kwargs={"k": 3})

        SYSTEM_PROMPT = '''\
You are a helpful assistant.
Use the context to answer the question in max three sentences.
If you don't know , just say don't know.
Context: {context}
Chat History: {chat_history}
'''

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )

        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        rag_enabled = True
        print("RAG chain created.")
    except Exception as e:
        print("RAG initialization failed:", e)
        rag_enabled = False

# Do not initialize RAG on import/startup to avoid heavy blocking operations.
# RAG will be initialized lazily within endpoints when needed by calling `init_rag()`.

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
def get_db_conn():
    if psycopg2 is None:
        raise RuntimeError("Database driver 'psycopg2' is not installed. Install dependencies: pip install -r requirements.txt")
    conn = psycopg2.connect(DB_URL)
    return conn


class QueryRequest(BaseModel):
    user_id: int
    text: str


class HistoryRequest(BaseModel):
    user_id: int


class UserRequest(BaseModel):
    username: str

#api endpoint

#login/signup

@app.post("/get_or_create_user")
def get_or_create_user(req: UserRequest):
    conn = get_db_conn()
    cur = conn.cursor()
    
    # 1. Try to find the user
    cur.execute("SELECT id FROM users WHERE username = %s", (req.username,))
    user_row = cur.fetchone() #(1,)

    
    if user_row:
        user_id = user_row[0] #1
    else:
        # 2. If not found, create them
        cur.execute("INSERT INTO users (username) VALUES (%s) RETURNING id", (req.username,))
        conn.commit()
        user_id = cur.fetchone()[0] #2
        
    cur.close()
    conn.close()
    return {"user_id": user_id, "username": req.username}

#chat history

@app.post("/get_history")
def get_history(req:HistoryRequest):
    conn = get_db_conn()
    cur = conn.cursor()
    
    cur.execute("SELECT prompt, answer FROM chat_history WHERE user_id = %s ORDER BY id ASC", (req.user_id,))
    history = cur.fetchall() #[("hi","hello how i can help you"),("hi","hello how i can help you")]
    
    cur.close()
    conn.close()
    
    # Format for frontend
    formatted_history = []
    for p, a in history:
        formatted_history.append({"role": "human", "content": p})
        formatted_history.append({"role": "ai", "content": a})

    #[{"role": "human", "content": "hi"}, {"role": "ai", "content": "hello how i can help you"}, {"role": "human", "content": "hi"}, {"role": "ai", "content": "hello how i can help you"}]
    return {"history": formatted_history}
# ... (FastAPI setup remains same)

@app.post("/query")
def query_rag(req: QueryRequest):
    # Ensure RAG is initialized (lazy initialization). If initialization fails,
    # return a clear JSON response instead of raising an internal exception.
    if not rag_enabled:
        try:
            init_rag()
        except Exception as e:
            return JSONResponse({"error": "RAG initialization failed.", "detail": str(e)}, status_code=503)
        if not rag_enabled:
            return JSONResponse({"error": "RAG features are not initialized. Install required dependencies or check FAISS index."}, status_code=503)

    conn = get_db_conn()
    cur = conn.cursor()
    
    # Fetch history
    cur.execute("SELECT prompt, answer FROM chat_history WHERE user_id = %s ORDER BY id ASC", (req.user_id,))
    db_history = cur.fetchall()
    
    chat_history_messages = []
    for p, a in db_history:
        chat_history_messages.append(HumanMessage(content=p))
        chat_history_messages.append(AIMessage(content=a))
    
    # FIXED: The chain now correctly receives 'chat_history'
    try:
        response = rag_chain.invoke({
            "input": req.text,
            "chat_history": chat_history_messages
        })
        answer = response.get("answer", "I'm sorry, I couldn't process that.")
    except Exception as e:
        # Return the LLM/chain error to the caller for easier debugging in the UI
        return JSONResponse({"error": "LLM invocation failed", "detail": str(e)}, status_code=502)
    
    # Save to DB
    cur.execute("INSERT INTO chat_history (user_id, prompt, answer) VALUES (%s, %s, %s)", 
                (req.user_id, req.text, answer))
    conn.commit()  
    cur.close()
    conn.close()

    return {"answer": answer}

@app.get("/")
def read_root():
    return {"message": "welcome to fastapi.go to /docs to get started"}

# Inside your upload endpoint in app.py

@app.post("/upload")
def upload_file(file: UploadFile, username: Optional[str] = Form(None)):
    global process_uploaded_file
    try:
        # 1. Ensure upload directory exists
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        # 2. Determine saved filename and path
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        safe_name = f"{timestamp}_{file.filename}"
        saved_path = os.path.join(upload_dir, safe_name)

        # 3. Save file locally
        with open(saved_path, "wb") as f:
            f.write(file.file.read())

        # If frontend didn't provide a username, use an anonymous default
        if not username:
            username = f"anonymous_{timestamp}"

        # 4. Persist file metadata linked to user in DB
        conn = get_db_conn()
        cur = conn.cursor()

        # Find or create user
        cur.execute("SELECT id FROM users WHERE username = %s", (username,))
        user_row = cur.fetchone()
        if user_row:
            user_id = user_row[0]
        else:
            cur.execute("INSERT INTO users (username) VALUES (%s) RETURNING id", (username,))
            user_id = cur.fetchone()[0]
            conn.commit()

        # Insert file metadata
        cur.execute(
            "INSERT INTO files (user_id, filename, filepath, uploaded_at) VALUES (%s, %s, %s, %s) RETURNING id",
            (user_id, file.filename, saved_path, datetime.utcnow()),
        )
        file_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        # 5. Prepare a local processor reference; try to import at runtime if global isn't available
        proc = process_uploaded_file
        if proc is None:
            try:
                from file_service import process_uploaded_file as _proc
                proc = _proc
            except Exception as e:
                # Provide a minimal fallback for plain text files so uploads still work
                ext = os.path.splitext(saved_path)[-1].lower()
                if ext == ".txt":
                    class _SimpleDoc:
                        def __init__(self, text):
                            self.page_content = text
                            self.metadata = {}

                    with open(saved_path, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                    new_chunks = [_SimpleDoc(content)]
                    fallback_mode = True
                else:
                    # Include the import exception detail when available to aid debugging
                    detail = str(e) if 'e' in locals() else (globals().get('file_service_import_error') or "")
                    return JSONResponse({"error": "file_service.process_uploaded_file is not available. Install file parsing dependencies for PDF/DOCX support.", "detail": detail}, status_code=500)

        # 6. Try to initialize RAG if not ready
        if not rag_enabled:
            try:
                init_rag()
            except Exception:
                pass

        if not rag_enabled:
            return JSONResponse({"error": "RAG is not initialized. Install RAG deps or check FAISS index."}, status_code=500)

        # 7. Process into chunks using your service (documents may include metadata)
        if 'new_chunks' not in locals():
            if proc is None:
                return JSONResponse({"error": "No document processor available."}, status_code=500)
            new_chunks = proc(saved_path)

        # Optionally attach file/user metadata to each chunk's metadata
        for d in new_chunks:
            if not hasattr(d, "metadata"):
                d.metadata = {}
            d.metadata.update({"user_id": user_id, "file_id": file_id, "source": saved_path})

        # 8. Add new chunks to the existing FAISS index (skip if using simple fallback)
        if not locals().get('fallback_mode', False):
            global db  # Access the db loaded at startup
            try:
                db.add_documents(new_chunks)
            except Exception as e:
                return JSONResponse({"error": "Failed to add documents to FAISS index.", "detail": str(e)}, status_code=500)
        else:
            # When using the minimal text fallback, we skip adding to FAISS but return success
            return JSONResponse({"status": "File stored (text fallback), indexing skipped", "file_id": file_id, "user_id": user_id})

        # 9. Save it back to disk so it persists after restart
        try:
            db.save_local(FAISS_PATH)
        except Exception as e:
            # not fatal, but warn
            print("Warning: failed to save FAISS index:", e)

        return JSONResponse({"status": "File indexed successfully", "file_id": file_id, "user_id": user_id})
    except Exception as e:
        # Return JSON error to frontend so it can display a clear message
        return JSONResponse({"error": "Upload failed", "detail": str(e)}, status_code=500)