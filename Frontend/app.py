import streamlit as st
import requests

# Ensure this matches your FastAPI port
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG CHATBOT", layout="wide")
st.title("RAG CHATBOT 🤖")

# --- Session State Initialization ---
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Authentication Logic ---
if st.session_state.user_id is None:
    st.header("Welcome! Please login to continue to chat...")
    username = st.text_input("Enter your Username:", key="login_username")

    if st.button("Login/Signup", key="login_button"):
        if username:
            try:
                # 1. Login/Signup Request
                res = requests.post(f"{API_URL}/get_or_create_user", json={"username": username})
                res.raise_for_status()
                data = res.json()
                
                st.session_state.user_id = data["user_id"]
                st.session_state.username = data["username"]
                
                # 2. Fetch Chat History immediately after login
                res_hist = requests.post(f"{API_URL}/get_history", json={"user_id": data["user_id"]})
                if res_hist.status_code == 200:
                    st.session_state.messages = res_hist.json().get("history", [])
                
                st.rerun()
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
        else:
            st.warning("Please enter a username.")

# --- Main Chat Interface (Only if Logged In) ---
else:
    # --- Sidebar: User Info and File Upload ---
    with st.sidebar:
        st.header(f"👤 {st.session_state.username}")
        st.write(f"User ID: {st.session_state.user_id}")
        
        st.markdown("---")
        st.subheader("📁 Knowledge Base")
        st.info("Upload PDF, TXT, or DOCX to add to the chatbot's memory.")
        
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"])
        
        if st.button("Upload & Index"):
            if uploaded_file:
                with st.spinner("Processing file..."):
                    try:
                        # Prepare the file for the multipart/form-data request
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                        # Include the logged-in username as a form field so backend receives it
                        data = {"username": st.session_state.username}
                        response = requests.post(f"{API_URL}/upload", files=files, data=data)
                        
                        if response.status_code == 200:
                            st.success(f"Successfully indexed {uploaded_file.name}!")
                        else:
                            # Try to show useful error information returned by backend
                            try:
                                err = response.json()
                            except Exception:
                                err = response.text
                            # Prefer explicit fields if present
                            msg = None
                            if isinstance(err, dict):
                                msg = err.get('detail') or err.get('error') or str(err)
                            else:
                                msg = str(err)
                            st.error(f"Failed: {msg}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")
            else:
                st.warning("Please select a file first.")
        
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # --- Chat Window ---
    st.header("Conversation")
    st.caption("Ask questions based on your documents.")

    # Display History
    for chat in st.session_state.messages:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    # Chat Input
    if prompt := st.chat_input("Ask me anything..."):
        # 1. Display Human message
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)
        
        # 2. Get AI response from backend
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                try:
                    payload = {
                        "user_id": st.session_state.user_id,
                        "query": prompt
                    }
                    res = requests.post(f"{API_URL}/query", json=payload)
                    res.raise_for_status()
                    
                    answer = res.json().get("answer", "I couldn't retrieve an answer.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "ai", "content": answer})
                    
                except Exception as e:
                    st.error(f"Error fetching response: {e}")