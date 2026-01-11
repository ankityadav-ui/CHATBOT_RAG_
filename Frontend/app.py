import streamlit as st
import requests

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
                # Login/Signup Request
                res = requests.post(f"{API_URL}/get_or_create_user", json={"username": username})
                res.raise_for_status()
                data = res.json()
                st.session_state.user_id = data["user_id"]
                st.session_state.username = data["username"]
                
                # Fetch Chat History
                res_hist = requests.post(f"{API_URL}/get_history", json={"user_id": data["user_id"]})
                st.session_state.messages = res_hist.json()["history"]
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
        
        st.markdown("---")
        st.subheader("📁 Knowledge Base")
        st.info("Upload new documents (PDF/TXT) to the chatbot's brain.")
        
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"])
        
        if st.button("Upload & Index"):
            if uploaded_file:
                with st.spinner("Processing file..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                        response = requests.post(f"{API_URL}/upload", files=files)
                        if response.status_code == 200:
                            st.success(f"Successfully indexed {uploaded_file.name}!")
                        else:
                            st.error(f"Failed: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please select a file first.")
        
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.messages = []
            st.rerun()

    # --- Chat Window ---
    st.header("Chat Interface")
    st.caption("Ask questions about your uploaded documents.")

    # Display History
    for chat in st.session_state.messages:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    # Chat Input
    if prompt := st.chat_input("Ask me anything..."):
        # Display Human message
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                try:
                    res = requests.post(
                        f"{API_URL}/query", 
                        json={"user_id": st.session_state.user_id, "text": prompt}
                    )
                    res.raise_for_status()
                    answer = res.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "ai", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")