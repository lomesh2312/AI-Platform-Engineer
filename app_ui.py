import streamlit as st
import requests
import json
import time

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal RAG - Google Drive",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for Premium Look ─────────────────────────────────────────────
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    .source-box {
        background-color: #1e2227;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
    .status-ok {
        color: #4CAF50;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
API_BASE_URL = "https://personal-rag-system.onrender.com/api/v1"

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Control Center")
    st.markdown("---")
    
    st.subheader("Knowledge Base")
    if st.button("🔄 Sync Google Drive"):
        with st.status("Connecting to Drive...", expanded=True) as status:
            try:
                st.write("Fetching documents...")
                response = requests.post(f"{API_BASE_URL}/sync-drive", timeout=300)
                if response.status_code == 200:
                    data = response.json()
                    st.write(f"✅ Indexed {data['documents_synced']} documents.")
                    st.write(f"📊 Created {data['total_chunks']} searchable chunks.")
                    status.update(label="Sync Complete!", state="complete", expanded=False)
                    st.toast("Drive Sync Successful!", icon="✅")
                else:
                    st.error(f"Sync failed: {response.text}")
                    status.update(label="Sync Failed", state="error")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                status.update(label="Error Connecting to API", state="error")

    st.markdown("---")
    st.subheader("System Health")
    try:
        health = requests.get("https://personal-rag-system.onrender.com/health").json()
        st.markdown(f"Status: <span class='status-ok'>{health['status'].upper()}</span>", unsafe_allow_html=True)
    except:
        st.markdown("Status: <span style='color: #ff4b4b; font-weight: bold;'>OFFLINE</span>", unsafe_allow_html=True)
        st.warning("Make sure the FastAPI server is running (python main.py)")

# ─── Main Interface ──────────────────────────────────────────────────────────
st.title("🧠 Personal RAG System")
st.markdown("#### Chat with your Google Drive documents using Llama 3 & FAISS")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("🔍 View Sources"):
                for src in message["sources"]:
                    st.markdown(f"<div class='source-box'>📄 {src}</div>", unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("Ask something about your documents..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Thinking...")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/ask",
                json={"query": prompt},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]
                
                message_placeholder.markdown(answer)
                
                if sources:
                    with st.expander("🔍 View Sources"):
                        for src in sources:
                            st.markdown(f"<div class='source-box'>📄 {src}</div>", unsafe_allow_html=True)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            else:
                error_msg = response.json().get("detail", "Unknown error")
                message_placeholder.error(f"Error: {error_msg}")
        except Exception as e:
            message_placeholder.error(f"Connection Error: {str(e)}")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with FastAPI, Streamlit, FAISS, and Groq Llama 3.")
