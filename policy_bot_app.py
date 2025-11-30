# policy_bot_app.py - Final Production Code (Configuration Error Fix)

# --- 1. Library Imports ---
import streamlit as st
import faiss
import pickle
import numpy as np
import os
import textwrap
from google.genai import Client

# --- 2. Configuration & Global Variables ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 

EMBEDDING_MODEL = 'text-embedding-004' 
VECTOR_STORE_PATH = 'policy_faiss_index.bin'
CHUNKS_PATH = 'policy_chunks.pkl'

# Initialize global variables
POLICY_INDEX = None
POLICY_CHUNKS = None
ST_INITIALIZED = False
CLIENT = None 

# policy_bot_app.py (Line 25 ke aas paas ka code)

# --- 2. Configuration & Global Variables ---
# CRITICAL FIX: API Key ko ab Streamlit Secrets se uthaya ja raha hai
try:
    # st.secrets ko use karne ke liye, 'import streamlit as st' zaroori hai
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
except KeyError:
    # Agar Secrets mein key na mile, toh error dikhao
    st.error("Error: GEMINI_API_KEY not found in Streamlit Secrets.")
    st.stop()
    
EMBEDDING_MODEL = 'text-embedding-004' 
# ... (Baaki code same)
# --- 3. Initial Setup and Loading (Robust Caching) ---

# Caching decorator use kiya gaya hai. Isay 'main' function se pehly define karna zaroori hai.
@st.cache_resource(show_spinner="Initializing RAG System...")
def setup_rag_system(api_key, vector_path, chunks_path):
    """
    Loads API client, FAISS index, and policy chunks.
    """
    try:
        client_obj = Client(api_key=api_key)
        
        if not os.path.exists(vector_path) or not os.path.exists(chunks_path):
            st.error(f"Knowledge Base files not found: {vector_path} or {chunks_path}. Please run Jupyter setup first.")
            return client_obj, None, None
            
        index = faiss.read_index(vector_path)
        with open(chunks_path, 'rb') as f:
            policy_chunks = pickle.load(f)
            
        return client_obj, index, policy_chunks

    except Exception as e:
        st.error(f"Initialization Failed. Check API Key or library versions. Details: {e}")
        return None, None, None


# --- 4. RAG Functions (Logic) ---

def _get_query_embedding(query):
    """ Converts query to vector. """
    try:
        response = CLIENT.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[query]
        )
        return np.array(response.embeddings[0].values).astype('float32').reshape(1, -1)
    except Exception:
        return None

def retrieve_context(query_vector, k=8): 
    """ Retrieves the most relevant context chunks. """
    D, I = POLICY_INDEX.search(query_vector, k) 
    relevant_chunks = [POLICY_CHUNKS[i] for i in I[0] if i < len(POLICY_CHUNKS)]
    context = "\n---\n".join(relevant_chunks)
    return context

def generate_response(query, context):
    """ Generates the final answer using the strict prompt. """
    
    system_prompt = (
        "You are the PAF-IAST Academic Policy Assistant. Your sole purpose is to "
        "extract precise facts and rules from the POLICY CONTEXT to answer the student's question. "
        "Prioritize mandatory requirements like CGPA, attendance, and deadlines. "
        "If the specific answer (e.g., a numerical value or an exact rule) "
        "is NOT explicitly present in the context, strictly reply with: "
        "'I cannot find this information in the official policy documents.'"
    )
    
    full_prompt = (
        f"{system_prompt}\n\n"
        f"--- POLICY CONTEXT ---\n{context}\n\n"
        f"--- STUDENT QUESTION ---\n{query}"
    )
    
    try:
        response = CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt
        )
        return response.text
        
    except Exception as e:
        return f"❌ ERROR during response generation: {e}"

def ask_policy_bot(query):
    """ Main function to combine retrieval and generation. """
    if not ST_INITIALIZED:
         return "❌ Initialization Error: Knowledge base not ready."
         
    query_vector = _get_query_embedding(query)
    if query_vector is None:
        return "❌ Unable to process the request."
        
    context = retrieve_context(query_vector)
    return generate_response(query, context)

# --- 5. Streamlit App Interface (The Finishing) ---

def main():
    """ Defines the Streamlit UI and Chat Logic. """
    
    # CRITICAL FIX: set_page_config() yahan se hata diya gaya hai.

    st.title("PAF-IAST Policy Advisor 🎓")
    st.caption("RAG-Based AI Engine for Policy Retrieval and Compliance")
    
    # Initialization status sidebar mein dikhana
    global ST_INITIALIZED
    if ST_INITIALIZED:
        st.sidebar.success("RAG System Initialized.⚡")
    else:
        st.sidebar.error("RAG System Failed to Initialize.")
        st.stop() # Agar fail ho toh aagay execution rok do
        
    # Chat history initialize karna
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # History display karna
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input lena
    if prompt := st.chat_input("Ask a policy question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Bot response generate karna
        with st.chat_message("assistant"):
            with st.spinner("Consulting Official Academic Rulebook..."):
                response = ask_policy_bot(prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# --- FINAL EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Page configuration ko sab se pehly call karna
    st.set_page_config(page_title="PAF-IAST Policy Assistant 🤖", layout="wide")
    
    # 2. RAG System ko initialize karna
    CLIENT, POLICY_INDEX, POLICY_CHUNKS = setup_rag_system(GEMINI_API_KEY, VECTOR_STORE_PATH, CHUNKS_PATH)
    
    # 3. Global variables ko update karna
    if POLICY_INDEX is not None:
        ST_INITIALIZED = True
        
    # 4. Main app ko run karna

    main()


