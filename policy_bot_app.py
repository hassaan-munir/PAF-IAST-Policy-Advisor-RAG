# policy_bot_app.py - Final Production Code with Conditional Fallback
import streamlit as st

# Gemini API key ko Streamlit secrets se le rahe hain
GEMINI_API_KEY = st.secrets["gemini"]["api_key"]

# Example usage
st.write("API key successfully loaded!")
# Ab GEMINI_API_KEY ko apni API calls me use karo

# --- 1. Library Imports ---
import streamlit as st
import faiss
import pickle
import numpy as np
import os
from google.genai import Client

# --- 2. Configuration & Global Variables ---
GEMINI_API_KEY = "AIzaSyAFNsEDngZL-5bzzRHEidVfQv1vapyEmRE"  # <-- Add your API key here

EMBEDDING_MODEL = 'text-embedding-004'
VECTOR_STORE_PATH = 'policy_faiss_index.bin'
CHUNKS_PATH = 'policy_chunks.pkl'

POLICY_INDEX = None
POLICY_CHUNKS = None
ST_INITIALIZED = False
CLIENT = None 

# --- 3. Initial Setup and Loading (Robust Caching) ---
@st.cache_resource(show_spinner="Initializing RAG System...")
def setup_rag_system(api_key, vector_path, chunks_path):
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

def fallback_general_answer(query):
    """ Generates a short 2-3 sentence general answer from LLM for unknown queries. """
    try:
        fallback_prompt = (
            "Provide a short 2-3 sentence formal academic-style answer to the student's question. "
            "Do NOT mention missing information. "
            f"Question: {query}"
        )
        response = CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=fallback_prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"❌ ERROR in fallback: {e}"

def generate_response(query, context):
    """ Generates the final answer with conditional fallback call. """
    
    system_prompt = (
        "You are the PAF-IAST Academic Policy Assistant. Your sole purpose is to "
        "extract precise facts and rules from the POLICY CONTEXT to answer the student's question. "
        "Prioritize mandatory requirements like CGPA, attendance, and deadlines. "
        "If the specific answer (e.g., a numerical value or an exact rule) "
        "is NOT explicitly present in the context, strictly reply with: "
        "'I cannot find this information in the official policy documents.'"
    )
    
    full_prompt = f"{system_prompt}\n\n--- POLICY CONTEXT ---\n{context}\n\n--- STUDENT QUESTION ---\n{query}"
    
    try:
        # Step 1: Original RAG answer
        response = CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt
        )
        text = response.text.strip()

        # Step 2: Conditional fallback call
        fallback_trigger = "I cannot find this information in the official policy documents."
        if text == fallback_trigger:
            return fallback_general_answer(query)

        # Normal RAG output
        return text
        
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

# --- 5. Streamlit App Interface ---
def main():
    """ Defines the Streamlit UI and Chat Logic. """
    
    st.title("PAF-IAST Policy Advisor 🎓")
    st.caption("RAG-Based AI Engine for Policy Retrieval and Compliance")
    
    global ST_INITIALIZED
    if ST_INITIALIZED:
        st.sidebar.success("RAG System Initialized.⚡")
    else:
        st.sidebar.error("RAG System Failed to Initialize.")
        st.stop()
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a policy question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting Official Academic Rulebook..."):
                response = ask_policy_bot(prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# --- FINAL EXECUTION BLOCK ---
if __name__ == "__main__":
    st.set_page_config(page_title="PAF-IAST Policy Assistant 🤖", layout="wide")
    
    CLIENT, POLICY_INDEX, POLICY_CHUNKS = setup_rag_system(GEMINI_API_KEY, VECTOR_STORE_PATH, CHUNKS_PATH)
    
    if POLICY_INDEX is not None:
        ST_INITIALIZED = True
        
    main()

