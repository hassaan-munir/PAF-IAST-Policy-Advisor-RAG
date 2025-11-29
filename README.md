PAF IAST Policy Advisor RAG System

🏛️ PAF IAST Academic Compliance System [RAG Based Policy Advisor]

This project is a deployment ready Retrieval Augmented Generation chatbot created for the Programming for Artificial Intelligence [COMP 212] final project. The system works as an instant guidance platform for all PAF IAST student academic policy related questions.

Its main purpose is to extract accurate and verified information from the official university rulebooks, which are usually large unstructured PDF documents. These documents are converted into a clean structured knowledge base so that students can receive correct and context aware answers immediately.

🚀 Key Features and AI Implementation
Feature and Explanation
Feature	Implementation Detail	COMP 212 Link
RAG Architecture	Uses FAISS, Facebook AI Similarity Search, for fast vector based retrieval in low hardware environments	Data Structures, NumPy, Streamlit
Policy Ingestion	Uses pypdf for PDF loading, re for cleaning, langchain text splitters for chunking	Python Fundamentals, Data Cleaning
AI Backend	Uses Google Gemini API, text embedding 004 for embeddings, gemini 2.5 flash for generation	AI and LLM Integration
Scalability	Batch vectorization logic supports very large documents	OOP based design understanding
Deployment	User friendly chat interface built using Streamlit	Streamlit, final project finishing
🛠️ Installation and Setup

Follow these steps to install and run the application locally.

1. Requirements

You must have Python 3.10 or newer and a valid Gemini API Key.

2. Required Project Files

Your repository must contain these four files.

policy_bot_app.py, the main application file

requirements.txt

policy_faiss_index.bin

policy_chunks.pkl

These last two files are created during the vectorization stage in Jupyter Notebook.

3. Install Dependencies

Run this command in your terminal.

pip install -r requirements.txt

4. Configure the API Key [Secure Method]

Never upload your API key to GitHub.

Create a folder named .streamlit in your main project directory

Create a file inside it named secrets.toml

Add the following lines inside that file

GEMINI_API_KEY = "YOUR_ACTUAL_GEMINI_API_KEY"

5. Run the Application

Use the command below to start the interface.

streamlit run policy_bot_app.py


The browser will automatically open the chat based policy advisor.
