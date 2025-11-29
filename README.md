# PAF-IAST-Policy-Advisor-RAG

🏛️ PAF-IAST Academic Compliance System (RAG Policy Advisor)

Project Overview

This system is a sophisticated, deployment-ready Retrieval-Augmented Generation (RAG) chatbot developed for the Programming for Artificial Intelligence (COMP-212) final project. It is designed to serve as the official, instant guidance platform for all PAF-IAST student academic policy queries.

The primary challenge it solves is extracting precise, verified facts from extensive, unstructured official university rulebooks (PDFs) that are often difficult for students to navigate. By transforming these complex documents into a structured, searchable knowledge base, the system ensures students receive accurate, contextual answers instantly.

🚀 Key Features and AI Implementation

Feature

Implementation Detail

COMP-212 Link

RAG Architecture

Uses FAISS (Facebook AI Similarity Search) for high-speed vector retrieval, making the search engine fast and efficient on minimal hardware.

Data Structures, NumPy, Streamlit

Policy Ingestion

Robust Python logic utilizing pypdf for data loading, regular expressions (re) for data cleaning (removing headers/footers), and langchain-text-splitters for logical chunking.

Python Fundamentals, Data Cleaning

AI Backend

Integrates Google Gemini API for both high-precision Embeddings (text-embedding-004) and context-aware Generation (gemini-2.5-flash).

AI/LLM Integration

Scalability

Designed with batch processing to efficiently vectorize large document sets, ensuring scalability if new policy documents are added.

OOP Design Principles (Conceptual)

Deployment

Presented via a user-friendly, clean chat interface using Streamlit, making it ready for live deployment on the University's LMS or portal.

Streamlit, Final Product Finishing

🛠️ Installation and Setup

This guide covers setting up the application and configuring the API key locally.

1. Prerequisites

You must have Python (3.10+) installed and an active Gemini API Key.

2. Project Files

Ensure your repository contains these four critical files:

policy_bot_app.py (The main application code)

requirements.txt (List of dependencies)

policy_faiss_index.bin (The Vector Database, built in Jupyter)

policy_chunks.pkl (The original text segments)

3. Install Dependencies

Install all necessary libraries using the provided requirements.txt file:

pip install -r requirements.txt


4. Configure API Key (Securely)

The API key must be configured in a secrets file, which should NOT be committed to GitHub for security reasons.

Create a folder named .streamlit in your project's root directory.

Inside .streamlit, create a file named secrets.toml.

Paste your Gemini API key inside this file in the following format:

# .streamlit/secrets.toml
GEMINI_API_KEY = "YOUR_ACTUAL_GEMINI_API_KEY" 


5. Run the Application Locally

Execute the Streamlit application from your terminal:

streamlit run policy_bot_app.py


The application will launch in your browser, ready for testing.

👩‍💻 Developer

[Your Name] - [Your Student ID]

Course: Programming for Artificial Intelligence (COMP-212)
