# PAF IAST Policy Advisor RAG System

🏛️ **PAF IAST Academic Compliance System, RAG Based Policy Advisor**

This project is a deployment ready Retrieval Augmented Generation chatbot created for the Programming for Artificial Intelligence [COMP 212] final project. It provides instant, accurate and verified answers to student academic policy questions by converting large unstructured university rulebooks into a clean searchable knowledge base.

---

## 🚀 Key Features and AI Implementation

### System Breakdown

| Feature          | Implementation Detail                                                                      | Relevant COMP 212 Topic           |
| ---------------- | ------------------------------------------------------------------------------------------ | --------------------------------- |
| RAG Architecture | Uses FAISS, Facebook AI Similarity Search, for fast vector retrieval                       | Data Structures, NumPy, Streamlit |
| Policy Ingestion | Uses pypdf for PDF extraction, re for data cleaning, langchain text splitters for chunking | Python Basics, Data Cleaning      |
| AI Backend       | Uses Google Gemini API, text embedding 004 for embeddings, gemini 2.5 flash for generation | AI and LLM Integration            |
| Scalability      | Batch vectorization supports large document sets                                           | OOP Concepts                      |
| Deployment       | Streamlit based chat interface for smooth user experience                                  | Streamlit Development             |

---

## 🛠️ Installation and Setup

Follow these steps to run the system locally.

### 1. Requirements

You need Python 3.10 or higher and a valid Gemini API key.

### 2. Project Files

Your project repository must include these files.

* `policy_bot_app.py`
* `requirements.txt`
* `policy_faiss_index.bin`
* `policy_chunks.pkl`

These last two files are generated earlier during knowledge base creation.

### 3. Install Dependencies

Run this command to install everything.

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key

Do not upload your API key to GitHub.

1. Create a folder named `.streamlit` in your project root
2. Inside it create a file named `secrets.toml`
3. Add the following lines

```
GEMINI_API_KEY = "YOUR_ACTUAL_GEMINI_API_KEY"
```

### 5. Run the Application

Start the chatbot using this command.

```bash
streamlit run policy_bot_app.py
```

Your browser will open the policy advisor interface.

---

## 👩‍💻 Developer Information

**Name:** Muhammad Hassaan Munir<br>
**Live Chatbot:** [Click Here](https://paf-iast-policy-advisor.streamlit.app)


