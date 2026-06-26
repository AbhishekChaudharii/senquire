# Senquire Assignment
# 📄 Document Q&A RAG Chatbot

This application allows users to upload multiple documents (PDF, DOCX, TXT) simultaneously, parses and embeds their contents into an isolated vector database session, and enables natural conversational Q&A where the AI accurately grounds its answers in the uploaded files.

---

## 🏗️ Architecture Overview

The system architecture is divided into three modular Haystack 2.0 pipelines to separate file processing, conversational context optimization, and document retrieval.

### 1. Ingestion Pipeline (`final_ingest.py`)
Responsible for parsing and transforming user-uploaded files into searchable vector embeddings.
* **Routing:** Uses `FileTypeRouter` to handle incoming file arrays based on their MIME type (`text/plain`, `application/pdf`, and OpenXML Word documents).
* **Extraction:** Routes files to dedicated text extraction engines (`TextFileToDocument`, `PyPDFToDocument`, and `DOCXToDocument`).
* **Preprocessing:** Standardizes the text via `DocumentCleaner` and uses `DocumentSplitter` to break long texts into cohesive chunks (300 words per chunk with a 50-word sliding window overlap) to preserve local semantic context.
* **Embedding & Storage:** Passes chunks to the Hugging Face Serverless Inference API running the `BAAI/bge-small-en-v1.5` embedding model, and saves them to a `ChromaDocumentStore` instance using a `SKIP` duplication policy.

### 2. Query Rewriting Pipeline (`final_query.py`)
Resolves conversational dependencies and co-references to enable robust multi-turn conversations.
* **Contextualization:** Accepts the raw user input along with a window of the last 4 chat history messages formatted as a unified string.
* **Transformation:** Leverages **Llama-3.1-8b-instant** (via Groq) to rewrite conversational context. If a user asks "Who is the CEO of Apple?" followed by "How old is he?", the rewriter optimizes the latter into the standalone query: `"How old is Tim Cook?"`. This ensures downstream vector searches are highly optimized.

### 3. Query & Retrieval Pipeline (`final_query.py`)
Executes the target vector match and generates the finalized answer.
* **Embedding:** Converts the rewritten standalone query into a dense vector via `HuggingFaceAPITextEmbedder`.
* **Retrieval:** Uses `ChromaEmbeddingRetriever` to fetch the top 5 (`top_k=5`) most relevant document chunks matching the query tensor.
* **Generation:** Injects the retrieved documents into a highly guarded prompt template. It strictly instructs the LLM (**Llama-3.1-8b-instant** on Groq) to answer exclusively from the provided text chunks, fallback to an absolute refusal phrase if the context is insufficient, and output zero external commentary.

---

## ✨ Key Design Decisions

1. **Query Rewriting vs. History Condensing:** Instead of feeding full conversational histories into the primary vector retriever (which often clouds search keyword accuracy), a dedicated, low-latency LLM pass re-contextualizes the query. This maximizes embedding alignment for conversational follow-ups.
2. **Session-Isolated Chroma Collections:** To prevent cross-user data contamination and maintain individual user privacy, every new Streamlit session generates a unique `uuid`. This string acts as an ephemeral `collection_name` inside ChromaDB, automatically isolating vector instances per browser tab.
3. **Ultra-Low Latency Inference:** The application decouples local computation completely by using **Groq** for rapid-fire token generation alongside Hugging Face's serverless pipeline for text embeddings. The chatbot can run fully on standard CPU servers without local hardware bottlenecks.
4. **Source File Attribution:** The Streamlit app traces chunk metadata backwards post-retrieval. It extracts the original `file_path` variables from the active document windows to append exact source citations (`**📌 Sources:** file1.pdf, file2.docx`) underneath successful AI answers.

---

## 🚀 Setup Steps

### Prerequisites
* Python 3.10+
* A [Groq Cloud API Key](https://console.groq.com/)
* A [Hugging Face User Access Token](https://huggingface.co/settings/tokens)

### 1. Clone & Structure the Workspace
Ensure your workspace directory contains the following file structures:
```text
├── final_app.py          # Streamlit UI & Orchestration
├── final_ingest.py       # Document extraction & ingestion pipelines
├── final_query.py        # Query rewriting & RAG execution pipelines
└── .env                  # Environment secrets (to be created)
```
### 2. Install Dependencies
`pip install -r requirements.txt`

### 3. Create a .env file and add your api keys
```text 
GROQ_API_KEY="gsk_your_actual_groq_api_key_goes_here"

HF_API_TOKEN="hf_your_actual_huggingface_token_goes_here"

```

## 💻 How to Run
`streamlit run final_app.py`

## Application workflow

1. Open your browser to the local hosting port

2. Expand the sidebar panel titled "📂 Upload Documents".

3. Drag-and-drop or select any combination of .txt, .pdf, or .docx files.

4. Click "Ingest Documents". The system will notify you with a success badge indicating the total chunk count indexed.

5. Head to the primary chat interface at the bottom of the screen and begin querying your documentation dataset!