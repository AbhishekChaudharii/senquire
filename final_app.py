import streamlit as st
import os
import tempfile
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from final_ingest import build_preprocessing_pipeline
from final_query import build_query_pipeline


st.set_page_config(page_title="RAG Chat", page_icon="📄")
st.title("📄 Document Q&A Chatbot")

@st.cache_resource
def initialize_system():
    document_store = ChromaDocumentStore(collection_name="my_documents")
    ingest_pipeline = build_preprocessing_pipeline(document_store)
    query_pipeline = build_query_pipeline(document_store)
    return document_store, ingest_pipeline, query_pipeline

document_store, ingest_pipeline, query_pipeline = initialize_system()

with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload files (TXT, PDF, DOCX)", 
        type=["txt", "pdf", "docx"], 
        accept_multiple_files=True
    )
    
    if st.button("Ingest Documents"):
        if uploaded_files:
            with st.spinner("Processing and embedding documents..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(temp_file_path)
                    
                
                    ingest_pipeline.run({"file_type_router": {"sources": file_paths}})
            
            st.success(f"Successfully processed {len(uploaded_files)} file(s)!")
            st.info(f"Total documents in database: {document_store.count_documents()}")
        else:
            st.warning("Please upload files before ingesting.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = query_pipeline.run(
                    {
                        "query_embedder": {"text": prompt},
                        "prompt_builder": {"query": prompt}
                    },
                    include_outputs_from={"retriever"}
                )
                
                chat_message_obj = result["generator"]["replies"][0]
                llm_answer = chat_message_obj.text
                
                retrieved_chunks = result.get("retriever", {}).get("documents", [])
                source_files = set()
                for doc in retrieved_chunks:
                    file_path = doc.meta.get("file_path")
                    if file_path:
                        filename = os.path.basename(file_path)
                        source_files.add(filename)
                
                response_text = f"{llm_answer}\n\n"
                if source_files:
                    response_text += f"**📌 Sources:** {', '.join(source_files)}"
                else:
                    response_text += "**📌 Sources:** No document chunks retrieved."
                    
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")