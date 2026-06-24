from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import tempfile
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from final_ingest import build_preprocessing_pipeline
from final_query import build_query_pipeline,build_rewriter_pipeline
import mimetypes
mimetypes.add_type('application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.docx')


st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("📄 Document Q&A Chatbot")

@st.cache_resource
def initialize_system():
    document_store = ChromaDocumentStore(collection_name="my_documents")
    ingest_pipeline = build_preprocessing_pipeline(document_store)
    query_pipeline = build_query_pipeline(document_store)
    rewriter_pipeline = build_rewriter_pipeline()
    return document_store, ingest_pipeline, query_pipeline, rewriter_pipeline

document_store, ingest_pipeline, query_pipeline, rewriter_pipeline = initialize_system()

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
        if document_store.count_documents() == 0:
            fallback_msg = "I couldn't find this information because no documents have been uploaded yet. Please upload files in the sidebar."
            st.markdown(fallback_msg)
            st.session_state.messages.append({"role": "assistant", "content": fallback_msg})
            
        else:
            with st.spinner("Thinking..."):
                try:
                    # 1. Format last few chat messages to give the rewriter context
                    history_string = ""
                    # Will grab up to the last 4 messages (excluding current prompt)
                    recent_history = st.session_state.messages[-5:-1]
                    if recent_history:
                        for msg in recent_history:
                            history_string += f"{msg['role'].capitalize()}: {msg['content']}\n"
                    else:
                        history_string = "No previous history."

                    # 2. Running rewriter piepline
                    rewrite_result = rewriter_pipeline.run({
                        "prompt_builder": {
                            "history": history_string,
                            "query": prompt
                        }
                    })
                    
                    # Extract the standalone string out of the ChatMessage object wrapper
                    standalone_query = rewrite_result["generator"]["replies"][0].text.strip()
                    # TODO: Debug print to terminal to trace how llm optimizes the query
                    # print(f"\n[DEBUG CONTEXT] Original: {prompt}")
                    # print(f"[DEBUG CONTEXT] Rewritten: {standalone_query}\n")

                    result = query_pipeline.run(
                        {
                            "query_embedder": {"text": standalone_query},
                            "prompt_builder": {"query": standalone_query}
                        },
                        include_outputs_from={"retriever"}
                    )
                    
                    #* Extracting sources
                    retrieved_chunks = result.get("retriever", {}).get("documents", [])
                    
                    if len(retrieved_chunks) == 0:
                        fallback_msg = "I couldn't find this information in the uploaded documents."
                        st.markdown(fallback_msg)
                        st.session_state.messages.append({"role": "assistant", "content": fallback_msg})
                    
                    else:
                        chat_message_obj = result["generator"]["replies"][0]
                        llm_answer = chat_message_obj.text
                        
                        source_files = set()
                        for doc in retrieved_chunks:
                            file_path = doc.meta.get("file_path")
                            if file_path:
                                filename = os.path.basename(file_path)
                                source_files.add(filename)
                        refusal_phrase="I couldn't find this information in the uploaded documents."
                        if refusal_phrase in llm_answer.strip():
                            response_text=refusal_phrase
                        else:
                            response_text = f"{llm_answer}\n\n**📌 Sources:** {', '.join(source_files)}"
                                                    
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        
                except Exception as e:
                    st.error(f"An error occurred during generation: {e}")