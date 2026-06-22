from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.embedders import SentenceTransformersTextEmbedder
import os

# Prompt template
template = """
You are a helpful assistant.

Answer the question only from the context below.
If the answer is not present in the context, say:
"I couldn't find this information in the uploaded documents."

Context:
{% for doc in documents %}
{{ doc.content }}

{% endfor %}

Question: {{ query }}

Answer:
"""
def build_query_pipeline(document_store: ChromaDocumentStore) -> Pipeline:
    # Query embedder
    query_embedder = SentenceTransformersTextEmbedder(model="BAAI/bge-small-en-v1.5")
    retriever = ChromaEmbeddingRetriever(document_store=document_store,top_k=5)
    generator = OllamaChatGenerator(model="llama3", url="http://localhost:11434")
    prompt_builder = PromptBuilder(template=template, required_variables=["documents", "query"])

    # Pipeline
    query_pipeline = Pipeline()

    query_pipeline.add_component("query_embedder", query_embedder)
    query_pipeline.add_component("retriever", retriever)
    query_pipeline.add_component("prompt_builder", prompt_builder)
    query_pipeline.add_component("generator", generator)

    query_pipeline.connect("query_embedder.embedding","retriever.query_embedding")
    query_pipeline.connect("retriever.documents","prompt_builder.documents")
    query_pipeline.connect("prompt_builder.prompt","generator.messages")

    return query_pipeline

if __name__ == "__main__":
    # Point to the exact same collection and database directory
    document_store = ChromaDocumentStore(persist_path="./chroma_db", collection_name="my_documents")
    
    # Check if database actually has documents before starting
    print(f"Total documents found in Chroma storage: {document_store.count_documents()}")
    
    pipeline = build_query_pipeline(document_store)
    
    # Clean, simple question targeting your text contents
    user_query = "What is the plain text sample file used for?"
    # user_query = "what is the objective of the assignment?"
    
    result = pipeline.run(
        {
            "query_embedder": {
                "text": user_query
            },
            "prompt_builder": {
                "query": user_query
            }
        },
        include_outputs_from={"retriever"}
    )
    
    # print("\n--- Answer ---")
    # 1. Extract the raw string text out of the ChatMessage object wrapper
    chat_message = result["generator"]["replies"][0]
    llm_answer = chat_message.text

    # 2. Extract the retrieved document chunks safely from the retriever output cache
    retrieved_chunks = result.get("retriever", {}).get("documents", [])

    # 3. Collect unique file names from the chunks metadata
    source_files = set()
    for doc in retrieved_chunks:
        file_path = doc.meta.get("file_path")
        if file_path:
            filename = os.path.basename(file_path)
            source_files.add(filename)

    # 4. Print the final beautifully formatted output
    print("\n--- Answer ---")
    print(f"💡 Answer:\n{llm_answer}")

    if source_files:
        print(f"\n📌 Source: {', '.join(source_files)}")
    else:
        print("\n📌 Source: No document chunks retrieved.")

