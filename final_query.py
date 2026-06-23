from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.embedders import SentenceTransformersTextEmbedder


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