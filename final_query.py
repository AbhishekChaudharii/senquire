from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.embedders import HuggingFaceAPITextEmbedder
from haystack.utils import Secret


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
    query_embedder = HuggingFaceAPITextEmbedder(api_type="serverless_inference_api",api_params={"model": "BAAI/bge-small-en-v1.5"})
    retriever = ChromaEmbeddingRetriever(document_store=document_store,top_k=5)
    generator = OpenAIChatGenerator(
    api_key=Secret.from_env_var("GROQ_API_KEY"),
    api_base_url="https://api.groq.com/openai/v1",
    model="llama-3.1-8b-instant",
    generation_kwargs = {"max_tokens": 512}
)
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