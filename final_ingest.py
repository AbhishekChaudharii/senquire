from haystack.components.writers import DocumentWriter
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.converters.docx import DOCXToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPIDocumentEmbedder
from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.document_stores.types import DuplicatePolicy
# from haystack.utils import Secret
import mimetypes
mimetypes.add_type('application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.docx')

def build_preprocessing_pipeline(document_store: ChromaDocumentStore) -> Pipeline:
    
    DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", DOCX_MIME])
    text_file_converter = TextFileToDocument()
    docx_converter = DOCXToDocument()
    pdf_converter = PyPDFToDocument()
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="word", split_length=300, split_overlap=50)
    text_embedder = HuggingFaceAPIDocumentEmbedder(
    api_type="serverless_inference_api",
    api_params={"model": "BAAI/bge-small-en-v1.5"})
    document_writer = DocumentWriter(document_store,policy=DuplicatePolicy.SKIP)


    preprocessing_pipeline = Pipeline()
    preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
    preprocessing_pipeline.add_component(instance=text_file_converter, name="text_file_converter")
    preprocessing_pipeline.add_component(instance=docx_converter, name="docx_converter")
    preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
    preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
    preprocessing_pipeline.add_component(instance=text_embedder, name="text_embedder")
    preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

    preprocessing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
    preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
    preprocessing_pipeline.connect("file_type_router." + DOCX_MIME, "docx_converter.sources")
    preprocessing_pipeline.connect("text_file_converter", "document_cleaner")
    preprocessing_pipeline.connect("pypdf_converter", "document_cleaner")
    preprocessing_pipeline.connect("docx_converter", "document_cleaner")
    preprocessing_pipeline.connect("document_cleaner", "document_splitter")
    preprocessing_pipeline.connect("document_splitter", "text_embedder")
    preprocessing_pipeline.connect("text_embedder", "document_writer")

    return preprocessing_pipeline