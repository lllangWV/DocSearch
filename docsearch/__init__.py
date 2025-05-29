from docsearch._version import __version__
from docsearch.doc_processor import DocProcessor
from docsearch.doc_search import DocSearch
from docsearch.pdf_processing import PDFProcessor
from docsearch.utils.log_utils import setup_logging
from docsearch.vector_store import VectorStore

setup_logging()

__all__ = ["DocSearch", "DocProcessor", "PDFProcessor", "VectorStore", "__version__"]
