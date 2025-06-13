from docsearch._version import __version__
from docsearch.core.doc_search import DocSearch
from docsearch.core.document import Document, Page
from docsearch.core.vector_store import VectorStore
from docsearch.utils.log_utils import setup_logging

setup_logging()

__all__ = ["DocSearch", "DocProcessor", "PDFProcessor", "VectorStore", "__version__"]
