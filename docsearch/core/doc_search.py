import logging
import os
from glob import glob
from pathlib import Path
from typing import List, Optional, Union

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from llama_index import core as llama_core

from docsearch.core import Document
from docsearch.core.vector_store import VectorStore, create_document_from_pdf_directory

logger = logging.getLogger(__name__)


class DocSearch:
    """
    Main interface for the DocSearch package.

    This class provides a simple interface to add PDFs, process them,
    and query the resulting vector database.
    """

    def __init__(
        self,
        base_path: Union[str, Path] = "data",
        embed_model: str = "text-embedding-3-small",
        llm: str = "gpt-4o-mini",
        max_tokens: int = 3000,
    ):
        """
        Initialize DocSearch with the specified base directory.

        Args:
            base_path: Path to the main directory for storing data
            embed_model: Embedding model to use for vector storage
            llm: Language model to use for processing and querying
            max_tokens: Maximum tokens for LLM processing
        """
        self.base_path = Path(base_path)
        self.embed_model = embed_model
        self.llm = llm
        self.max_tokens = max_tokens

        # Set up directory structure
        self.vector_store_dir = self.base_path / "vector_stores" / "default"
        self.pages_path = self.base_path / "pages"
        self.output_dir = self.base_path / "output"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pages_path.mkdir(parents=True, exist_ok=True)

        # Initialize vector store
        self.vector_store = VectorStore(
            index_store_dir=str(self.vector_store_dir),
            embed_model=self.embed_model,
            llm=self.llm,
        )

        self.engine = None

    @property
    def page_indices(self) -> List[int]:
        """
        Get all page indices from the document pages path.
        """
        return [int(p.stem.split("_")[-1]) for p in self.pages_path.glob("*.parquet")]

    def add_pdfs(
        self,
        pdf_paths: Union[str, Path, List[Union[str, Path]]],
        extraction_method: str = "llm",
        auto_load: bool = True,
    ) -> None:
        """
        Add and process PDF files.

        Args:
            pdf_paths: Single PDF path or list of PDF paths to process
            extraction_method: Method to use for PDF processing ('llm' or 'text_then_llm')
            auto_load: Whether to automatically load processed PDFs into vector store
        """
        # Convert single path to list and ensure all are Path objects
        if isinstance(pdf_paths, (str, Path)):
            pdf_paths = [Path(pdf_paths)]
        else:
            pdf_paths = [Path(p) for p in pdf_paths]

        # Process each PDF
        page_indices = self.page_indices.copy()

        table_files = list(self.pages_path.glob("*.parquet"))
        if len(table_files) > 0:
            dataset = ds.dataset(self.pages_path, format="parquet")
            pages_table = dataset.to_table(columns=["pdf_id"])
            pdf_ids = pages_table["pdf_id"].to_pylist()
        else:
            pdf_ids = []

        for pdf_path in pdf_paths:
            if not pdf_path.exists():
                logger.warning(f"Warning: PDF file not found: {pdf_path}")
                continue

            logger.info(f"Processing PDF: {pdf_path}")
            # Get the next available page index
            next_page_index = max(page_indices) + 1 if page_indices else 0
            next_pdf_index = max(pdf_ids) + 1 if pdf_ids else 0
            page_indices.append(next_page_index)
            pdf_ids.append(next_pdf_index)

            out_filepath = self.pages_path / f"pages_{next_page_index}.parquet"
            # Use export_full to process and save both JSON and images
            document = Document.from_pdf(pdf_path, dpi=150, pdf_id=next_pdf_index)
            document.to_pyarrow(filepath=out_filepath)

        # Auto-load processed PDFs if requested
        if auto_load:
            self.load_processed_pdfs()

    def load_processed_pdfs(self) -> None:
        """
        Load all processed PDFs from interim directory into vector store.
        """
        logger.info("Loading processed PDFs into vector store")

        dataset = ds.dataset(self.pages_path, format="parquet")
        table = dataset.to_table(columns=["pdf_id", "page_id", "markdown", "pdf_path"])
        df = table.to_pandas()

        for index, row in df.iterrows():
            pdf_path = Path(row["pdf_path"])
            pdf_name = pdf_path.stem
            metadata = {
                "pdf_id": row["pdf_id"],
                "page_id": row["page_id"],
                "pdf_path": row["pdf_path"],
                "pdf_name": pdf_name,
            }

            doc = llama_core.Document(
                text=row["markdown"],
                metadata=metadata,
                id_=f"pdf_id-{row['pdf_id']}_page_id-{row['page_id']}",
            )
            self.vector_store.load_docs(docs=[doc])

    def query(
        self,
        query_text: str,
        engine_type: str = "citation_query",
        similarity_top_k: int = 20,
        save_response: bool = True,
        **engine_kwargs,
    ) -> object:
        """
        Query the vector database.

        Args:
            query_text: The query string
            engine_type: Type of engine to use ('query', 'citation_query', or 'retriever')
            similarity_top_k: Number of similar documents to retrieve
            save_response: Whether to save the response to output directory
            **engine_kwargs: Additional arguments for engine creation

        Returns:
            Query response object
        """
        # Create or update engine if needed
        if (
            self.engine is None
            or getattr(self, "_last_engine_type", None) != engine_type
        ):
            logger.info(f"Creating {engine_type} engine")

            # Set default citation engine parameters
            if engine_type == "citation_query":
                engine_kwargs.setdefault("citation_chunk_size", 2048)
                engine_kwargs.setdefault("citation_chunk_overlap", 0)

            self.engine = self.vector_store.create_engine(
                engine_type=engine_type,
                similarity_top_k=similarity_top_k,
                llm=self.llm,
                **engine_kwargs,
            )
            self._last_engine_type = engine_type

        logger.info("Executing query")
        response = self.engine.query(query_text)

        if save_response:
            self.vector_store.save_response(
                response, query_text, output_dir=str(self.output_dir)
            )
            logger.info(f"Response saved to: {self.output_dir}")

        return response

    def get_stats(self) -> dict:
        """
        Get statistics about the current DocSearch instance.

        Returns:
            Dictionary containing statistics
        """
        document_ids = []
        # Count raw PDFs
        if self.pages_path.exists():
            document_ids = ds.dataset(self.pages_path, format="parquet").to_table(
                columns=["pdf_id"]
            )
            total_pages = len(document_ids)
            total_pdfs = len(pc.unique(document_ids["pdf_id"].combine_chunks()))
        else:
            total_pages = 0
            total_pdfs = 0

        # Check if vector store exists
        vector_store_exists = self.vector_store.exists()

        return {
            "base_path": str(self.base_path),
            "total_pages": total_pages,
            "total_pdfs": total_pdfs,
            "vector_store_exists": vector_store_exists,
            "embed_model": self.embed_model,
            "llm": self.llm,
        }

    def reset(self, confirm: bool = False) -> None:
        """
        Reset the DocSearch instance by clearing all data.

        Args:
            confirm: Must be True to actually perform the reset
        """
        if not confirm:
            logger.info("Reset not performed. Set confirm=True to actually reset.")
            return

        import shutil

        # Remove directories if they exist
        for directory in [self.interim_dir, self.vector_store_dir, self.output_dir]:
            if directory.exists():
                shutil.rmtree(directory)
                logger.info(f"Removed: {directory}")

        # Recreate directories
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Reset engine
        self.engine = None

        # Reinitialize vector store
        self.vector_store = VectorStore(
            index_store_dir=str(self.vector_store_dir),
            embed_model=self.embed_model,
            llm=self.llm,
        )

        logger.info("DocSearch instance reset successfully")
