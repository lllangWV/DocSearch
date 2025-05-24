import logging
import os
from glob import glob
from pathlib import Path
from typing import List, Optional, Union

from docsearch.pdf_processing import PDFProcessor
from docsearch.vector_store import VectorStore, create_document_from_pdf_directory

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
        self.raw_dir = self.base_path / "raw"
        self.interim_dir = self.base_path / "interim"
        self.vector_store_dir = self.base_path / "vector_stores" / "default"
        self.output_dir = self.base_path / "output"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(model=self.llm, max_tokens=self.max_tokens)

        # Initialize vector store
        self.vector_store = VectorStore(
            index_store_dir=str(self.vector_store_dir),
            embed_model=self.embed_model,
            llm=self.llm,
        )

        self.engine = None

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
        for pdf_path in pdf_paths:
            if not pdf_path.exists():
                logger.warning(f"Warning: PDF file not found: {pdf_path}")
                continue

            logger.info(f"Processing PDF: {pdf_path}")

            # Create output directory for this PDF
            pdf_output_dir = self.interim_dir / pdf_path.stem

            # Use export_full to process and save both JSON and images
            pdf_data, image_paths = self.pdf_processor.export_full(
                pdf_path=pdf_path,
                output_dir=pdf_output_dir,
                method=extraction_method,
                include_images=True,
            )

        # Auto-load processed PDFs if requested
        if auto_load:
            self.load_processed_pdfs()

    def load_processed_pdfs(self) -> None:
        """
        Load all processed PDFs from interim directory into vector store.
        """
        logger.info("Loading processed PDFs into vector store")

        # Get all directories in interim_dir
        if self.interim_dir.exists():
            pdf_dirs = [p for p in self.interim_dir.iterdir() if p.is_dir()]
        else:
            pdf_dirs = []

        for pdf_dir in pdf_dirs:
            try:
                docs = create_document_from_pdf_directory(pdf_dir=str(pdf_dir))
                self.vector_store.load_docs(docs=docs)
                logger.info(f"Loaded documents from: {pdf_dir.name}")
            except Exception as e:
                logger.error(f"Error loading documents from {pdf_dir}: {e}")

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

    def add_pdfs_from_directory(
        self,
        directory_path: Union[str, Path],
        pattern: str = "*.pdf",
        extraction_method: str = "llm",
        auto_load: bool = True,
    ) -> None:
        """
        Add all PDFs from a directory.

        Args:
            directory_path: Directory containing PDF files
            pattern: File pattern to match (default: "*.pdf")
            extraction_method: Method to use for PDF processing
            auto_load: Whether to automatically load processed PDFs into vector store
        """
        directory_path = Path(directory_path)

        if not directory_path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return

        pdf_files = list(directory_path.glob(pattern))
        if not pdf_files:
            logger.error(
                f"No PDF files found in {directory_path} matching pattern {pattern}"
            )
            return

        logger.info(f"Found {len(pdf_files)} PDF files to process")
        self.add_pdfs(
            pdf_files, extraction_method=extraction_method, auto_load=auto_load
        )

    def get_stats(self) -> dict:
        """
        Get statistics about the current DocSearch instance.

        Returns:
            Dictionary containing statistics
        """
        # Count processed PDFs
        if self.interim_dir.exists():
            interim_dirs = [p for p in self.interim_dir.iterdir() if p.is_dir()]
            processed_pdfs = len(interim_dirs)
        else:
            processed_pdfs = 0

        # Count raw PDFs
        if self.raw_dir.exists():
            raw_pdfs = len(list(self.raw_dir.glob("*.pdf")))
        else:
            raw_pdfs = 0

        # Count output runs
        output_runs = 0
        if self.output_dir.exists():
            output_runs = len(list(self.output_dir.glob("run_*")))

        # Check if vector store exists
        vector_store_exists = self.vector_store.exists()

        return {
            "base_path": str(self.base_path),
            "raw_pdfs": raw_pdfs,
            "processed_pdfs": processed_pdfs,
            "vector_store_exists": vector_store_exists,
            "output_runs": output_runs,
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
