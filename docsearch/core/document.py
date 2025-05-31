import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union

from pdf2image import convert_from_path
from PIL import Image

from docsearch import llm_processing
from docsearch.core.page import Page

logger = logging.getLogger(__name__)


class Document:
    """
    A class representing a document containing multiple pages.

    This class manages a collection of Page objects and provides methods
    for processing PDF documents by extracting pages and analyzing content.
    """

    def __init__(self, pdf_path: Union[str, Path], pages: List[Page] = None, **kwargs):
        """
        Initialize Document with a list of Page objects.

        Args:
            pdf_path: Path to the PDF file
            pages: List of Page objects
        """
        self._pdf_path = pdf_path
        self._pages = pages

        if pages is None:
            self._pages = Document._process_pages(
                pdf_path,
                dpi=kwargs.get("dpi", 300),
                verbose=kwargs.get("verbose", True),
                model_weights=kwargs.get(
                    "model_weights", "doclayout_yolo_docstructbench_imgsz1024.pt"
                ),
                model=kwargs.get("model", None),
                generate_config=kwargs.get("generate_config", None),
            )

    def __len__(self):
        """Return the number of pages in the document."""
        return len(self._pages)

    def __getitem__(self, index):
        """Allow indexing into the pages list."""
        return self._pages[index]

    def __iter__(self):
        """Allow iteration over pages."""
        return iter(self._pages)

    def __repr__(self):
        return f"Document(pages={len(self._pages)})"

    def __str__(self):
        return self.to_markdown()

    # Properties for aggregated content
    @property
    def figures(self):
        """Get all figures from all pages."""
        all_figures = []
        for page_num, page in enumerate(self._pages, 1):
            all_figures.extend(page.figures)
        return all_figures

    @property
    def tables(self):
        """Get all tables from all pages."""
        all_tables = []
        for page_num, page in enumerate(self._pages, 1):
            all_tables.extend(page.tables)
        return all_tables

    @property
    def formulas(self):
        """Get all formulas from all pages."""
        all_formulas = []
        for page_num, page in enumerate(self._pages, 1):
            all_formulas.extend(page.formulas)
        return all_formulas

    @property
    def elements(self):
        """Get all elements from all pages."""
        all_elements = []
        for page_num, page in enumerate(self._pages, 1):
            all_elements.extend(page.elements)
        return all_elements

    @property
    def text(self):
        """Get all text from all pages."""
        all_text = []
        for page_num, page in enumerate(self._pages, 1):
            all_text.extend(page.text)
        return all_text

    @property
    def titles(self):
        """Get all titles from all pages."""
        all_titles = []
        for page_num, page in enumerate(self._pages, 1):
            all_titles.extend(page.titles)
        return all_titles

    @property
    def markdown(self):
        """Get combined markdown from all pages."""
        return self.to_markdown()

    @property
    def md(self):
        """Get combined markdown from all pages."""
        return self.to_markdown()

    # Content aggregation methods
    def get_page(self, page_number: int) -> Optional[Page]:
        """
        Get a specific page by number (1-indexed).

        Args:
            page_number: Page number (1-indexed)

        Returns:
            Page object or None if page doesn't exist
        """
        if 1 <= page_number <= len(self._pages):
            return self._pages[page_number - 1]
        return None

    def get_figures_by_page(self, page_number: int) -> List:
        """Get all figures from a specific page."""
        page = self.get_page(page_number)
        return page.figures if page else []

    def get_tables_by_page(self, page_number: int) -> List:
        """Get all tables from a specific page."""
        page = self.get_page(page_number)
        return page.tables if page else []

    def get_formulas_by_page(self, page_number: int) -> List:
        """Get all formulas from a specific page."""
        page = self.get_page(page_number)
        return page.formulas if page else []

    # Output methods
    def to_markdown(
        self,
        filepath: Union[str, Path] = None,
        page_kwargs: Dict = None,
    ) -> str:
        """
        Convert all pages to markdown format.

        Args:
            filepath: Optional path to save markdown file

        Returns:
            Combined markdown string from all pages
        """
        markdown_content = []
        if page_kwargs is None:
            page_kwargs = {}

        for page_num, page in enumerate(self._pages, 1):
            markdown_content.append(f"# Page {page_num}\n")
            page_md = page.to_markdown(**page_kwargs)
            if page_md.strip():
                markdown_content.append(page_md)
            markdown_content.append("\n")

        combined_markdown = "\n".join(markdown_content)

        if filepath:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(combined_markdown)

        return combined_markdown

    def to_dict(self) -> Dict:
        """
        Convert document to dictionary format.

        Returns:
            Dictionary containing all document data
        """
        return {
            "total_pages": len(self._pages),
            "pages": [page.to_dict() for page in self._pages],
            "summary": {
                "total_figures": len(self.figures),
                "total_tables": len(self.tables),
                "total_formulas": len(self.formulas),
            },
        }

    def to_json(self, filepath: Union[str, Path] = None, indent: int = 2) -> str:
        """
        Convert document to JSON format.

        Args:
            filepath: Optional path to save JSON file
            indent: JSON indentation

        Returns:
            JSON string
        """
        data = self.to_dict()

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent)

        return json.dumps(data, indent=indent)

    @classmethod
    def from_pdf(
        cls,
        pdf_path: Union[str, Path],
        dpi: int = 300,
        model_weights: Union[Path, str] = "doclayout_yolo_docstructbench_imgsz1024.pt",
        model=None,
        generate_config: Dict = None,
        verbose: bool = True,
    ):
        """
        Create a Document from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for PDF to image conversion
            model_weights: Path to YOLO model weights
            model: LLM model for content parsing
            generate_config: Configuration for content generation
            verbose: Whether to print progress information

        Returns:
            Document object with all pages processed
        """
        pages = Document._process_pages(
            pdf_path=pdf_path,
            dpi=dpi,
            model_weights=model_weights,
            model=model,
            generate_config=generate_config,
            verbose=verbose,
        )
        return cls(pdf_path=pdf_path, pages=pages)

    @classmethod
    async def from_pdf_async(
        cls,
        pdf_path: Union[str, Path],
        dpi: int = 300,
        model_weights: Union[Path, str] = "doclayout_yolo_docstructbench_imgsz1024.pt",
        model=None,
        generate_config: Dict = None,
        verbose: bool = True,
    ):
        """
        Create a Document from a PDF file asynchronously.

        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for PDF to image conversion
            model_weights: Path to YOLO model weights
            model: LLM model for content parsing
            generate_config: Configuration for content generation
            verbose: Whether to print progress information

        Returns:
            Document object with all pages processed
        """
        pages = await Document._process_pages_async(
            pdf_path=pdf_path,
            dpi=dpi,
            model_weights=model_weights,
            model=model,
            generate_config=generate_config,
            verbose=verbose,
        )
        return cls(pdf_path=pdf_path, pages=pages)

    # PDF extraction methods (adapted from DocProcessor)
    @staticmethod
    def _extract_pages_as_images(
        pdf_path: Union[str, Path], dpi: int = 300, verbose: bool = True
    ) -> List[Image.Image]:
        """
        Extract all pages from PDF as PIL Images.

        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for PDF to image conversion
            verbose: Whether to print progress information

        Returns:
            List of PIL Image objects, one per page
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if verbose:
            print(f"Extracting pages from PDF at {dpi} DPI...")

        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            if verbose:
                print(f"Extracted {len(images)} pages")
            return images
        except Exception as e:
            logger.error(f"Error extracting pages from PDF: {e}")
            raise

    @staticmethod
    async def _process_single_page(
        page_image: Image.Image,
        model_weights: Union[Path, str] = "doclayout_yolo_docstructbench_imgsz1024.pt",
        model=None,
        generate_config: Dict = None,
    ):

        try:
            page = await Page.from_image_async(
                image=page_image,
                model_weights=model_weights,
                model=model,
                generate_config=generate_config,
            )

            return page

        except Exception as e:
            logger.error(f"Error processing page: {e}")
            return None

    @staticmethod
    async def _process_pages_async(
        pdf_path: Union[str, Path],
        dpi: int = 300,
        model_weights: Union[Path, str] = "doclayout_yolo_docstructbench_imgsz1024.pt",
        model=None,
        generate_config: Dict = None,
        verbose: bool = True,
    ):
        """
        Create a Document from a PDF file asynchronously.

        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for PDF to image conversion
            model_weights: Path to YOLO model weights
            model: LLM model for content parsing
            generate_config: Configuration for content generation
            verbose: Whether to print progress information

        Returns:
            Document object with all pages processed
        """
        if model is None:
            model = llm_processing.MODELS[2]

        # Extract page images from PDF
        page_images = Document._extract_pages_as_images(
            pdf_path, dpi=dpi, verbose=verbose
        )

        # Create tasks for all pages
        tasks = [
            Document._process_single_page(
                page_image,
                model_weights=model_weights,
                model=model,
                generate_config=generate_config,
            )
            for page_image in page_images
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Sort results by page number and filter out failed pages
        pages = []
        for page in results:
            if page is not None:
                pages.append(page)

        if verbose:
            print(
                f"✓ Async document processing complete: {len(pages)}/{len(page_images)} pages processed successfully"
            )

        return pages

    @staticmethod
    def _process_pages(
        pdf_path: Union[str, Path],
        dpi: int = 300,
        model_weights: Union[Path, str] = "doclayout_yolo_docstructbench_imgsz1024.pt",
        model=None,
        generate_config: Dict = None,
        verbose: bool = True,
    ):
        """
        Create a Document from a PDF file asynchronously.

        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for PDF to image conversion
            model_weights: Path to YOLO model weights
            model: LLM model for content parsing
            generate_config: Configuration for content generation
            verbose: Whether to print progress information

        Returns:
            Document object with all pages processed
        """
        try:
            asyncio.get_running_loop()  # Triggers RuntimeError if no running event loop
            # Create a separate thread so we can block before returning
            with ThreadPoolExecutor(1) as pool:
                pages = pool.submit(
                    lambda: asyncio.run(
                        Document._process_pages_async(
                            pdf_path,
                            dpi,
                            model_weights,
                            model,
                            generate_config,
                            verbose,
                        )
                    )
                ).result()
        except RuntimeError:
            pages = asyncio.run(
                Document._process_pages_async(
                    pdf_path,
                    dpi,
                    model_weights,
                    model,
                    generate_config,
                    verbose,
                )
            )

        return pages
