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

    def __init__(self, pages: List[Page]):
        """
        Initialize Document with a list of Page objects.

        Args:
            pages: List of Page objects
        """
        self.pages = pages

    def __len__(self):
        """Return the number of pages in the document."""
        return len(self.pages)

    def __getitem__(self, index):
        """Allow indexing into the pages list."""
        return self.pages[index]

    def __iter__(self):
        """Allow iteration over pages."""
        return iter(self.pages)

    def __repr__(self):
        return f"Document(pages={len(self.pages)})"

    def __str__(self):
        return self.to_markdown()

    # Properties for aggregated content
    @property
    def figures(self):
        """Get all figures from all pages."""
        all_figures = []
        for page_num, page in enumerate(self.pages, 1):
            for figure in page.figures:
                # Add page number information
                figure_dict = figure.to_dict()
                figure_dict["page_number"] = page_num
                all_figures.append(figure_dict)
        return all_figures

    @property
    def tables(self):
        """Get all tables from all pages."""
        all_tables = []
        for page_num, page in enumerate(self.pages, 1):
            for table in page.tables:
                # Add page number information
                table_dict = table.to_dict()
                table_dict["page_number"] = page_num
                all_tables.append(table_dict)
        return all_tables

    @property
    def formulas(self):
        """Get all formulas from all pages."""
        all_formulas = []
        for page_num, page in enumerate(self.pages, 1):
            for formula in page.formulas:
                # Add page number information
                formula_dict = formula.to_dict()
                formula_dict["page_number"] = page_num
                all_formulas.append(formula_dict)
        return all_formulas

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
        if 1 <= page_number <= len(self.pages):
            return self.pages[page_number - 1]
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
    def to_markdown(self, filepath: Union[str, Path] = None) -> str:
        """
        Convert all pages to markdown format.

        Args:
            filepath: Optional path to save markdown file

        Returns:
            Combined markdown string from all pages
        """
        markdown_content = []

        for page_num, page in enumerate(self.pages, 1):
            markdown_content.append(f"# Page {page_num}\n")
            page_md = page.to_markdown()
            if page_md.strip():
                markdown_content.append(page_md)
            markdown_content.append("\n")

        combined_markdown = "\n".join(markdown_content)

        if filepath:
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
            "total_pages": len(self.pages),
            "pages": [page.to_dict() for page in self.pages],
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
        if model is None:
            model = llm_processing.MODELS[2]

        # Extract page images from PDF
        page_images = cls._extract_pages_as_images(pdf_path, dpi=dpi, verbose=verbose)

        # Create Page objects for each image
        pages = []
        for i, page_image in enumerate(page_images, 1):
            if verbose:
                print(f"Processing page {i}/{len(page_images)}")

            try:
                page = Page.from_image(
                    image=page_image,
                    model_weights=model_weights,
                    model=model,
                    generate_config=generate_config,
                )
                pages.append(page)

                if verbose:
                    print(f"✓ Completed page {i}")

            except Exception as e:
                logger.error(f"Error processing page {i}: {e}")
                if verbose:
                    print(f"✗ Failed to process page {i}: {e}")
                # Continue with other pages even if one fails
                continue

        if verbose:
            print(f"✓ Document processing complete: {len(pages)} pages processed")

        return cls(pages)

    @classmethod
    async def from_pdf_async(
        cls,
        pdf_path: Union[str, Path],
        dpi: int = 300,
        model_weights: Union[Path, str] = "doclayout_yolo_docstructbench_imgsz1024.pt",
        model=None,
        generate_config: Dict = None,
        max_concurrent: int = 5,
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
            max_concurrent: Maximum number of concurrent page processing
            verbose: Whether to print progress information

        Returns:
            Document object with all pages processed
        """
        if model is None:
            model = llm_processing.MODELS[2]

        # Extract page images from PDF
        page_images = cls._extract_pages_as_images(pdf_path, dpi=dpi, verbose=verbose)

        if verbose:
            print(
                f"Processing {len(page_images)} pages asynchronously (max concurrent: {max_concurrent})..."
            )

        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single_page(page_num: int, page_image: Image.Image):
            async with semaphore:
                if verbose:
                    print(f"Processing page {page_num}/{len(page_images)}")

                try:
                    page = await Page.from_image_async(
                        image=page_image,
                        model_weights=model_weights,
                        model=model,
                        generate_config=generate_config,
                    )

                    if verbose:
                        print(f"✓ Completed page {page_num}")

                    return page_num, page

                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    if verbose:
                        print(f"✗ Failed to process page {page_num}: {e}")
                    return page_num, None

        # Create tasks for all pages
        tasks = [
            process_single_page(i, page_image)
            for i, page_image in enumerate(page_images, 1)
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Sort results by page number and filter out failed pages
        pages = []
        for page_num, page in sorted(results):
            if page is not None:
                pages.append(page)

        if verbose:
            print(
                f"✓ Async document processing complete: {len(pages)}/{len(page_images)} pages processed successfully"
            )

        return cls(pages)

    @classmethod
    def from_pdf_with_executor(
        cls,
        pdf_path: Union[str, Path],
        dpi: int = 300,
        model_weights: Union[Path, str] = "doclayout_yolo_docstructbench_imgsz1024.pt",
        model=None,
        generate_config: Dict = None,
        max_concurrent: int = 5,
        verbose: bool = True,
    ):
        """
        Create a Document from a PDF file using async processing in a sync context.

        This method is useful when you want async performance but are calling
        from a synchronous context (like Jupyter notebooks).

        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for PDF to image conversion
            model_weights: Path to YOLO model weights
            model: LLM model for content parsing
            generate_config: Configuration for content generation
            max_concurrent: Maximum number of concurrent page processing
            verbose: Whether to print progress information

        Returns:
            Document object with all pages processed
        """
        try:
            # Check if we're in an existing event loop
            asyncio.get_running_loop()
            # If we are, use ThreadPoolExecutor to run async code
            with ThreadPoolExecutor(1) as executor:
                future = executor.submit(
                    lambda: asyncio.run(
                        cls.from_pdf_async(
                            pdf_path=pdf_path,
                            dpi=dpi,
                            model_weights=model_weights,
                            model=model,
                            generate_config=generate_config,
                            max_concurrent=max_concurrent,
                            verbose=verbose,
                        )
                    )
                )
                return future.result()
        except RuntimeError:
            # No existing event loop, we can use asyncio.run directly
            return asyncio.run(
                cls.from_pdf_async(
                    pdf_path=pdf_path,
                    dpi=dpi,
                    model_weights=model_weights,
                    model=model,
                    generate_config=generate_config,
                    max_concurrent=max_concurrent,
                    verbose=verbose,
                )
            )
