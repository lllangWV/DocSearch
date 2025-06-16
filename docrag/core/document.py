import asyncio
import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq
from pdf2image import convert_from_path
from PIL import Image

from docrag.core import llm_processing
from docrag.core.page import Page

logger = logging.getLogger(__name__)


class Document:
    """
    A class representing a document containing multiple pages.

    This class manages a collection of Page objects and provides methods
    for processing PDF documents by extracting pages and analyzing content.
    """

    def __init__(
        self,
        pdf_path: Union[str, Path],
        pages: List[Page] = None,
        pdf_id: str = 0,
        **kwargs,
    ):
        """
        Initialize Document with a list of Page objects.

        Args:
            pdf_path: Path to the PDF file
            pages: List of Page objects
        """
        self._pdf_path = pdf_path
        self._pages = pages
        self._pdf_id = pdf_id

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

    def pages(self):
        return self._pages

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

    @staticmethod
    def pyarrow_struct():
        return pa.struct(
            [
                pa.field("pages", pa.list_(Page.pyarrow_struct())),
                pa.field("pdf_id", pa.str()),
            ]
        )

    def to_dict(
        self, include_images: bool = False, image_as_base64: bool = False, **kwargs
    ) -> Dict:
        """
        Convert document to dictionary format.

        Returns:
            Dictionary containing all document data
        """
        pages = []
        for page in self._pages:
            pages.append(
                page.to_dict(
                    include_images=include_images,
                    image_as_base64=image_as_base64,
                    **kwargs,
                )
            )
        return {
            "pdf_id": self._pdf_id,
            "pdf_path": self._pdf_path,
            "pages": pages,
        }

    def to_page_per_row(self):
        pdf_dict = self.to_dict(include_images=True, image_as_base64=True)
        pydict = {
            "pdf_id": [],
            "pdf_path": [],
            "page_id": [],
        }
        for page_dict in pdf_dict["pages"]:
            for key, value in page_dict.items():
                if key not in pydict:
                    pydict[key] = []
                pydict[key].append(value)
            pydict["pdf_id"].append(pdf_dict["pdf_id"])
            pydict["pdf_path"].append(str(pdf_dict["pdf_path"]))

        page_struct = Page.get_pyarrow_struct()
        pdf_struct = {
            "pdf_id": pa.int32(),
            "pdf_path": pa.string(),
        }
        for field in page_struct:
            pdf_struct[field.name] = field.type

        schema = pa.schema(pdf_struct)
        table = pa.Table.from_pydict(pydict, schema=schema)
        return table

    def to_pdf_per_row(self):
        pdf_dict = self.to_dict(include_images=True, image_as_base64=True)
        page_struct = Page.get_pyarrow_struct()

        pdf_struct = {
            "pdf_id": pa.int32(),
            "pdf_path": pa.string(),
            "pages": pa.list_(page_struct),
        }
        schema = pa.schema(pdf_struct)
        table = pa.Table.from_pylist([pdf_dict], schema=schema)
        return table

    def to_pyarrow(self, filepath: Union[str, Path] = None, page_per_row: bool = True):
        if page_per_row:
            table = self.to_page_per_row()
        else:
            table = self.to_pdf_per_row()

        if filepath:
            pq.write_table(table, filepath)

        return table

    def save(
        self,
        dirpath: Union[str, Path],
        save_pdf: bool = False,
        save_json: bool = False,
        **kwargs,
    ):
        """
        Save document to a file.
        """

        if dirpath:
            dirpath = Path(dirpath)
            dirpath.mkdir(parents=True, exist_ok=True)

        for page_num, page in enumerate(self._pages):
            page.full_save(dirpath / f"page_{page_num}", **kwargs)

        if save_pdf:
            pdf_path = Path(self._pdf_path)
            shutil.copy(pdf_path, dirpath / pdf_path.name)

        if save_json:
            self.to_json(dirpath / "document.json", **kwargs)

    def to_json(
        self,
        filepath: Union[str, Path] = None,
        indent: int = 2,
        encoding: str = "utf-8",
        **kwargs,
    ) -> str:
        """
        Convert document to JSON format.

        Args:
            filepath: Optional path to save JSON file
            indent: JSON indentation

        Returns:
            JSON string
        """
        data = self.to_dict(include_images=kwargs.get("include_images", False))

        if filepath:
            with open(filepath, "w", encoding=encoding) as f:
                json.dump(data, f, indent=indent)

        return json.dumps(data, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict):
        page_per_row = False
        if "page_id" in data:
            page_per_row = True

        if page_per_row:
            pages = []
            page_ids = data["page_id"]
            for page_id in page_ids:
                page_dict = {}
                for key, value in data.items():
                    if key in ["pdf_id", "pdf_path"]:
                        continue
                    page_dict[key] = value[page_id]
                pages.append(Page.from_dict(page_dict))
            return cls(
                pdf_path=data["pdf_path"][0],
                pages=pages,
                pdf_id=data["pdf_id"][0],
            )
        else:
            pages = [Page.from_dict(page) for page in data["pages"][0]]
            return cls(
                pdf_path=data["pdf_path"][0],
                pages=pages,
                pdf_id=data["pdf_id"][0],
            )

    @classmethod
    def from_pyarrow(cls, filepath: Union[str, Path]):
        table = pq.read_table(filepath)
        return cls.from_dict(table.to_pydict())

    @classmethod
    def from_pdf(
        cls,
        pdf_path: Union[str, Path],
        dpi: int = 300,
        model_weights: Union[Path, str] = "doclayout_yolo_docstructbench_imgsz1024.pt",
        model=None,
        generate_config: Dict = None,
        verbose: bool = True,
        pdf_id: int = 0,
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
        return cls(pdf_path=pdf_path, pages=pages, pdf_id=pdf_id)

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
        page_id: int = None,
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
                page_id=page_id,
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
                page_id=page_id,
            )
            for page_id, page_image in enumerate(page_images)
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
