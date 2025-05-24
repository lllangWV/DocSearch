import copy
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from enum import Enum
from glob import glob
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import PyPDF2
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

from docsearch.image_processing import encode_image, extract_text_from_image
from docsearch.utils.log_utils import set_verbose_level

logger = logging.getLogger(__name__)

load_dotenv()


class PDFExtractionMethods(Enum):
    """Enumeration of available PDF extraction methods."""

    TEXT_THEN_LLM = "text_then_llm"
    LLM = "llm"

    @classmethod
    def list_modes(cls) -> str:
        """
        List all available extraction modes.

        Returns
        -------
        str
            Comma-separated string of available extraction modes.
        """
        return ", ".join([mode.value for mode in cls])


@dataclass
class PDFMetadata:
    """
    Schema for the PDF metadata.

    Attributes
    ----------
    pdf_name : str
        Name of the PDF file without extension
    pdf_rel_path : str
        Relative path to the PDF file
    num_pages : int
        Total number of pages in the PDF
    image_prompt : str
        Prompt used for LLM image processing
    """

    pdf_name: str
    pdf_rel_path: str
    num_pages: int
    error_reading_pdf: bool = False
    image_prompt: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PDFPageMetadata:
    """
    Defines the PDF page metadata.

    Attributes
    ----------
    page_number : int
        Page number (1-indexed)
    process_as_image : bool
        Whether this page was processed using LLM image analysis
    has_images : bool
        Whether the page contains embedded images
    has_text : bool
        Whether the page has extractable text
    """

    page_num: int
    processed_as_image: bool = False
    has_images: bool = False
    has_text: bool = True
    error_extracting_text: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PDFPage:
    """
    Defines the PDF page data.

    Attributes
    ----------
    text : str
        Extracted text content from the page
    metadata : PDFPageMetadata
        Page-specific metadata
    image : Optional[bytes]
        Page image as bytes (PNG format)
    """

    text: str
    metadata: PDFPageMetadata
    image: Optional[bytes] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PDF:
    """
    Defines the PDF data.

    Attributes
    ----------
    metadata : PDFMetadata
        PDF-level metadata
    pages : List[PDFPage]
        List of page objects containing text and metadata
    """

    metadata: PDFMetadata
    pages: List[PDFPage]

    def to_dict(self) -> Dict:
        return asdict(self)


class PDFProcessor:
    """
    A class for processing PDF files and extracting text content.

    This class provides functionality to extract text from PDF files using
    different methods including LLM-based extraction and traditional text
    extraction with LLM fallback. Processing and exporting are separated
    to allow flexible output options. Uses dataclasses for structured data.

    Parameters
    ----------
    model : str, optional
        LLM model to use for text extraction, by default "gpt-4o-mini"
    max_tokens : int, optional
        Maximum number of tokens for LLM processing, by default 1000
    verbose : int, optional
        Verbosity level, by default 1
    """

    def __init__(
        self, model: str = "gpt-4o-mini", max_tokens: int = 1000, verbose: int = 1
    ) -> None:
        set_verbose_level(verbose)
        self.verbose = verbose

        self.model = model
        self.max_tokens = max_tokens
        self.pdf_data: Optional[PDF] = None

        if self.verbose == 1:
            print(f"verbose level: {self.verbose}")
            print(f"Extraction methods:")
            print(
                f"'llm': Process all pages using LLM. This means extract information by processing the pages as images and then using llm for text extraction."
            )
            print(
                f"'text_then_llm': Process all pages using text extraction first, then LLM for problematic pages."
            )

    def process(self, path: Union[str, Path], method: str = "llm") -> PDF:
        """
        Process a PDF file using the specified extraction method.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the PDF file to process
        method : str, optional
            Extraction method to use ('llm' or 'text_then_llm'), by default "llm"

        Returns
        -------
        PDF
            PDF dataclass containing extracted text and metadata

        Raises
        ------
        ValueError
            If an invalid extraction method is provided
        """
        extraction_modes = [mode.value for mode in PDFExtractionMethods]
        if method not in extraction_modes:
            raise ValueError(
                f"Invalid extraction method: {method}. Valid methods are: {extraction_modes}"
            )

        self.method = method

        path = Path(path)

        logger.info(f"Processing PDF: {path}")

        # Extract images from PDF
        page_images = self.extract_images(path)

        with open(path, "rb") as pdf_stream:
            self._pdf_stream = pdf_stream

            pdf_pages = []
            for i_page, page_image in enumerate(page_images):
                pdf_page = self._process_page(i_page, page_image)
                pdf_pages.append(pdf_page)

            pdf_metadata = PDFMetadata(
                pdf_name=path.stem,
                pdf_rel_path=str(path),
                num_pages=len(pdf_pages),
                error_reading_pdf=self.error_reading_pdf,
            )
            self.pdf_data = PDF(metadata=pdf_metadata, pages=pdf_pages)

        return self.pdf_data

    def _process_page(self, i_page: int, page_image: Image.Image) -> None:
        """
        Process a page image and extract text.

        Parameters
        ----------
        i_page : int
            Page number (1-indexed)
        page_image : Image.Image
            Page image as PIL Image object

        Returns
        -------
        PDFPage
            PDF page dataclass containing text and metadata
        """
        self.error_reading_pdf = False
        try:
            pdf_reader = PyPDF2.PdfReader(self._pdf_stream)

            page = pdf_reader.pages[i_page]

            page_text = page.extract_text()

            # Check for images in the page
            has_images = page.images is not None and len(page.images) > 0

            # Check for presentation-style pages with XObjects
            resources = page["/Resources"]
            is_a_pa_attachment = False
            if "/XObject" in resources:
                xobjects = page["/Resources"]["/XObject"]
                is_a_pa_attachment = True

            process_as_image = False
            if is_a_pa_attachment or has_images or len(page_text) == 0:
                process_as_image = True

        except Exception as e:
            logger.error(f"Error processing page: {e}")
            process_as_image = True
            self.error_reading_pdf = True

        if self.method == PDFExtractionMethods.LLM.value:
            image_prompt, page_text, error_extracting_text = self._process_page_llm(
                page_image
            )

        elif self.method == PDFExtractionMethods.TEXT_THEN_LLM.value:
            page_text = page.extract_text()
            if self.error_reading_pdf or process_as_image:
                image_prompt, llm_page_text, error_extracting_text = (
                    self._process_page_llm(page_image)
                )
                page_text += llm_page_text
        else:
            raise ValueError(f"Invalid extraction method: {self.method}")

        has_text = len(page_text) > 0

        page_metadata = PDFPageMetadata(
            page_num=i_page,
            processed_as_image=process_as_image,
            has_images=has_images,
            has_text=has_text,
            error_extracting_text=error_extracting_text,
        )

        page = PDFPage(
            text=page_text,
            image=page_image,
            metadata=page_metadata,
        )

        return page

    def _process_page_llm(self, page_image: Image.Image) -> None:
        prompt_and_response = None
        image_prompt = ""
        page_text = ""
        error_extracting_text = False
        try:
            prompt_and_response = extract_text_from_image(
                page_image,
                model=self.model,
                max_tokens=self.max_tokens,
                image_type="png",
            )
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            error_extracting_text = True

        if prompt_and_response is not None:
            image_prompt = prompt_and_response[0]
            page_text = prompt_and_response[1]

        return image_prompt, page_text, error_extracting_text

    def extract_images(
        self,
        path: Union[str, Path],
        dpi: int = 300,
        out_dir: Optional[Union[str, Path]] = None,
    ) -> List[PDFPage]:
        """
        Extract images from PDF pages.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the PDF file
        dpi : int, optional
            Resolution for image conversion, by default 300
        out_dir : Optional[Union[str, Path]], optional
            Directory to save the images, by default None

        Returns
        -------
        List[Image.Image]
            List of PIL Image objects, one per page
        """
        path = Path(path)
        logger.info(f"Extracting images from PDF: {path}")

        images = convert_from_path(path, dpi=dpi)
        image_bytes_list = []

        for image in images:
            image_bytes_list.append(encode_image(image, format="PNG"))

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for i_page, image in enumerate(images):
                image.save(out_dir / f"page_{i_page+1}.png", "PNG")
        return image_bytes_list

    def to_json(self, filepath: Union[str, Path]) -> None:
        """
        Convert the PDF data to a JSON file.
        """
        filepath = Path(filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.pdf_data.to_dict(), f, indent=2)

    @staticmethod
    def fix_filename(filename: str) -> str:
        """
        Clean and fix problematic characters in filenames.

        Parameters
        ----------
        filename : str
            Original filename to clean

        Returns
        -------
        str
            Cleaned filename suitable for filesystem use
        """
        # Removing bad characters
        n_dot = filename.count(".")
        if n_dot < 2:
            filename = filename.split(".")[0]
        else:
            filename = filename.replace(".", "", 1).split(".")[0]

        filename = filename.replace("%", "_").replace("(", "").replace(")", "").strip()
        return filename
