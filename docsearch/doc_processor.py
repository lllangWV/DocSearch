import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from pdf2image import convert_from_path
from PIL import Image

from docsearch.figure_extraction import DocumentPageAnalyzer

logger = logging.getLogger(__name__)


class DocProcessor:
    """
    A class to process PDF documents page by page using DocumentPageAnalyzer.

    This class extracts PDF pages as images, saves them in organized directories,
    and performs document analysis using the DocumentPageAnalyzer class with
    options for both synchronous and asynchronous processing.
    """

    def __init__(
        self,
        model_weights: Optional[Path] = None,
        dpi: int = 300,
        confidence_threshold: float = 0.2,
        device: str = "cpu",
        verbose: bool = True,
    ):
        """
        Initialize DocProcessor.

        Args:
            model_weights: Path to YOLO model weights for DocumentPageAnalyzer
            dpi: Resolution for PDF to image conversion
            confidence_threshold: Confidence threshold for element detection
            device: Device to use for inference ('cpu' or 'cuda')
            verbose: Whether to print progress information
        """
        self.model_weights = model_weights
        self.dpi = dpi
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.verbose = verbose

        # Will store the page analyzers for each processed page
        self.page_analyzers: Dict[int, DocumentPageAnalyzer] = {}
        self.pdf_path: Optional[Path] = None
        self.output_dir: Optional[Path] = None

    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        async_mode: bool = False,
        max_concurrent: int = 5,
        extract_elements: bool = True,
        parse_content: bool = True,
    ) -> Dict[str, any]:
        """
        Process a PDF document page by page.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted pages and analysis results.
                       If None, uses the same directory as the PDF with PDF filename as folder name.
            async_mode: Whether to run analysis asynchronously
            max_concurrent: Maximum number of concurrent operations for async mode
            extract_elements: Whether to extract elements from pages
            parse_content: Whether to parse content using LLM

        Returns:
            Dictionary containing processing results and metadata
        """
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        # Set up output directory
        if output_dir is None:
            self.output_dir = self.pdf_path.parent / self.pdf_path.stem
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"Processing PDF: {self.pdf_path}")
            print(f"Output directory: {self.output_dir}")

        # Extract pages as images
        page_images = self._extract_pages_as_images()

        # Save page images in organized directories
        page_paths = self._save_page_images(page_images)

        # Process pages with DocumentPageAnalyzer
        if async_mode:
            results = asyncio.run(
                self._process_pages_async(
                    page_paths, max_concurrent, extract_elements, parse_content
                )
            )
        else:
            results = self._process_pages_sync(
                page_paths, extract_elements, parse_content
            )

        # Create summary
        summary = self._create_processing_summary(results)

        # Save summary to file
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        if self.verbose:
            print(f"Processing complete. Summary saved to: {summary_path}")

        return {
            "pdf_path": self.pdf_path,
            "output_dir": self.output_dir,
            "page_count": len(page_images),
            "page_analyzers": self.page_analyzers,
            "summary": summary,
            "page_paths": page_paths,
        }

    def _extract_pages_as_images(self) -> List[Image.Image]:
        """Extract all pages from PDF as PIL Images."""
        if self.verbose:
            print(f"Extracting pages from PDF at {self.dpi} DPI...")

        try:
            images = convert_from_path(self.pdf_path, dpi=self.dpi)
            if self.verbose:
                print(f"Extracted {len(images)} pages")
            return images
        except Exception as e:
            logger.error(f"Error extracting pages from PDF: {e}")
            raise

    def _save_page_images(self, page_images: List[Image.Image]) -> List[Path]:
        """
        Save page images in organized directory structure.

        Structure: output_dir/pages/page_N/page_N.png

        Returns:
            List of paths to saved page images
        """
        if self.verbose:
            print("Saving page images...")

        pages_dir = self.output_dir / "pages"
        pages_dir.mkdir(exist_ok=True)

        page_paths = []

        for i, image in enumerate(page_images, 1):
            # Create directory for this page
            page_dir = pages_dir / f"page_{i}"
            page_dir.mkdir(exist_ok=True)

            # Save the page image
            page_path = page_dir / f"page_{i}.png"
            image.save(page_path, "PNG")
            page_paths.append(page_path)

            if self.verbose:
                print(f"Saved page {i} to: {page_path}")

        return page_paths

    def _process_pages_sync(
        self,
        page_images: List[Image.Image],
        extract_elements: bool,
        parse_content: bool,
    ) -> Dict[int, Dict]:
        """Process pages synchronously using DocumentPageAnalyzer."""
        if self.verbose:
            print("Processing pages synchronously...")

        results = {}

        for i, page_image in enumerate(page_images, 1):
            if self.verbose:
                print(f"Processing page {i}/{len(page_images)}")

            try:
                # Create DocumentPageAnalyzer for this page
                analyzer = DocumentPageAnalyzer(
                    image=page_image,
                    model_weights=self.model_weights,
                    extract_elements=extract_elements,
                    confidence_threshold=self.confidence_threshold,
                    device=self.device,
                )

                self.page_analyzers[i] = analyzer

                # Parse content if requested
                if parse_content and extract_elements:
                    analyzer.parse_all_tables()
                    analyzer.parse_all_figures()
                    analyzer.parse_all_formulas()
                    analyzer.parse_all_text()
                    analyzer.parse_text()

                # Save analysis results
                page_dir = self.output_dir / f"page_{i}"
                page_path = page_dir / f"page_{i}.png"
                analyzer.save_elements(page_dir)

                results[i] = {
                    "page_path": page_path,
                    "page_dir": page_dir,
                    "extraction_summary": analyzer.extraction_summary,
                    "elements_found": {
                        "tables": len(analyzer.tables),
                        "figures": len(analyzer.figures),
                        "formulas": len(analyzer.formulas),
                    },
                    "status": "success",
                    "error": None,
                }

            except Exception as e:
                logger.error(f"Error processing page {i}: {e}")
                results[i] = {
                    "page_path": page_path,
                    "page_dir": page_path.parent,
                    "extraction_summary": {},
                    "elements_found": {},
                    "status": "error",
                    "error": str(e),
                }

        return results

    async def _process_pages_async(
        self,
        page_images: List[Image.Image],
        max_concurrent: int,
        extract_elements: bool,
        parse_content: bool,
    ) -> Dict[int, Dict]:
        """Process pages asynchronously using DocumentPageAnalyzer."""
        if self.verbose:
            print(
                f"Processing pages asynchronously (max concurrent: {max_concurrent})..."
            )

        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}

        async def process_single_page(page_num: int, page_path: Path):
            async with semaphore:
                if self.verbose:
                    print(f"Processing page {page_num}/{len(page_images)}")

                try:
                    # Create DocumentPageAnalyzer for this page
                    analyzer = DocumentPageAnalyzer(
                        image=page_images[page_num - 1],
                        model_weights=self.model_weights,
                        extract_elements=extract_elements,
                        confidence_threshold=self.confidence_threshold,
                        device=self.device,
                    )

                    self.page_analyzers[page_num] = analyzer

                    # Parse content if requested
                    if parse_content and extract_elements:
                        await analyzer.parse_all_elements_async(max_concurrent=3)

                    # Save analysis results
                    page_dir = self.output_dir / f"page_{page_num}"
                    analyzer.save_elements(page_dir)

                    results[page_num] = {
                        "page_path": page_path,
                        "page_dir": page_dir,
                        "extraction_summary": analyzer.extraction_summary,
                        "elements_found": {
                            "tables": len(analyzer.tables),
                            "figures": len(analyzer.figures),
                            "formulas": len(analyzer.formulas),
                        },
                        "status": "success",
                        "error": None,
                    }

                    if self.verbose:
                        print(f"✓ Completed page {page_num}")

                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    results[page_num] = {
                        "page_path": page_path,
                        "page_dir": page_path.parent,
                        "extraction_summary": {},
                        "elements_found": {},
                        "status": "error",
                        "error": str(e),
                    }

        # Create tasks for all pages
        tasks = [
            process_single_page(i, page_image)
            for i, page_image in enumerate(page_images, 1)
        ]

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)

        if self.verbose:
            print(f"✓ Completed async processing of {len(page_images)} pages")

        return results

    def _create_processing_summary(self, results: Dict[int, Dict]) -> Dict:
        """Create a summary of the processing results."""
        total_pages = len(results)
        successful_pages = sum(1 for r in results.values() if r["status"] == "success")
        failed_pages = total_pages - successful_pages

        total_elements = {
            "tables": 0,
            "figures": 0,
            "formulas": 0,
        }

        for result in results.values():
            if result["status"] == "success":
                elements = result.get("elements_found", {})
                for element_type in total_elements:
                    total_elements[element_type] += elements.get(element_type, 0)

        summary = {
            "pdf_path": str(self.pdf_path),
            "output_dir": str(self.output_dir),
            "total_pages": total_pages,
            "successful_pages": successful_pages,
            "failed_pages": failed_pages,
            "success_rate": successful_pages / total_pages if total_pages > 0 else 0,
            "total_elements_found": total_elements,
            "processing_details": results,
        }

        return summary

    def get_page_analyzer(self, page_number: int) -> Optional[DocumentPageAnalyzer]:
        """Get the DocumentPageAnalyzer for a specific page."""
        return self.page_analyzers.get(page_number)

    def get_all_tables(self) -> List[Dict]:
        """Get all tables from all processed pages."""
        all_tables = []
        for page_num, analyzer in self.page_analyzers.items():
            tables = analyzer.tables
            for table in tables:
                table["page_number"] = page_num
                all_tables.append(table)
        return all_tables

    def get_all_figures(self) -> List[Dict]:
        """Get all figures from all processed pages."""
        all_figures = []
        for page_num, analyzer in self.page_analyzers.items():
            figures = analyzer.figures
            for figure in figures:
                figure["page_number"] = page_num
                all_figures.append(figure)
        return all_figures

    def get_all_formulas(self) -> List[Dict]:
        """Get all formulas from all processed pages."""
        all_formulas = []
        for page_num, analyzer in self.page_analyzers.items():
            formulas = analyzer.formulas
            for formula in formulas:
                formula["page_number"] = page_num
                all_formulas.append(formula)
        return all_formulas

    def export_markdown_summary(
        self, output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Export a markdown summary of all extracted content.

        Args:
            output_path: Path to save the markdown file. If None, saves to output_dir.

        Returns:
            Path to the saved markdown file
        """
        if output_path is None:
            output_path = self.output_dir / f"{self.pdf_path.stem}_summary.md"
        else:
            output_path = Path(output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Document Analysis Summary: {self.pdf_path.name}\n\n")

            # Write summary statistics
            if hasattr(self, "page_analyzers") and self.page_analyzers:
                total_tables = len(self.get_all_tables())
                total_figures = len(self.get_all_figures())
                total_formulas = len(self.get_all_formulas())

                f.write("## Summary Statistics\n\n")
                f.write(f"- **Total Pages**: {len(self.page_analyzers)}\n")
                f.write(f"- **Total Tables**: {total_tables}\n")
                f.write(f"- **Total Figures**: {total_figures}\n")
                f.write(f"- **Total Formulas**: {total_formulas}\n\n")

            # Write content by page
            for page_num in sorted(self.page_analyzers.keys()):
                analyzer = self.page_analyzers[page_num]
                f.write(f"## Page {page_num}\n\n")

                # Page text if available
                if analyzer.parsed_text and analyzer.parsed_text.get("md"):
                    f.write("### Page Text\n\n")
                    f.write(analyzer.parsed_text["md"])
                    f.write("\n\n")

                # Tables
                if analyzer.tables:
                    f.write("### Tables\n\n")
                    for i, table in enumerate(analyzer.tables):
                        f.write(f"#### Table {i+1}\n\n")
                        if table.get("md"):
                            f.write(table["md"])
                        f.write("\n\n")

                # Figures
                if analyzer.figures:
                    f.write("### Figures\n\n")
                    for i, figure in enumerate(analyzer.figures):
                        f.write(f"#### Figure {i+1}\n\n")
                        if figure.get("md"):
                            f.write(figure["md"])
                        f.write("\n\n")

                # Formulas
                if analyzer.formulas:
                    f.write("### Formulas\n\n")
                    for i, formula in enumerate(analyzer.formulas):
                        f.write(f"#### Formula {i+1}\n\n")
                        if formula.get("md"):
                            f.write(formula["md"])
                        f.write("\n\n")

        if self.verbose:
            print(f"Markdown summary exported to: {output_path}")

        return output_path
