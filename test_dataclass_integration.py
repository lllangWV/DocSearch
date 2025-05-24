#!/usr/bin/env python3
"""
Test script to demonstrate the new dataclass-based PDFProcessor.
"""

import logging
from pathlib import Path

from docsearch.pdf_processing import PDF, PDFProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pdf_processor():
    """Test the dataclass-based PDFProcessor."""

    # Initialize processor
    processor = PDFProcessor(model="gpt-4o-mini", max_tokens=3000, verbose=1)

    # Test data directory
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)

    print("=== PDFProcessor Dataclass Test ===")
    print(f"Model: {processor.model}")
    print(f"Max tokens: {processor.max_tokens}")

    # Example with a dummy PDF path (this would normally be a real PDF)
    example_pdf = Path("example.pdf")

    if example_pdf.exists():
        print(f"\nProcessing: {example_pdf}")

        # Test 1: Process PDF and get dataclass
        pdf_data = processor.process(example_pdf, method="llm")

        print(f"Processed PDF: {pdf_data.metadata.pdf_name}")
        print(f"Number of pages: {pdf_data.metadata.num_pages}")
        print(
            f"First page text length: {len(pdf_data.pages[0].text) if pdf_data.pages else 0}"
        )
        print(
            f"First page has image: {pdf_data.pages[0].image is not None if pdf_data.pages else False}"
        )

        # Test 2: Save JSON
        json_path = test_dir / "test_output.json"
        processor.save_json(pdf_data, json_path)
        print(f"JSON saved to: {json_path}")

        # Test 3: Export images
        images_dir = test_dir / "images"
        image_paths = processor.export_images(pdf_data=pdf_data, output_dir=images_dir)
        print(f"Exported {len(image_paths)} images to: {images_dir}")

        # Test 4: Full export
        full_export_dir = test_dir / "full_export"
        pdf_data_full, image_paths_full = processor.export_full(
            example_pdf, full_export_dir, method="llm", include_images=True
        )
        print(f"Full export completed to: {full_export_dir}")

        print("\n=== Dataclass Structure ===")
        print(f"PDF Metadata type: {type(pdf_data.metadata)}")
        print(f"Pages type: {type(pdf_data.pages)}")
        if pdf_data.pages:
            print(f"First page type: {type(pdf_data.pages[0])}")
            print(f"First page metadata type: {type(pdf_data.pages[0].metadata)}")
            print(
                f"First page image bytes length: {len(pdf_data.pages[0].image) if pdf_data.pages[0].image else 0}"
            )

    else:
        print(f"No example PDF found at {example_pdf}")
        print("Skipping actual processing tests")

        # Just test the dataclass structure
        from docsearch.pdf_processing import PDF, PDFMetadata, PDFPage, PDFPageMetadata

        # Create sample data
        metadata = PDFMetadata(
            pdf_name="test",
            pdf_rel_path="test.pdf",
            num_pages=1,
            image_prompt="test prompt",
        )

        page_metadata = PDFPageMetadata(
            page_number=1, process_as_image=True, has_images=True, has_text=True
        )

        page = PDFPage(
            text="Sample text", metadata=page_metadata, image=b"fake_image_bytes"
        )

        pdf_data = PDF(metadata=metadata, pages=[page])

        print(f"Sample PDF data created: {pdf_data.metadata.pdf_name}")
        print(f"Sample page image bytes: {len(pdf_data.pages[0].image)}")

        # Test JSON export
        json_path = test_dir / "sample_output.json"
        processor.save_json(pdf_data, json_path)
        print(f"Sample JSON saved to: {json_path}")


if __name__ == "__main__":
    test_pdf_processor()
