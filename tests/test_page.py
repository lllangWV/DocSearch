import os

import pytest
from PIL import Image

from docrag.core.page import Page
from tests.utils import DATA_DIR

# @pytest.fixture
# def tmp_dir(page_dirpath):
#     return page_dirpath / "test.parquet"


@pytest.fixture
def page_dirpath():
    return DATA_DIR / "pages"


@pytest.fixture
def page_image(page_dirpath):
    filepath = list(page_dirpath.glob("*.png"))[0]
    return Image.open(filepath)


class TestPage:

    def test_pyarrow_support(self, tmp_path, page_image):
        page = Page.from_image(page_image)
        print(len(page.elements))
        parquet_path = tmp_path / "test.parquet"
        table = page.to_pyarrow(parquet_path)
        assert table.shape == (1,5)
        assert len(table['elements'].combine_chunks()[0]) == len(page.elements)
        
        page = Page.from_parquet(parquet_path)
        assert table.shape == (1,5)
        assert len(table['elements'].combine_chunks()[0]) == len(page.elements)
        
        
        
        







def create_pages():

    # Save images to pages directory
    import os

    from PIL import Image
    
    sample_doc = DATA_DIR / "documents" / "sample_document_small.pdf"
    from docrag.core.document import Document


    images = Document._extract_pages_as_images(sample_doc)


    # Create pages directory in the same location as the PDF
    pdf_dir = sample_doc.parent
    pages_dir = pdf_dir / "pages"
    pages_dir.mkdir(exist_ok=True)

    # Save each image with page number
    for i, image in enumerate(images):
        page_filename = pages_dir / f"page_{i+1:03d}.png"
        image.save(page_filename, "PNG")
        print(f"Saved {page_filename}")


