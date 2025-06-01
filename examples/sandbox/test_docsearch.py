import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from rich.console import Console
from rich.markdown import Markdown

from docsearch.core import Document
from docsearch.doc_search import DocSearch

console = Console()

ROOT_DIR = Path(os.path.abspath(__file__)).parent
DATA_DIR = ROOT_DIR / "data"
PDF_PATH = (
    ROOT_DIR.parent.parent
    / "data"
    / "test-dataset"
    / "Fang et al. - 2022 - Molecular Contrastive Learning with Chemical Eleme.pdf"
)


PDF_PATH = ROOT_DIR.parent.parent / "data" / "test-dataset" / "phi-xps-2021-impact.pdf"


# from docsearch.core import Page

# # # from docsearch.core.element import Page

# # # page = Page.from_image(sample_filepaths[5], model_weights=MODEL_WEIGHTS)

# # doc = Document.from_pdf(PDF_PATH, dpi=150)
# # md = Markdown(doc.md)
# # console.print(md, crop=False)

# # # print(page.tables[0].metadata)
# # print(page.tables[1].metadata)

# # print(page.page_layout.element_list)

# docsearch = DocSearch(base_path=DATA_DIR)
# pdf_paths = [PDF_PATH]
# docsearch.add_pdfs(pdf_paths)

# # doc.to_markdown(filepath=DATA_DIR / "samples" / "test_pdf.md")
