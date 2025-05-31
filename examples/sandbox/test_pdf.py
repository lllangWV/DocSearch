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
from docsearch.figure_extraction import DocumentPageAnalyzer

console = Console()

ROOT_DIR = Path(os.path.abspath(__file__)).parent
DATA_DIR = ROOT_DIR / "data"
MODEL_WEIGHTS = (
    ROOT_DIR.parent.parent / "data" / "doclayout_yolo_docstructbench_imgsz1024.pt"
)
PDF_PATH = (
    ROOT_DIR.parent.parent
    / "data"
    / "test-dataset"
    / "Fang et al. - 2022 - Molecular Contrastive Learning with Chemical Eleme.pdf"
)
# print(f"ROOT_DIR: {ROOT_DIR}")
# print(f"DATA_DIR: {DATA_DIR}")
# print(f"MODEL_WEIGHTS: {MODEL_WEIGHTS}")
# print(f"SAMPLES_DIR: {SAMPLES_DIR}")
# for filepath in sample_filepaths:
#     print(f"sample_filepath: {filepath}")


from docsearch.core import Page

# from docsearch.core.element import Page

# page = Page.from_image(sample_filepaths[5], model_weights=MODEL_WEIGHTS)

doc = Document.from_pdf(PDF_PATH, dpi=150)
md = Markdown(doc.md)
console.print(md, crop=False)

# print(page.tables[0].metadata)
# print(page.tables[1].metadata)

# print(page.page_layout.element_list)

doc.to_markdown(filepath=DATA_DIR / "samples" / "test_pdf.md")
# page.to_sorted_markdown(
#     filepath=SAMPLES_DIR / sample_filepaths[3].stem / "page_sorted.md"
# )
