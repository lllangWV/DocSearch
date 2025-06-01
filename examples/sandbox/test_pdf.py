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

PDF_PATH = ROOT_DIR.parent.parent / "data" / "test-dataset" / "phi-xps-2021-impact.pdf"

# doc = Document.from_pdf(PDF_PATH, dpi=150)
# md = Markdown(doc.md)
# console.print(md, crop=False)

# doc.save(DATA_DIR / PDF_PATH.stem, save_pdf=True)


# doc.to_pyarrow(filepath=DATA_DIR / PDF_PATH.stem / "pdf.parquet")

# doc = Document.from_pyarrow(DATA_DIR / PDF_PATH.stem / "pdf.parquet")
# table = doc.to_pyarrow()

# table = doc.to_pyarrow(
#     filepath=DATA_DIR / PDF_PATH.stem / "pdf_per_row.parquet", page_per_row=False
# )
# import pyarrow.compute as pc

# print(table.schema)

# page_0 = table.filter(pc.field("page_id") == 1)

# ["markdown"].combine_chunks()


# print(page_0)

# print(table.shape)

doc = Document.from_pyarrow(DATA_DIR / PDF_PATH.stem / "pdf_per_row.parquet")
table = doc.to_pyarrow(
    filepath=DATA_DIR / PDF_PATH.stem / "pdf_per_row.parquet", page_per_row=True
)
table = doc.to_pyarrow(page_per_row=True)["markdown"].combine_chunks()[0]
# print(table.schema)
print(table)


# print(page.tables[0].metadata)
# print(page.tables[1].metadata)

# print(page.page_layout.element_list)

# doc.to_markdown(filepath=DATA_DIR / "samples" / "test_pdf.md")
# page.to_sorted_markdown(
#     filepath=SAMPLES_DIR / sample_filepaths[3].stem / "page_sorted.md"
# )
