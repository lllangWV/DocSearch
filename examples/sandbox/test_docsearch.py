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
from docsearch.core.doc_search import DocSearch

console = Console()

ROOT_DIR = Path(os.path.abspath(__file__)).parent
DATA_DIR = ROOT_DIR / "data"

DOCSEARCH_DIR = DATA_DIR / "DocSearch"
PDF_PATH = (
    ROOT_DIR.parent.parent
    / "data"
    / "test-dataset"
    / "Fang et al. - 2022 - Molecular Contrastive Learning with Chemical Eleme.pdf"
)


PDF_PATH = ROOT_DIR.parent.parent / "data" / "test-dataset" / "phi-xps-2021-impact.pdf"
PDF_PATH_2 = (
    ROOT_DIR.parent.parent
    / "data"
    / "test-dataset"
    / "xps-profiling-of-organic-pv-films.pdf"
)

# from docsearch.core import Page

# # # from docsearch.core.element import Page

# # # page = Page.from_image(sample_filepaths[5], model_weights=MODEL_WEIGHTS)

# # doc = Document.from_pdf(PDF_PATH, dpi=150)
# # md = Markdown(doc.md)
# # console.print(md, crop=False)

# # # print(page.tables[0].metadata)
# # print(page.tables[1].metadata)

# # print(page.page_layout.element_list)

# docsearch = DocSearch(base_path=DOCSEARCH_DIR)
# pdf_paths = [PDF_PATH, PDF_PATH_2]
# docsearch.add_pdfs(pdf_paths, auto_load=False)


docsearch = DocSearch(base_path=DOCSEARCH_DIR)
# docsearch.load_processed_pdfs()

print("Initial stats:")
stats = docsearch.get_stats()
for key, value in stats.items():
    print(f"  {key}: {value}")

docsearch.query(
    "What has XPS been used for recently?",
    save_response=True,
    similarity_top_k=2,
)
# # doc.to_markdown(filepath=DATA_DIR / "samples" / "test_pdf.md")

# import pyarrow.compute as pc
# import pyarrow.dataset as ds

# dataset = ds.dataset(DOCSEARCH_DIR / "pages", format="parquet")

# # print(dataset.schema)

# table = dataset.to_table(columns=["pdf_id", "page_id", "markdown"])
# df = table.to_pandas()

# for index, row in df.iterrows():
#     print(row["pdf_id"])
#     print(row["page_id"])
#     print(row["markdown"])
#     print("\n")


# print(ds.to_pyarrow_dataset())
