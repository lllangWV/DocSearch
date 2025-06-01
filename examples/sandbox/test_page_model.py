import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
from IPython.display import display
from PIL import Image
from rich.console import Console
from rich.markdown import Markdown

from docsearch.core import Page

console = Console()

ROOT_DIR = Path(os.path.abspath(__file__)).parent
DATA_DIR = ROOT_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
sample_filepaths = list(SAMPLES_DIR.glob("*.png"))

PDF_PATH = ROOT_DIR.parent.parent / "data" / "test-dataset" / "phi-xps-2021-impact.pdf"
PDF_PAGE = DATA_DIR / "phi-xps-2021-impact" / "page_2" / "page.png"

# page = Page.from_image(PDF_PAGE)
# md = Markdown(page.md)
# # console.print(md, crop=False)

# table = page.to_pyarrow()


# # # print(table)
# print(table.shape)


# pq.write_table(table, DATA_DIR / "phi-xps-2021-impact" / "page.parquet")

table = pq.read_table(DATA_DIR / "phi-xps-2021-impact" / "page.parquet")

page = Page.from_parquet(DATA_DIR / "phi-xps-2021-impact" / "page.parquet")

print(page)


table = pq.read_table(DATA_DIR / "phi-xps-2021-impact" / "page.parquet")
data = table.to_pandas().to_dict(orient="records")
page = Page.from_dict(data[0])

print(page)


# table = pa.table([page_data], schema=pa.schema(page_struct))


# print(page.tables[0].metadata)
# print(page.tables[1].metadata)

# print(page.page_layout.element_list)

# page.to_markdown(filepath=SAMPLES_DIR / sample_filepaths[3].stem / "page.md")
# page.to_sorted_markdown(
#     filepath=SAMPLES_DIR / sample_filepaths[3].stem / "page_sorted.md"
# )
