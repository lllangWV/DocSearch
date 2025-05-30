import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from rich.console import Console
from rich.markdown import Markdown

from docsearch.figure_extraction import DocumentPageAnalyzer

console = Console()

ROOT_DIR = Path(os.path.abspath(__file__)).parent
DATA_DIR = ROOT_DIR / "data"
MODEL_WEIGHTS = (
    ROOT_DIR.parent.parent / "data" / "doclayout_yolo_docstructbench_imgsz1024.pt"
)
SAMPLES_DIR = DATA_DIR / "samples"
sample_filepaths = list(SAMPLES_DIR.glob("*.png"))

# print(f"ROOT_DIR: {ROOT_DIR}")
# print(f"DATA_DIR: {DATA_DIR}")
# print(f"MODEL_WEIGHTS: {MODEL_WEIGHTS}")
# print(f"SAMPLES_DIR: {SAMPLES_DIR}")
# for filepath in sample_filepaths:
#     print(f"sample_filepath: {filepath}")


from docsearch.core import Page

page = Page.from_image(sample_filepaths[5], model_weights=MODEL_WEIGHTS)
print(page)
md = Markdown(page.md)
console.print(md, crop=False)
