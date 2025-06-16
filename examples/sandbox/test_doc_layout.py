import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from rich.console import Console
from rich.markdown import Markdown

from docrag.core.page_layout import PageLayout
from docrag.figure_extraction import DocumentPageAnalyzer

console = Console()

ROOT_DIR = Path(os.path.abspath(__file__)).parent
DATA_DIR = ROOT_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
sample_filepaths = list(SAMPLES_DIR.glob("*.png"))

print(f"ROOT_DIR: {ROOT_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"SAMPLES_DIR: {SAMPLES_DIR}")
for filepath in sample_filepaths:
    print(f"sample_filepath: {filepath}")


from docrag.core import Page

# page = Page.from_image(sample_filepaths[5], model_weights=MODEL_WEIGHTS)

page_layout = PageLayout()
sample_layout = page_layout.extract_elements(sample_filepaths[3])


for element_type, elements in sample_layout.elements.items():
    print(element_type)
    for element in elements:
        print(element)
sample_layout.save(SAMPLES_DIR / sample_filepaths[3].stem)
print(sample_layout)
