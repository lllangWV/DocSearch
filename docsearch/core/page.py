import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from PIL import Image

from docsearch import llm_processing
from docsearch.core.data import Figure, Formula, Table, Text, Title, Undefined
from docsearch.core.page_layout import PageLayout
from docsearch.utils.config import MODELS_DIR


class Page:

    def __init__(
        self,
        image: Image.Image,
        figures: List[Figure] = None,
        tables: List[Table] = None,
        formulas: List[Formula] = None,
        text: List[Text] = None,
        title: List[Title] = None,
        undefined: List[Undefined] = None,
        annotated_image: Image.Image = None,
        elements: Dict[str, List[Dict]] = None,
        page_layout: PageLayout = None,
    ):
        self._image = image
        self._figures = figures or []
        self._tables = tables or []
        self._formulas = formulas or []
        self._text = text or []
        self._title = title or []
        self._undefined = undefined or []
        self._annotated_image = annotated_image
        self._elements = elements

    def __repr__(self):
        return f"Page(\nimage={self._image}, \nfigures={self._figures}, \ntables={self._tables}, \nformulas={self._formulas}, \nannotated_image={self._annotated_image}, \nelements={self._elements})"

    def __str__(self):
        return self.to_markdown()

    @property
    def markdown(self):
        return self.to_markdown()

    @property
    def md(self):
        return self.to_markdown()

    @property
    def image(self):
        return self._image

    @property
    def annotated_image(self):
        return self._annotated_image

    @property
    def figures(self):
        return self._figures

    @property
    def tables(self):
        return self._tables

    @property
    def formulas(self):
        return self._formulas

    @property
    def title(self):
        return self._title

    @property
    def text(self):
        return self._text

    @property
    def undefined(self):
        return self._undefined

    @property
    def elements(self):
        return self._elements

    @property
    def description(self):
        return self.__repr__()

    def full_save(self, out_dir: Union[str, Path]):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        self.to_json(out_dir / "page.json")
        self.to_markdown(out_dir / "page.md")
        self._image.save(out_dir / "page.png")
        self._annotated_image.save(out_dir / "page_annotated.png")

    def to_markdown(
        self,
        filepath: Union[str, Path] = None,
        include_caption=False,
        include_summary=False,
        include_section_header=True,
    ):
        tmp_str = ""
        if include_section_header:
            tmp_str += "# Page\n\n"

        tmp_str += "## Text\n\n" if self._text else ""
        for text in self._text:
            tmp_str += text.to_markdown(
                include_caption=include_caption, include_summary=include_summary
            )
            tmp_str += "\n\n"

        tmp_str += "## Title\n\n" if self._title else ""
        for title in self._title:
            tmp_str += title.to_markdown(
                include_caption=include_caption, include_summary=include_summary
            )
            tmp_str += "\n\n"

        tmp_str += "## Figures\n\n" if self._figures else ""
        for fig in self._figures:
            tmp_str += fig.to_markdown(
                include_caption=include_caption, include_summary=include_summary
            )
            tmp_str += "\n\n"

        tmp_str += "## Tables\n\n" if self._tables else ""
        for table in self._tables:
            tmp_str += table.to_markdown(
                include_caption=include_caption, include_summary=include_summary
            )
            tmp_str += "\n\n"

        tmp_str += "## Formulas\n\n" if self._formulas else ""
        for formula in self._formulas:
            tmp_str += formula.to_markdown(
                include_caption=include_caption, include_summary=include_summary
            )
            tmp_str += "\n\n"

        tmp_str += "## Undefined\n\n" if self._undefined else ""
        for undefined in self._undefined:
            tmp_str += undefined.to_markdown(
                include_caption=include_caption, include_summary=include_summary
            )
            tmp_str += "\n\n"

        if filepath:
            with open(filepath, "w") as f:
                f.write(tmp_str)
        return tmp_str

    def to_dict(self):

        return {
            "figures": [fig.to_dict() for fig in self._figures],
            "tables": [table.to_dict() for table in self._tables],
            "formulas": [formula.to_dict() for formula in self._formulas],
            "text": [text.to_dict() for text in self._text],
            "title": [title.to_dict() for title in self._title],
            "undefined": [undefined.to_dict() for undefined in self._undefined],
        }

    def to_json(self, filepath: Union[str, Path] = None, indent: int = 2):
        if filepath:
            with open(filepath, "w") as f:
                json.dump(self.to_dict(), f, indent=indent)
        return json.dumps(self.to_dict())

    @classmethod
    def _validate_image(cls, image):
        if isinstance(image, str):
            image = Path(image)

        if isinstance(image, Path):
            image = Image.open(image)
        return image

    @classmethod
    async def parse(
        cls,
        page_layout_dict,
        model=llm_processing.MODELS[2],
        generate_config: Dict = None,
    ):
        tasks = []
        for element_type, elements in page_layout_dict["elements"].items():
            if element_type == "table":
                class_type = Table
            elif element_type == "formula":
                class_type = Formula
            elif element_type == "figure":
                class_type = Figure
            elif element_type == "text":
                class_type = Text
            elif element_type == "title":
                class_type = Title
            elif element_type == "unknown":
                class_type = Undefined
            else:
                continue
            for element in elements:
                caption = None
                if element.caption:
                    caption = element.caption.image
                if element.footnote:
                    caption = element.footnote.image
                tasks.append(
                    class_type.from_image_async(
                        element.image,
                        model=model,
                        generate_config=generate_config,
                        caption=caption,
                    )
                )
        results = await asyncio.gather(*tasks)

        return results

    @staticmethod
    def _gather_results(results):
        gathered_results = {
            "tables": [],
            "figures": [],
            "formulas": [],
            "text": [],
            "title": [],
            "undefined": [],
        }
        for result in results:
            if isinstance(result, Table):
                gathered_results["tables"].append(result)
            elif isinstance(result, Figure):
                gathered_results["figures"].append(result)
            elif isinstance(result, Formula):
                gathered_results["formulas"].append(result)
            elif isinstance(result, Text):
                gathered_results["text"].append(result)
            elif isinstance(result, Title):
                gathered_results["title"].append(result)
            elif isinstance(result, Undefined):
                gathered_results["undefined"].append(result)
            else:
                raise ValueError(f"Unknown result type: {type(result)}\n {result}")
        return gathered_results

    # Class methods remain the same
    @classmethod
    def from_image(
        cls,
        image: Union[str, Path, Image.Image],
        model_weights: Union[Path, str] = "doclayout_yolo_docstructbench_imgsz1024.pt",
        model=llm_processing.MODELS[2],
        generate_config: Dict = None,
    ):
        image = cls._validate_image(image)

        # Use DocLayout class instead of extract_image_elements function
        page_layout = PageLayout(model_weights=model_weights)
        page_layout = page_layout.extract_elements(image)
        page_layout_dict = page_layout.to_dict()

        # Usually asyncio.run() is used to run an async function, but in a jupyter notbook this does not work.
        # So we need to run it in a separate thread so we can block before returning.
        try:
            asyncio.get_running_loop()  # Triggers RuntimeError if no running event loop
            # Create a separate thread so we can block before returning
            with ThreadPoolExecutor(1) as pool:
                results = pool.submit(
                    lambda: asyncio.run(
                        cls.parse(
                            page_layout_dict,
                            model=model,
                            generate_config=generate_config,
                        )
                    )
                ).result()
        except RuntimeError:
            results = asyncio.run(
                cls.parse(
                    page_layout_dict, model=model, generate_config=generate_config
                )
            )

        if not isinstance(results, list):
            results = [results]

        gathered_results = Page._gather_results(results)
        return cls(
            image=image,
            **gathered_results,
            annotated_image=page_layout_dict.get("annotated_image", None),
            elements=page_layout_dict.get("elements", {}),
            page_layout=page_layout,
        )

    @classmethod
    async def from_image_async(
        cls,
        image: Union[str, Path, Image.Image],
        model_weights: Union[Path, str] = "doclayout_yolo_docstructbench_imgsz1024.pt",
        model=llm_processing.MODELS[2],
        generate_config: Dict = None,
    ):
        image = cls._validate_image(image)

        # Use DocLayout class instead of extract_image_elements function
        page_layout = PageLayout(model_weights=model_weights)
        page_layout.extract_elements(image)
        extraction_results = page_layout.to_dict()

        # Usually asyncio.run() is used to run an async function, but in a jupyter notbook this does not work.
        # So we need to run it in a separate thread so we can block before returning.
        results = await cls.parse(
            extraction_results, model=model, generate_config=generate_config
        )
        if not isinstance(results, list):
            results = [results]

        gathered_results = Page._gather_results(results)
        return cls(
            image=image,
            **gathered_results,
            annotated_image=extraction_results.get("annotated_image", None),
            elements=extraction_results.get("elements", {}),
        )
