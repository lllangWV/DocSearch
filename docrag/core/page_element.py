import asyncio
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from PIL import Image

from docrag.core import llm_processing
from docrag.core.utils import pil_image_to_bytes
from docrag.utils.config import MODELS_DIR

logger = logging.getLogger(__name__)

class PageElementType(Enum):
    """Type of element."""

    FIGURE = "figure"
    TABLE = "table"
    FORMULA = "formula"
    TEXT = "text"
    TITLE = "title"
    TABLE_CAPTION = "table_caption"
    FORMULA_CAPTION = "formula_caption"
    TABLE_FOOTNOTE = "table_footnote"
    FIGURE_CAPTION = "figure_caption"
    UNKNOWN = "unknown"


@dataclass
class PageElement:
    """Metadata for an element."""

    element_type: PageElementType
    confidence: float
    bbox: List[int]
    image: Image.Image
    caption: Optional["PageElement"] = None
    footnote: Optional["PageElement"] = None
    markdown: str = ""
    summary: str = ""

    def to_markdown(
        self,
        include_caption=True,
        include_summary=True,
        include_footnote=True,
    ):
        tmp_str = ""
        tmp_str += self.markdown
        if include_caption and self.caption:
            tmp_str += f"\n\n{self.caption.markdown}"
        if include_footnote and self.footnote:
            tmp_str += f"\n\n{self.footnote.markdown}"
        if include_summary and self.summary:
            tmp_str += f"\n\nSummary: {self.summary}"
        return tmp_str

    def to_dict(self, include_image: bool = False, image_as_base64: bool = False):

        data = {
            "element_type": self.element_type,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "markdown": self.markdown,
            "summary": self.summary,
            "caption": (
                self.caption.to_dict(include_image, image_as_base64)
                if self.caption
                else None
            ),
            "footnote": (
                self.footnote.to_dict(include_image, image_as_base64)
                if self.footnote
                else None
            ),
        }
        if include_image:
            if image_as_base64:
                data["image"] = pil_image_to_bytes(self.image)
            else:
                data["image"] = self.image
        return data

    @classmethod
    def from_dict(cls, data: Dict):
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        if not "image" in data:
            raise ValueError("Image not found in data")

        if isinstance(data["image"], bytes):
            data["image"] = Image.open(io.BytesIO(data["image"]))
        if "caption" in data and data["caption"]:
            data["caption"] = cls.from_dict(data["caption"])
        if "footnote" in data and data["footnote"]:
            data["footnote"] = cls.from_dict(data["footnote"])

        return cls(**data)

    @staticmethod
    def get_pyarrow_struct(
        include_image: bool = True,
        include_captions: bool = True,
        include_footnotes: bool = True,
    ):
        struct_dict = {
            "element_type": pa.string(),
            "confidence": pa.float32(),
            "bbox": pa.list_(pa.int32()),
            "markdown": pa.string(),
            "summary": pa.string(),
        }
        if include_image:
            struct_dict["image"] = pa.binary()
        if include_captions:
            struct_dict["caption"] = PageElement.get_pyarrow_struct(
                include_image=include_image, include_captions=False, include_footnotes=False
            )
        if include_footnotes:
            struct_dict["footnote"] = PageElement.get_pyarrow_struct(
                include_image=include_image, include_captions=False, include_footnotes=False
            )
        return pa.struct(struct_dict)

    @staticmethod
    def get_pyarrow_empty_data(
        include_captions: bool = True, include_footnotes: bool = True
    ):
        struct_dict = {
            "element_type": None,
            "confidence": None,
            "bbox": None,
            "image": None,
            "markdown": None,
            "summary": None,
        }
        if include_captions:
            struct_dict["caption"] = PageElement.get_pyarrow_empty_data(
                include_captions=False, include_footnotes=False
            )
        if include_footnotes:
            struct_dict["footnote"] = PageElement.get_pyarrow_empty_data(
                include_captions=False, include_footnotes=False
            )
        return struct_dict

    async def parse_content(self, model=None, generate_config=None):
        """Parse content based on element type."""
        # Create tasks for parallel execution
        tasks = []

        # Main element parsing task
        tasks.append(
            self._parse_main_content(model=model, generate_config=generate_config)
        )

        # Caption parsing task
        if self.caption is not None:
            tasks.append(
                self.caption._parse_caption_or_footnote(
                    element=self.caption,
                    model=model,
                    generate_config=generate_config,
                )
            )

        # Footnote parsing task
        if self.footnote is not None:
            tasks.append(
                self.footnote._parse_caption_or_footnote(
                    element=self.footnote,
                    model=model,
                    generate_config=generate_config,
                )
            )

        # Execute all parsing tasks in parallel
        await asyncio.gather(*tasks)

        # Set results for caption/footnote (they handle their own assignment in _parse_main_content)

    async def _parse_main_content(self, model=None, generate_config=None):
        """Parse content based on element type."""
        if self.element_type == PageElementType.FIGURE.value:
            result = await llm_processing.parse_figure(
                self.image, model=model, generate_config=generate_config
            )
        elif self.element_type == PageElementType.TABLE.value:
            result = await llm_processing.parse_table(
                self.image, model=model, generate_config=generate_config
            )
        elif self.element_type == PageElementType.FORMULA.value:
            result = await llm_processing.parse_formula(
                self.image, model=model, generate_config=generate_config
            )
        elif self.element_type in [
            PageElementType.TEXT.value,
            PageElementType.TITLE.value,
            PageElementType.UNKNOWN.value,
        ]:
            result = await llm_processing.parse_text(
                self.image, model=model, generate_config=generate_config
            )

        else:
            result = await llm_processing.parse_text(
                self.image, model=model, generate_config=generate_config
            )

        # Set parsed content for this element
        self.markdown = result.get("md", "")
        self.summary = result.get("summary", "")

        return result

    async def _parse_caption_or_footnote(self, element, model, generate_config):
        """Helper method to parse caption or footnote element."""
        result = await llm_processing.parse_text(
            element.image, model=model, generate_config=generate_config
        )
        element.markdown = result.get("md", "")
        element.summary = result.get("summary", "")


