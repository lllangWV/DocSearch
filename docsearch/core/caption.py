import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from PIL import Image

from docsearch import llm_processing


class Caption:
    def __init__(
        self,
        md: str = None,
        image: Image.Image = None,
    ):
        self._image = image
        self._md = md

    def __repr__(self):
        return f"Caption(\nmd={self._md}, \nimage={self._image})"

    def __str__(self):
        return self._md

    @property
    def markdown(self):
        return self._md

    @property
    def md(self):
        return self._md

    @property
    def image(self):
        return self._image

    # Conversion methods
    def to_dict(self) -> Union[Dict, List[Dict]]:
        """Convert the table to a dictionary."""

        data = {
            "md": self._md,
        }
        return data

    def to_json(
        self,
        filepath: Union[str, Path] = None,
        indent: int = 2,
    ) -> str:
        """Convert the table to JSON string.

        Args:
            filepath: If provided, save to this file path
            indent: Number of spaces for indentation
        """
        data = self.to_dict()
        if filepath:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=indent)
        return json.dumps(data, indent=indent)

    # Class methods remain the same
    @classmethod
    def from_image(
        cls,
        image: Union[str, Path, Image.Image],
        model=llm_processing.MODELS[2],
        generate_config: Dict = None,
    ):
        if isinstance(image, str):
            image = Path(image)

        if isinstance(image, Path):
            image = Image.open(image)

        # Usually asyncio.run() is used to run an async function, but in a jupyter notbook this does not work.
        # So we need to run it in a separate thread so we can block before returning.
        try:
            asyncio.get_running_loop()  # Triggers RuntimeError if no running event loop
            # Create a separate thread so we can block before returning
            with ThreadPoolExecutor(1) as pool:
                result = pool.submit(
                    lambda: asyncio.run(
                        llm_processing.parse_text(
                            image, model=model, generate_config=generate_config
                        )
                    )
                ).result()
        except RuntimeError:
            result = asyncio.run(
                llm_processing.parse_text(
                    image, model=model, generate_config=generate_config
                )
            )
        return cls(md=result["md"], image=image)

    @classmethod
    async def from_image_async(
        cls,
        image: Union[str, Path, Image.Image],
        model=llm_processing.MODELS[2],
        generate_config: Dict = None,
    ):
        if isinstance(image, str):
            image = Path(image)

        if isinstance(image, Path):
            image = Image.open(image)

        result = await llm_processing.parse_text(
            image, model=model, generate_config=generate_config
        )
        return cls(md=result["md"], image=image)

    @classmethod
    def from_md(
        cls,
        md: Union[str | Path],
        image: Image.Image = None,
    ):
        if isinstance(md, Path):
            if not md.suffix.lower() == ".md":
                raise ValueError(f"File must have .md extension, got {md.suffix}")
            with open(md, "r") as f:
                md = f.read()
        return cls(md=md, image=image)
