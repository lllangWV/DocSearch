import asyncio
import json
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from PIL import Image

from docsearch import llm_processing


def markdown_to_dataframe(markdown_table: str):
    # Remove leading/trailing whitespace and split into lines
    try:
        lines = [
            line.strip() for line in markdown_table.strip().split("\n") if line.strip()
        ]

        # Remove the separator line (the one with |---|---|)
        lines = [
            line for line in lines if not re.match(r"^\s*\|[\s\-\|:]+\|\s*$", line)
        ]

        # Convert to CSV format
        csv_lines = []
        for line in lines:
            # Remove leading and trailing |
            line = line.strip("|").strip()
            # Split by | and clean each cell
            cells = [cell.strip() for cell in line.split("|")]
            csv_lines.append(",".join(f'"{cell}"' for cell in cells))

        # Create DataFrame
        csv_string = "\n".join(csv_lines)
        df = pd.read_csv(StringIO(csv_string))
    except Exception as e:
        return pd.DataFrame({})

    return df


class ImageElementBase(ABC):

    def __init__(self, image: Image.Image, md: str, summary: str = None, caption=None):
        self._image = image
        self._md = md
        self._summary = summary
        self._caption = caption

    @abstractmethod
    async def parse(cls, image, *args, caption=None, **kwargs):
        pass

    def __repr__(self):
        return f"ImageElementBase(\nmd={self._md}, \nimage={self._image})"

    @property
    def markdown(self):
        return self._md

    @property
    def md(self):
        return self._md

    @property
    def image(self):
        return self._image

    @property
    def caption(self):
        return self._caption

    @property
    def description(self):
        return self.to_markdown(include_caption=True, include_summary=True)

    def to_markdown(
        self,
        filepath: Union[str, Path] = None,
        include_caption: bool = True,
        include_summary: bool = True,
    ) -> str:
        """Convert the element to a complete markdown string including element, summary, and caption.

        Args:
            include_caption: Whether to include the caption in the output
            include_summary: Whether to include the summary in the output

        Returns:
            Combined markdown string with element, summary, and caption
        """
        combined_markdown = self._md

        if include_caption and self._caption:
            combined_markdown += f"\n\n{self._caption.md}"

        if include_summary and self._summary:
            combined_markdown += f"\n\n*Summary: {self._summary}*"

        if filepath:
            with open(filepath, "w") as f:
                f.write(combined_markdown)

        return combined_markdown

    def to_dict(self) -> Union[Dict, List[Dict]]:
        """Convert the element to a dictionary.

        Args:
            orient: How to orient the dictionary ('records', 'dict', 'list', etc.)
        """

        data = {
            "md": self._md,
            "summary": self._summary,
            "caption": self._caption.to_dict() if self._caption else None,
        }
        return data

    def to_json(
        self,
        filepath: Union[str, Path] = None,
        indent: int = 2,
    ) -> str:
        """Convert the element to JSON string.

        Args:
            filepath: If provided, save to this file path
            indent: Number of spaces for indentation
        """
        data = self.to_dict()
        if filepath:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=indent)
        return json.dumps(data, indent=indent)

    @classmethod
    def _validate_caption(cls, caption):
        if isinstance(caption, str):
            possible_caption_path = Path(caption)
            if possible_caption_path.exists():
                caption = possible_caption_path
        return caption

    @classmethod
    def _validate_image(cls, image):
        if isinstance(image, str):
            image = Path(image)
        if isinstance(image, Path):
            image = Image.open(image)
        return image

    @classmethod
    def _validate_results(cls, results):
        if isinstance(results, list) and len(results) == 2:
            element_results = results[0]
            caption = results[1]
        elif isinstance(results, list) and len(results) == 1:
            element_results = results[0]
            caption = None
        else:
            element_results = results
            caption = None
        return element_results, caption

    @classmethod
    def from_image(
        cls,
        image: Union[str, Path, Image.Image],
        model=llm_processing.MODELS[2],
        generate_config: Dict = None,
        caption: Union[str, Path, "Caption", Image.Image] = None,
    ):
        image = cls._validate_image(image)
        caption = cls._validate_caption(caption)

        # Usually asyncio.run() is used to run an async function, but in a jupyter notbook this does not work.
        # So we need to run it in a separate thread so we can block before returning.
        try:
            asyncio.get_running_loop()  # Triggers RuntimeError if no running event loop
            # Create a separate thread so we can block before returning
            with ThreadPoolExecutor(1) as pool:
                results = pool.submit(
                    lambda: asyncio.run(
                        cls.parse(
                            image,
                            caption=caption,
                            model=model,
                            generate_config=generate_config,
                        )
                    )
                ).result()
        except RuntimeError:
            results = asyncio.run(
                cls.parse(image, caption, model=model, generate_config=generate_config)
            )
        element_results, caption = cls._validate_results(results)
        return cls(
            image=image,
            md=element_results["md"],
            summary=element_results["summary"],
            caption=caption,
        )

    @classmethod
    async def from_image_async(
        cls,
        image: Union[str, Path, Image.Image],
        model=llm_processing.MODELS[2],
        generate_config: Dict = None,
        caption: Union[str, Path, "Caption", Image.Image] = None,
    ):
        image = cls._validate_image(image)
        caption = cls._validate_caption(caption)
        results = await cls.parse(
            image, caption=caption, model=model, generate_config=generate_config
        )
        element_results, caption = cls._validate_results(results)

        return cls(
            image=image,
            md=element_results["md"],
            summary=element_results["summary"],
            caption=caption,
        )


class Caption(ImageElementBase):
    @classmethod
    async def parse(
        cls,
        image: Image.Image,
        **kwargs,
    ):
        model = kwargs.get("model", llm_processing.MODELS[2])
        generate_config = kwargs.get("generate_config", None)
        result = await llm_processing.parse_text(
            image, model=model, generate_config=generate_config
        )
        return result


class Figure(ImageElementBase):
    def __init__(
        self,
        image: Image.Image,
        md: str,
        summary: str,
        caption: Caption = None,
    ):
        self._image = image
        self._md = md
        self._summary = summary
        self._caption = caption

    @classmethod
    async def parse(cls, image, caption=None, **kwargs):
        tasks = []
        model = kwargs.get("model", llm_processing.MODELS[2])
        generate_config = kwargs.get("generate_config", None)
        tasks.append(
            asyncio.create_task(
                llm_processing.parse_figure(
                    image, model=model, generate_config=generate_config
                )
            )
        )
        if isinstance(caption, Image.Image) or isinstance(caption, Path):
            tasks.append(
                Caption.from_image_async(
                    caption, model=model, generate_config=generate_config
                )
            )
        results = await asyncio.gather(*tasks)
        return results


class Table(ImageElementBase):

    def __init__(
        self,
        image: Image.Image,
        md: str,
        summary: str = None,
        caption: Caption = None,
    ):
        super().__init__(image, md, summary, caption)
        self._df = markdown_to_dataframe(md)

    @classmethod
    async def parse(cls, image, caption, **kwargs):
        tasks = []
        model = kwargs.get("model", llm_processing.MODELS[2])
        generate_config = kwargs.get("generate_config", None)
        tasks.append(
            asyncio.create_task(
                llm_processing.parse_figure(
                    image, model=model, generate_config=generate_config
                )
            )
        )
        if isinstance(caption, Image.Image) or isinstance(caption, Path):
            tasks.append(
                Caption.from_image_async(
                    caption, model=model, generate_config=generate_config
                )
            )
        results = await asyncio.gather(*tasks)
        return results

    def __len__(self):
        """Return the number of rows in the table."""
        return len(self._df)

    def __getitem__(self, key):
        """Allow indexing into the DataFrame directly."""
        return self._df[key]

    # Properties for easy access
    @property
    def df(self) -> pd.DataFrame:
        """Get the pandas DataFrame."""
        return self._df

    @property
    def shape(self) -> tuple:
        """Get the shape of the table (rows, columns)."""
        return self._df.shape

    @property
    def columns(self) -> pd.Index:
        """Get the column names."""
        return self._df.columns

    @property
    def index(self) -> pd.Index:
        """Get the row index."""
        return self._df.index

    # Data access and manipulation methods
    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the first n rows."""
        return self._df.head(n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        """Return the last n rows."""
        return self._df.tail(n)

    def to_csv(
        self,
        filepath: Union[str, Path] = None,
        **kwargs,
    ) -> str:
        """Convert the table to CSV.

        Args:
            filepath: If provided, save to this file path
            **kwargs: Additional arguments passed to pandas.to_csv()
        """
        if filepath:
            return self._df.to_csv(filepath, index=False, **kwargs)
        else:
            return self._df.to_csv(index=False, **kwargs)


class Formula(ImageElementBase):
    def __init__(
        self,
        image: Image.Image,
        md: str,
        summary: str,
        caption: Caption = None,
    ):
        self._image = image
        self._md = md
        self._caption = caption
        self._summary = summary

    @classmethod
    async def parse(cls, image, caption=None, **kwargs):
        tasks = []
        model = kwargs.get("model", llm_processing.MODELS[2])
        generate_config = kwargs.get("generate_config", None)
        tasks.append(
            asyncio.create_task(
                llm_processing.parse_formula(
                    image, model=model, generate_config=generate_config
                )
            )
        )
        if isinstance(caption, Image.Image) or isinstance(caption, Path):
            tasks.append(
                Caption.from_image_async(
                    caption, model=model, generate_config=generate_config
                )
            )
        results = await asyncio.gather(*tasks)
        return results


class Text(ImageElementBase):
    def __init__(
        self, image: Image.Image, md: str, summary: str = None, caption: Caption = None
    ):
        super().__init__(image, md, summary, caption)

    @classmethod
    async def parse(cls, image, caption, **kwargs):
        model = kwargs.get("model", llm_processing.MODELS[2])
        generate_config = kwargs.get("generate_config", None)
        result = await llm_processing.parse_text(
            image, model=model, generate_config=generate_config
        )

        return result


class Title(ImageElementBase):
    def __init__(
        self, image: Image.Image, md: str, summary: str = None, caption: Caption = None
    ):
        super().__init__(image, md, summary, caption)

    @classmethod
    async def parse(cls, image, caption, **kwargs):
        model = kwargs.get("model", llm_processing.MODELS[2])
        generate_config = kwargs.get("generate_config", None)
        result = await llm_processing.parse_text(
            image, model=model, generate_config=generate_config
        )
        return result


class Undefined(ImageElementBase):
    def __init__(
        self, image: Image.Image, md: str, summary: str = None, caption: Caption = None
    ):
        super().__init__(image, md, summary, caption)

    @classmethod
    async def parse(cls, image, caption, **kwargs):
        model = kwargs.get("model", llm_processing.MODELS[2])
        generate_config = kwargs.get("generate_config", None)
        result = await llm_processing.parse_text(
            image, model=model, generate_config=generate_config
        )
        return result
