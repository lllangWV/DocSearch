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
from docsearch.core.caption import Caption


def markdown_to_dataframe(markdown_table: str):
    # Remove leading/trailing whitespace and split into lines
    lines = [
        line.strip() for line in markdown_table.strip().split("\n") if line.strip()
    ]

    # Remove the separator line (the one with |---|---|)
    lines = [line for line in lines if not re.match(r"^\s*\|[\s\-\|:]+\|\s*$", line)]

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

    return df


class Figure:
    def __init__(
        self,
        md: str = None,
        summary: str = None,
        image: Image.Image = None,
        caption: Caption = None,
    ):
        self._md = md
        self._summary = summary
        self._image = image
        self._caption = caption

    def __repr__(self):
        return f"Figure(\nmd={self._md}, \nsummary={self._summary}, \nimage={self._image}, \ncaption={self._caption})"

    @property
    def markdown(self) -> Optional[str]:
        """Get the markdown representation."""
        return self._md

    @property
    def md(self) -> Optional[str]:
        """Get the markdown representation."""
        return self._md

    @property
    def summary(self) -> Optional[str]:
        """Get the table summary."""
        return self._summary

    @property
    def image(self) -> Image.Image:
        """Get the image."""
        return self._image

    @property
    def caption(self) -> Optional[Caption]:
        """Get the caption."""
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
        """Convert the table to a complete markdown string including table, summary, and caption.

        Args:
            include_caption: Whether to include the caption in the output
            include_summary: Whether to include the summary in the output

        Returns:
            Combined markdown string with table, summary, and caption
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
        """Convert the figure to a dictionary representation."""

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
        """Convert the table to JSON string.

        Args:
            filepath: If provided, save to this file path
            orient: How to orient the JSON ('records', 'index', 'values', etc.)
            indent: Number of spaces for indentation
        """
        data = self.to_dict()
        if filepath:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=indent)
        return json.dumps(data, indent=indent)

    @classmethod
    def from_image(
        cls,
        image: Union[str, Path, Image.Image],
        model=llm_processing.MODELS[2],
        generate_config: Dict = None,
        caption: Union[str, Path, Caption, Image.Image] = None,
    ):
        if isinstance(image, str):
            image = Path(image)

        if isinstance(image, Path):
            image = Image.open(image)

        if isinstance(caption, str):
            possible_caption_path = Path(caption)
            if possible_caption_path.exists():
                caption = possible_caption_path

        tasks = []
        tasks

        async def parse_all_images():
            tasks = []
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

        # Usually asyncio.run() is used to run an async function, but in a jupyter notbook this does not work.
        # So we need to run it in a separate thread so we can block before returning.
        try:
            asyncio.get_running_loop()  # Triggers RuntimeError if no running event loop
            # Create a separate thread so we can block before returning
            with ThreadPoolExecutor(1) as pool:
                results = pool.submit(lambda: asyncio.run(parse_all_images())).result()
        except RuntimeError:
            results = asyncio.run(parse_all_images())

        if isinstance(results, list):
            figure_results = results[0]
            caption = results[1]
        else:
            figure_results = results

        if caption and isinstance(caption, str):
            caption = Caption.from_md(caption)

        return cls(
            md=figure_results["md"],
            summary=figure_results["summary"],
            image=image,
            caption=caption,
        )

    @classmethod
    async def from_image_async(
        cls,
        image: Union[str, Path, Image.Image],
        model=llm_processing.MODELS[2],
        generate_config: Dict = None,
        caption: Union[str, Path, Caption, Image.Image] = None,
    ):
        if isinstance(image, str):
            image = Path(image)

        if isinstance(image, Path):
            image = Image.open(image)

        if isinstance(caption, str):
            possible_caption_path = Path(caption)
            if possible_caption_path.exists():
                caption = possible_caption_path

        async def parse_all_images():
            tasks = []
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

        result = await parse_all_images()
        if isinstance(result, list):
            figure_results = result[0]
            caption = result[1]
        else:
            figure_results = result

        if caption and isinstance(caption, str):
            caption = Caption.from_md(caption)

        return cls(
            md=figure_results["md"],
            summary=figure_results["summary"],
            image=image,
            caption=caption,
        )
