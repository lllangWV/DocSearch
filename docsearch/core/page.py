import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
from doclayout_yolo import YOLOv10
from PIL import Image

from docsearch import llm_processing
from docsearch.core.caption import Caption
from docsearch.core.figure import Figure
from docsearch.core.formula import Formula
from docsearch.core.table import Table


def find_nearest_caption(
    element_box: List[float],
    caption_boxes: List[List[float]],
    max_distance: float = 100,
) -> Optional[List[float]]:
    """Find the nearest caption to a given element based on proximity and vertical alignment."""
    best_caption = None
    best_score = float("inf")

    element_center_x = (element_box[0] + element_box[2]) / 2
    element_bottom = element_box[3]

    for caption in caption_boxes:
        caption_box = caption
        caption_center_x = (caption_box[0] + caption_box[2]) / 2
        caption_top = caption_box[1]

        # Calculate horizontal alignment (prefer captions that are horizontally aligned)
        horizontal_distance = abs(element_center_x - caption_center_x)

        # Calculate vertical distance (prefer captions below the element)
        vertical_distance = caption_top - element_bottom

        # Prefer captions that are below the element but not too far
        if vertical_distance > max_distance:
            continue

        # Combined score: prioritize vertical proximity and horizontal alignment
        score = vertical_distance + horizontal_distance * 0.5

        if score < best_score:
            best_score = score
            best_caption = caption

    return best_caption


def get_element_type(class_name: str) -> str:
    """Determine the element type from class name."""
    if "figure" in class_name and "caption" not in class_name:
        return "figure"
    elif (
        "table" in class_name
        and "caption" not in class_name
        and "footnote" not in class_name
    ):
        return "table"
    elif "formula" in class_name and "caption" not in class_name:
        return "formula"
    elif "text" in class_name:
        return "text"
    elif "title" in class_name:
        return "title"
    elif "abandon" in class_name:
        return "abandon"
    elif "caption" in class_name:
        return "caption"
    elif "footnote" in class_name:
        return "footnote"


def extract_image_elements(
    image: Image.Image,
    model_weights: Union[Path, str] = "doclayout_yolo_docstructbench_imgsz1024.pt",
    confidence_threshold: float = 0.2,
    image_size: int = 1024,
    device: str = "cpu",
) -> None:
    """
    Extract elements from the image and store them in memory.

    Args:
        confidence_threshold: Confidence threshold for detection
        image_size: Image size for prediction
        device: Device to use for inference
    """

    extraction_results = {}
    model_weights = Path(model_weights)
    # Load original image

    model = YOLOv10(model_weights)
    original_width, original_height = image.size

    # Run YOLO prediction
    det_res = model.predict(
        image,
        imgsz=image_size,
        conf=confidence_threshold,
        device=device,
    )

    detection_results = det_res[0]
    boxes = detection_results.boxes
    names = detection_results.names
    name_map = {i: name.replace(" ", "-") for i, name in names.items()}

    if boxes is None or len(boxes) == 0:
        print("No detections found")
        return

    # Initialize storage
    elements = {name: [] for name in name_map.values()}

    # Find captions and footnotes first
    figure_captions = []
    table_footnotes = []
    for i, box in enumerate(boxes.xyxy):
        if "caption" in names[int(boxes.cls[i])]:
            figure_captions.append(box.tolist())
        if "footnote" in names[int(boxes.cls[i])]:
            table_footnotes.append(box.tolist())

    # Process each detection
    class_counts = {name: 0 for name in name_map.values()}

    for i, box in enumerate(boxes.xyxy):
        # Get class information
        cls_id = int(boxes.cls[i])
        confidence = float(boxes.conf[i])
        class_name = name_map[cls_id]

        if "caption" in class_name or "footnote" in class_name:
            continue

        # Extract bounding box coordinates
        x1, y1, x2, y2 = box.tolist()

        # Ensure coordinates are within image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(original_width, int(x2))
        y2 = min(original_height, int(y2))

        # Determine element type
        element_type = get_element_type(class_name)

        # Crop the element
        cropped_element = image.crop((x1, y1, x2, y2))

        # Create element info
        element_info = {
            "image": cropped_element,
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence,
            "class_name": class_name,
            "type": element_type,
            "index": class_counts[class_name],
            "name": f"{class_name}_{class_counts[class_name]:03d}",
            "caption": {"image": None, "key": None},
            "footnote": {"image": None, "key": None},
        }

        # Handle captions for figures, tables, and formulas
        if class_name in ["figure", "isolate_formula", "table"] and figure_captions:
            caption_box = find_nearest_caption(box.tolist(), figure_captions)
            if caption_box:
                cx1, cy1, cx2, cy2 = caption_box
                cropped_caption = image.crop((cx1, cy1, cx2, cy2))
                caption_key = f"{class_name}_{class_counts[class_name]:03d}"
                element_info["caption"]["image"] = cropped_caption
                element_info["caption"]["key"] = caption_key

        # Handle footnotes for tables
        if class_name in ["table"] and table_footnotes:
            footnote_box = find_nearest_caption(box.tolist(), table_footnotes)
            if footnote_box:
                fx1, fy1, fx2, fy2 = footnote_box
                cropped_footnote = image.crop((fx1, fy1, fx2, fy2))
                footnote_key = f"{class_name}_{class_counts[class_name]:03d}"
                element_info["footnote"]["image"] = cropped_footnote
                element_info["footnote"]["key"] = footnote_key

        # Store the cropped element
        elements[class_name].append(element_info)

        class_counts[class_name] += 1

    # Create annotated image
    annotated_frame = detection_results.plot(pil=True, line_width=3, font_size=16)

    # Convert from BGR to RGB if needed
    if isinstance(annotated_frame, np.ndarray):
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        annotated_image = Image.fromarray(annotated_frame)
    else:
        annotated_image = annotated_frame

    extraction_results["annotated_image"] = annotated_image
    extraction_results["elements"] = elements

    return extraction_results


class Page:

    def __init__(
        self,
        image: Image.Image,
        figures: List[Figure] = None,
        tables: List[Table] = None,
        formulas: List[Formula] = None,
        annotated_image: Image.Image = None,
        elements: Dict[str, List[Dict]] = None,
    ):
        self._image = image
        self._figures = figures or []
        self._tables = tables or []
        self._formulas = formulas or []
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
    def elements(self):
        return self._elements

    @property
    def description(self):
        return self.__repr__()

    def to_markdown(self, filepath: Union[str, Path] = None):
        tmp_str = ""
        for fig in self._figures:
            tmp_str += fig.to_markdown()
            tmp_str += "\n\n"
        for table in self._tables:
            tmp_str += table.to_markdown()
            tmp_str += "\n\n"
        for formula in self._formulas:
            tmp_str += formula.to_markdown()
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
        }

    def to_json(self, filepath: Union[str, Path] = None, indent: int = 2):
        if filepath:
            with open(filepath, "w") as f:
                json.dump(self.to_dict(), f, indent=indent)
        return json.dumps(self.to_dict())

    # Class methods remain the same
    @classmethod
    def from_image(
        cls,
        image: Union[str, Path, Image.Image],
        model_weights: Union[Path, str] = "doclayout_yolo_docstructbench_imgsz1024.pt",
        model=llm_processing.MODELS[2],
        generate_config: Dict = None,
    ):
        if isinstance(image, str):
            image = Path(image)

        if isinstance(image, Path):
            image = Image.open(image)

        extraction_results = extract_image_elements(image, model_weights=model_weights)

        async def parse_all_images():
            tasks = []
            for element_type, elements in extraction_results["elements"].items():
                if element_type == "table":
                    class_type = Table
                elif element_type == "formula":
                    class_type = Formula
                elif element_type == "figure":
                    class_type = Figure
                else:
                    continue
                for element in elements:
                    tasks.append(
                        class_type.from_image_async(
                            element["image"],
                            model=model,
                            generate_config=generate_config,
                            caption=element["caption"]["image"],
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

        if not isinstance(results, list):
            results = [results]

        tables = []
        figures = []
        formulas = []
        for result in results:
            if isinstance(result, Table):
                tables.append(result)
            elif isinstance(result, Figure):
                figures.append(result)
            elif isinstance(result, Formula):
                formulas.append(result)
        return cls(
            tables=tables,
            figures=figures,
            formulas=formulas,
            image=image,
            annotated_image=extraction_results.get("annotated_image", None),
            elements=extraction_results.get("elements", {}),
        )
