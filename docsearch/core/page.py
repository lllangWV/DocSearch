import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from PIL import Image

from docsearch import llm_processing
from docsearch.utils.config import MODELS_DIR


class ElementType(Enum):
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
class Element:
    """Metadata for an element."""

    element_type: ElementType
    confidence: float
    bbox: List[int]
    image: Image.Image
    caption: Optional["Element"] = None
    footnote: Optional["Element"] = None
    global_sort_index: Optional[int] = None
    type_sort_index: Optional[int] = None

    markdown: str = None
    summary: str = None

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

    def to_dict(self):
        return {
            "element_type": self.element_type,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "caption": self.caption.to_dict() if self.caption else None,
            "footnote": self.footnote.to_dict() if self.footnote else None,
            "global_sort_index": self.global_sort_index,
            "type_sort_index": self.type_sort_index,
        }

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
        if self.element_type == ElementType.FIGURE.value:
            result = await llm_processing.parse_figure(
                self.image, model=model, generate_config=generate_config
            )
        elif self.element_type == ElementType.TABLE.value:
            result = await llm_processing.parse_table(
                self.image, model=model, generate_config=generate_config
            )
        elif self.element_type == ElementType.FORMULA.value:
            result = await llm_processing.parse_formula(
                self.image, model=model, generate_config=generate_config
            )
        elif self.element_type in [
            ElementType.TEXT.value,
            ElementType.TITLE.value,
            ElementType.UNKNOWN.value,
        ]:
            result = await llm_processing.parse_text(
                self.image, model=model, generate_config=generate_config
            )

        else:
            result = await llm_processing.parse_text(
                self.image, model=model, generate_config=generate_config
            )

        # Set parsed content for this element
        self.markdown = result.get("md", None)
        self.summary = result.get("summary", None)

        return result

    async def _parse_caption_or_footnote(self, element, model, generate_config):
        """Helper method to parse caption or footnote element."""
        result = await llm_processing.parse_text(
            element.image, model=model, generate_config=generate_config
        )
        element.markdown = result.get("md", None)
        element.summary = result.get("summary", None)


def get_doclayout_model(
    model_weights: str = "doclayout_yolo_docstructbench_imgsz1024.pt",
):
    model_weights = Path(model_weights)
    if model_weights.exists():
        return model_weights

    model_weights = MODELS_DIR / "doclayout_yolo_docstructbench_imgsz1024.pt"
    model_name = model_weights.name
    model_repo = "juliozhao/PageLayout-YOLO-PageStructBench"
    if (
        not model_weights.exists() or not model_weights.is_file()
    ):  # Check if dir exists and is not empty
        print(
            f"Model not found locally. Downloading from Hugging Face Hub: {model_repo}"
        )
        try:
            hf_hub_download(
                repo_id=model_repo,
                filename=model_name,
                local_dir=MODELS_DIR,
                local_dir_use_symlinks=False,
            )
            print("Model download complete.")
        except Exception as e:
            print(f"Failed to download model: {e}")
            return model_weights

    return model_weights


class Page:

    def __init__(
        self,
        image: Image.Image,
        elements: List[Element],
        annotated_image: Image.Image = None,
    ):
        self._image = image
        self._elements = elements
        self._annotated_image = annotated_image

    def __repr__(self):
        return f"Page(\nimage={self._image}, \nelements={self._elements})"

    def __str__(self):
        return self.to_markdown()

    @property
    def annotated_image(self):
        return self._annotated_image

    @property
    def elements(self):
        return self._elements

    @property
    def elements_by_type(self):
        elements_by_type = {}
        for element in self._elements:
            element_type = element.element_type
            if element_type not in elements_by_type:
                elements_by_type[element_type] = []
            elements_by_type[element_type].append(element)
        return elements_by_type

    @property
    def tables(self):
        return self.elements_by_type[ElementType.TABLE.value]

    @property
    def figures(self):
        return self.elements_by_type[ElementType.FIGURE.value]

    @property
    def formulas(self):
        return self.elements_by_type[ElementType.FORMULA.value]

    @property
    def text(self):
        return self.elements_by_type[ElementType.TEXT.value]

    @property
    def titles(self):
        return self.elements_by_type[ElementType.TITLE.value]

    @property
    def unknown(self):
        return self.elements_by_type[ElementType.UNKNOWN.value]

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
    def description(self):
        return self.__repr__()

    def full_save(self, out_dir: Union[str, Path]):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        self.to_json(out_dir / "page.json")
        self.to_markdown(out_dir / "page.md")
        self._image.save(out_dir / "page.png")
        self._annotated_image.save(out_dir / "page_annotated.png")

    def get_markdown_by_type(
        self,
        include_caption_by_type: Dict[str, bool] = None,
        include_summary_by_type: Dict[str, bool] = None,
        include_footnote_by_type: Dict[str, bool] = None,
    ):
        tmp_str = ""
        tmp_str += "## Text\n\n" if self.text else ""
        for text in self.text:
            tmp_str += text.to_markdown(
                include_caption=include_caption_by_type[text.element_type],
                include_summary=include_summary_by_type[text.element_type],
                include_footnote=include_footnote_by_type[text.element_type],
            )
            tmp_str += "\n\n"

        tmp_str += "## Title\n\n" if self.titles else ""
        for title in self.titles:
            tmp_str += title.to_markdown(
                include_caption=include_caption_by_type[title.element_type],
                include_summary=include_summary_by_type[title.element_type],
                include_footnote=include_footnote_by_type[title.element_type],
            )
            tmp_str += "\n\n"

        tmp_str += "## Figures\n\n" if self.figures else ""
        for fig in self.figures:
            tmp_str += fig.to_markdown(
                include_caption=include_caption_by_type[fig.element_type],
                include_summary=include_summary_by_type[fig.element_type],
                include_footnote=include_footnote_by_type[fig.element_type],
            )
            tmp_str += "\n\n"

        tmp_str += "## Tables\n\n" if self.tables else ""
        for table in self.tables:
            tmp_str += table.to_markdown(
                include_caption=include_caption_by_type[table.element_type],
                include_summary=include_summary_by_type[table.element_type],
                include_footnote=include_footnote_by_type[table.element_type],
            )
            tmp_str += "\n\n"

        tmp_str += "## Formulas\n\n" if self.formulas else ""
        for formula in self.formulas:
            tmp_str += formula.to_markdown(
                include_caption=include_caption_by_type[formula.element_type],
                include_summary=include_summary_by_type[formula.element_type],
                include_footnote=include_footnote_by_type[formula.element_type],
            )
            tmp_str += "\n\n"

        tmp_str += "## Undefined\n\n" if self.unknown else ""
        for undefined in self.unknown:
            tmp_str += undefined.to_markdown(
                include_caption=include_caption_by_type[undefined.element_type],
                include_summary=include_summary_by_type[undefined.element_type],
                include_footnote=include_footnote_by_type[undefined.element_type],
            )
            tmp_str += "\n\n"

        return tmp_str

    def to_markdown(
        self,
        filepath: Union[str, Path] = None,
        by_type: bool = False,
        include_caption_by_type: Dict[str, bool] = None,
        include_footnote_by_type: Dict[str, bool] = None,
        include_summary_by_type: Dict[str, bool] = None,
        include_section_header=True,
    ):
        element_types = []
        for element in self.elements:
            element_types.append(element.element_type)
        element_types = set(element_types)
        if include_caption_by_type is None:
            include_caption_by_type = {
                element_type: True for element_type in element_types
            }
        if include_summary_by_type is None:
            include_summary_by_type = {
                element_type: True for element_type in element_types
            }
            include_summary_by_type[ElementType.TITLE.value] = False
            include_summary_by_type[ElementType.TEXT.value] = False
            include_summary_by_type[ElementType.UNKNOWN.value] = False
            include_summary_by_type[ElementType.FORMULA.value] = False
            include_summary_by_type[ElementType.TABLE.value] = False
        if include_footnote_by_type is None:
            include_footnote_by_type = {
                element_type: True for element_type in element_types
            }

        if include_section_header:
            tmp_str = "# Page\n\n"
        else:
            tmp_str = ""

        if by_type:
            tmp_str += self.get_markdown_by_type(
                include_caption_by_type=include_caption_by_type,
                include_summary_by_type=include_summary_by_type,
                include_footnote_by_type=include_footnote_by_type,
            )
        else:
            for element in self.elements:
                tmp_str += element.to_markdown(
                    include_caption=include_caption_by_type[element.element_type],
                    include_summary=include_summary_by_type[element.element_type],
                    include_footnote=include_footnote_by_type[element.element_type],
                )
                tmp_str += "\n\n"

        if filepath:
            with open(filepath, "w") as f:
                f.write(tmp_str)
        return tmp_str

    def to_dict(self):

        return {
            "markdown": self.to_markdown(),
            "elements": [element.to_dict() for element in self._elements],
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
        elements: List[Element],
        model=llm_processing.MODELS[2],
        generate_config: Dict = None,
    ):
        tasks = []
        for element in elements:
            tasks.append(
                element.parse_content(model=model, generate_config=generate_config)
            )
        results = await asyncio.gather(*tasks)

        return results

    # Class methods remain the same
    @classmethod
    def from_image(
        cls,
        image: Union[str, Path, Image.Image],
        model_weights: Union[Path, str] = "doclayout_yolo_docstructbench_imgsz1024.pt",
        image_size=1024,
        confidence_threshold=0.2,
        device="cpu",
        model=llm_processing.MODELS[2],
        generate_config: Dict = None,
        **kwargs,
    ):
        image = cls._validate_image(image)

        # Use DocLayout class instead of extract_image_elements function
        elements, annotated_image = Page.extract_elements(
            image,
            model_weights=model_weights,
            image_size=image_size,
            confidence_threshold=confidence_threshold,
            device=device,
        )

        # Usually asyncio.run() is used to run an async function, but in a jupyter notbook this does not work.
        # So we need to run it in a separate thread so we can block before returning.
        try:
            asyncio.get_running_loop()  # Triggers RuntimeError if no running event loop
            # Create a separate thread so we can block before returning
            with ThreadPoolExecutor(1) as pool:
                pool.submit(
                    lambda: asyncio.run(
                        cls.parse(
                            elements, model=model, generate_config=generate_config
                        )
                    )
                ).result()
        except RuntimeError:
            asyncio.run(
                cls.parse(elements, model=model, generate_config=generate_config)
            )

        # if not isinstance(results, list):
        #     results = [results]

        return cls(
            image=image,
            elements=elements,
            annotated_image=annotated_image,
        )

    @classmethod
    async def from_image_async(
        cls,
        image: Union[str, Path, Image.Image],
        model_weights: Union[Path, str] = "doclayout_yolo_docstructbench_imgsz1024.pt",
        image_size=1024,
        confidence_threshold=0.2,
        device="cpu",
        model=llm_processing.MODELS[2],
        generate_config: Dict = None,
    ):
        image = cls._validate_image(image)

        # Use DocLayout class instead of extract_image_elements function
        elements, annotated_image = cls.extract_elements(
            image,
            model_weights=model_weights,
            image_size=image_size,
            confidence_threshold=confidence_threshold,
            device=device,
        )

        # Usually asyncio.run() is used to run an async function, but in a jupyter notbook this does not work.
        # So we need to run it in a separate thread so we can block before returning.
        await cls.parse(elements, model=model, generate_config=generate_config)

        return cls(
            image=image,
            elements=elements,
            annotated_image=annotated_image,
        )

    @staticmethod
    def extract_elements(
        image: Union[str, Path, Image.Image],
        **kwargs,
    ):
        """
        Extract elements from the image and store results.

        Args:
            image: PIL Image to analyze
            kwargs:
                model_weights: Path to model weights
                confidence_threshold: Confidence threshold
                image_size: Image size
                device: Device to run the model on

        Returns:
            elements: List of elements
            annotated_image: PIL Image with bounding boxes
        """
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(Path(image))
        image = image

        # Process each detection
        elements, annotated_image = Page._process_detections(
            image,
            model_weights=kwargs.get(
                "model_weights", "doclayout_yolo_docstructbench_imgsz1024.pt"
            ),
            confidence_threshold=kwargs.get("confidence_threshold", 0.2),
            image_size=kwargs.get("image_size", 1024),
            device=kwargs.get("device", "cpu"),
        )
        elements = Page._match_captions_and_footnotes(elements)
        elements = Page._sort_image_elements(image, elements)

        return elements, annotated_image

    @staticmethod
    def _process_detections(
        image,
        model_weights="doclayout_yolo_docstructbench_imgsz1024.pt",
        image_size=1024,
        confidence_threshold=0.2,
        device="cpu",
    ) -> Dict[ElementType, List[Element]]:
        """Process each detection and create element info."""

        model_weights = get_doclayout_model(model_weights)

        model = YOLOv10(model_weights)
        # Run YOLO prediction
        det_res = model.predict(
            image,
            imgsz=image_size,
            conf=confidence_threshold,
            device=device,
        )

        class_element_type_map = {
            "figure": ElementType.FIGURE.value,
            "table": ElementType.TABLE.value,
            "isolate_formula": ElementType.FORMULA.value,
            "plain text": ElementType.TEXT.value,
            "title": ElementType.TITLE.value,
            "table_caption": ElementType.TABLE_CAPTION.value,
            "formula_caption": ElementType.FORMULA_CAPTION.value,
            "figure_caption": ElementType.FIGURE_CAPTION.value,
            "table_footnote": ElementType.TABLE_FOOTNOTE.value,
            "abandon": ElementType.UNKNOWN.value,
        }

        results = det_res[0]
        boxes = results.boxes
        class_names = results.names
        class_name_map = {i: class_name for i, class_name in class_names.items()}
        element_name_map = {
            i: class_element_type_map[class_name]
            for i, class_name in class_name_map.items()
        }

        if boxes is None or len(boxes) == 0:
            print("No detections found")
            elements = []
            annotated_image = image
            return elements, annotated_image

        original_width, original_height = image.size
        # elements = {element_type: [] for element_type in element_name_map.values()}
        elements = []
        for i, box in enumerate(boxes.xyxy):
            # Get class information
            cls_id = int(boxes.cls[i])
            confidence = float(boxes.conf[i])
            element_type = element_name_map[cls_id]

            # Extract and validate bounding box coordinates
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(original_width, int(x2))
            y2 = min(original_height, int(y2))

            # Crop the element
            cropped_element = image.crop((x1, y1, x2, y2))

            # Create element info
            element = Element(
                element_type=element_type,
                confidence=confidence,
                bbox=[x1, y1, x2, y2],
                image=cropped_element,
            )

            # Store the element
            elements.append(element)

        """Create annotated image with bounding boxes."""
        annotated_frame = results.plot(pil=True, line_width=3, font_size=16)

        # Convert from BGR to RGB if needed
        if isinstance(annotated_frame, np.ndarray):
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            annotated_image = Image.fromarray(annotated_frame)
        else:
            annotated_image = annotated_frame
        return elements, annotated_image

    @staticmethod
    def _match_captions_and_footnotes(elements: Dict[ElementType, List[Element]]):
        """Find and store caption and footnote boxes."""

        table_captions = []
        table_footnotes = []
        figure_captions = []
        formula_captions = []

        new_elements = []
        # Separate caption and footnote elements
        for element in elements:
            if element.element_type == ElementType.TABLE_CAPTION.value:
                table_captions.append(element)
            elif element.element_type == ElementType.TABLE_FOOTNOTE.value:
                table_footnotes.append(element)
            elif element.element_type == ElementType.FIGURE_CAPTION.value:
                figure_captions.append(element)
            elif element.element_type == ElementType.FORMULA_CAPTION.value:
                formula_captions.append(element)
            else:
                new_elements.append(element)
        # Match captions and footnotes to elements
        for element in new_elements:
            if element.element_type == ElementType.TABLE.value:
                best_caption_neighbor_element, table_captions = (
                    find_nearest_neighbor_element(element, table_captions)
                )
                best_footnote_neighbor_element, table_footnotes = (
                    find_nearest_neighbor_element(element, table_footnotes)
                )
                element.caption = best_caption_neighbor_element
                element.footnote = best_footnote_neighbor_element
            elif element.element_type == ElementType.FORMULA.value:
                best_caption_neighbor_element, formula_captions = (
                    find_nearest_neighbor_element(element, formula_captions)
                )
            elif element.element_type == ElementType.FIGURE.value:
                best_caption_neighbor_element, figure_captions = (
                    find_nearest_neighbor_element(element, figure_captions)
                )
                element.caption = best_caption_neighbor_element

        return new_elements

    @staticmethod
    def _sort_image_elements(image, elements):
        """Sort elements by logical reading order."""
        sort_indices = find_element_logical_reading_order(elements, image)
        elements = [elements[i] for i in sort_indices]
        return elements


def find_nearest_neighbor_element(
    element: Element,
    neighbor_elements: List[Element],
    max_distance: float = 100,
) -> Optional[Element]:
    """Find the nearest caption to a given element based on proximity and vertical alignment."""

    if len(neighbor_elements) == 0:
        return None, neighbor_elements

    best_neighbor_element = None
    best_score = float("inf")

    element_box = element.bbox

    element_center_x = (element_box[0] + element_box[2]) / 2
    element_left = element_box[0]
    element_right = element_box[2]
    element_top = element_box[1]
    element_bottom = element_box[3]

    for i, neighbor_element in enumerate(neighbor_elements):
        neighbor_element_box = neighbor_element.bbox
        neighbor_element_left = neighbor_element_box[0]
        neighbor_element_right = neighbor_element_box[2]
        neighbor_element_top = neighbor_element_box[1]
        neighbor_element_bottom = neighbor_element_box[3]

        # Calculate vertical distance (prefer captions below the element)
        element_top_neighbor_element_bottom = abs(neighbor_element_bottom - element_top)

        # Calculate vertical distance (prefer captions above the element)
        element_bottom_neighbor_element_top = abs(neighbor_element_top - element_bottom)

        smallest_vertical_distance = min(
            element_top_neighbor_element_bottom, element_bottom_neighbor_element_top
        )
        if smallest_vertical_distance < best_score:
            best_score = smallest_vertical_distance
            best_neighbor_element = neighbor_element
            best_neighbor_element_index = i

    neighbor_elements.pop(best_neighbor_element_index)
    return best_neighbor_element, neighbor_elements


def find_element_logical_reading_order(
    elements: List[Element], image: Image.Image
) -> List[int]:
    """Sort elements by logical reading order: multi-column elements by x then y,
    with full-width elements inserted based on y-coordinate."""

    if not elements:
        return []

    # Calculate center coordinates and width for each element
    original_width, original_height = image.size

    element_info = []
    for i, element in enumerate(elements):
        element_box = element.bbox
        element_center_x = (element_box[0] + element_box[2]) / 2
        element_center_y = (element_box[1] + element_box[3]) / 2
        element_width = element_box[2] - element_box[0]
        element_info.append((i, element_center_x, element_center_y, element_width))

    # Separate elements into full-width and column elements
    full_width_elements = []
    column_elements = []

    for i, center_x, center_y, width in element_info:
        if width > original_width * 0.55:  # If element spans >55% of page width
            full_width_elements.append((i, center_y))
        else:
            column_elements.append((i, center_x, center_y))

    # Sort column elements by x coordinate first, then y coordinate
    column_elements.sort(key=lambda x: (x[1], x[2]))  # Sort by x then y

    # Sort full-width elements by y coordinate
    full_width_elements.sort(key=lambda x: x[1])  # Sort by y

    # Create the final sorted list by interleaving based on y-coordinates
    result_indices = []
    column_idx = 0
    full_width_idx = 0

    while column_idx < len(column_elements) or full_width_idx < len(
        full_width_elements
    ):
        # Check if we should insert a full-width element
        if full_width_idx < len(full_width_elements) and (
            column_idx >= len(column_elements)
            or full_width_elements[full_width_idx][1] <= column_elements[column_idx][2]
        ):
            # Insert full-width element
            result_indices.append(full_width_elements[full_width_idx][0])
            full_width_idx += 1
        else:
            # Insert column element
            result_indices.append(column_elements[column_idx][0])
            column_idx += 1

    return result_indices
