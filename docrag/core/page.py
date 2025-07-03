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
from docrag.core.page_element import PageElement, PageElementType
from docrag.core.utils import pil_image_to_bytes
from docrag.utils.config import MODELS_DIR

logger = logging.getLogger(__name__)



@dataclass
class SerializationConfig:
    include_page_image: bool = True
    include_annotated_image: bool = True
    include_element_images: bool = True
    include_captions: bool = True
    include_footnotes: bool = True
    image_as_base64: bool = True

class Page:

    def __init__(
        self,
        image: Image.Image,
        elements: List[PageElement],
        annotated_image: Image.Image = None,
        page_id: int = 0,
    ):
        self._image = image
        self._elements = elements
        self._annotated_image = annotated_image
        self._page_id = page_id

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
        elements_by_type = {
            PageElementType.FIGURE.value: [],
            PageElementType.TABLE.value: [],
            PageElementType.FORMULA.value: [],
            PageElementType.TEXT.value: [],
            PageElementType.TITLE.value: [],
            PageElementType.UNKNOWN.value: [],
        }
        for element in self._elements:
            element_type = element.element_type
            elements_by_type[element_type].append(element)
        return elements_by_type

    @property
    def tables(self):
        return self.elements_by_type[PageElementType.TABLE.value]

    @property
    def figures(self):
        return self.elements_by_type[PageElementType.FIGURE.value]

    @property
    def formulas(self):
        return self.elements_by_type[PageElementType.FORMULA.value]

    @property
    def text(self):
        return self.elements_by_type[PageElementType.TEXT.value]

    @property
    def titles(self):
        return self.elements_by_type[PageElementType.TITLE.value]

    @property
    def unknown(self):
        return self.elements_by_type[PageElementType.UNKNOWN.value]

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

    def full_save(self, out_dir: Union[str, Path], **kwargs):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        self.to_json(out_dir / "page.json", **kwargs)
        self.to_markdown(out_dir / "page.md", **kwargs)
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
        encoding: str = "utf-8",
        **kwargs,
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
            include_summary_by_type[PageElementType.TITLE.value] = False
            include_summary_by_type[PageElementType.TEXT.value] = False
            include_summary_by_type[PageElementType.UNKNOWN.value] = False
            include_summary_by_type[PageElementType.FORMULA.value] = False
            include_summary_by_type[PageElementType.TABLE.value] = False
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
            with open(filepath, "w", encoding=encoding) as f:
                f.write(tmp_str)
        return tmp_str

    @staticmethod
    def get_pyarrow_struct(serialization_config: SerializationConfig):
        element_struct = PageElement.get_pyarrow_struct(
            include_image=serialization_config.include_element_images,
            include_captions=serialization_config.include_captions,
            include_footnotes=serialization_config.include_footnotes,
        )

        page_struct = {
            "markdown": pa.string(),
            "elements": pa.list_(element_struct),
            "page_id": pa.int32(),
        }
        if serialization_config.include_page_image:
            page_struct["image"] = pa.binary()
        if serialization_config.include_annotated_image:
            page_struct["annotated_image"] = pa.binary()

        return pa.struct(page_struct)

    def to_pyarrow(self, filepath: Union[str, Path] = None, 
                   serialization_config: SerializationConfig = SerializationConfig(),
                   ):
        page_struct = Page.get_pyarrow_struct(serialization_config=serialization_config)
        data = [self.to_dict(serialization_config=serialization_config)
                ]
        page_schema = pa.schema(page_struct)
        table = pa.Table.from_pylist(data, schema=page_schema)

        if filepath:
            pq.write_table(table, filepath)
        return table

    def to_dict(self, serialization_config: SerializationConfig):
        data = {
            "markdown": self.to_markdown(),
            "elements": [
                element.to_dict(
                    include_image=serialization_config.include_element_images, image_as_base64=serialization_config.image_as_base64
                )
                for element in self._elements
            ],
            "page_id": self._page_id,
        }
        
        if serialization_config.include_page_image:
            data["image"] = pil_image_to_bytes(self._image) if serialization_config.image_as_base64 else self._image
        
        if serialization_config.include_annotated_image:
            data["annotated_image"] = pil_image_to_bytes(self._annotated_image) if serialization_config.image_as_base64 else self._annotated_image
     
        return data

    def to_json(
        self,
        filepath: Union[str, Path] = None,
        indent: int = 2,
        encoding: str = "utf-8",
        **kwargs,
    ):
        if filepath:
            with open(filepath, "w", encoding=encoding) as f:
                json.dump(
                    self.to_dict(
                        include_images=kwargs.get("include_images", False),
                        image_as_base64=kwargs.get("image_as_base64", False),
                    ),
                    f,
                    indent=indent,
                )
        return json.dumps(
            self.to_dict(
                include_images=kwargs.get("include_images", False),
                image_as_base64=kwargs.get("image_as_base64", False),
            )
        )

    @classmethod
    def from_dict(cls, data: Dict):
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        if not "elements" in data:
            raise ValueError("PageElements not found in data")
        elements = [PageElement.from_dict(element) for element in data["elements"]]
        image = None
        annotated_image = None
        if "image" in data:
            image = Image.open(io.BytesIO(data["image"]))
        if "annotated_image" in data:
            annotated_image = Image.open(io.BytesIO(data["annotated_image"]))
        return cls(elements=elements, image=image, annotated_image=annotated_image)

    @classmethod
    def from_parquet(cls, filepath: Union[str, Path]):
        table = pq.read_table(filepath)
        data = table.to_pandas().to_dict(orient="records")
        return cls.from_dict(data[0])

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
        elements: List[PageElement],
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
        page_id: int = None,
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
            page_id=page_id,
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
    ) -> Dict[PageElementType, List[PageElement]]:
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
            "figure": PageElementType.FIGURE.value,
            "table": PageElementType.TABLE.value,
            "isolate_formula": PageElementType.FORMULA.value,
            "plain text": PageElementType.TEXT.value,
            "title": PageElementType.TITLE.value,
            "table_caption": PageElementType.TABLE_CAPTION.value,
            "formula_caption": PageElementType.FORMULA_CAPTION.value,
            "figure_caption": PageElementType.FIGURE_CAPTION.value,
            "table_footnote": PageElementType.TABLE_FOOTNOTE.value,
            "abandon": PageElementType.UNKNOWN.value,
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
            logger.info("No detections found")
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
            element = PageElement(
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
    def _match_captions_and_footnotes(elements: Dict[PageElementType, List[PageElement]]):
        """Find and store caption and footnote boxes."""

        table_captions = []
        table_footnotes = []
        figure_captions = []
        formula_captions = []

        new_elements = []
        # Separate caption and footnote elements
        for element in elements:
            if element.element_type == PageElementType.TABLE_CAPTION.value:
                table_captions.append(element)
            elif element.element_type == PageElementType.TABLE_FOOTNOTE.value:
                table_footnotes.append(element)
            elif element.element_type == PageElementType.FIGURE_CAPTION.value:
                figure_captions.append(element)
            elif element.element_type == PageElementType.FORMULA_CAPTION.value:
                formula_captions.append(element)
            else:
                new_elements.append(element)
        # Match captions and footnotes to elements
        for element in new_elements:
            if element.element_type == PageElementType.TABLE.value:
                best_caption_neighbor_element, table_captions = (
                    find_nearest_neighbor_element(element, table_captions)
                )
                best_footnote_neighbor_element, table_footnotes = (
                    find_nearest_neighbor_element(element, table_footnotes)
                )
                element.caption = best_caption_neighbor_element
                element.footnote = best_footnote_neighbor_element
            elif element.element_type == PageElementType.FORMULA.value:
                best_caption_neighbor_element, formula_captions = (
                    find_nearest_neighbor_element(element, formula_captions)
                )
            elif element.element_type == PageElementType.FIGURE.value:
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
    element: PageElement,
    neighbor_elements: List[PageElement],
    max_distance: float = 100,
) -> Optional[PageElement]:
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




def get_doclayout_model(
    model_weights: str = "doclayout_yolo_docstructbench_imgsz1024.pt",
):
    model_weights = Path(model_weights)
    if model_weights.exists():
        return model_weights

    model_weights = MODELS_DIR / "doclayout_yolo_docstructbench_imgsz1024.pt"
    model_name = model_weights.name
    model_repo = "juliozhao/DocLayout-YOLO-DocStructBench"
    
    logger.info(f"Model repo: {model_repo}")
    logger.info(f"Model name: {model_name}")
    if (
        not model_weights.exists() or not model_weights.is_file()
    ):  # Check if dir exists and is not empty
        logger.info(
            f"Model not found locally. Downloading from Hugging Face Hub: {model_repo}"
        )
        try:
            hf_hub_download(
                repo_id=model_repo,
                filename=model_name,
                local_dir=MODELS_DIR,
            )
            logger.info("Model download complete.")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return model_weights

    return model_weights

def projection_by_bboxes(boxes: np.array, axis: int) -> np.ndarray:
    """
    Args:
        boxes: [N, 4]
        axis: 

    Returns:
        1D 

    """
    assert axis in [0, 1]
    length = np.max(boxes[:, axis::2])
    res = np.zeros(length, dtype=int)
    # TODO: how to remove for loop?
    for start, end in boxes[:, axis::2]:
        res[start:end] += 1
    return res




def split_projection_profile(arr_values: np.array, min_value: float, min_gap: float):
    """Split projection profile:

    ```
                              ┌──┐
         arr_values           │  │       ┌─┐───
             ┌──┐             │  │       │ │ |
             │  │             │  │ ┌───┐ │ │min_value
             │  │<- min_gap ->│  │ │   │ │ │ |
         ────┴──┴─────────────┴──┴─┴───┴─┴─┴─┴───
         0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    ```

    Args:
        arr_values (np.array): 1-d array representing the projection profile.
        min_value (float): Ignore the profile if `arr_value` is less than `min_value`.
        min_gap (float): Ignore the gap if less than this value.

    Returns:
        tuple: Start indexes and end indexes of split groups.
    """
    # all indexes with projection height exceeding the threshold
    arr_index = np.where(arr_values > min_value)[0]
    if not len(arr_index):
        return

    # find zero intervals between adjacent projections
    # |  |                    ||
    # ||||<- zero-interval -> |||||
    arr_diff = arr_index[1:] - arr_index[0:-1]
    arr_diff_index = np.where(arr_diff > min_gap)[0]
    arr_zero_intvl_start = arr_index[arr_diff_index]
    arr_zero_intvl_end = arr_index[arr_diff_index + 1]

    # convert to index of projection range:
    # the start index of zero interval is the end index of projection
    arr_start = np.insert(arr_zero_intvl_end, 0, arr_index[0])
    arr_end = np.append(arr_zero_intvl_start, arr_index[-1])
    arr_end += 1  # end index will be excluded as index slice

    return arr_start, arr_end

def recursive_xy_cut(boxes: np.ndarray, indices: List[int], res: List[int]):
    """

    Args:
        boxes: (N, 4)
        indices: 
        res: 

    """

    assert len(boxes) == len(indices)

    _indices = boxes[:, 1].argsort()
    y_sorted_boxes = boxes[_indices]
    y_sorted_indices = indices[_indices]



    y_projection = projection_by_bboxes(boxes=y_sorted_boxes, axis=1)
    pos_y = split_projection_profile(y_projection, 0, 1)
    if not pos_y:
        return

    arr_y0, arr_y1 = pos_y
    for r0, r1 in zip(arr_y0, arr_y1):

        _indices = (r0 <= y_sorted_boxes[:, 1]) & (y_sorted_boxes[:, 1] < r1)

        y_sorted_boxes_chunk = y_sorted_boxes[_indices]
        y_sorted_indices_chunk = y_sorted_indices[_indices]

        _indices = y_sorted_boxes_chunk[:, 0].argsort()
        x_sorted_boxes_chunk = y_sorted_boxes_chunk[_indices]
        x_sorted_indices_chunk = y_sorted_indices_chunk[_indices]


        x_projection = projection_by_bboxes(boxes=x_sorted_boxes_chunk, axis=0)
        pos_x = split_projection_profile(x_projection, 0, 1)
        if not pos_x:
            continue

        arr_x0, arr_x1 = pos_x
        if len(arr_x0) == 1:
 
            res.extend(x_sorted_indices_chunk)
            continue

        for c0, c1 in zip(arr_x0, arr_x1):
            _indices = (c0 <= x_sorted_boxes_chunk[:, 0]) & (
                x_sorted_boxes_chunk[:, 0] < c1
            )
            recursive_xy_cut(
                x_sorted_boxes_chunk[_indices], x_sorted_indices_chunk[_indices], res
            )



def find_element_logical_reading_order(
    elements: List[PageElement], image: Image.Image, half_width_threshold: float = 0.50
) -> List[int]:
    """Sort elements by logical reading order: multi-column elements by x then y,
    with full-width elements inserted based on y-coordinate."""
    from scipy.cluster.vq import kmeans, vq
    if not elements:
        return []

    # Calculate center coordinates and width for each element
    element_bboxes=[]
    for i, element in enumerate(elements):
        element_bboxes.append(element.bbox)

    res=[]
    recursive_xy_cut(np.array(element_bboxes), np.arange(len(element_bboxes)), res)
    return res


