from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from PIL import Image

from docsearch.utils.config import MODELS_DIR


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


class PageLayout:
    """Page layout analyzer using YOLO for element detection and extraction."""

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
    caption_types = [
        ElementType.TABLE_CAPTION.value,
        ElementType.FORMULA_CAPTION.value,
        ElementType.FIGURE_CAPTION.value,
    ]
    footnote_types = [
        ElementType.TABLE_FOOTNOTE.value,
    ]

    def __init__(
        self,
        model_weights: Union[Path, str] = "doclayout_yolo_docstructbench_imgsz1024.pt",
        confidence_threshold: float = 0.2,
        image_size: int = 1024,
        device: str = "cpu",
    ):
        """
        Initialize PageLayout analyzer.

        Args:
            model_weights: Path to YOLO model weights
            confidence_threshold: Confidence threshold for detection
            image_size: Image size for prediction
            device: Device to use for inference ('cpu' or 'cuda')
        """
        self.model_weights = get_doclayout_model(model_weights)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device

        # Initialize model
        self.model = YOLOv10(self.model_weights)
        # Results storage
        self._image = None
        self._annotated_image = None
        self._class_counts = {}
        self.results = None

    @property
    def image(self) -> Optional[Image.Image]:
        """Original input image."""
        return self._image

    @property
    def annotated_image(self) -> Optional[Image.Image]:
        """Image with detection annotations."""
        return self._annotated_image

    @property
    def elements(self) -> Dict[ElementType, List[Element]]:
        """Extracted elements organized by element type."""
        return self._elements

    @property
    def figures(self) -> List[Dict]:
        """Extracted figure elements."""
        return self._elements.get(ElementType.FIGURE, [])

    @property
    def tables(self) -> List[Dict]:
        """Extracted table elements."""
        return self._elements.get(ElementType.TABLE, [])

    @property
    def formulas(self) -> List[Dict]:
        """Extracted formula elements."""
        return self._elements.get(ElementType.FORMULA, [])

    @property
    def text(self) -> List[Dict]:
        """Extracted text elements."""
        return self._elements.get(ElementType.TEXT, [])

    @property
    def titles(self) -> List[Dict]:
        """Extracted title elements."""
        return self._elements.get(ElementType.TITLE, [])

    @property
    def abandoned(self) -> List[Dict]:
        """Abandoned/undefined elements."""
        return self._elements.get(ElementType.UNKNOWN, [])

    def extract_elements(self, image: Union[str, Path, Image.Image]) -> "PageLayout":
        """
        Extract elements from the image and store results.

        Args:
            image: PIL Image to analyze

        Returns:
            Self for method chaining
        """
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(Path(image))
        self._image = image

        # Run YOLO prediction
        det_res = self.model.predict(
            self.image,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            device=self.device,
        )

        self.results = det_res[0]
        boxes = self.results.boxes
        class_names = self.results.names
        class_name_map = {i: class_name for i, class_name in class_names.items()}
        element_name_map = {
            i: self.class_element_type_map[class_name]
            for i, class_name in class_name_map.items()
        }

        if boxes is None or len(boxes) == 0:
            print("No detections found")
            self._elements = {
                element_type: [] for element_type in element_name_map.values()
            }
            self._annotated_image = self.image
            return self

        # Process each detection
        elements = self._process_detections(boxes, element_name_map)
        elements = self._match_captions_and_footnotes(elements)
        self._elements = elements
        # Create annotated image
        self._create_annotated_image(self.results)

        return self

    def _process_detections(
        self, boxes, element_name_map
    ) -> Dict[ElementType, List[Element]]:
        """Process each detection and create element info."""
        original_width, original_height = self._image.size
        elements = {element_type: [] for element_type in element_name_map.values()}
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
            cropped_element = self._image.crop((x1, y1, x2, y2))

            # Create element info
            element = Element(
                element_type=element_type,
                confidence=confidence,
                bbox=[x1, y1, x2, y2],
                image=cropped_element,
            )

            # Store the element
            elements[element_type].append(element)

        return elements

    def _match_captions_and_footnotes(self, elements: Dict[ElementType, List[Element]]):
        """Find and store caption and footnote boxes."""
        tables = elements[ElementType.TABLE.value]
        table_captions = elements[ElementType.TABLE_CAPTION.value]
        table_footnotes = elements[ElementType.TABLE_FOOTNOTE.value]

        for i, table in enumerate(tables):
            best_caption_neighbor_element, table_captions = (
                find_nearest_neighbor_element(table, table_captions)
            )
            best_footnote_neighbor_element, table_footnotes = (
                find_nearest_neighbor_element(table, table_footnotes)
            )
            elements[ElementType.TABLE.value][i].caption = best_caption_neighbor_element
            elements[ElementType.TABLE.value][
                i
            ].footnote = best_footnote_neighbor_element

        figures = elements[ElementType.FIGURE.value]
        figure_captions = elements[ElementType.FIGURE_CAPTION.value]
        for i, figure in enumerate(figures):
            best_caption_neighbor_element, figure_captions = (
                find_nearest_neighbor_element(figure, figure_captions)
            )
            elements[ElementType.FIGURE.value][
                i
            ].caption = best_caption_neighbor_element

        formulas = elements[ElementType.FORMULA.value]
        formula_captions = elements[ElementType.FORMULA_CAPTION.value]
        for i, formula in enumerate(formulas):
            best_caption_neighbor_element, formula_captions = (
                find_nearest_neighbor_element(formula, formula_captions)
            )
            elements[ElementType.FORMULA.value][
                i
            ].caption = best_caption_neighbor_element

        return elements

    def _create_annotated_image(self, detection_results):
        """Create annotated image with bounding boxes."""
        annotated_frame = detection_results.plot(pil=True, line_width=3, font_size=16)

        # Convert from BGR to RGB if needed
        if isinstance(annotated_frame, np.ndarray):
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            self._annotated_image = Image.fromarray(annotated_frame)
        else:
            self._annotated_image = annotated_frame

    def to_dict(self) -> Dict:
        """Convert results to dictionary format (for backward compatibility)."""
        return {
            "annotated_image": self.annotated_image,
            "elements": self.elements,
        }

    def save(self, out_dir: Union[str, Path]):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        self._image.save(out_dir / "page.png")
        self._annotated_image.save(out_dir / "page_annotated.png")
        for element_type, elements in self.elements.items():
            if (
                element_type in self.caption_types
                or element_type in self.footnote_types
            ):
                continue
            element_type_dir = out_dir / element_type
            element_type_dir.mkdir(exist_ok=True)
            for i, element in enumerate(elements):
                element.image.save(element_type_dir / f"{element.element_type}_{i}.png")
                if element.caption:
                    element.caption.image.save(
                        element_type_dir / f"{element.caption.element_type}_{i}.png"
                    )
                if element.footnote:
                    element.footnote.image.save(
                        element_type_dir / f"{element.footnote.element_type}_{i}.png"
                    )

    def __repr__(self):
        return (
            f"PageLayout("
            f"model_weights={self.model_weights}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"image_size={self.image_size}, "
            f"device='{self.device}', "
            f"elements_detected={len(self._elements) if self._elements else 0})"
        )


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
