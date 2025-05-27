import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from doclayout_yolo import YOLOv10
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageColor, ImageDraw, ImageFont
from pydantic import BaseModel

load_dotenv()

MODELS = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


TEXT_EXTRACT_PROMPT = """Extarct the text from this image. 

Follow these intructions:

- FORMAT THE TEXT IN MARKDOWN.
- Ignore any figures, diagrams, tables, or images.
- The order should be top to bottom, left to right.
- The EQUATIONS SHOULD BE MARKDOWN.
- The summary should be a summary of the content
- The md should be the text in markdown format.
"""


class PageImageText(BaseModel):
    md: str
    summary: str


TABLE_EXTRACT_PROMPT = """Extract the infromation from the image of the table into a markdown table.

Follow these intructions:
- MAKE SURE TO EXTRACT ALL THE INFORMATION FROM THE TABLE
- Write the table in markdown format.
"""


class TableImage(BaseModel):
    md: str
    summary: str


FIGURE_EXTRACT_PROMPT = """Extract the infromation from the image of the figure into a markdown figure.

Follow these intructions:
- MAKE SURE TO EXTRACT ALL THE INFORMATION FROM THE FIGURE
- Write the figure in markdown format.
- The summary should be a summary of the content
- The md should be the text in markdown format.
"""


class FigureImage(BaseModel):
    md: str
    summary: str


FORMULA_EXTRACT_PROMPT = """Extract the infromation from the image of the formula into a markdown formula.

Follow these intructions:
- MAKE SURE TO EXTRACT ALL THE INFORMATION FROM THE FORMULA
- Write the formula in markdown format.
- The summary should be a summary of the content
- The md should be the text in markdown format.
"""


class FormulaImage(BaseModel):
    md: str
    summary: str


def parse_image(
    image_input: Union[Path, Image.Image],
    prompt: str,
    response_schema: BaseModel,
    model: str = MODELS[0],
):

    print(f"Processing: {image_input}")

    if isinstance(image_input, Path):
        # Handle file path input
        with open(image_input, "rb") as f:
            image_bytes = f.read()

        file_ext = image_input.suffix
        if file_ext == ".jpg":
            mime_type = "image/jpeg"
        elif file_ext == ".png":
            mime_type = "image/png"
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")

    elif isinstance(image_input, Image.Image):
        # Handle PIL Image input
        import io

        buffer = io.BytesIO()
        # Save as PNG to ensure consistent format
        image_input.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        mime_type = "image/png"

    else:
        raise ValueError(
            f"Unsupported input type: {type(image_input)}. Expected Path or PIL Image."
        )

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    responses = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            prompt,
        ],
        config={
            "system_instruction": "You are a helpful assistant that extracts text from images.",
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        },
    )

    return json.loads(responses.text)


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


class DocumentPageAnalyzer:
    """
    A class to extract and manage figures, tables, and other elements from a document page image.
    Keeps cropped images in memory and provides methods to save them.
    """

    def __init__(
        self,
        image: Union[Path, Image.Image],
        model_weights: Path = "doclayout_yolo_docstructbench_imgsz1024.pt",
        extract_elements: bool = True,
        **kwargs,
    ):
        """
        Initialize PageImage with an image path and YOLO model.

        Args:
            image_path: Path to the input image
            model_weights: Path to the YOLO model weights
            extract_elements: Whether to extract elements from the image
        """
        self.image = image if isinstance(image, Image.Image) else Image.open(image)
        self.model = YOLOv10(model_weights)
        self.annotated_image = None
        self.detection_results = None

        # Storage for cropped elements
        self.cropped_elements = {}
        self.captions = {}
        self.footnotes = {}
        self.extraction_summary = {}

        # Storage for parsed content
        self.parsed_text = None
        self.tables_parsed = {}

        if extract_elements:
            self.extract_elements(**kwargs)

    @property
    def tables(self) -> List[Dict]:
        return self.get_elements_by_type("table")

    @property
    def figures(self) -> List[Dict]:
        return self.get_elements_by_type("figure")

    @property
    def formulas(self) -> List[Dict]:
        return self.get_elements_by_type("formula")

    def extract_elements(
        self,
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
        if len(self.cropped_elements) > 0:
            print("Elements already extracted")
            return None
        # Load original image

        original_width, original_height = self.image.size

        # Run YOLO prediction
        det_res = self.model.predict(
            self.image,
            imgsz=image_size,
            conf=confidence_threshold,
            device=device,
        )

        self.detection_results = det_res[0]
        result = self.detection_results
        boxes = result.boxes
        names = result.names
        name_map = {i: name.replace(" ", "-") for i, name in names.items()}

        if boxes is None or len(boxes) == 0:
            print("No detections found")
            return

        # Initialize storage
        self.cropped_elements = {name: [] for name in name_map.values()}
        self.captions = {}
        self.footnotes = {}

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
            element_type = self._get_element_type(class_name)

            # Crop the element
            cropped_element = self.image.crop((x1, y1, x2, y2))

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
                    cropped_caption = self.image.crop((cx1, cy1, cx2, cy2))
                    caption_key = f"{class_name}_{class_counts[class_name]:03d}"
                    self.captions[caption_key] = cropped_caption
                    element_info["caption"]["image"] = cropped_caption
                    element_info["caption"]["key"] = caption_key

            # Handle footnotes for tables
            if class_name in ["table"] and table_footnotes:
                footnote_box = find_nearest_caption(box.tolist(), table_footnotes)
                if footnote_box:
                    fx1, fy1, fx2, fy2 = footnote_box
                    cropped_footnote = self.image.crop((fx1, fy1, fx2, fy2))
                    footnote_key = f"{class_name}_{class_counts[class_name]:03d}"
                    self.footnotes[footnote_key] = cropped_footnote
                    element_info["footnote"]["image"] = cropped_footnote
                    element_info["footnote"]["key"] = footnote_key

            # Store the cropped element
            self.cropped_elements[class_name].append(element_info)

            class_counts[class_name] += 1

        # Create annotated image
        annotated_frame = result.plot(pil=True, line_width=3, font_size=16)

        # Convert from BGR to RGB if needed
        if isinstance(annotated_frame, np.ndarray):
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            self.annotated_image = Image.fromarray(annotated_frame)
        else:
            self.annotated_image = annotated_frame

        # Create extraction summary
        self.extraction_summary = self._create_summary()

    def _get_element_type(self, class_name: str) -> str:
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

    def _create_summary(self) -> Dict[str, int]:
        """Create extraction summary."""
        summary = {}
        for class_name, elements in self.cropped_elements.items():
            element_type = self._get_element_type(class_name)
            key = f"{element_type}s_found"
            summary[key] = summary.get(key, 0) + len(elements)
        return summary

    def get_elements_by_type(self, element_type: str) -> List[Dict]:
        """
        Get all elements of a specific type.

        Args:
            element_type: Type of elements to retrieve ('figure', 'table', 'formula', etc.)

        Returns:
            List of element dictionaries containing image and metadata
        """
        elements = []
        for class_name, class_elements in self.cropped_elements.items():
            for element in class_elements:
                if element["type"] == element_type:
                    elements.append(element)
        return elements

    def get_elements_by_class(self, class_name: str) -> List[Dict]:
        """
        Get all elements of a specific class.
        """
        return self.cropped_elements[class_name]

    def parse_text_element(
        self, element: Dict, model: str = MODELS[1]
    ) -> PageImageText:
        """
        Parse text from a specific element using LLM.

        Args:
            element: Element dictionary containing the image
            model: Model to use for text extraction

        Returns:
            PageImageText object with markdown and summary
        """
        response_dict = parse_image(
            element["image"],
            model=model,
            prompt=TEXT_EXTRACT_PROMPT,
            response_schema=PageImageText,
        )
        return response_dict

    def parse_table_element(self, element: Dict, model: str = MODELS[0]) -> TableImage:
        """
        Parse table from a specific element using LLM.

        Args:
            element: Element dictionary containing the table image
            model: Model to use for table extraction

        Returns:
            TableImage object with markdown table and summary
        """
        response_dict = parse_image(
            element["image"],
            model=model,
            prompt=TABLE_EXTRACT_PROMPT,
            response_schema=TableImage,
        )
        return response_dict

    def parse_formula_element(
        self, element: Dict, model: str = MODELS[0]
    ) -> FormulaImage:
        """
        Parse formula from a specific element using LLM.

        Args:
            element: Element dictionary containing the formula image
            model: Model to use for formula extraction
        """
        response_dict = parse_image(
            element["image"],
            model=model,
            prompt=FORMULA_EXTRACT_PROMPT,
            response_schema=FormulaImage,
        )
        return response_dict

    def parse_figure_element(
        self, element: Dict, model: str = MODELS[0]
    ) -> FigureImage:
        """
        Parse figure from a specific element using LLM.
        """
        response_dict = parse_image(
            element["image"],
            model=model,
            prompt=FIGURE_EXTRACT_PROMPT,
            response_schema=FigureImage,
        )
        return response_dict

    def parse_all_tables(self, model: str = MODELS[0]) -> List[Dict]:
        """
        Parse all table elements and return results with metadata.

        Args:
            model: Model to use for table extraction

        Returns:
            List of dictionaries containing original element info and parsed results
        """
        for class_name, class_elements in self.cropped_elements.items():
            if class_name != "table":
                continue
            for table_element in class_elements:
                try:
                    parsed_table_dict = self.parse_table_element(
                        table_element, model=model
                    )
                    for key, value in parsed_table_dict.items():
                        table_element[key] = value
                except Exception as e:
                    print(f"Error parsing table {table_element['index']}: {e}")

    def parse_all_figures(self, model: str = MODELS[0]) -> List[Dict]:
        """
        Parse all figure elements and return results with metadata.

        Args:
            model: Model to use for table extraction

        Returns:
            List of dictionaries containing original element info and parsed results
        """
        for class_name, class_elements in self.cropped_elements.items():
            if class_name != "figure":
                continue
            for figure_element in class_elements:
                try:
                    parsed_figure_dict = self.parse_figure_element(
                        figure_element, model=model
                    )
                    for key, value in parsed_figure_dict.items():
                        figure_element[key] = value
                except Exception as e:
                    print(f"Error parsing figure {figure_element['index']}: {e}")

    def parse_all_formulas(self, model: str = MODELS[0]) -> List[Dict]:
        """
        Parse all formula elements and return results with metadata.

        Args:
            model: Model to use for formula extraction
        """
        for class_name, class_elements in self.cropped_elements.items():
            if class_name != "isolate_formula":
                continue
            for formula_element in class_elements:
                try:
                    parsed_formula_dict = self.parse_formula_element(
                        formula_element, model=model
                    )
                    for key, value in parsed_formula_dict.items():
                        formula_element[key] = value
                except Exception as e:
                    print(f"Error parsing formula {formula_element['index']}: {e}")

    def parse_all_text(self, model: str = MODELS[1]) -> List[Dict]:
        """
        Parse text from all elements of specified types.

        Args:
            element_types: List of element types to parse (default: ['text', 'title'])
            model: Model to use for text extraction

        Returns:
            List of dictionaries containing original element info and parsed results
        """
        for class_name, class_elements in self.cropped_elements.items():
            if class_name != "plain-text" and class_name != "title":
                continue
            for text_element in class_elements:
                try:
                    parsed_text_dict = self.parse_text_element(
                        text_element, model=model
                    )
                    for key, value in parsed_text_dict.items():
                        text_element[key] = value
                except Exception as e:
                    print(f"Error parsing text {text_element['index']}: {e}")
        return None

    def parse_text(self, model: str = MODELS[1]) -> PageImageText:
        """
        Parse text from the original image using LLM and optionally store in memory.

        Args:
            model: Model to use for text extraction

        Returns:
            PageImageText object with markdown and summary
        """
        response_dict = parse_image(
            self.image,
            model=model,
            prompt=TEXT_EXTRACT_PROMPT,
            response_schema=PageImageText,
        )

        self.parsed_text = response_dict

        return response_dict

    def save_elements(self, output_dir: Union[Path, str]) -> Dict[str, List[Path]]:
        """
        Save all cropped elements to disk in the same structure as extract_info.

        Args:
            output_dir: Directory to save extracted elements (default: same as image directory)

        Returns:
            Dictionary with paths to saved elements
        """
        if not self.cropped_elements:
            raise ValueError("No elements extracted. Call extract_elements() first.")

        output_dir = Path(output_dir)

        # Create output folders
        image_name = output_dir.stem
        base_output_dir = output_dir / image_name

        saved_paths = {"figures": [], "tables": [], "formula_captions": []}

        # Save cropped elements
        for class_name, elements in self.cropped_elements.items():
            if not elements:
                continue

            if "caption" in class_name or "footnote" in class_name:
                continue

            element_type = self._get_element_type(class_name)

            # Create directories
            class_dir = base_output_dir / class_name.replace(" ", "-")
            class_dir.mkdir(parents=True, exist_ok=True)

            element_metadata = {}

            for element in elements:
                filename = element["name"]

                element_metadata[filename] = {
                    "bbox": element["bbox"],
                    "confidence": element["confidence"],
                    "class_name": element["class_name"],
                    "type": element["type"],
                    "index": element["index"],
                    "name": element["name"],
                    "md": element.get("md", None),
                    "summary": element.get("summary", None),
                }

                # Save to both class-specific and type-specific directories
                class_path = class_dir / f"{filename}.png"

                element["image"].save(class_path)

                caption = element["caption"]
                caption_key = caption.get("key", None)
                caption_image = caption.get("image", None)
                if caption_key and caption_image:
                    caption_file_name = f"{filename}_caption.png"
                    caption_image.save(class_dir / caption_file_name)

                footnote = element["footnote"]
                footnote_key = footnote.get("key", None)
                footnote_image = footnote.get("image", None)
                if footnote_key and footnote_image:
                    footnote_file_name = f"{filename}_footnote.png"
                    footnote_image.save(class_dir / footnote_file_name)

            with open(class_dir / "metadata.json", "w") as f:
                json.dump(element_metadata, f)

        # Save annotated image
        if self.annotated_image:
            annotated_path = base_output_dir / f"{image_name}_annotated.png"
            self.annotated_image.save(annotated_path)
            saved_paths["annotated_image"] = annotated_path

        if self.image:
            original_path = base_output_dir / f"{image_name}.png"
            self.image.save(original_path)
            if self.parsed_text:
                original_json_path = base_output_dir / f"{image_name}.json"
                with open(original_json_path, "w") as f:
                    json.dump(self.parsed_text, f)
            saved_paths["original_image"] = original_path

        saved_paths["output_directory"] = base_output_dir
        saved_paths["extraction_summary"] = self.extraction_summary

        with open(base_output_dir / "extraction_summary.json", "w") as f:
            json.dump(self.extraction_summary, f)

        return saved_paths


if __name__ == "__main__":
    # Example usage
    current_dir = Path(__file__).parent
    root_dir = current_dir.parent
    data_dir = root_dir / "data"
    test_dir = data_dir / "test-dataset"
    image_dir = test_dir / "images"

    page_filename = image_dir / "page_13.png"

    # minutes_dir = data_dir / "minutes"
    # interim_dir = minutes_dir / "interim"
    # december_dir = interim_dir / "december-5-2013-emergency-meeting-minutes"

    # minute_dir = interim_dir / "august-22-2023-bog-minutes-with-attachments"
    # image_dir = minute_dir / "images"

    # # Load model
    model_weights = data_dir / "doclayout_yolo_docstructbench_imgsz1024.pt"

    # # Process a single image
    # page_filename = image_dir / "page_1.png"

    # # page_filename = image_dir / "page_1.png"
    # page_filename = image_dir / "page_26.png"

    # # Example 1: Using the original extract_info function
    # print("=== Extracting Figures and Tables (Original Function) ===")
    # results = extract_info(
    #     image_path=page_filename,
    #     model=model,
    #     output_dir=data_dir / "extracted-minutes",
    #     confidence_threshold=0.2,
    #     device="cpu",
    # )

    # Example 2: Using the new PageImage class
    print("\n=== Using PageImage Class ===")
    page_image = DocumentPageAnalyzer(page_filename, model_weights=model_weights)

    # Extract elements and keep them in memory
    page_image.extract_elements(confidence_threshold=0.2, device="cpu")
    page_image.parse_all_tables(model=MODELS[1])
    page_image.parse_all_text(model=MODELS[1])
    page_image.parse_all_formulas(model=MODELS[1])
    page_image.parse_all_figures(model=MODELS[1])
    page_image.parse_text(model=MODELS[1])

    # Access elements in memory
    figures = page_image.figures
    tables = page_image.tables
    formulas = page_image.formulas

    print(figures)

    print(f"Found {len(figures)} figures and {len(tables)} tables in memory")
    print(f"Extraction summary: {page_image.extraction_summary}")

    print("Saving elements to disk")
    saved_results = page_image.save_elements(
        output_dir=data_dir / "extracted-minutes-class"
    )
    print(f"Saved elements to: {saved_results['output_directory']}")

    out_dir = saved_results["output_directory"]
    table_json = out_dir / "table" / "metadata.json"
    table_md = out_dir / "table" / "tables.md"
    with open(table_json, "r") as f:
        table_metadata = json.load(f)
    with open(table_md, "w") as f:
        for key, value in table_metadata.items():
            f.write(f"{key}\n")
            f.write(f"{value['md']}\n")
            f.write("\n")

    text_json = out_dir / "plain-text" / "metadata.json"
    text_md = out_dir / "plain-text" / "text.md"
    with open(text_json, "r") as f:
        text_metadata = json.load(f)
    with open(text_md, "w") as f:
        for key, value in text_metadata.items():
            f.write(f"{key}\n")
            f.write(f"{value['md']}\n")
    # # Parse original image text and store in memory
    # print("\n=== Parsing Original Image Text ===")
    # original_text = page_image.parse_original_text()
    # print(f"Original text parsed and stored in memory")
    # print(f"Summary: {original_text['summary']}")
