import asyncio
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


# --- Helper for Image Preparation (No Duplication Here) ---
def _prepare_image_data(image_input: Union[Path, Image.Image]) -> Tuple[bytes, str]:
    """Prepares image bytes and mime type from Path or PIL Image."""
    if isinstance(image_input, Path):
        with open(image_input, "rb") as f:
            image_bytes = f.read()
        file_ext = image_input.suffix.lower()
        if file_ext == ".jpg" or file_ext == ".jpeg":
            mime_type = "image/jpeg"
        elif file_ext == ".png":
            mime_type = "image/png"
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    elif isinstance(image_input, Image.Image):
        buffer = io.BytesIO()
        image_input.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        mime_type = "image/png"
    else:
        raise ValueError(
            f"Unsupported input type: {type(image_input)}. Expected Path or PIL Image."
        )
    return image_bytes, mime_type


# --- Core Synchronous API Call Logic (Single Source of Truth) ---
def _make_api_call(
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    response_schema: BaseModel,
    model: str,
) -> Dict:
    """Makes the synchronous API call to Google GenAI."""
    print(f"Making API call (Model: {model})...")
    # client = genai.GenerativeModel(model_name=model)  # Adjusted for current SDK
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
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

    # We need to make sure the response is JSON and load it.
    # Gemini might return ```json ... ```, so we need to extract it.
    try:
        text_response = response.text
        # Basic extraction if wrapped in markdown
        if text_response.strip().startswith("```json"):
            text_response = text_response.strip()[7:-3].strip()

        parsed_json = json.loads(text_response)
        # Optional: Validate with Pydantic
        # validated_data = response_schema.model_validate(parsed_json)
        # return validated_data.model_dump()
        return parsed_json

    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw response: {getattr(response, 'text', 'N/A')}")
        # Return a default/error structure or re-raise
        return {"error": str(e), "raw_text": getattr(response, "text", "N/A")}


# --- Public Synchronous Function ---
def parse_image(
    image_input: Union[Path, Image.Image],
    prompt: str,
    response_schema: BaseModel,
    model: str = MODELS[0],
) -> Dict:
    """Parses an image synchronously."""
    print(f"Processing sync: {image_input}")
    image_bytes, mime_type = _prepare_image_data(image_input)
    return _make_api_call(image_bytes, mime_type, prompt, response_schema, model)


# --- Public Asynchronous Function ---
async def parse_image_async(
    image_input: Union[Path, Image.Image],
    prompt: str,
    response_schema: BaseModel,
    model: str = MODELS[0],
) -> Dict:
    """Parses an image asynchronously."""
    print(f"Processing async: {image_input}")
    image_bytes, mime_type = _prepare_image_data(image_input)

    # Run the blocking API call in an executor
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,  # Use the default thread pool executor
        _make_api_call,
        image_bytes,
        mime_type,
        prompt,
        response_schema,
        model,
    )
    return result


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

    async def parse_text_element_async(
        self, element: Dict, model: str = MODELS[1]
    ) -> PageImageText:
        """
        Async version of parse_text_element for concurrent processing.
        """
        response_dict = await parse_image_async(
            element["image"],
            model=model,
            prompt=TEXT_EXTRACT_PROMPT,
            response_schema=PageImageText,
        )
        return response_dict

    async def parse_table_element_async(
        self, element: Dict, model: str = MODELS[0]
    ) -> TableImage:
        """
        Async version of parse_table_element for concurrent processing.
        """
        response_dict = await parse_image_async(
            element["image"],
            model=model,
            prompt=TABLE_EXTRACT_PROMPT,
            response_schema=TableImage,
        )
        return response_dict

    async def parse_formula_element_async(
        self, element: Dict, model: str = MODELS[0]
    ) -> FormulaImage:
        """
        Async version of parse_formula_element for concurrent processing.
        """
        response_dict = await parse_image_async(
            element["image"],
            model=model,
            prompt=FORMULA_EXTRACT_PROMPT,
            response_schema=FormulaImage,
        )
        return response_dict

    async def parse_figure_element_async(
        self, element: Dict, model: str = MODELS[0]
    ) -> FigureImage:
        """
        Async version of parse_figure_element for concurrent processing.
        """
        response_dict = await parse_image_async(
            element["image"],
            model=model,
            prompt=FIGURE_EXTRACT_PROMPT,
            response_schema=FigureImage,
        )
        return response_dict

    async def parse_all_tables_async(
        self, model: str = MODELS[0], max_concurrent: int = 5
    ) -> None:
        """
        Parse all table elements asynchronously with concurrency control.

        Args:
            model: Model to use for table extraction
            max_concurrent: Maximum number of concurrent API calls
        """
        table_elements = []
        for class_name, class_elements in self.cropped_elements.items():
            if class_name == "table":
                table_elements.extend(class_elements)

        if not table_elements:
            print("No tables found to parse")
            return

        semaphore = asyncio.Semaphore(max_concurrent)

        async def parse_single_table(table_element):
            async with semaphore:
                try:
                    parsed_table_dict = await self.parse_table_element_async(
                        table_element, model=model
                    )
                    for key, value in parsed_table_dict.items():
                        table_element[key] = value
                    print(f"✓ Parsed table {table_element['index']}")
                except Exception as e:
                    print(f"✗ Error parsing table {table_element['index']}: {e}")

        # Create tasks for all tables
        tasks = [parse_single_table(element) for element in table_elements]

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)
        print(f"Completed parsing {len(table_elements)} tables")

    async def parse_all_figures_async(
        self, model: str = MODELS[0], max_concurrent: int = 5
    ) -> None:
        """
        Parse all figure elements asynchronously with concurrency control.

        Args:
            model: Model to use for figure extraction
            max_concurrent: Maximum number of concurrent API calls
        """
        figure_elements = []
        for class_name, class_elements in self.cropped_elements.items():
            if class_name == "figure":
                figure_elements.extend(class_elements)

        if not figure_elements:
            print("No figures found to parse")
            return

        semaphore = asyncio.Semaphore(max_concurrent)

        async def parse_single_figure(figure_element):
            async with semaphore:
                try:
                    parsed_figure_dict = await self.parse_figure_element_async(
                        figure_element, model=model
                    )
                    for key, value in parsed_figure_dict.items():
                        figure_element[key] = value
                    print(f"✓ Parsed figure {figure_element['index']}")
                except Exception as e:
                    print(f"✗ Error parsing figure {figure_element['index']}: {e}")

        # Create tasks for all figures
        tasks = [parse_single_figure(element) for element in figure_elements]

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)
        print(f"Completed parsing {len(figure_elements)} figures")

    async def parse_all_formulas_async(
        self, model: str = MODELS[0], max_concurrent: int = 5
    ) -> None:
        """
        Parse all formula elements asynchronously with concurrency control.

        Args:
            model: Model to use for formula extraction
            max_concurrent: Maximum number of concurrent API calls
        """
        formula_elements = []
        for class_name, class_elements in self.cropped_elements.items():
            if class_name == "isolate_formula":
                formula_elements.extend(class_elements)

        if not formula_elements:
            print("No formulas found to parse")
            return

        semaphore = asyncio.Semaphore(max_concurrent)

        async def parse_single_formula(formula_element):
            async with semaphore:
                try:
                    parsed_formula_dict = await self.parse_formula_element_async(
                        formula_element, model=model
                    )
                    for key, value in parsed_formula_dict.items():
                        formula_element[key] = value
                    print(f"✓ Parsed formula {formula_element['index']}")
                except Exception as e:
                    print(f"✗ Error parsing formula {formula_element['index']}: {e}")

        # Create tasks for all formulas
        tasks = [parse_single_formula(element) for element in formula_elements]

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)
        print(f"Completed parsing {len(formula_elements)} formulas")

    async def parse_all_text_async(
        self, model: str = MODELS[1], max_concurrent: int = 5
    ) -> None:
        """
        Parse all text elements asynchronously with concurrency control.

        Args:
            model: Model to use for text extraction
            max_concurrent: Maximum number of concurrent API calls
        """
        text_elements = []
        for class_name, class_elements in self.cropped_elements.items():
            if class_name in ["plain-text", "title"]:
                text_elements.extend(class_elements)

        if not text_elements:
            print("No text elements found to parse")
            return

        semaphore = asyncio.Semaphore(max_concurrent)

        async def parse_single_text(text_element):
            async with semaphore:
                try:
                    parsed_text_dict = await self.parse_text_element_async(
                        text_element, model=model
                    )
                    for key, value in parsed_text_dict.items():
                        text_element[key] = value
                    print(f"✓ Parsed text {text_element['index']}")
                except Exception as e:
                    print(f"✗ Error parsing text {text_element['index']}: {e}")

        # Create tasks for all text elements
        tasks = [parse_single_text(element) for element in text_elements]

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)
        print(f"Completed parsing {len(text_elements)} text elements")

    async def parse_all_elements_async(
        self,
        model: str = MODELS[1],
        text_model: str = MODELS[1],
        max_concurrent: int = 5,
    ) -> None:
        """
        Parse all elements (tables, figures, formulas, text) asynchronously.

        Args:
            model: Model to use for tables, figures, and formulas
            text_model: Model to use for text extraction
            max_concurrent: Maximum number of concurrent API calls
        """
        print("Starting async parsing of all elements...")

        # Parse all element types concurrently
        await asyncio.gather(
            self.parse_all_tables_async(model=model, max_concurrent=max_concurrent),
            self.parse_all_figures_async(model=model, max_concurrent=max_concurrent),
            self.parse_all_formulas_async(model=model, max_concurrent=max_concurrent),
            self.parse_all_text_async(model=text_model, max_concurrent=max_concurrent),
        )

        print("✓ Completed async parsing of all elements")

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

    # Load model
    model_weights = data_dir / "doclayout_yolo_docstructbench_imgsz1024.pt"

    # Example 1: Using the new PageImage class (Synchronous)
    print("=== Using DocumentPageAnalyzer Class (Synchronous) ===")
    page_image = DocumentPageAnalyzer(page_filename, model_weights=model_weights)

    # Extract elements and keep them in memory
    page_image.extract_elements(confidence_threshold=0.2, device="cpu")

    # Synchronous parsing (original way)
    print("\n--- Synchronous Parsing ---")
    import time

    start_time = time.time()

    page_image.parse_all_tables(model=MODELS[1])
    page_image.parse_all_text(model=MODELS[1])
    page_image.parse_all_formulas(model=MODELS[1])
    page_image.parse_all_figures(model=MODELS[1])
    page_image.parse_text(model=MODELS[1])

    sync_time = time.time() - start_time
    print(f"Synchronous parsing completed in {sync_time:.2f} seconds")

    # Access elements in memory
    figures = page_image.figures
    tables = page_image.tables
    formulas = page_image.formulas

    print(
        f"Found {len(figures)} figures, {len(tables)} tables, and {len(formulas)} formulas"
    )
    print(f"Extraction summary: {page_image.extraction_summary}")

    # Example 2: Using async parsing for better performance
    print("\n=== Using DocumentPageAnalyzer Class (Asynchronous) ===")

    # Create a new instance for async testing
    page_image_async = DocumentPageAnalyzer(page_filename, model_weights=model_weights)
    page_image_async.extract_elements(confidence_threshold=0.2, device="cpu")

    async def async_parsing_example():
        print("\n--- Asynchronous Parsing ---")
        start_time = time.time()

        # Parse all elements asynchronously with concurrency control
        await page_image_async.parse_all_elements_async(
            model=MODELS[1],
            text_model=MODELS[1],
            max_concurrent=3,  # Limit concurrent API calls
        )

        async_time = time.time() - start_time
        print(f"Asynchronous parsing completed in {async_time:.2f} seconds")
        print(f"Speed improvement: {sync_time/async_time:.2f}x faster")

        return async_time

    # Run the async example
    async_time = asyncio.run(async_parsing_example())

    # Example 3: Individual async parsing methods
    # print("\n=== Individual Async Parsing Methods ===")

    # page_image_individual = DocumentPageAnalyzer(
    #     page_filename, model_weights=model_weights
    # )
    # page_image_individual.extract_elements(confidence_threshold=0.2, device="cpu")

    # async def individual_async_example():
    #     print("Parsing tables asynchronously...")
    #     await page_image_individual.parse_all_tables_async(
    #         model=MODELS[1], max_concurrent=2
    #     )

    #     print("Parsing figures asynchronously...")
    #     await page_image_individual.parse_all_figures_async(
    #         model=MODELS[1], max_concurrent=2
    #     )

    #     print("Individual async parsing completed")

    # asyncio.run(individual_async_example())

    print("\nSaving elements to disk")
    saved_results = page_image.save_elements(
        output_dir=data_dir / "extracted-minutes-class"
    )
    print(f"Saved elements to: {saved_results['output_directory']}")

    # Save markdown files as before
    out_dir = saved_results["output_directory"]
    table_json = out_dir / "table" / "metadata.json"
    table_md = out_dir / "table" / "tables.md"

    if table_json.exists():
        with open(table_json, "r") as f:
            table_metadata = json.load(f)
        with open(table_md, "w") as f:
            for key, value in table_metadata.items():
                f.write(f"{key}\n")
                f.write(f"{value['md']}\n")
                f.write("\n")

    text_json = out_dir / "plain-text" / "metadata.json"
    text_md = out_dir / "plain-text" / "text.md"

    if text_json.exists():
        with open(text_json, "r") as f:
            text_metadata = json.load(f)
        with open(text_md, "w") as f:
            for key, value in text_metadata.items():
                f.write(f"{key}\n")
                f.write(f"{value['md']}\n")
