import json
import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import cv2
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

MODEL = MODELS[1]


def plot_bounding_boxes(im, bounding_boxes):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "pink",
        "purple",
        "brown",
        "gray",
        "beige",
        "turquoise",
        "cyan",
        "magenta",
        "lime",
        "navy",
        "maroon",
        "teal",
        "olive",
        "coral",
        "lavender",
        "violet",
        "gold",
        "silver",
    ]

    # Parsing out the markdown fencing
    # bounding_boxes = parse_json(bounding_boxes)

    # font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(bounding_boxes):
        # Select a color from the list
        color = colors[i % len(colors)]

        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["box_2d"][0] / 1000 * height)
        abs_x1 = int(bounding_box["box_2d"][1] / 1000 * width)
        abs_y2 = int(bounding_box["box_2d"][2] / 1000 * height)
        abs_x2 = int(bounding_box["box_2d"][3] / 1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw the bounding box
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)

        # Draw the text
        if "label" in bounding_box:
            draw.text(
                (abs_x1 + 8, abs_y1 + 6),
                bounding_box["label"],
                fill=color,
            )

    # Display the image
    img.show()


def extract_images_from_bounding_boxes(im, bounding_boxes):
    """
    Extracts cropped images from bounding boxes.

    Args:
        im: The PIL Image object.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.

    Returns:
        List of PIL Image objects cropped from the bounding boxes.
    """
    extracted_images = []
    width, height = im.size

    for bounding_box in bounding_boxes:
        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["box_2d"][0] / 1000 * height)
        abs_x1 = int(bounding_box["box_2d"][1] / 1000 * width)
        abs_y2 = int(bounding_box["box_2d"][2] / 1000 * height)
        abs_x2 = int(bounding_box["box_2d"][3] / 1000 * width)

        # Ensure coordinates are in correct order
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Crop the image using the bounding box coordinates
        cropped_image = im.crop((abs_x1, abs_y1, abs_x2, abs_y2))
        extracted_images.append(cropped_image)

    return extracted_images


# FIGURE_EXTRACT_PROMPT = """The goal is to detect the all of the figures, diagrams, images, tables from this image.

# Follow these intructions:

# - The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.
#     - The box should NOT include the caption or description of the image.
#     - Make sure the box includes the entire figure, diagram, image, or table.
# - The index should be based on the logical order of the images, tables, when reading the text.
# - The caption should be the caption or description of the image, table, or diagram.
# """


# class ImageAnnotation(BaseModel):
#     box_2d: List[int]
#     index: int
#     caption: str


# class TableAnnotation(BaseModel):
#     box_2d: List[int]
#     table: str
#     caption: str


# # class FigureAnnotation(BaseModel):
# #     box_2d: List[int]
# #     index: int
# #     type: Literal["image", "table"]
# #     caption: str


# # class PDFPageFigures(BaseModel):
# #     images: List[ImageAnnotation] = None
# #     tables: List[TableAnnotation] = None

# import enum


# class FigureType(str, enum.Enum):
#     TABLE = "table"
#     IMAGE = "image"


# class FigureAnnotation(BaseModel):
#     box_2d: List[int]
#     index: int
#     type: FigureType
#     caption: str


# class PDFPageFigures(BaseModel):
#     figures: List[FigureAnnotation] = None


# def parse_figures(image_file: Path, model: str = MODELS[0]) -> PDFPageFigures:
#     print(f"Processing :{image_file}")
#     with open(image_file, "rb") as f:
#         image_bytes = f.read()

#     client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

#     responses = client.models.generate_content(
#         model=model,
#         contents=[
#             types.Part.from_bytes(
#                 data=image_bytes,
#                 mime_type="image/png",
#             ),
#             FIGURE_EXTRACT_PROMPT,
#         ],
#         config={
#             "response_mime_type": "application/json",
#             "response_schema": PDFPageFigures,
#         },
#     )
#     return json.loads(responses.text)


# FIGURE_EXTRACT_PROMPT = """Detect the 2d bounding boxes for figures from this image.

# Follow these intructions:

# - Figures can be images, tables, or diagrams.
# - The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.
# - The 2d bounding box should NOT include the caption or description of the figure.
# - The index should be based on the logical order of the figures when reading the text.
# - The caption should be the caption or description of the figure.
# """

# import enum


# class FigureType(str, enum.Enum):
#     TABLE = "table"
#     IMAGE = "image"


# class FigureAnnotation(BaseModel):
#     box_2d: List[int]
#     index: int
#     caption: str


FIGURE_EXTRACT_PROMPT = """Detect the 2d bounding boxes for figures from this image.

Follow these intructions:

- Figures can be images, tables, or diagrams.
- The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000. 
- The 2d bounding box should NOT include the caption or description of the figure.
- The caption should be the caption or description of the figure.
"""

import enum


class FigureType(str, enum.Enum):
    TABLE = "table"
    IMAGE = "image"


class FigureAnnotation(BaseModel):
    box_2d: List[int]
    caption: str


class PDFPageFigures(BaseModel):
    figures: List[FigureAnnotation] = None


def parse_figures(image_file: Path, model: str = MODELS[0]) -> PDFPageFigures:
    print(f"Processing :{image_file}")
    with open(image_file, "rb") as f:
        image_bytes = f.read()

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    responses = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png",
            ),
        ],
        config={
            "system_instruction": FIGURE_EXTRACT_PROMPT,
            "response_mime_type": "application/json",
            "response_schema": PDFPageFigures,
        },
    )
    return json.loads(responses.text)


TEXT_EXTRACT_PROMPT = """Extarct the text from this image. 

Follow these intructions:

- FORMAT THE TEXT IN MARKDOWN.
- Ignore any figures, diagrams, tables, or images.
- The order should be top to bottom, left to right.
- The EQUATIONS SHOULD BE MARKDOWN.
- The summary should be a summary of the content
- The md should be the text in markdown format.
"""


class PDFPageText(BaseModel):
    md: str
    summary: str


def parse_text(image_file: Path, model: str = MODELS[1]) -> PDFPageText:

    print(f"Processing :{image_file}")
    with open(image_file, "rb") as f:
        image_bytes = f.read()

    file_ext = image_file.suffix

    if file_ext == ".jpg":
        mime_type = "image/jpeg"
    elif file_ext == ".png":
        mime_type = "image/png"
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    responses = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            TEXT_EXTRACT_PROMPT,
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": PDFPageText,
        },
    )

    return json.loads(responses.text)


TABLE_EXTRACT_PROMPT = """Extract the infromation from the image of the table into a markdown table.

Follow these intructions:
- Write the table in markdown format.
"""


class TableImage(BaseModel):
    md: str
    summary: str


def parse_table_image(image_file: Path, model: str = MODELS[0]) -> TableImage:

    print(f"Processing :{image_file}")
    with open(image_file, "rb") as f:
        image_bytes = f.read()

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    responses = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png",
            ),
            TABLE_EXTRACT_PROMPT,
        ],
        config={
            "system_instruction": "You are a helpful assistant that extracts text from images.",
            "response_mime_type": "application/json",
            "response_schema": TableImage,
        },
    )

    return json.loads(responses.text)


# if __name__ == "__main__":

#     current_dir = Path(__file__).parent
#     root_dir = current_dir.parent
#     data_dir = root_dir / "data"
#     test_dir = data_dir / "test-dataset"
#     image_dir = test_dir / "images"

#     # parse_images(image_dir=image_dir)
#     page_filename = image_dir / "page_1.png"
#     response_dict = None
#     response_dict = parse_page_text(image_file=page_filename)
#     # print(page)

#     # # Save the page as JSON
#     output_file = page_filename.with_suffix(".json")
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(response_dict, f, indent=2)
#     md_file = page_filename.with_suffix(".md")
#     with open(md_file, "w", encoding="utf-8") as f:
#         f.write(response_dict["md"])


# if __name__ == "__main__":

#     current_dir = Path(__file__).parent
#     root_dir = current_dir.parent
#     data_dir = root_dir / "data"
#     test_dir = data_dir / "test-dataset"
#     image_dir = test_dir / "images"

#     # parse_images(image_dir=image_dir)
#     page_filename = image_dir / "page_13.png"
#     response_dict = None
#     response_dict = parse_figures(image_file=page_filename, model=MODELS[0])
#     # print(page)

#     # # Save the page as JSON
#     output_file = page_filename.with_suffix(".json")

#     if response_dict:
#         with open(output_file, "w", encoding="utf-8") as f:
#             json.dump(response_dict, f, indent=2)

#     im = Image.open(BytesIO(open(page_filename, "rb").read()))

#     with open(output_file, "r", encoding="utf-8") as f:
#         response_dict = json.load(f)
#     plot_bounding_boxes(im, response_dict["images"])
#     plot_bounding_boxes(im, response_dict["tables"])

#     extracted_images = extract_images_from_bounding_boxes(im, response_dict["images"])
#     # Create output directory for extracted images
#     image_dir = image_dir / "page_2" / "images"
#     image_dir.mkdir(exist_ok=True, parents=True)

#     for i, image in enumerate(extracted_images):
#         output_path = image_dir / f"image_{i}.png"
#         image.save(output_path)
#         print(f"Saved extracted image to: {output_path}")

#     extracted_tables = extract_images_from_bounding_boxes(im, response_dict["tables"])
#     table_dir = image_dir / "tables"
#     table_dir.mkdir(exist_ok=True, parents=True)

#     for i, table in enumerate(extracted_tables):
#         output_path = table_dir / f"table_{i}.png"
#         table.save(output_path)

#         table_response = parse_table_image(output_path)

#         md_out_path = output_path.with_suffix(".md")
#         table_md = table_response["md"]
#         with open(md_out_path, "w", encoding="utf-8") as f:
#             f.write(table_md)


if __name__ == "__main__":

    current_dir = Path(__file__).parent
    root_dir = current_dir.parent
    data_dir = root_dir / "data"
    test_dir = data_dir / "test-dataset"
    image_dir = test_dir / "images"

    # parse_images(image_dir=image_dir)
    page_filename = image_dir / "page_13.png"

    from doclayout_yolo import YOLOv10

    model = YOLOv10(data_dir / "doclayout_yolo_docstructbench_imgsz1024.pt")

    page_filenames = [
        image_dir / "page_1.png",
        image_dir / "page_2.png",
        image_dir / "page_3.png",
    ]

    det_res = model.predict(
        page_filename,  # Image to predict
        imgsz=1024,  # Prediction image size
        conf=0.2,  # Confidence threshold
        device="cpu",  # Device to use (e.g., 'cuda:0' or 'cpu')
    )

    print(det_res)

    print(det_res[0].boxes)
    print(det_res[0].names)
    annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
    cv2.imwrite(test_dir / "result.jpg", annotated_frame)

    # page_filenames = [
    #     image_dir / "page_1.png",
    #     image_dir / "page_2.png",
    #     image_dir / "page_3.png",
    # ]

    # det_res = model.predict(
    #     page_filenames,  # Image to predict
    #     imgsz=1024,  # Prediction image size
    #     conf=0.2,  # Confidence threshold
    #     device="cuda:0",  # Device to use (e.g., 'cuda:0' or 'cpu')
    # )

    # for i, det_res in enumerate(det_res):
    #     annotated_frame = det_res.plot(pil=True, line_width=5, font_size=20)
    #     cv2.imwrite(test_dir / f"result_{i}.jpg", annotated_frame)

# response_dict = None
# response_dict = parse_figures(image_file=page_filename, model=MODELS[1])
# # print(page)
# # # Save the page as JSON
# output_file = page_filename.with_suffix(".json")
# if response_dict:
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(response_dict, f, indent=2)
# im = Image.open(BytesIO(open(page_filename, "rb").read()))
# with open(output_file, "r", encoding="utf-8") as f:
#     response_dict = json.load(f)
# plot_bounding_boxes(im, response_dict["figures"])
# # plot_bounding_boxes(im, response_dict["tables"])
# extracted_images = extract_images_from_bounding_boxes(im, response_dict["images"])
# # Create output directory for extracted images
# image_dir = image_dir / "page_2" / "images"
# image_dir.mkdir(exist_ok=True, parents=True)
# for i, image in enumerate(extracted_images):
#     output_path = image_dir / f"image_{i}.png"
#     image.save(output_path)
#     print(f"Saved extracted image to: {output_path}")
# extracted_tables = extract_images_from_bounding_boxes(im, response_dict["tables"])
# table_dir = image_dir / "tables"
# table_dir.mkdir(exist_ok=True, parents=True)
# for i, table in enumerate(extracted_tables):
#     output_path = table_dir / f"table_{i}.png"
#     table.save(output_path)
#     table_response = parse_table_image(output_path)
#     md_out_path = output_path.with_suffix(".md")
#     table_md = table_response["md"]
#     with open(md_out_path, "w", encoding="utf-8") as f:
#         f.write(table_md)


# if __name__ == "__main__":
#     current_dir = Path(__file__).parent
#     root_dir = current_dir.parent
#     data_dir = root_dir / "data"
#     test_dir = data_dir / "test-dataset"
#     image_dir = test_dir / "images"

#     # page_filename = test_dir / "result_0.jpg"
#     page_filename = image_dir / "page_1.png"
#     response_dict = parse_text(image_file=page_filename, model=MODELS[2])

#     md_file = page_filename.with_suffix(".md")
#     with open(md_file, "w", encoding="utf-8") as f:
#         f.write(response_dict["md"])
