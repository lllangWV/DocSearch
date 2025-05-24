import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
import PyPDF2
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image

IMAGE_EXTRACTION_PROMPT = """Extarct the text from this image. Also provide a summarization of the image content.
The response should be in the following format:

### Extracted Text:

Extracted text goes here

### Summary:

Summary goes here.
"""


# IMAGE_EXTRACTION_MARKDOWN_PROMPT = """Extarct the text from this image. Follow these intructions:
# 1. Provide a summarization of the image content.
# 2. If there are any mathematical equations, represent them in markdown format.
# 3. If there are any tables, represent them in markdown format.
# 4. If there are any figure, diagram, or images, describe them. Write None if there are no figures, diagrams, or images present in the extracted text.

# The response should be in the following format:

# ### Extracted Text:

# Extracted text goes here

# ### Figure, Diagram, or Image Description:

# Figure number 1: Description goes here
# Diagram number 1: Description goes here
# Figure number 2: Description goes here
# Image number 1: Description goes here
# ... continue for all figures, diagrams, and images

# ### Summary:

# Summary goes here.
# """

IMAGE_EXTRACTION_MARKDOWN_PROMPT = """Extarct the text from this image. Follow these intructions: 
1. Make sure to extract all the text. This can come from abstract, captions, footnotes, etc.
2. Provide a summarization of the image content. 
3. If there are any mathematical equations, represent them in markdown format.

The response should be in the following format:

### Extracted Text:

Extracted text goes here

### Summary:

Summary goes here.
"""


# Function to encode the image
def encode_image(
    image_input: Union[str, Path, Image.Image, np.ndarray], format: str = "PNG"
) -> str:
    """
    Encode different image input types to base64 string.

    Parameters
    ----------
    image_input : Union[str, Path, Image.Image, np.ndarray]
        Input image in various formats:
        - str/Path: Path to image file
        - Image.Image: PIL Image object
        - np.ndarray: Numpy array representing image
    format : str, optional
        Output image format for PIL/numpy inputs, by default "PNG"

    Returns
    -------
    str
        Base64 encoded image string

    Raises
    ------
    ValueError
        If input type is not supported
    TypeError
        If numpy array has invalid shape or dtype
    FileNotFoundError
        If image file path does not exist
    """
    # Handle string/Path input (image file path)
    if isinstance(image_input, (str, Path)):
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Handle PIL Image input
    elif isinstance(image_input, Image.Image):
        img_byte_arr = BytesIO()
        # Convert to RGB if necessary (for formats that don't support transparency)
        if format.upper() in ["JPEG", "JPG"] and image_input.mode in ["RGBA", "LA"]:
            image_input = image_input.convert("RGB")
        image_input.save(img_byte_arr, format=format.upper())
        img_bytes = img_byte_arr.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

    # Handle numpy array input
    elif isinstance(image_input, np.ndarray):
        # Validate array shape
        if image_input.ndim not in [2, 3]:
            raise TypeError(f"Numpy array must be 2D or 3D, got {image_input.ndim}D")

        # Handle different array types
        if image_input.ndim == 2:
            # Grayscale image
            if image_input.dtype != np.uint8:
                # Normalize to 0-255 range if needed
                if image_input.max() <= 1.0:
                    image_input = (image_input * 255).astype(np.uint8)
                else:
                    image_input = image_input.astype(np.uint8)
            img = Image.fromarray(image_input, mode="L")

        elif image_input.ndim == 3:
            # Color image
            if image_input.shape[2] not in [3, 4]:
                raise TypeError(
                    f"3D array must have 3 (RGB) or 4 (RGBA) channels, got {image_input.shape[2]}"
                )

            if image_input.dtype != np.uint8:
                # Normalize to 0-255 range if needed
                if image_input.max() <= 1.0:
                    image_input = (image_input * 255).astype(np.uint8)
                else:
                    image_input = image_input.astype(np.uint8)

            if image_input.shape[2] == 3:
                img = Image.fromarray(image_input, mode="RGB")
            else:  # 4 channels
                img = Image.fromarray(image_input, mode="RGBA")

        # Convert and encode
        img_byte_arr = BytesIO()
        # Convert to RGB if saving as JPEG
        if format.upper() in ["JPEG", "JPG"] and img.mode in ["RGBA", "LA"]:
            img = img.convert("RGB")
        img.save(img_byte_arr, format=format.upper())
        img_bytes = img_byte_arr.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

    else:
        raise ValueError(
            f"Unsupported image input type: {type(image_input)}. "
            f"Supported types: str, Path, PIL.Image.Image, numpy.ndarray"
        )


def extract_text_from_image(
    base64_image: str,
    model: str = "gpt-4o",
    max_tokens: int = 1000,
    image_type: str = "png",
    prompt: str = None,
    cache_results: bool = True,
):
    """
    Extract text from image using OpenAI Vision API.

    Parameters
    ----------
    image_input : Union[str, Path, Image.Image, np.ndarray]
        Input image in various formats:
        - str/Path: Path to image file
        - Image.Image: PIL Image object
        - np.ndarray: Numpy array representing image
    model : str, optional
        OpenAI model to use, by default 'gpt-4o'
    max_tokens : int, optional
        Maximum tokens for response, by default 1000
    image_type : str, optional
        Image type for URL formatting, by default 'png'
    prompt : str, optional
        Custom prompt, by default None (uses IMAGE_EXTRACTION_MARKDOWN_PROMPT)
    cache_results : bool, optional
        Whether to cache results, by default True

    Returns
    -------
    tuple
        Tuple of (prompt, response_message)
    """
    client = OpenAI()

    if prompt is None:
        prompt = IMAGE_EXTRACTION_MARKDOWN_PROMPT

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{base64_image}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens
    )
    data = response.model_dump(mode="python")

    message = data["choices"][0]["message"]["content"]

    prompt_and_response = (prompt, message)

    return prompt_and_response
