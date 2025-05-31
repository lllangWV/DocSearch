import asyncio
import inspect
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

load_dotenv()


MODELS = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


LLM_API_RATE_LIMITS = {
    "gemini-2.5-flash-preview-05-20": {"rpm": 1000, "rpd": 10000},
    "gemini-2.5-flash-preview-04-17": {"rpm": 1000, "rpd": 10000},
    "gemini-2.0-flash": {"rpm": 2000, "rpd": 20000},
    "gemini-2.0-flash-lite": {"rpm": 4000, "rpd": 40000},
}

RATE_LIMITERS = {
    "gemini-2.5-flash-preview-05-20": AsyncLimiter(1000),
    "gemini-2.5-flash-preview-04-17": AsyncLimiter(1000),
    "gemini-2.0-flash": AsyncLimiter(2000),
    "gemini-2.0-flash-lite": AsyncLimiter(4000),
}


TEXT_EXTRACT_PROMPT = """Extarct the text from this image. 

Follow these intructions:
- FORMAT THE TEXT IN MARKDOWN.
- Ignore any figures, diagrams, tables, or images.
- The order should be top to bottom, left to right.
- The EQUATIONS SHOULD BE MARKDOWN.
- The summary should be a summary of the content
- The md should be the text in markdown format.
"""


class TextImage(BaseModel):
    md: str
    summary: str


TABLE_EXTRACT_PROMPT = """Extract the infromation from the image of the table into a markdown table.

Follow these intructions:
- MAKE SURE TO EXTRACT ALL THE INFORMATION FROM THE TABLE
- Write the table in markdown format.
- Make sure the number of columns for earch row is consistent so it can render in markdown.
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
    generate_config: Dict = None,
) -> Dict:
    """Makes the synchronous API call to Google GenAI."""
    print(f"Making API call (Model: {model})...")
    # client = genai.GenerativeModel(model_name=model)  # Adjusted for current SDK
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    generate_config = generate_config or {}

    default_config = {
        "response_mime_type": "application/json",
        "response_schema": response_schema,
        "temperature": 0.0,
    }

    default_config.update(generate_config)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            prompt,
        ],
        config=default_config,
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
def parse_image_sync(
    image_input: Union[Path, Image.Image],
    prompt: str,
    response_schema: BaseModel,
    model: str = MODELS[0],
    generate_config: Dict = None,
) -> Dict:
    """Parses an image synchronously."""
    print(f"Processing sync: {image_input}")
    image_bytes, mime_type = _prepare_image_data(image_input)
    return _make_api_call(
        image_bytes, mime_type, prompt, response_schema, model, generate_config
    )


def parse_images_sync(
    images: List[Union[Path, Image.Image]],
    **kwargs,
) -> Dict:
    results = [parse_image_sync(image, **kwargs) for image in images]
    return results


async def parse_image(
    image_input: Union[Path, Image.Image],
    prompt: str,
    response_schema: BaseModel,
    model: str = MODELS[0],
    generate_config: Dict = None,
) -> Dict:
    """Parses an image asynchronously."""
    print(f"Processing async: {image_input}")
    image_bytes, mime_type = _prepare_image_data(image_input)

    rate_limiter = RATE_LIMITERS[model]
    async with rate_limiter:  # Acquire semaphore
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
            generate_config,
        )
    return result


async def parse_images(
    images: List[Union[Path, Image.Image]],
    **kwargs,
) -> Dict:
    tasks = [asyncio.create_task(parse_image(image, **kwargs)) for image in images]
    results = await asyncio.gather(*tasks)
    return results


async def parse_table(
    image: Union[Path, Image.Image],
    model: str = MODELS[0],
    generate_config: Dict = None,
) -> Dict:
    """Parses an image asynchronously."""
    result = await parse_image(
        image,
        prompt=TABLE_EXTRACT_PROMPT,
        response_schema=TableImage,
        model=model,
        generate_config=generate_config,
    )
    return result


async def parse_tables(
    images: List[Union[Path, Image.Image]],
    **kwargs,
) -> Dict:
    results = await parse_images(
        images, prompt=TABLE_EXTRACT_PROMPT, response_schema=TableImage, **kwargs
    )
    return results


async def parse_figure(
    image: Union[Path, Image.Image],
    **kwargs,
) -> Dict:
    result = await parse_image(
        image,
        prompt=FIGURE_EXTRACT_PROMPT,
        response_schema=FigureImage,
        **kwargs,
    )
    return result


async def parse_figures(
    images: List[Union[Path, Image.Image]],
    **kwargs,
) -> Dict:
    results = await parse_images(
        images, prompt=FIGURE_EXTRACT_PROMPT, response_schema=FigureImage, **kwargs
    )
    return results


async def parse_formula(
    image: Union[Path, Image.Image],
    **kwargs,
) -> Dict:
    result = await parse_image(
        image, prompt=FORMULA_EXTRACT_PROMPT, response_schema=FormulaImage, **kwargs
    )
    return result


async def parse_formulas(
    images: List[Union[Path, Image.Image]],
    **kwargs,
) -> Dict:
    results = await parse_images(
        images, prompt=FORMULA_EXTRACT_PROMPT, response_schema=FormulaImage, **kwargs
    )
    return results


async def parse_text(
    image: Union[Path, Image.Image],
    **kwargs,
) -> Dict:
    result = await parse_image(
        image, prompt=TEXT_EXTRACT_PROMPT, response_schema=TextImage, **kwargs
    )
    return result


async def parse_texts(
    images: List[Union[Path, Image.Image]],
    **kwargs,
) -> Dict:
    results = await parse_images(
        images, prompt=TEXT_EXTRACT_PROMPT, response_schema=TextImage, **kwargs
    )
    return results
