import os 
import json

import base64
import PyPDF2
import numpy as np  
from pdf2image import convert_from_path
from openai import OpenAI


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
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_from_image(image_path, 
                            model='gpt-4o', 
                            max_tokens=1000, 
                            image_type='png',
                            prompt=None,
                            cahce_results=True):
    client = OpenAI()

    # Getting the base64 string
    base64_image = encode_image(image_path)

    if prompt is None:
        prompt = IMAGE_EXTRACTION_MARKDOWN_PROMPT
        
    messages= [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": prompt
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/{image_type};base64,{base64_image}",
              "detail": "high"
            }
          }
        ]
      }
    ]

    page_num=os.path.basename(image_path).split('.')[0]
    pdf_directory = os.path.dirname(os.path.dirname(image_path))
  

    response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
    data=response.model_dump(mode='python')

    message=None


    message = data['choices'][0]['message']['content']

    prompt_and_response = (prompt,message)

    return prompt_and_response
    