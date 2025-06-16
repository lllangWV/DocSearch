# DocRag

DocRag is an advanced document search and retrieval system that leverages Retrieval-Augmented Generation (RAG) to provide intelligent natural language search capabilities across PDF document collections. This system combines sophisticated PDF processing, vector embeddings, and large language models to enable semantic understanding of document content and context-aware responses to complex queries.

## Installation

```bash
pip install docrag
```

## Getting Started

For a comprehensive tutorial on using DocSearch, check out our [Getting Started notebook](examples/01-Getting%20Started.ipynb) which covers:

- Setting up your Google Gemini API key
- Creating Document objects from PDFs
- Exploring document content (figures, tables, text, formulas)
- Converting documents to markdown
- Saving and exporting processed data

The notebook provides step-by-step examples and explanations to help you get up and running quickly with DocSearch.

## Quick Example

```python
import os
from docrag import Document

# Set your Google Gemini API key
os.environ['GEMINI_API_KEY'] = 'your-api-key-here'

# Process a PDF document
doc = Document.from_pdf('your_document.pdf')

# Access different content types
print(f"Found {len(doc.figures)} figures, {len(doc.tables)} tables")

# Convert to markdown
markdown = doc.to_markdown()
print(markdown)

# Save processed document
doc.save('output_directory')
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.


## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.


## Authors
Logan Lang