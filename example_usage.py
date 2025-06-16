#!/usr/bin/env python3
"""
Example usage of the DocSearch class.

This script demonstrates how to use the DocSearch class to:
1. Initialize a DocSearch instance
2. Add PDFs for processing
3. Query the resulting vector database
"""

from docrag import DocSearch


def main():
    # Initialize DocSearch with a custom data directory
    # This will create the necessary subdirectories (raw, interim, vector_stores, output)
    doc_search = DocSearch(
        base_path="my_documents",
        embed_model="text-embedding-3-small",
        llm="gpt-4o-mini",
        max_tokens=3000,
    )

    # Get initial statistics
    print("Initial stats:")
    stats = doc_search.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Example 1: Add a single PDF
    # doc_search.add_pdfs("path/to/your/document.pdf")

    # Example 2: Add multiple PDFs
    # pdf_files = [
    #     "path/to/document1.pdf",
    #     "path/to/document2.pdf",
    #     "path/to/document3.pdf"
    # ]
    # doc_search.add_pdfs(pdf_files)

    # Example 3: Add all PDFs from a directory
    # doc_search.add_pdfs_from_directory("path/to/pdf/directory")

    # For demonstration, let's check stats after adding PDFs
    # print("\nStats after adding PDFs:")
    # stats = doc_search.get_stats()
    # for key, value in stats.items():
    #     print(f"  {key}: {value}")

    # Example queries
    example_queries = [
        "What are the main topics discussed in these documents?",
        "Find information about methodology or approach",
        "What conclusions were drawn from the research?",
        "Are there any references to specific dates or events?",
    ]

    # Note: Uncomment the following section after adding PDFs
    # print("\nExample queries:")
    # for i, query in enumerate(example_queries, 1):
    #     print(f"\n{i}. Query: {query}")
    #     try:
    #         response = doc_search.query(
    #             query_text=query,
    #             engine_type='citation_query',  # or 'query' for simple retrieval
    #             similarity_top_k=10,
    #             save_response=True
    #         )
    #         print(f"   Response: {response.response[:200]}...")
    #     except Exception as e:
    #         print(f"   Error: {e}")

    # Advanced usage examples
    print("\nAdvanced usage examples:")

    # Custom engine parameters for citation queries
    print("- Custom citation query with specific chunk size:")
    print("  response = doc_search.query(")
    print("      query_text='Your query here',")
    print("      engine_type='citation_query',")
    print("      citation_chunk_size=1024,")
    print("      citation_chunk_overlap=100")
    print("  )")

    # Using different extraction methods
    print("\n- Using different PDF extraction methods:")
    print("  doc_search.add_pdfs('document.pdf', extraction_method='text_then_llm')")

    # Manual control over loading
    print("\n- Manual control over PDF loading:")
    print("  doc_search.add_pdfs('document.pdf', auto_load=False)")
    print("  doc_search.load_processed_pdfs()  # Load when ready")

    # Reset functionality
    print("\n- Reset the DocSearch instance:")
    print("  doc_search.reset(confirm=True)  # Clears all data")


if __name__ == "__main__":
    main()
