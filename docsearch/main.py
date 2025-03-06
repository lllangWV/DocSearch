import os
from glob import glob

from pdf_processing import PDFProcessor
from download import parse_and_download_wvu_minutes
from vector_store import VectorStore, create_document_from_pdf_directory

def main():

    # Parameters
    DOWNLOAD_DATA=False
    PROCESS_PDFS=False
    LOAD_PDFS_INTO_STORE=False
    USE_QUERY_ENGINE=False
    USE_CITATION_ENGINE=True
    RETURN_RESPONSE=True

    raw_dir = os.path.join('data/minutes/raw')
    interim_dir=os.path.join('data/minutes/interim')
    index_store_dir=os.path.join('data','minutes','vector_stores','test_1')
    output_dir=os.path.join('data','minutes','output')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(interim_dir, exist_ok=True)
    # os.makedirs(index_store_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # Pdf processing parameters
    llm='gpt-4o-mini'
    extraction_method='llm'
    max_tokens=3000

    # Querys to run

    # query="""Find evidence of that the board of governors deliberates adjustments to faculty promotion and tenure. 
    # Provide this in a markdown table where the coulmn order should be Date, Evidence."""

    # query="""Find evidence that the board of governors deliberates adjustments to faculty promotion and tenure. 
    # Provide this in a markdown table where the coulmn order should be Date, Description of the adjustments, Evidence.
    # """

    # query="""Find evidence that the board of governors deliberates adjustments to faculty promotion and tenure. Give 10 pieces of evidence.
    # Provide this in a markdown table where the coulmn order should be Date, Description of the adjustments, Evidence.
    # """

    # query="""Are there any references to associate provosts providing reports during the Board meetings? Give 10 references.
    # Provide this in a markdown table where the coulmn order should be Date, Description, Context of the reference.
    # """

    query="""What other personnel, apart from the President and Provost, provide regular reports to the Board committees? Give 10 references.
    Provide this in a markdown table where the coulmn order should be Date, Personal, Context of the report provided.
    """
    ################################################################################################
    # query parameters
    similarity_top_k=20
    embed_model='text-embedding-3-small'
    llm='gpt-4o-mini'

    # Citation query parameters
    similarity_top_k=30
    citation_chunk_size=2048
    citation_chunk_overlap=0
    embed_model='text-embedding-3-small'
    llm='gpt-4o-mini'

    ###########################################################################################################################
    # Logic for query engine
    ###########################################################################################################################
    # Initialize the vector store
    store=VectorStore(
                index_store_dir=index_store_dir,
                embed_model=embed_model,
                llm=llm
                )
    
    if DOWNLOAD_DATA:
        print('Downloading minutes')
        parse_and_download_wvu_minutes(download_dir=raw_dir, url = "https://bog.wvu.edu/minutes")

    if PROCESS_PDFS:
        print('Processing pdfs')
        processor=PDFProcessor(
                        output_dir=interim_dir, 
                        model=llm,
                        max_tokens=max_tokens)

        pdf_files=glob(os.path.join(raw_dir,'*.pdf'))
        for pdf_file in pdf_files:
            print(pdf_file)
            processor.process(pdf_file, method=extraction_method)

    if LOAD_PDFS_INTO_STORE: 
        print('Loading pdfs into store')
        pdf_dirs=glob(os.path.join(interim_dir,'*'))

        for pdf_dir in pdf_dirs:
            docs=create_document_from_pdf_directory(pdf_dir=pdf_dir)
            store.load_docs(docs=docs)

    if USE_QUERY_ENGINE:
        print('Using query engine')
        engine=store.create_engine(
                        engine_type='query',
                        similarity_top_k=similarity_top_k,
                        llm=llm,
                        )
    if USE_CITATION_ENGINE:
        print('Using citation engine')
        engine=store.create_engine(
                        engine_type='citation_query',
                        similarity_top_k=similarity_top_k,
                        citation_chunk_size=citation_chunk_size,
                        citation_chunk_overlap=citation_chunk_overlap,
                        llm=llm,
                        )
    if RETURN_RESPONSE:
        print('Returning response')
        response=engine.query(query)
        store.save_response(response,query,output_dir=output_dir)


if __name__ == "__main__":
    main()