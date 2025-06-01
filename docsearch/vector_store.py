import json
import logging
import sys
from pathlib import Path

import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    ServiceContext,
    Settings,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.query_engine import CitationQueryEngine, RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)

load_dotenv()


class VectorStore:

    def __init__(
        self,
        docs=None,
        index_store_dir="vector_store",
        output_dir=None,
        embed_model="text-embedding-3-small",
        llm="gpt-4o-mini",
    ):

        self.index_store_dir = Path(index_store_dir)

        self.index = None
        self.engine = None
        self.embed_model = embed_model
        self.llm = llm

        self.output_dir = Path(output_dir) if output_dir else None

        self.metadata = {"embed_model": embed_model, "llm": llm}
        self.set_settings()
        if docs:
            self.load_docs(docs)

        self.load_embed_model(model=self.metadata["embed_model"])
        self.load_llm(model=self.metadata["llm"])

    def exists(self):
        return self.index_store_dir.exists()

    def set_settings(self):
        Settings.embed_model = OpenAIEmbedding(model=self.metadata["embed_model"])
        Settings.llm = OpenAI(model=self.metadata["llm"])
        return None

    def save_metadata(self):
        metadata_file = self.index_store_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f)

    def load_metadata(self):
        metadata_file = self.index_store_dir / "metadata.json"
        if not metadata_file.exists():
            return self.metadata

        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

        self.load_embed_model()
        self.load_llm()

    def load_llm(self, model=None):
        if model:
            self.llm = OpenAI(model=model)
        else:
            self.llm = OpenAI(model=self.metadata["llm"])
        return self.llm

    def load_embed_model(self, model=None):
        if model:
            self.embed_model = OpenAIEmbedding(model=model)
        else:
            self.embed_model = OpenAIEmbedding(model=self.metadata["embed_model"])
        return None

    def load_docs(self, docs=None):
        if not self.exists():
            self.create_index(docs)
        else:
            self.load_index(docs)
        return self.index

    def create_index(self, docs, **kwargs):
        embed_model = Settings.embed_model
        print(f"Creating index {self.index_store_dir}")

        self.index = VectorStoreIndex.from_documents(
            docs,
            show_progress=kwargs.get("show_progress", True),
            embed_model=embed_model,
        )
        self.index.storage_context.persist(persist_dir=str(self.index_store_dir))

        self.save_metadata()

    def load_index(self, docs=None):
        print(f"Loading index {self.index_store_dir}")
        # self.load_metadata()
        self.set_settings()
        self.index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(self.index_store_dir))
        )

        if docs:
            print(f"Refreshing index {self.index_store_dir}")
            refreshed_docs = self.index.refresh(docs)
            if sum(refreshed_docs) != 0:
                self.index.storage_context.persist(
                    persist_dir=str(self.index_store_dir)
                )

    def get_llama_debugger(self, debug=False):
        self.llama_debug = None
        if debug:
            nest_asyncio.apply()
            # Using the LlamaDebugHandler to print the trace of the underlying steps
            # regarding each sub-process for document ingestion part
            self.llama_debug = LlamaDebugHandler(print_trace_on_end=True)
            self.callback_manager.add_handler(self.llama_debug)

    def create_engine(
        self,
        engine_type="query",
        llm=None,
        similarity_top_k=10,
        callbacks=[],
        node_postprocessors=None,
        **kwargs,
    ):

        self.load_index()
        possible_engine_types = ["query", "citation_query", "retriever"]
        if engine_type not in possible_engine_types:
            raise ValueError(f"engine_type must be either {possible_engine_types}")

        # Define callback manager
        self.callback_manager = CallbackManager()
        for callback in callbacks:
            self.callback_manager.add_handler(callback)

        # Define llama debugger
        self.get_llama_debugger(debug=kwargs.get("debug", False))

        # Define node postprocessors
        self.node_postprocessors = node_postprocessors

        # configure retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k,
            embed_model=self.embed_model,
        )
        # configure response synthesizer

        llm = self.load_llm(model=llm)

        response_synthesizer = get_response_synthesizer(llm=llm)

        # Define the query engine
        if engine_type == "query":
            self.engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=self.node_postprocessors,
                callback_manager=self.callback_manager,
            )
        elif engine_type == "citation_query":
            self.engine = CitationQueryEngine.from_args(
                self.index,
                llm=llm,
                similarity_top_k=similarity_top_k,
                # here we can control how granular citation sources are, the default is 512
                citation_chunk_size=kwargs.get("citation_chunk_size", 512),
                citation_chunk_overlap=kwargs.get("citation_chunk_overlap", 20),
                text_splitter=kwargs.get("text_splitter", None),
                # callback_manager=self.callback_manager,
                node_postprocessors=self.node_postprocessors,
            )
        elif engine_type == "retriever":
            self.engine = retriever

        return self.engine

    def save_response(self, response, query, output_dir=None):

        # Creating a directory for the run
        if output_dir is None:
            output_dir = self.output_dir

        output_path = Path(output_dir)
        dirs = list(output_path.iterdir()) if output_path.exists() else []
        n_runs = len(dirs)

        run_dir = output_path / f"run_{n_runs}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Seperating the response into context and response
        context = " ".join(
            [node.dict()["node"]["text"] for node in response.source_nodes]
        )
        source_nodes = [node.dict() for node in response.source_nodes]

        # Defining the file names
        context_file = run_dir / "prompt.txt"
        response_file = run_dir / "response.md"
        source_file = run_dir / "source.json"
        source_summary_file = run_dir / "source_summary.txt"
        response_source_file = run_dir / "response_source.md"

        with open(context_file, "w", encoding="utf-8") as f:
            f.write(context)
            f.write("-" * 300)
            f.write(query)

        with open(response_file, "w", encoding="utf-8") as f:
            f.write(response.response)

        with open(source_file, "w", encoding="utf-8") as f:
            json.dump(source_nodes, f)

        with open(source_summary_file, "w", encoding="utf-8") as f:
            for i, node in enumerate(source_nodes):
                pdf_name = node["node"]["metadata"]["pdf_name"]
                score = node["score"]
                text = node["node"]["text"]

                id = node["node"]["id_"]
                page_name = node["node"]["metadata"]["page_name"]
                pdf_name = node["node"]["metadata"]["pdf_name"]

                f.write(
                    f"Source {i+1} | Score - {score} | pdf_name - {pdf_name} | page_name - {page_name}"
                )
                f.write("\n")

        with open(response_source_file, "w", encoding="utf-8") as f:
            f.write("## Query\n\n")
            f.write(query)
            f.write("\n")
            f.write("---\n")
            f.write("## Response\n\n")
            f.write(response.response)
            f.write("\n")
            f.write("---\n")
            f.write("## Sources\n\n")
            for i, node in enumerate(source_nodes):
                pdf_name = node["node"]["metadata"]["pdf_name"]
                score = node["score"]
                text = node["node"]["text"]

                id = node["node"]["id_"]
                page_name = node["node"]["metadata"]["page_name"]
                pdf_name = node["node"]["metadata"]["pdf_name"]

                f.write(
                    f"Source {i+1} | Score - {score:0.4f} | pdf_name - {pdf_name} | page_name - {page_name}\n"
                )


# This should be moved to some class
def create_document_from_pdf_directory(pdf_dir):

    pdf_dir_path = Path(pdf_dir)
    pdf_title = pdf_dir_path.name
    json_file = pdf_dir_path / "pdf_info.json"

    with open(json_file, "r") as f:
        data = json.load(f)

    docs = []
    metadata = data.get("metadata", {})
    pages_dict = data.get("pages", {})
    pdf_title = metadata.get("pdf_name", pdf_title)

    for key, page_dict in pages_dict.items():
        text = page_dict["text"]
        page_number = key
        metadata["page_name"] = page_number
        doc = Document(text=text, metadata=metadata, id_=f"{pdf_title}_{page_number}")
        docs.append(doc)

    return docs


if __name__ == "__main__":
    ################################################################################################
    # Initialize the vector store
    store = VectorStore(
        index_store_dir=Path("data") / "dft" / "vector_stores" / "dft",
        embed_model="text-embedding-3-small",
        llm="gpt-4o-mini",
    )
    ################################################################################################
    # Load singular document
    # pdf_dir = Path("data") / "dft" / "interim" / "Thomas_1927"
    # docs=create_document_from_pdf_directory(pdf_dir=pdf_dir)
    # index=store.load_docs(docs=docs)

    # Load multiple documents
    # pdf_dirs = list((Path("data") / "dft" / "interim").glob("*"))
    # for pdf_dir in pdf_dirs:
    #     docs=create_document_from_pdf_directory(pdf_dir=pdf_dir)
    #     store.load_docs(docs=docs)
    ################################################################################################
    # Query Engine
    engine = store.create_engine(
        engine_type="query",
        similarity_top_k=125,
        llm="gpt-4o-mini",
    )

    # Citation Query Engine
    # engine=store.create_engine(
    #                     engine_type='citation_query',
    #                     similarity_top_k=3,
    #                     citation_chunk_size=1024,
    #                     citation_chunk_overlap=64,
    #                     llm='gpt-4o-mini',
    #                     )
    ################################################################################################
    # Using the query engine
    query = ()
    response = engine.query(query)
    output_dir = Path("data") / "dft" / "output"
    store.save_response(response, output_dir=output_dir)
