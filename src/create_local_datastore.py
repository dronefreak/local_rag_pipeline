# src/data_ingestion.py
"""Handles all data ingestion, processing, and vector store management.

This module is responsible for discovering files, loading them into memory, processing
their content using a parent-document retrieval strategy, and persisting the resulting
vector store and document store to disk for efficient reuse. It uses parallel processing
to speed up file loading.
"""
import concurrent.futures
import pickle
import sys
import uuid
from pathlib import Path

import hydra

# LangChain and ChromaDB Imports for core RAG components
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredPDFLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from omegaconf import DictConfig, OmegaConf

# Rich library for enhanced console output
from rich.progress import Progress

from src.utils import RichConsoleManager


def process_file_for_loading(file_path: Path, config):
    """Loads a single file from disk into a LangChain Document object.

    This function acts as a worker for parallel processing. It inspects the
    file's extension and uses the appropriate loader as defined in the Hydra
    config. It's designed to be robust, returning an empty list if a single
    file fails to load, preventing a complete failure of the ingestion pipeline.

    Args:
        file_path (Path): The pathlib.Path object pointing to the file.
        config: The Hydra DictConfig object containing pipeline settings.

    Returns:
        list: A list of LangChain Document objects, or an empty list if loading fails.
    """
    try:
        # Check the file extension to determine the correct loader.
        if file_path.suffix == ".pdf":
            # Use the PDF parsing library specified in the config
            # (e.g., 'unstructured' or 'pymupdf').
            strategy = config.data_ingestion.pdf_parsing.library
            if strategy == "unstructured":
                loader = UnstructuredPDFLoader(
                    str(file_path),
                    mode=config.data_ingestion.pdf_parsing.mode,
                    strategy=config.data_ingestion.pdf_parsing.strategy,
                )
            else:  # Default to PyMuPDF for a faster, simpler alternative.
                loader = PyMuPDFLoader(str(file_path))
            return loader.load()
        elif file_path.suffix == ".txt":
            return TextLoader(str(file_path), encoding="utf-8").load()
        elif file_path.suffix == ".csv":
            return CSVLoader(file_path=str(file_path), encoding="utf-8").load()
        else:
            # If the file type is not supported, return an empty list.
            return []
    except Exception:
        # Gracefully handle any exceptions during file loading, preventing a crash.
        return []


@hydra.main(version_base=None, config_path="../configs", config_name="rag_pipeline")
def setup_retriever(config: DictConfig):
    """Sets up the ParentDocumentRetriever, orchestrating the entire data ingestion
    process.

    This function is the main entry point for data handling. It checks if a
    persisted document store (`docstore`) exists.
    - If it doesn't, it builds the entire database from scratch by:
        1. Discovering all files in the source data directory.
        2. Loading their content in parallel using worker threads.
        3. Splitting the documents into large "parent" chunks and
           small "child" chunks.
        4. Storing the parent documents in a docstore and embedding/indexing
           the child chunks in a ChromaDB vector store.
        5. Persisting the populated docstore and vectorstore to disk.
    - If the docstore exists, it loads the persisted stores from disk to ensure
      fast startup times on subsequent runs.

    Args:
        config: The Hydra DictConfig object containing all pipeline parameters.
        console: The Rich console object for formatted output.

    Returns:
        ParentDocumentRetriever: A fully configured retriever instance ready for use.
    """
    # Initialize a Rich console for clean and styled terminal output.
    console = RichConsoleManager.get_console()
    # Print the full configuration for the current run for reproducibility.
    console.print(OmegaConf.to_yaml(config), style="warning")
    # Define paths using pathlib for robust, cross-platform path management.
    vectorstore_path = Path(config.vectorstore_path)
    docstore_path = Path(config.docstore_path)

    # Initialize the embedding model specified in the config.
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model, model_kwargs={"device": config.device}
    )

    # Use the existence of the persisted docstore as the trigger for a full rebuild.
    if not docstore_path.exists():
        console.print(
            "--- Docstore not found. Building from scratch... ---", style="bold yellow"
        )

        # --- Step 1: Discover and Load all documents in parallel ---
        root_data_path = Path(config.dataset_path)
        # Recursively find all files in the data directory.
        files_to_process = [p for p in root_data_path.rglob("*") if p.is_file()]

        all_loaded_docs = []
        # Use Rich's Progress for a clean, visual progress bar
        # during the slow loading phase.
        with Progress(console=console) as progress:
            task = progress.add_task(
                "[green]Loading documents...", total=len(files_to_process)
            )
            # Use a thread pool to load multiple files concurrently.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=config.data_ingestion.max_workers
            ) as executor:
                # Submit all file processing jobs to the executor.
                future_to_docs = {
                    executor.submit(process_file_for_loading, path, config): path
                    for path in files_to_process
                }
                # Collect results as they are completed.
                for future in concurrent.futures.as_completed(future_to_docs):
                    docs = future.result()
                    if docs:
                        all_loaded_docs.extend(docs)
                    progress.update(task, advance=1)

        # A critical guard rail to halt the build if no documents were loaded.
        if not all_loaded_docs:
            console.print(
                "!!! CRITICAL: No documents were loaded. Halting build. !!!",
                style="bold red",
            )
            return None

        # --- Step 2: Manually perform the Parent/Child Splitting ---
        console.print("--- Splitting documents into parent/child chunks... ---")
        # The parent splitter creates large chunks that provide
        # rich context for the LLM.
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.data_ingestion.parent_splitter.chunk_size
        )
        # The child splitter creates small, specific chunks ideal for semantic search.
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.data_ingestion.child_splitter.chunk_size
        )

        # Create the large parent documents.
        parent_documents = parent_splitter.split_documents(all_loaded_docs)
        # Generate a unique ID for each parent document.
        doc_ids = [str(uuid.uuid4()) for _ in parent_documents]

        child_documents = []
        # Create smaller child documents linked to their parent by a metadata ID.
        for i, doc in enumerate(parent_documents):
            _id = doc_ids[i]
            _sub_docs = child_splitter.split_documents([doc])
            for _doc in _sub_docs:
                _doc.metadata["doc_id"] = _id
            child_documents.extend(_sub_docs)

        # --- Step 3: Manually populate the document and vector stores ---
        console.print("--- Populating docstore and vectorstore... ---")
        # The docstore holds the large parent documents in memory (and later, on disk).
        docstore = InMemoryStore()
        docstore.mset(list(zip(doc_ids, parent_documents)))

        # The vectorstore holds the embeddings of the small child documents.
        vectorstore = Chroma(
            collection_name="parent_document_retriever",
            embedding_function=embeddings,
            persist_directory=str(vectorstore_path),
        )

        # --- Step 4: Add child documents to the vector store in
        # batches for scalability ---
        batch_size = config.data_ingestion.batch_size
        console.print(
            f"--- Adding {len(child_documents)} child chunks to ChromaDB... ---"
        )
        with Progress(console=console) as progress:
            task = progress.add_task(
                "[green]Embedding and storing...", total=len(child_documents)
            )
            for i in range(0, len(child_documents), batch_size):
                batch = child_documents[i : i + batch_size]
                # This is the step where embedding happens.
                vectorstore.add_documents(batch)
                progress.update(task, advance=len(batch))

        # --- Step 5: Save the populated docstore to disk for persistence ---
        console.print(
            f"--- Saving parent document store to [cyan]{docstore_path}[/cyan]... ---"
        )
        # Ensure the parent directory exists before writing the file.
        docstore_path.parent.mkdir(parents=True, exist_ok=True)
        with open(docstore_path, "wb") as f:
            pickle.dump(docstore.store, f)

    # --- Loading logic for subsequent runs ---
    else:
        console.print(
            "--- Found existing vector store and docstore. Loading... ---",
            style="bold green",
        )
        # Load the persisted parent documents.
        with open(docstore_path, "rb") as f:
            store = pickle.load(f)
        docstore = InMemoryStore()
        docstore.store = store

        # Load the persisted vector store of child documents.
        vectorstore = Chroma(
            collection_name="parent_document_retriever",
            persist_directory=str(vectorstore_path),
            embedding_function=embeddings,
        )

    # --- Final Retriever Instantiation ---
    # The splitters are still required by the retriever's constructor for validation,
    # even when loading from a persisted state.
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.data_ingestion.parent_splitter.chunk_size
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.data_ingestion.child_splitter.chunk_size
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    return retriever


if __name__ == "__main__":
    # Standard entry point for the script.
    # The following lines are a workaround for Hydra's default behavior of
    # creating new output directories on each run. This keeps the project clean.
    research_dir = Path(__file__).resolve().parent
    sys.argv.append(f"hydra.run.dir={research_dir}")
    sys.argv.append("hydra.output_subdir=null")
    # This call executes the main function, with Hydra handling the config injection.
    setup_retriever()
