# src/data_ingestion.py

import concurrent.futures
import os
import pickle
import uuid  # <-- NEW IMPORT for unique IDs

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma

# LangChain Imports
from langchain_community.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredPDFLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.progress import Progress


# Worker function (correct and unchanged)
def process_file_for_loading(file_path, config):
    try:
        if file_path.endswith(".pdf"):
            strategy = config.pdf_parsing.library
            if strategy == "unstructured":
                loader = UnstructuredPDFLoader(
                    file_path,
                    mode=config.pdf_parsing.mode,
                    strategy=config.pdf_parsing.strategy,
                )
            else:  # Default to PyMuPDF
                loader = PyMuPDFLoader(file_path)
            return loader.load()
        elif file_path.endswith(".txt"):
            return TextLoader(file_path, encoding="utf-8").load()
        elif file_path.endswith(".csv"):
            return CSVLoader(file_path=file_path, encoding="utf-8").load()
        else:
            return []
    except Exception:
        return []


# --- Main Data Ingestion and Retriever Setup Function (with Manual Batching) ---
def setup_retriever(config, console):
    """Initializes the ParentDocumentRetriever by manually splitting, embedding, and
    batch-inserting documents to handle large-scale datasets."""
    vectorstore_path = config.vectorstore_path
    docstore_path = config.docstore_path

    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model, model_kwargs={"device": config.device}
    )

    if not os.path.exists(docstore_path):
        console.print(
            "--- Docstore not found. Building from scratch... ---", style="bold yellow"
        )

        # --- Step 1: Load all documents in parallel (same as before) ---
        files_to_process = [
            os.path.join(root, filename)
            for root, _, files in os.walk(config.dataset_path)
            for filename in files
        ]
        all_loaded_docs = []
        with Progress(console=console) as progress:
            task = progress.add_task(
                "[green]Loading documents...", total=len(files_to_process)
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=config.data_ingestion.max_workers
            ) as executor:
                future_to_docs = {
                    executor.submit(process_file_for_loading, path, config): path
                    for path in files_to_process
                }
                for future in concurrent.futures.as_completed(future_to_docs):
                    docs = future.result()
                    if docs:
                        all_loaded_docs.extend(docs)
                    progress.update(task, advance=1)

        if not all_loaded_docs:
            console.print(
                "!!! CRITICAL: No documents were loaded. Halting build. !!!",
                style="bold red",
            )
            return None

        # --- Step 2: Manually perform the Parent/Child Splitting ---
        console.print("--- Splitting documents into parent/child chunks... ---")
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.parent_splitter.chunk_size
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.child_splitter.chunk_size
        )

        parent_documents = parent_splitter.split_documents(all_loaded_docs)
        doc_ids = [str(uuid.uuid4()) for _ in parent_documents]

        child_documents = []
        for i, doc in enumerate(parent_documents):
            _id = doc_ids[i]
            _sub_docs = child_splitter.split_documents([doc])
            for _doc in _sub_docs:
                _doc.metadata["doc_id"] = _id
            child_documents.extend(_sub_docs)

        # --- Step 3: Manually populate the stores ---
        console.print("--- Populating docstore and vectorstore... ---")
        docstore = InMemoryStore()
        docstore.mset(list(zip(doc_ids, parent_documents)))

        vectorstore = Chroma(
            collection_name="parent_document_retriever",
            embedding_function=embeddings,
            persist_directory=vectorstore_path,
        )

        # --- Step 4: Add child documents to the vector store IN BATCHES ---
        batch_size = config.data_ingestion.batch_size
        console.print(
            f"--- Adding {len(child_documents)} child chunks to ChromaDB in batches of {batch_size}... ---"
        )
        with Progress(console=console) as progress:
            task = progress.add_task(
                "[green]Embedding and storing...", total=len(child_documents)
            )
            for i in range(0, len(child_documents), batch_size):
                batch = child_documents[i : i + batch_size]
                vectorstore.add_documents(batch)
                progress.update(task, advance=len(batch))

        # --- Step 5: Save the populated docstore ---
        console.print(
            f"--- Saving parent document store to [cyan]{docstore_path}[/cyan]... ---"
        )
        # os.makedirs(os.path.dirname(docstore_path), exist_ok=True)
        with open(docstore_path, "wb") as f:
            pickle.dump(docstore.store, f)

    # --- Loading logic remains the same ---
    else:
        console.print(
            f"--- Found existing vector store and docstore. Loading... ---",
            style="bold green",
        )
        with open(docstore_path, "rb") as f:
            store = pickle.load(f)
        docstore = InMemoryStore()
        docstore.store = store
        vectorstore = Chroma(
            collection_name="parent_document_retriever",
            persist_directory=vectorstore_path,
            embedding_function=embeddings,
        )

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.parent_splitter.chunk_size
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.child_splitter.chunk_size
        )

        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

    return retriever
