# src/data_ingestion.py

import concurrent.futures
import os
import pickle
import uuid
from pathlib import Path

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings

# LangChain Imports
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from omegaconf import DictConfig
from rich.progress import Progress


# --- Worker Function (Loads and Merges Pages) ---
def load_and_merge_pages(file_path: Path, config: DictConfig):
    """Worker function to load a single file and merge all its pages/parts into a single
    LangChain Document object to preserve contextual integrity."""
    try:
        # We prioritize pre-processed files, so PyMuPDF is just a fallback
        if file_path.suffix == ".pdf":
            loader = PyMuPDFLoader(str(file_path))
        elif file_path.suffix in [".txt", ".md"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif file_path.suffix == ".csv":
            loader = CSVLoader(file_path=str(file_path), encoding="utf-8")
        elif file_path.suffix == ".json":
            loader = JSONLoader(
                file_path=str(file_path), jq_schema=".", text_content=False
            )
        else:
            return None

        pages = loader.load()
        if not pages:
            return None

        full_text = "\n\f\n".join([doc.page_content for doc in pages])
        # The metadata of the first page is usually the most representative
        return Document(page_content=full_text, metadata=pages[0].metadata)
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return None


# --- Main Data Ingestion and Retriever Setup Function (Final Architecture) ---
def setup_retriever(config: DictConfig, console):
    """Initializes a retriever with a hybrid chunking strategy, file prioritization,
    manual batching, and persistent storage for both vector and document stores."""
    vectorstore_path = Path(config.vectorstore_path)
    docstore_path = Path(config.docstore_path)
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model, model_kwargs={"device": config.device}
    )

    # Define splitters here as they are needed for both building and loading
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.parent_splitter.chunk_size
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.child_splitter.chunk_size,
        chunk_overlap=config.child_splitter.chunk_overlap,
    )

    # Use the existence of the docstore file as the source of truth for a full build
    if not docstore_path.exists():
        console.print(
            "--- Docstore not found. Building from scratch... ---", style="bold yellow"
        )

        # --- STEP 1: FILE PRIORITIZATION ---
        console.print("--- Scanning and prioritizing data sources... ---")
        all_files = {}
        priority_order = config.data_ingestion.file_priority_order
        root_data_path = Path(config.dataset_path)
        for file_path in root_data_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in priority_order:
                base_key = file_path.parent / file_path.stem
                current_priority = priority_order.index(file_path.suffix)
                if (
                    base_key not in all_files
                    or current_priority < all_files[base_key]["priority"]
                ):
                    all_files[base_key] = {
                        "path": file_path,
                        "priority": current_priority,
                    }
        files_to_process = [data["path"] for data in all_files.values()]

        # --- STEP 2: PARALLEL DOCUMENT LOADING ---
        all_loaded_docs = []
        with Progress(console=console) as progress:
            task = progress.add_task(
                "[green]Loading documents...", total=len(files_to_process)
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=config.data_ingestion.max_workers
            ) as executor:
                future_to_doc = {
                    executor.submit(load_and_merge_pages, path, config): path
                    for path in files_to_process
                }
                for future in concurrent.futures.as_completed(future_to_doc):
                    doc = future.result()
                    if doc:
                        all_loaded_docs.append(doc)
                    progress.update(task, advance=1)

        if not all_loaded_docs:
            console.print(
                "!!! CRITICAL: No documents were loaded. Halting build. !!!",
                style="bold red",
            )
            return None

        # --- STEP 3: HYBRID CHUNKING STRATEGY ---
        console.print("--- Applying hybrid chunking strategy... ---")
        final_chunks_for_db = []
        docstore = InMemoryStore()

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )

        docs_for_parent_retriever = []
        for doc in all_loaded_docs:
            if doc.metadata.get("source", "").endswith((".md", ".json")):
                # For pre-processed files, use the precise Markdown splitter
                md_chunks = markdown_splitter.split_text(doc.page_content)
                for chunk in md_chunks:
                    chunk.metadata.update(doc.metadata)
                final_chunks_for_db.extend(md_chunks)
            else:
                # For raw files (.pdf, .txt), add them to be processed by ParentDocumentRetriever
                docs_for_parent_retriever.append(doc)

        if docs_for_parent_retriever:
            console.print(
                f"--- Processing {len(docs_for_parent_retriever)} raw documents with Parent/Child splitting... ---"
            )
            parent_documents = parent_splitter.split_documents(
                docs_for_parent_retriever
            )
            doc_ids = [str(uuid.uuid4()) for _ in parent_documents]
            child_documents = []
            for i, p_doc in enumerate(parent_documents):
                _id = doc_ids[i]
                _sub_docs = child_splitter.split_documents([p_doc])
                for _doc in _sub_docs:
                    _doc.metadata["doc_id"] = _id
                child_documents.extend(_sub_docs)

            docstore.mset(list(zip(doc_ids, parent_documents)))
            final_chunks_for_db.extend(child_documents)

        # --- STEP 4: BATCH EMBEDDING AND STORAGE ---
        vectorstore = Chroma(
            collection_name=config.collection_name,
            embedding_function=embeddings,
            persist_directory=str(vectorstore_path),
        )
        batch_size = config.data_ingestion.batch_size
        console.print(
            f"--- Adding {len(final_chunks_for_db)} final chunks to ChromaDB in batches of {batch_size}... ---"
        )
        with Progress(console=console) as progress:
            task = progress.add_task(
                "[green]Embedding and storing...", total=len(final_chunks_for_db)
            )
            for i in range(0, len(final_chunks_for_db), batch_size):
                batch = final_chunks_for_db[i : i + batch_size]
                vectorstore.add_documents(batch)
                progress.update(task, advance=len(batch))

        # --- STEP 5: PERSIST THE DOCSTORE ---
        console.print(
            f"--- Saving parent document store to [cyan]{docstore_path}[/cyan]... ---"
        )
        docstore_path.parent.mkdir(parents=True, exist_ok=True)
        with open(docstore_path, "wb") as f:
            pickle.dump(docstore.store, f)

    # --- Loading from disk ---
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
            collection_name=config.collection_name,
            persist_directory=str(vectorstore_path),
            embedding_function=embeddings,
        )

    # --- Final retriever creation (now correct for both paths) ---
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    return retriever
