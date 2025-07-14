# --- Data Ingestion and Vector Store Imports ---

import concurrent.futures
import os

from langchain.load import dumps, loads

# Can also use Semantic Chunker for intelligent semantic splitting
# from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (  # For PDF loading
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredPDFLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings


# Hierarchical Data Ingestion
def process_file(file_path, doc_metadata, console=None):
    """A helper function to process a single file.

    This will be run in parallel. Returns a list of Document objects.
    """
    try:
        if file_path.endswith(".pdf"):
            # Try the fast loader first
            fast_loader = PyMuPDFLoader(file_path)
            docs = fast_loader.load()
            # If fast loader fails (no text), fall back to OCR
            if not docs or not docs[0].page_content.strip():
                console.print(
                    f"  - PyMuPDF failed for {os.path.basename(file_path)},"
                    " falling back to OCR...",
                    style="warning",
                )
                ocr_loader = UnstructuredPDFLoader(
                    file_path, mode="single", strategy="ocr_only"
                )
                docs = ocr_loader.load()
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path=file_path, encoding="utf-8")
            docs = loader.load()
        else:
            return []  # Return empty list for unsupported files

        # Add the hierarchical metadata to the loaded documents
        for doc in docs:
            doc.metadata.update(doc_metadata)
        return docs
    except Exception as e:
        console.print(
            f"  - CRITICAL ERROR loading {os.path.basename(file_path)}: {e}",
            style="danger",
        )
        return []


def create_vector_store(config, console=None):
    """Creates a Chroma vector store by processing files in parallel."""
    console.print(
        f"--- Creating new parallelized vector store at {config.vectorstore_path} ---"
    )

    # 1. Gather all file paths and their associated metadata
    files_to_process = []
    root_data_path = config.dataset_path
    if not os.path.exists(root_data_path):
        console.print(
            f"!!! Dataset path {root_data_path} does not exist. Exiting. !!!",
            style="danger",
        )
        return None
    for root, dirs, files in os.walk(root_data_path):
        if not files:
            continue
        relative_path = os.path.relpath(root, root_data_path)
        if relative_path == ".":
            continue
        path_parts = relative_path.split(os.sep)
        doc_metadata = {}
        if len(path_parts) > 0:
            doc_metadata["topic"] = path_parts[0]
        if len(path_parts) > 1:
            doc_metadata["sub_topic"] = path_parts[1]

        for filename in files:
            files_to_process.append((os.path.join(root, filename), doc_metadata))

    # 2. Process files in parallel using a thread pool
    all_documents = []
    # Adjust max_workers based on your CPU cores, but don't go too high for I/O tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Use partial to "pre-fill" the process_file function
        future_to_docs = {
            executor.submit(process_file, file_path, metadata, console): (
                file_path,
                metadata,
                console,
            )
            for file_path, metadata in files_to_process
        }

        for future in concurrent.futures.as_completed(future_to_docs):
            docs = future.result()
            if docs:
                all_documents.extend(docs)

    if not all_documents:
        console.print("!!! No documents found to process. !!!", style="warning")
        return

    # 3. Chunk, Embed, and Store (same as before)
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model, model_kwargs={"device": config.device}
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.recursive_text_splitter.chunk_size,
        chunk_overlap=config.recursive_text_splitter.chunk_overlap,
    )
    texts = text_splitter.split_documents(all_documents)
    console.print(
        f"\n--- Total documents loaded: {len(all_documents)},"
        f" split into {len(texts)} chunks ---",
        style="info",
    )

    db = Chroma.from_documents(
        documents=texts, embedding=embeddings, persist_directory=config.vectorstore_path
    )
    console.print(
        "--- Parallelized Chroma vector store created successfully. ---", style="info"
    )
    return db


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [
        loads(doc)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results
