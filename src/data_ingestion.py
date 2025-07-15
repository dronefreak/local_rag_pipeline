# Data Ingestion and Vector Store Imports

import concurrent.futures
import os

from langchain.load import dumps, loads

# Can also use Semantic Chunker for intelligent semantic splitting
# from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredPDFLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from src.utils import get_rich_console


# The worker function now also handles the splitting.
def process_and_split_file(
    file_path, doc_metadata, text_splitter, config, console=get_rich_console()
):
    """Worker function to load a single file, process its content, and then split it
    into chunks.

    Returns a list of chunked Document objects.
    """
    try:
        # Step 1: Load the full document based on its type
        if file_path.endswith(".pdf"):
            if config.pdf_parsing.library == "unstructured":
                # Use UnstructuredPDFLoader with high-resolution strategy
                loader = UnstructuredPDFLoader(
                    file_path,
                    mode=config.pdf_parsing.mode,  # "single" or "elements"
                    strategy=config.pdf_parsing.strategy,  # "hi_res" or "low_res"
                    infer_table_structure=config.pdf_parsing.infer_table_structure,
                    extract_images=config.pdf_parsing.extract_images,
                )
            else:
                # Fallback to PyMuPDFLoader if Unstructured is not used
                # Note: PyMuPDFLoader is generally faster but less accurate
                # for complex layouts. It does not support high-resolution
                # strategies like Unstructured.
                loader = PyMuPDFLoader(file_path)
            # Load the document
            docs = loader.load()
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path=file_path, encoding="utf-8")
            docs = loader.load()
        else:
            return []

        # Step 2: Apply metadata to every page/doc loaded from this single file
        for doc in docs:
            doc.metadata.update(doc_metadata)

        # Step 3: Perform text splitting ON THIS DOCUMENT ONLY
        # The splitter will see all pages from this single
        # file as one continuous sequence
        chunked_docs = text_splitter.split_documents(docs)

        return chunked_docs
    except Exception as e:
        # Log the error with the specific file that caused it
        console.print(
            f"\nError processing file {os.path.basename(file_path)}: {e}",
            style="danger",
        )
        return []


# The main orchestrator function
def create_vector_store(config, console):
    """Creates a Chroma vector store by processing and splitting each file individually
    in parallel, ensuring contextual integrity."""
    console.print(
        f"Creating/updating vector store at [cyan]{config.vectorstore_path}[/cyan]"
    )

    # 1. Gather all file paths and their associated metadata
    files_to_process = []
    # (This logic is the same as before, omitted for brevity)
    root_data_path = config.dataset_path
    if not os.path.exists(root_data_path):
        console.print(
            f"!!! Dataset path {root_data_path} does not exist. !!!", style="bold red"
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

    # Initialize the text splitter here, so we can pass it to the workers
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.recursive_text_splitter.chunk_size,
        chunk_overlap=config.recursive_text_splitter.chunk_overlap,
    )

    all_chunked_texts = []

    # 2. Process and split files in parallel
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[green]Chunking documents...", total=len(files_to_process)
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.concurrency
        ) as executor:
            # We pass the splitter instance to each worker
            future_to_chunks = {
                executor.submit(
                    process_and_split_file, path, meta, text_splitter, config
                ): path
                for path, meta in files_to_process
            }

            for future in concurrent.futures.as_completed(future_to_chunks):
                chunked_docs = future.result()
                if chunked_docs:
                    all_chunked_texts.extend(chunked_docs)
                progress.update(task, advance=1)

    if not all_chunked_texts:
        console.print(
            "!!! No documents were successfully processed. !!!", style="bold yellow"
        )
        return None

    # 3. Embed and Store the final list of chunks
    console.print(f"\nEmbedding {len(all_chunked_texts)} total text chunks...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model, model_kwargs={"device": config.device}
    )

    db = Chroma.from_documents(
        documents=all_chunked_texts,  # We now use the final list of chunks
        embedding=embeddings,
        persist_directory=config.vectorstore_path,
    )
    console.print(
        "Vector store created successfully with per-document chunking.",
        style="bold green",
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
