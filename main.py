import concurrent.futures
import os

from langchain.load import dumps, loads

# --- Core LangChain and RAG Imports ---
from langchain.prompts import PromptTemplate

# --- Data Ingestion and Vector Store Imports ---
# Can also use Semantic Chunker for intelligent semantic splitting
# from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import (  # For PDF loading
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredPDFLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Global Constants ---
DATA_PATH = "data/"
DB_PATH = "vectorstore_semantic_fusion/"  # New DB path for the final architecture
MODEL_NAME = "llama3-8b-q6k"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --- 1. Hierarchical Data Ingestion ---
def process_file(file_path, doc_metadata):
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
                print(
                    f"  - PyMuPDF failed for {os.path.basename(file_path)},"
                    " falling back to OCR..."
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
        print(f"  - CRITICAL ERROR loading {os.path.basename(file_path)}: {e}")
        return []


def create_vector_store(persist_path):
    """Creates a Chroma vector store by processing files in parallel."""
    print(f"--- Creating new parallelized vector store at {persist_path} ---")

    # 1. Gather all file paths and their associated metadata
    files_to_process = []
    root_data_path = DATA_PATH
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
            executor.submit(process_file, file_path, metadata): (file_path, metadata)
            for file_path, metadata in files_to_process
        }

        for future in concurrent.futures.as_completed(future_to_docs):
            docs = future.result()
            if docs:
                all_documents.extend(docs)

    if not all_documents:
        print("!!! No documents found to process. !!!")
        return

    # 3. Chunk, Embed, and Store (same as before)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cuda"}
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    texts = text_splitter.split_documents(all_documents)
    print(
        f"\n--- Total documents loaded: {len(all_documents)},"
        f" split into {len(texts)} chunks ---"
    )

    db = Chroma.from_documents(
        documents=texts, embedding=embeddings, persist_directory=persist_path
    )
    print("--- Parallelized Chroma vector store created successfully. ---")
    return db


# --- 2. Final Answer Formatter ---
def get_final_formatter_chain(llm):
    """Creates the chain that polishes the final answer."""
    formatter_template = """You are the final response generation layer.
    Your job is to take the raw information gathered by a research system
    and format it into a clean, polite, and helpful response for the user.
    Do not add any new facts. If the raw information indicates the system could not
    find an answer, state that politely.

Raw Information from System:
{raw_answer}

Formatted, Polite Final Answer:"""
    prompt = PromptTemplate.from_template(formatter_template)
    return prompt | llm | StrOutputParser()


# --- 3. Main Orchestration Block with RAG Fusion ---
if __name__ == "__main__":
    # --- Initial Setup ---
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cuda"}
    )
    if not os.path.exists(DB_PATH):
        create_vector_store(DB_PATH)
    print("--- Loading existing Chroma vector store... ---")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = db.as_retriever()

    llm = ChatOllama(model=MODEL_NAME, temperature=0, top_p=1, seed=42)

    # --- RAG Fusion Implementation ---
    query_gen_template = """You are a helpful assistant that generates multiple
    search queries based on a single input query.
    Generate 4 other queries that are similar to the original one.
    The queries should be diverse and cover different aspects or phrasings
    of the original question. Provide ONLY the queries, separated by newlines.
    Original Query: {question}
    Generated Queries:"""
    query_gen_prompt = PromptTemplate.from_template(query_gen_template)
    query_generator = (
        query_gen_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    )

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
            for doc, score in sorted(
                fused_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]
        return reranked_results

    fusion_chain = query_generator | retriever.map() | reciprocal_rank_fusion

    final_qa_template = """You are a specialist assistant.
    Answer the user's question based ONLY on the following context.
    If the context does not contain the answer, state that you could
    not find the information in the provided documents.
    Be concise and precise.
    Context: {context}
    Question: {question}
    Answer:"""
    final_qa_prompt = PromptTemplate.from_template(final_qa_template)

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # This chain produces the raw, factual answer
    rag_chain_for_raw_answer = (
        {"context": fusion_chain, "question": RunnablePassthrough()}
        | final_qa_prompt
        | llm
        | StrOutputParser()
    )

    # --- Initialize the Final Formatter ---
    final_formatter_chain = get_final_formatter_chain(llm)

    # --- Main Chat Loop ---
    print("\n--- Advanced RAG Fusion Assistant is Ready ---")
    print("Ask questions about the provided documents. Type 'exit' to quit.")
    print("----------------------------------------------------------\n")

    while True:
        query = input("You: ")
        if not query.strip():
            continue
        if query.lower() in ["exit", "quit"]:
            break
        try:
            print("--- Generating multiple queries and retrieving documents...")

            # 1. Generate alternative queries
            generated_queries = query_generator.invoke({"question": query})

            # 2. Add the original query to the list
            all_queries = generated_queries + [query]
            print(f"--- Searching with {len(all_queries)} queries: {all_queries} ---")

            # 3. Retrieve documents for all queries in parallel
            retrieved_docs = retriever.batch(all_queries)

            # 4. Fuse the results
            fused_docs = reciprocal_rank_fusion(retrieved_docs)

            # 5. Get the raw answer using the fused context
            raw_answer = rag_chain_for_raw_answer.invoke(
                {"context": fused_docs, "question": query}
            )

            # 6. Format the final response
            print("--- Formatting final response... ---")
            final_answer = final_formatter_chain.invoke({"raw_answer": raw_answer})

            print("\nAssistant:", final_answer)
        except Exception as e:
            print(f"An error occurred: {e}")
    print("Assistant: Goodbye!")
