import os
import sys
from pathlib import Path

import hydra

# --- Core LangChain and RAG Imports ---
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from omegaconf import DictConfig, OmegaConf

from src.chains import get_final_formatter_chain
from src.data_ingestion import create_vector_store, reciprocal_rank_fusion
from src.utils import get_rich_console


@hydra.main(version_base=None, config_path="../configs", config_name="rag_pipeline")
def run_rag_pipeline(config: DictConfig):
    """Run the main RAG pipeline. This pipeline runs the selected RAG fusion model on
    the provided queries and documents, retrieves relevant information, and formats the
    final answer using the specified LLM and output parser.

    Parameters
    ----------
    config : DictConfig
        Hydra configuration containing all the parameters to be used for the pipeline
    """
    console = get_rich_console()
    console.print(OmegaConf.to_yaml(config), style="warning")
    console.print("\n--- Advanced RAG Fusion Assistant ---", style="info")
    console.print("Loading documents and initializing vector store...\n", style="info")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model, model_kwargs={"device": config.device}
    )
    if not os.path.exists(config.vectorstore_path):
        create_vector_store(config, console=console)
    console.print("--- Loading existing Chroma vector store... ---")
    db = Chroma(
        persist_directory=config.vectorstore_path, embedding_function=embeddings
    )
    retriever = db.as_retriever()

    llm = ChatOllama(
        model=config.model.name,
        temperature=config.model.temperature,
        top_p=config.model.top_p,
        seed=config.model.seed,
    )

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
    console.print("\n--- RAG Fusion Assistant is Ready ---", style="info")
    console.print("Ask questions about the provided documents. Type 'exit' to quit.")
    console.print("----------------------------------------------------------\n")

    while True:
        query = input("You: ")
        if not query.strip():
            continue
        if query.lower() in ["exit", "quit"]:
            break
        try:
            console.print(
                "--- Generating queries and retrieving relevant documents...",
                style="info",
            )

            # 1. Generate alternative queries
            generated_queries = query_generator.invoke({"question": query})

            # 2. Add the original query to the list
            all_queries = generated_queries + [query]
            console.print(
                f"--- Searching with {len(all_queries)} queries: {all_queries} ---"
            )

            # 3. Retrieve documents for all queries in parallel
            retrieved_docs = retriever.batch(all_queries)

            # 4. Fuse the results
            fused_docs = reciprocal_rank_fusion(retrieved_docs)

            # 5. Get the raw answer using the fused context
            raw_answer = rag_chain_for_raw_answer.invoke(
                {"context": fused_docs, "question": query}
            )

            # 6. Format the final response
            console.print("--- Formatting final response... ---")
            final_answer = final_formatter_chain.invoke({"raw_answer": raw_answer})

            console.print("\nAssistant:", final_answer)
        except Exception as e:
            console.print(f"An error occurred: {e}", style="danger")
    console.print("Assistant: Goodbye!", style="info")


if __name__ == "__main__":
    # This hack is required to prevent hydra from creating output directories.
    research_dir = Path(__file__).resolve().parent
    sys.argv.append(f"hydra.run.dir={research_dir}")
    sys.argv.append("hydra.output_subdir=null")
    # Disable Pylint warning about missing parameter, config is passed by Hydra.
    # pylint: disable=no-value-for-parameter
    run_rag_pipeline()
