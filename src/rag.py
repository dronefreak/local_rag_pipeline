# src/rag.py
"""The main entry point and orchestration logic for the RAG pipeline.

This script uses Hydra for configuration management and orchestrates the entire process,
including data ingestion, retriever setup, RAG Fusion, and the interactive user chat
loop.
"""

import logging
import sys
from pathlib import Path

import hydra

# LangChain Imports for building the RAG chain and interacting with the LLM
from langchain.load import dumps, loads
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from omegaconf import DictConfig, OmegaConf

# Local module imports for different pipeline components
from src.chains import get_final_formatter_chain
from src.create_local_datastore import setup_retriever
from src.utils import RichConsoleManager


def reciprocal_rank_fusion(results: list[list], k=60):
    """Applies the Reciprocal Rank Fusion (RRF) algorithm to a list of search results.

    RRF is a method for combining multiple ranked lists of documents into a single,
    more robust ranking. It gives a higher score to documents that appear
    frequently and in high ranks across the different result sets.

    Args:
        results (list[list]): A list where each element is a ranked list of
                               LangChain Document objects from a retriever.
        k (int): A constant used in the RRF formula to control the influence
                 of lower-ranked documents. Defaults to 60.

    Returns:
        list: A single, re-ranked list of LangChain Document objects.
    """
    # Dictionary to store the fused scores for each unique document.
    fused_scores = {}
    # Iterate through each list of retrieved documents.
    for docs in results:
        # Iterate through each document in the list, with its rank.
        for rank, doc in enumerate(docs):
            # Serialize the document object to use it as a dictionary key.
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Add the RRF score to the document's total score.
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their final fused scores in descending order.
    reranked_results = [
        loads(doc)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


@hydra.main(version_base=None, config_path="../configs", config_name="rag_pipeline")
def run_rag_pipeline(config: DictConfig):
    """The main function that orchestrates the entire RAG pipeline.

    This function is decorated with Hydra's `@hydra.main`, which automatically
    loads the configuration from the YAML file specified in `config_path` and
    `config_name` and passes it as a `DictConfig` object.

    The pipeline performs the following steps:
    1. Initializes the console, LLM, and retriever.
    2. Constructs the RAG Fusion chain for advanced document retrieval.
    3. Constructs the final question-answering and formatting chains.
    4. Enters an interactive chat loop to process user queries.

    Args:
        config (DictConfig): The configuration object provided by Hydra,
                             containing all settings for the pipeline.
    """
    # Get the logger for the httpx library
    httpx_logger = logging.getLogger("httpx")
    # Set its level to WARNING to hide INFO messages about successful HTTP requests.
    httpx_logger.setLevel(logging.WARNING)
    # Initialize a Rich console for clean and styled terminal output.
    console = RichConsoleManager.get_console()
    # Print the full configuration for the current run for reproducibility.
    console.print(OmegaConf.to_yaml(config), style="warning")

    # --- Setup Phase ---
    # Initialize the Language Model (LLM) from Ollama with parameters from the config.
    llm = ChatOllama(
        model=config.model.name,
        temperature=config.model.temperature,
        top_p=config.model.top_p,
        seed=config.model.seed,
    )

    # Set up the retriever. This function handles both building the vector store
    # from scratch and loading it from disk if it already exists.
    retriever = setup_retriever(config)
    if not retriever:
        console.print("!!! Failed to setup retriever. Exiting. !!!", style="bold red")
        return

    # --- RAG Fusion and QA Chain Implementation ---
    # Define the prompt for the query generator, which creates multiple perspectives
    # on the user's original question to improve retrieval diversity.
    query_gen_template = f"""You are a helpful assistant that generates multiple search
    queries based on a single input query.
    Generate {config.rag_fusion.generated_query_count}
    other queries that are similar to the original one. The queries should be diverse
    and cover different aspects or phrasings of the original question.
    Provide ONLY the queries, separated by newlines.
    Original Query: {{question}}
    Generated Queries:"""
    query_gen_prompt = PromptTemplate.from_template(query_gen_template)

    # The query generator chain pipes the prompt to the LLM and then splits the output
    # into a list of separate query strings.
    query_generator = (
        query_gen_prompt | llm | StrOutputParser() | (lambda x: x.strip().split("\n"))
    )

    # The RAG Fusion chain combines the query generator, parallel retrieval, and RRF.
    # retriever.map() runs the retriever for each generated query in parallel.
    fusion_chain = query_generator | retriever.map() | reciprocal_rank_fusion

    # Define the prompt for the final question-answering step, which takes the
    # retrieved context and the original question to generate a raw answer.
    final_qa_template = """You are a specialist assistant.
    Answer the user's question based ONLY on the following context.
    If the context does not contain the answer, state that you could
    not find the information in the provided documents.
    Be concise and precise.
    Context: {context}
    Question: {question}
    Answer:"""
    final_qa_prompt = PromptTemplate.from_template(final_qa_template)

    # This is the final RAG chain that produces the raw, factual answer.
    # It takes the user's question, passes it to the fusion_chain to get context,
    # and then passes both to the final QA prompt and LLM.
    rag_chain_for_raw_answer = (
        {"context": fusion_chain, "question": RunnablePassthrough()}
        | final_qa_prompt
        | llm
        | StrOutputParser()
    )

    # Initialize the final formatting chain for polishing the output.
    final_formatter_chain = get_final_formatter_chain(llm)

    # --- Main Interactive Chat Loop ---
    console.print("\n--- RAG Assistant is Ready ---", style="bold magenta")
    console.print("Ask questions about your documents. Type 'exit' to quit.")
    console.print("-" * 75, style="magenta")

    while True:
        query = input("You: ")
        # Handle empty input.
        if not query.strip():
            continue
        # Provide a way to exit the loop.
        if query.lower() in ["exit", "quit"]:
            break

        try:
            # Use Rich's status indicator for a better user
            # experience during processing.
            if config.rag_fusion.stream_final_answer:
                # The spinner is not needed as the user gets immediate feedback.
                console.print("\nAssistant:", style="bold green")

                # Using the .stream() method to get a generator of output chunks.
                # We iterate through the generator and print each chunk as it arrives.
                for chunk in rag_chain_for_raw_answer.stream(query):
                    # Print the chunk to the console without a newline.
                    console.print(chunk, end="", style="bold green")

                # Print a final newline to clean up the output.
                console.print()
            else:
                with console.status("[bold green]Thinking...[/bold green]"):
                    # Invoke the main RAG chain to get the raw answer.
                    raw_answer = rag_chain_for_raw_answer.invoke(query)

                    # Invoke the formatter chain to polish the answer for the user.
                    final_answer = final_formatter_chain.invoke(
                        {"raw_answer": raw_answer}
                    )

                console.print("\nAssistant:", style="bold green")
                console.print(final_answer)
        except Exception as e:
            # Catch and display any errors that occur during the process.
            console.print(f"An error occurred: {e}", style="bold red")

    console.print("Assistant: Goodbye!", style="bold magenta")


if __name__ == "__main__":
    # Standard entry point for the script.
    # The following lines are a workaround for Hydra's default behavior of
    # creating new output directories on each run. This keeps the project clean.
    research_dir = Path(__file__).resolve().parent
    sys.argv.append(f"hydra.run.dir={research_dir}")
    sys.argv.append("hydra.output_subdir=null")
    # This call executes the main function, with Hydra handling the config injection.
    run_rag_pipeline()
