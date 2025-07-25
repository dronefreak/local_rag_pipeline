# src/rag.py
import sys
from pathlib import Path

import hydra
from langchain.load import dumps, loads
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LangChain Imports
from langchain_ollama import ChatOllama
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

# Local Imports
from src.chains import get_final_formatter_chain
from src.data_ingestion_working import (  # <-- The only import we need from here now
    setup_retriever,
)
from src.utils import RichConsoleManager


# RRF function belongs here in the main application logic
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


@hydra.main(version_base=None, config_path="../configs", config_name="rag_pipeline")
def run_rag_pipeline(config: DictConfig):
    """Run the main RAG pipeline using ParentDocumentRetriever and RAG Fusion."""
    console = RichConsoleManager.get_console()
    console.print(OmegaConf.to_yaml(config), style="warning")

    # --- Setup Phase ---
    llm = ChatOllama(
        model=config.model.name,
        temperature=config.model.temperature,
        top_p=config.model.top_p,
        seed=config.model.seed,
    )
    # This single function now handles all data ingestion and retriever setup
    retriever = setup_retriever(config, console)
    if not retriever:
        console.print("!!! Failed to setup retriever. Exiting. !!!", style="bold red")
        return

    # --- RAG Fusion and QA Chain Implementation (no changes here) ---
    query_gen_template = f"""You are a helpful assistant that generates multiple search queries...
    Generate {config.rag_fusion.generated_query_count} other queries...
    Original Query: {{question}}
    Generated Queries:"""
    query_gen_prompt = PromptTemplate.from_template(query_gen_template)
    query_generator = (
        query_gen_prompt | llm | StrOutputParser() | (lambda x: x.strip().split("\n"))
    )

    fusion_chain = query_generator | retriever.map() | reciprocal_rank_fusion

    final_qa_template = """You are a specialist assistant...
    Context: {context}
    Question: {question}
    Answer:"""
    final_qa_prompt = PromptTemplate.from_template(final_qa_template)

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs[:5])

    # The main RAG chain now uses the fusion chain for context
    rag_chain_for_raw_answer = (
        {"context": fusion_chain, "question": RunnablePassthrough()}
        | final_qa_prompt
        | llm
        | StrOutputParser()
    )

    final_formatter_chain = get_final_formatter_chain(llm)

    # --- Main Chat Loop (Updated to use the new chain structure) ---
    console.print("\n--- RAG Assistant is Ready ---", style="bold magenta")
    console.print("Ask complex questions about your documents. Type 'exit' to quit.")
    console.print("-" * 75, style="magenta")

    while True:
        query = input("You: ")
        if not query.strip():
            continue
        if query.lower() in ["exit", "quit"]:
            break

        try:
            with console.status("[bold green]Thinking...[/bold green]"):
                # The input to the final chain is now just the query
                raw_answer = rag_chain_for_raw_answer.invoke(query)

                # Polish the final response
                final_answer = final_formatter_chain.invoke({"raw_answer": raw_answer})

            console.print("\nAssistant:", style="bold green")
            console.print(final_answer)
        except Exception as e:
            console.print(f"An error occurred: {e}", style="bold red")

    console.print("Assistant: Goodbye!", style="bold magenta")


if __name__ == "__main__":
    research_dir = Path(__file__).resolve().parent
    sys.argv.append(f"hydra.run.dir={research_dir}")
    sys.argv.append("hydra.output_subdir=null")
    run_rag_pipeline()
