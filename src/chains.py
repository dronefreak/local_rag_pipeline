# --- Core LangChain and RAG Imports ---
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
