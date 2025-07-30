# src/llm_parser.py
"""This script performs an advanced, LLM-based parsing of PDF and other document types.

It processes files in a source directory including PDFs, DOCX, PPTX, CSVs, images,
plain text, HTML, and EPUB, performs OCR (if needed),
and uses a large language model (LLM) to generate structured summaries.

The output is saved in two formats:
1. A detailed JSON file containing all results.
2. Individual Markdown files for each document, providing a human-readable
   summary with metadata and collapsible sections for the raw extracted text.
"""

import json
import sys
from pathlib import Path

import hydra
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

from src.utils import (
    RichConsoleManager,
    generate_markdown_from_json,
    process_single_document,
)


@hydra.main(version_base=None, config_path="../configs", config_name="rag_pipeline")
def parse_data_with_llms(config: DictConfig):
    console = RichConsoleManager.get_console()
    console.print(OmegaConf.to_yaml(config), style="warning")

    input_folder = Path(config.dataset_path)
    output_markdown_dir = Path(config.dataset_path)
    output_json_path = Path(config.llm_parser.output_json)
    output_markdown_dir.mkdir(exist_ok=True)

    if config.llm_parser.save_imgs:
        imgs_dir = Path(config.llm_parser.imgs_dir)
        imgs_dir.mkdir(exist_ok=True)
    else:
        imgs_dir = None

    llm = OllamaLLM(model=config.model.name)
    prompt = PromptTemplate.from_template(
        """
    You are a smart assistant tasked with reading diverse documents, such as:

    - Technical hardware datasheets
    - Sensor specifications
    - Instruction manuals
    - Books (educational, dietary, technical)
    - Game rulebooks
    - Research guidelines

    The source is a document titled: **{document_name}**,
    found in the folder: **{category}**

    The document may be written in English, Spanish, French,
    German, Italian, Portuguese, or another language.
    Please process the text accordingly and respond in the same language.

    Below is the content from page {page_number}:
    ---
    {page_text}
    ---

    Please summarize the key ideas or rules in **clear bullet points**.
    Adapt to the document type:
    - If it’s a **datasheet**, extract specs, features, and usage notes.
    - If it’s a **rulebook**, extract gameplay rules, setup, and conditions.
    - If it’s a **book**, extract core arguments, teachings, or steps.
    - If it’s a **guide/manual**, summarize the procedures or best practices.
    - If it’s a **scientific paper**, extract objective, methodology, and findings.

    Also try to extract metadata such as the author(s), publication date,
    or source if available.

    Avoid page numbers, headers, or formatting artifacts.
    """
    )
    runnable = prompt | llm
    supported_exts = [
        ".pdf",
        ".docx",
        ".pptx",
        ".csv",
        ".jpg",
        ".jpeg",
        ".png",
        ".txt",
        ".html",
        ".htm",
        ".epub",
    ]

    # === Summarization Process ===
    all_summary_results = []  # We will still collect all results for the final big JSON

    # Use pathlib's rglob for recursive file discovery.
    # We will now use a progress bar from `rich` instead of `tqdm` for consistency
    files_to_process = [
        p for p in input_folder.rglob("*") if p.suffix.lower() in supported_exts
    ]

    # We iterate over files first
    for path in track(files_to_process, description="Processing all documents..."):
        json_output_filename = path.parent / f"{path.name}.json"
        md_filename = path.parent / f"{path.name}.md"

        if json_output_filename.exists():
            console.print(f"Skipping {path.name} - JSON already exists.", style="info")
            if not md_filename.exists():
                generate_markdown_from_json(json_output_filename, md_filename)
            continue

        page_texts, metadata = process_single_document(path, config, console)

        if not page_texts:
            console.print(
                f"No text extracted from {path.name}. Skipping.", style="warning"
            )
            continue

        doc_page_results = []
        for i in range(0, len(page_texts), config.llm_parser.batch_size):
            batch = page_texts[i : i + config.llm_parser.batch_size]

            combined_text = "\n\n--- PAGE BREAK ---\n\n".join(
                [f"Content from Page {p[0]}:\n{p[1]}" for p in batch]
            )
            page_numbers = ", ".join([str(p[0]) for p in batch])

            summary = runnable.invoke(
                {
                    "document_name": path.name,
                    "category": path.parent.name,
                    "page_number": page_numbers,
                    "page_text": combined_text,
                }
            )

            for j, (page_num, page_text) in enumerate(batch):
                result = {
                    "document": path.name,
                    "category": path.parent.name,
                    "page": page_num,
                    "summary": (
                        summary
                        if j == 0
                        else "See summary for page range starting at "
                        + str(batch[0][0])
                    ),
                    "ocr_text": page_text,
                    "source_metadata": metadata,
                }
                doc_page_results.append(result)

        # Save results for this document
        if doc_page_results:
            with open(json_output_filename, "w", encoding="utf-8") as f:
                json.dump(doc_page_results, f, ensure_ascii=False, indent=2)
            generate_markdown_from_json(json_output_filename, md_filename)
            all_summary_results.extend(doc_page_results)

    # Output the final compilation JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_summary_results, f, ensure_ascii=False, indent=2)

    console.print(
        f"\n Individual summaries saved in `{output_markdown_dir}`.\n"
        f" Full data compilation saved in `{output_json_path}`."
    )


if __name__ == "__main__":
    research_dir = Path(__file__).resolve().parent
    sys.argv.append(f"hydra.run.dir={research_dir}")
    sys.argv.append("hydra.output_subdir=null")
    parse_data_with_llms()
