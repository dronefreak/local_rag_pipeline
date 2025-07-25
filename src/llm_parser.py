import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import hydra
import numpy as np
import pytesseract
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from omegaconf import DictConfig, OmegaConf
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from tqdm import tqdm

from src.utils import RichConsoleManager


@hydra.main(version_base=None, config_path="../configs", config_name="rag_pipeline")
def parse_data_with_llms(config: DictConfig):
    # Initialize a Rich console for clean and styled terminal output.
    console = RichConsoleManager.get_console()
    # Print the full configuration for the current run for reproducibility.
    console.print(OmegaConf.to_yaml(config), style="warning")

    if config.llm_parser.save_imgs:
        os.makedirs(config.llm_parser.imgs_dir, exist_ok=True)
    os.makedirs(config.llm_parser.output_markdown_dir, exist_ok=True)

    # === LLM Setup ===
    llm = OllamaLLM(model=config.model.name)  # More multilingual than mistral
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

    # === Summarization Process ===
    summary_results = []

    for root, dirs, files in os.walk(config.dataset_path):
        for filename in files:
            if not filename.lower().endswith(".pdf"):
                continue

            md_filename = os.path.join(
                config.llm_parser.output_markdown_dir, f"{filename}.md"
            )
            if os.path.exists(md_filename):
                console.print(f"Skipping {filename} - already processed.", style="info")
                continue

            category = os.path.basename(root)
            pdf_path = os.path.join(root, filename)
            console.print(f"\nProcessing: {category}/{filename}", style="info")
            if config.llm_parser.save_imgs:
                images = convert_from_path(
                    pdf_path,
                    dpi=300,
                    output_folder=config.llm_parser.imgs_dir,
                    fmt="png",
                )

            # Extract metadata from PDF
            metadata = {}
            try:
                reader = PdfReader(pdf_path)
                doc_info = reader.metadata
                if doc_info:
                    metadata = {
                        "author": doc_info.author,
                        "creator": doc_info.creator,
                        "producer": doc_info.producer,
                        "subject": doc_info.subject,
                        "title": doc_info.title,
                        "created": (
                            str(doc_info.creation_date)
                            if doc_info.creation_date
                            else None
                        ),
                    }
            except Exception as e:
                metadata = {"error": f"Metadata extraction failed: {e}"}

            doc_summaries = []
            toc_entries = []

            for i, img in tqdm(enumerate(images)):
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # --- Orientation and Skew Correction ---
                try:
                    # Get orientation data from Tesseract
                    osd = pytesseract.image_to_osd(
                        img_cv, output_type=pytesseract.Output.DICT
                    )
                    rotation = osd["rotate"]
                    # Rotate the image to the correct orientation
                    if rotation > 0:
                        console.print(
                            f"  - Rotating page {i+1} by {rotation} degrees...",
                            style="warning",
                        )
                        (h, w) = img_cv.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, -rotation, 1.0)
                        img_cv = cv2.warpAffine(
                            img_cv,
                            M,
                            (w, h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE,
                        )
                except Exception as e:
                    console.print(
                        f"  - Could not get orientation for page {i+1}: {e}",
                        style="info",
                    )
                # --- END of Correction ---

                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                enhanced = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15
                )

                text = pytesseract.image_to_string(
                    enhanced, lang=config.llm_parser.tesseract_languages
                ).strip()
                if not text:
                    continue

                summary = runnable.invoke(
                    {
                        "document_name": filename,
                        "category": category,
                        "page_number": i + 1,
                        "page_text": text,
                    }
                )

                result = {
                    "document": filename,
                    "category": category,
                    "page": i + 1,
                    "summary": summary,
                    "ocr_text": text,
                    "metadata": metadata,
                }

                summary_results.append(result)
                doc_summaries.append(result)
                toc_entries.append(f"- [Page {i + 1}](#page-{i + 1})")

            # Save per-document markdown
            with open(md_filename, "w", encoding="utf-8") as f:
                f.write(f"# Summary for {filename}\n")
                f.write(f"_Generated: {datetime.now().isoformat()}\n\n")

                if metadata:
                    f.write("## Document Metadata\n")
                    for key, val in metadata.items():
                        if val:
                            f.write(f"- **{key.capitalize()}**: {val}\n")
                    f.write("\n")

                f.write("## Table of Contents\n")
                f.write("\n".join(toc_entries) + "\n\n")

                for item in doc_summaries:
                    f.write(f"## Page {item['page']}\n\n")
                    f.write(f"**Summary:**\n\n{item['summary']}\n\n")
                    f.write(
                        f"<details><summary>Raw OCR Text</summary>\n\n"
                        f" ```{item['ocr_text']}```\n</details>\n\n"
                    )

    # === Output all to JSON ===
    with open(config.llm_parser.output_json, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)

    console.print(
        f"\n Markdown summaries saved in `{config.llm_parser.output_markdown_dir}`"
        f" Full data saved in `{config.llm_parser.output_json}`."
    )


if __name__ == "__main__":
    # Standard entry point for the script.
    # The following lines are a workaround for Hydra's default behavior of
    # creating new output directories on each run. This keeps the project clean.
    research_dir = Path(__file__).resolve().parent
    sys.argv.append(f"hydra.run.dir={research_dir}")
    sys.argv.append("hydra.output_subdir=null")
    # This call executes the main function, with Hydra handling the config injection.
    parse_data_with_llms()
