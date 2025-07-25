import json
import os
from datetime import datetime

import cv2
import numpy as np
import pytesseract
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from tqdm import tqdm

# === CONFIGURATION ===
INPUT_FOLDER = "data"
TEMP_IMG_DIR = "ocr_temp"
OUTPUT_JSON = "summaries.json"
OUTPUT_MD_DIR = "summaries_md"
TESSERACT_LANG = "eng+fra+spa+de+ita+por"  # Multilingual OCR support

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MD_DIR, exist_ok=True)

# === LLM Setup ===
llm = OllamaLLM(model="llama3-8b-q6k")  # More multilingual than mistral
prompt = PromptTemplate.from_template(
    """
You are a smart assistant tasked with reading diverse documents, such as:

- Technical hardware datasheets
- Sensor specifications
- Instruction manuals
- Books (educational, dietary, technical)
- Game rulebooks
- Research guidelines

The source is a document titled: **{document_name}**, found in the folder: **{category}**

The document may be written in English, Spanish, French, German, Italian, Portuguese, or another language. Please process the text accordingly and respond in the same language.

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

Also try to extract metadata such as the author(s), publication date, or source if available.

Avoid page numbers, headers, or formatting artifacts.
"""
)
runnable = prompt | llm

# === Summarization Process ===
summary_results = []

for root, dirs, files in os.walk(INPUT_FOLDER):
    for filename in files:
        if not filename.lower().endswith(".pdf"):
            continue

        md_filename = os.path.join(OUTPUT_MD_DIR, f"{filename}.md")
        if os.path.exists(md_filename):
            print(f"⚠️ Skipping {filename} - already processed.")
            continue

        category = os.path.basename(root)
        pdf_path = os.path.join(root, filename)
        print(f"\n🔍 Processing: {category}/{filename}")
        images = convert_from_path(
            pdf_path, dpi=300, output_folder=TEMP_IMG_DIR, fmt="png"
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
                        str(doc_info.creation_date) if doc_info.creation_date else None
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
                    print(f"  - Rotating page {i+1} by {rotation} degrees...")
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
                print(f"  - Could not get orientation for page {i+1}: {e}")
            # --- END of Correction ---

            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15
            )

            text = pytesseract.image_to_string(enhanced, lang=TESSERACT_LANG).strip()
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
                f.write("## 📄 Document Metadata\n")
                for key, val in metadata.items():
                    if val:
                        f.write(f"- **{key.capitalize()}**: {val}\n")
                f.write("\n")

            f.write("## 📑 Table of Contents\n")
            f.write("\n".join(toc_entries) + "\n\n")

            for item in doc_summaries:
                f.write(f"## Page {item['page']}\n\n")
                f.write(f"**Summary:**\n\n{item['summary']}\n\n")
                f.write(
                    f"<details><summary>Raw OCR Text</summary>\n\n```{item['ocr_text']}```\n</details>\n\n"
                )

# === Output all to JSON ===
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary_results, f, ensure_ascii=False, indent=2)

print(
    f"\n✅ Done! Markdown summaries saved per document in `{OUTPUT_MD_DIR}` with metadata and TOC. Full data saved in `{OUTPUT_JSON}`."
)
