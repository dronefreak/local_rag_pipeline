# Local RAG Fusion AI Assistant

![AI Assistant Banner](httpsd://i.imgur.com/your-banner-image.png) <!-- Optional: Create and upload a banner image -->

Welcome to the **Local RAG Fusion AI Assistant**, a powerful, private, and highly advanced retrieval-augmented generation (RAG) pipeline that runs entirely on your local machine. This project is the culmination of building a robust system capable of understanding and answering questions from a complex, nested library of private documents, while also leveraging GPU acceleration for high performance.

This is not just a simple Q&A bot. It's a sophisticated system designed for enterprise-level challenges, featuring state-of-the-art techniques for data ingestion, retrieval, and response generation.

## ✨ Key Strengths & Advanced Features

This project showcases a modern, state-of-the-art local RAG architecture, moving beyond basic prototypes to a system built for accuracy and scalability.

- **🧠 Intelligent Retrieval with RAG Fusion:** Instead of a simple vector search, this pipeline uses RAG Fusion. It takes a single user query, generates multiple perspectives on it, searches for all of them in parallel, and uses a Reciprocal Rank Fusion algorithm to intelligently re-rank and fuse the results. This makes retrieval highly resilient to the user's phrasing and excellent at finding information buried deep within documents.

- **📂 Hierarchical Data Ingestion:** The system is designed to handle complex, real-world document structures. It automatically walks through nested subdirectories, extracting and tagging data with hierarchical metadata (e.g., `topic`, `sub_topic`). This allows for highly organized, department-like data segregation.

- **🤖 Robust & Resilient PDF Parsing:** The data ingestion pipeline uses a powerful fallback mechanism. It first attempts to parse PDFs with the extremely fast `PyMuPDFLoader`. If it detects a protected, image-based, or malformed PDF, it automatically falls back to a more powerful OCR (Optical Character Recognition) engine to ensure no document is left behind.

- **🚀 Fully GPU Accelerated:** Every computationally intensive part of the pipeline is optimized to run on a local NVIDIA GPU via CUDA:
  - **Text Embeddings:** Document and query embeddings are generated on the GPU for a significant speed-up during data ingestion and runtime.
  - **LLM Inference:** The language model is served by Ollama, which ensures all possible layers are offloaded to your GPU's VRAM for near-instantaneous response times (e.g., 70+ tokens/second).

- **🔒 100% Local and Private:** Your documents and your questions never leave your machine. The entire pipeline, from the embedding model to the powerful LLM, runs locally, guaranteeing complete data privacy and security.

- **💡 State-of-the-Art Local LLM:** The system is configured to use top-tier local models like Llama 3 or Mixtral via Ollama, providing high-quality reasoning and instruction-following capabilities without relying on external APIs.

- **✨ Polished Final Output:** A final "formatter" chain ensures that all responses are clean, polite, and consistently structured, creating a professional and user-friendly experience.

## 🛠️ Setup and Installation

Follow these steps to get your own local AI assistant running.

### 1. Prerequisites

- An **NVIDIA GPU** with CUDA installed (CUDA 12.1+ recommended).
- **Ollama** installed and running. Get it from [ollama.com](https://ollama.com).
- **Tesseract OCR Engine**. See installation instructions [here](https://tesseract-ocr.github.io/tessdoc/Installation.html).

### 2. Initial Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install all required dependencies:**
    This project uses specific versions for stability. It's recommended to install them with this command:
    ```bash
    pip install "nougat-ocr==0.1.17" "timm==0.5.4" "transformers==4.38.2" "sentence-transformers==2.7.0" "packaging==24.0" chromadb langchain langchain-community langchain-chroma langchain-experimental pydantic pymupdf pytesseract unstructured "unstructured[pdf]" --force-reinstall
    ```
4.  **Install `llama-cpp-python` with CUDA support:**
    - **For Windows:**
      ```cmd
      set CMAKE_ARGS="-DLLAMA_CUBLAS=on" && set FORCE_CMAKE=1 && pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
      ```
    - **For Linux/macOS:**
      ```bash
      export CMAKE_ARGS="-DLLAMA_CUBLAS=on" && export FORCE_CMAKE=1 && pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
      ```

### 3. Prepare Your Data

1.  Place all your documents inside the `data/` directory.
2.  Organize them into a nested folder structure. The top-level folder name will be used as the `topic` metadata.
    ```
    data/
    ├── topic_A/
    │   └── sub_topic_1/
    │       └── doc1.pdf
    └── topic_B/
        └── doc2.txt
    ```

### 4. Configure and Run

1.  **Download a model with Ollama:** We recommend starting with a powerful model like Llama 3 8B or Mixtral.
    ```bash
    ollama run llama3:8b # Or 'ollama run mixtral'
    ```
2.  **Update `main.py`:** Change the `MODEL_NAME` global constant to match the model you downloaded (e.g., `MODEL_NAME = "llama3:8b"`).
3.  **Run the application:**
    ```bash
    python main.py
    ```
4.  The first time you run it, the script will build the vector database. This may take some time, especially if you have many PDFs requiring OCR. Subsequent runs will be much faster.

## 🚀 Future Improvements & Next Steps

This project provides a solid foundation. Here are some exciting directions for future development:

- **Implement a Web UI:**
  - Wrap the application in a user-friendly web interface using **Streamlit** or **Gradio**. This would allow for easy interaction, displaying chat history, and visualizing source documents.

- **Upgrade to More Powerful Models:**
  - As new and better open-source models are released, you can easily test them. For example, moving from an 8B model to a 70B model (like `llama3:70b`) on a machine with more VRAM (e.g., RTX 4090 with 24GB) would provide a massive leap in reasoning capabilities.

- **Advanced Agentic Workflow with Tools:**
  - Re-integrate the agent architecture to give the bot more tools, such as a **Calculator**, a **Python REPL** for data analysis, or the **Tavily Web Search** for real-time information. The current RAG Fusion pipeline could become one powerful tool that the agent can choose to use.

- **Systematic Evaluation with RAGAs:**
  - Implement the **RAGAs** framework to quantitatively measure the performance of the pipeline. This allows you to objectively test changes (like different chunking strategies or embedding models) and see their impact on metrics like `Faithfulness` and `Answer Relevancy`.

- **Explore Alternative Embedding Models:**
  - While `all-MiniLM-L6-v2` is fast and efficient, you could experiment with larger, more powerful embedding models like `bge-large-en-v1.5` or Cohere's `embed-english-v3.0` to potentially improve retrieval accuracy, especially for nuanced or domain-specific language.

- **Implement a Fine-grained Metadata Cache for Nougat:**
  - The current Nougat implementation assigns a default department. A more advanced system could save the original hierarchical metadata for each PDF and re-apply it after the `.mmd` file is generated, allowing OCR'd documents to be filtered with the same precision as text-based ones.

---
