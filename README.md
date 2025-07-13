# Local RAG Fusion AI Assistant

Welcome to the **Local RAG Fusion AI Assistant**, a powerful, private, and highly advanced retrieval-augmented generation (RAG) pipeline that runs entirely on your local machine. This project is the culmination of building a robust system capable of understanding and answering questions from a complex, nested library of private documents, while also leveraging GPU acceleration for high performance.

This is a realtively simple Q&A bot but a few sophisticated implementations for for data ingestion, retrieval, and response generation allow for versatile use cases.

## Key Strengths & Advanced Features

This project showcases a modern, state-of-the-art local RAG architecture, moving beyond basic prototypes to a system built for accuracy and scalability.

- ** Intelligent Retrieval with RAG Fusion:** Instead of a simple vector search, this pipeline uses RAG Fusion. It takes a single user query, generates multiple perspectives on it, searches for all of them in parallel, and uses a Reciprocal Rank Fusion algorithm to intelligently re-rank and fuse the results. This makes retrieval highly resilient to the user's phrasing and excellent at finding information buried deep within documents.

- ** Hierarchical Data Ingestion:** The system is designed to handle complex, real-world document structures. It automatically walks through nested subdirectories, extracting and tagging data with hierarchical metadata (e.g., `topic`, `sub_topic`). This allows for highly organized, department-like data segregation. Please note that this can also be used for personal documents as well, even if they are segregated and nested deeply.

- ** Robust & Resilient PDF Parsing:** The data ingestion pipeline uses a powerful fallback mechanism. It first attempts to parse PDFs with the extremely fast `PyMuPDFLoader`. If it detects a protected, image-based, or malformed PDF, it automatically falls back to a more powerful OCR (Optical Character Recognition) engine to ensure no document is left behind.

- ** Fully GPU Accelerated:** Every computationally intensive part of the pipeline is optimized to run on a local NVIDIA GPU via CUDA (if available):
  - **Text Embeddings:** Document and query embeddings are generated on the GPU for a significant speed-up during data ingestion and runtime.
  - **LLM Inference:** The language model is served by Ollama, which ensures all possible layers are offloaded to your GPU's VRAM for near-instantaneous response times (e.g., 70+ tokens/second).

- ** 100% Local and Private:** Your documents and your questions never leave your machine. The entire pipeline, from the embedding model to the powerful LLM, runs locally, guaranteeing complete data privacy and security (especially requried in case of sensitive documents).

- ** State-of-the-Art Local LLM:** The system is configured to use top-tier local models like Llama 3 or Mixtral via Ollama, providing high-quality reasoning and instruction-following capabilities without relying on external APIs. The model choice can be configured externally via using the `Modelfile` file.

- ** Polished Final Output:** A final "formatter" chain ensures that all responses are clean, polite, and consistently structured, creating a professional and user-friendly experience.

## Setup and Installation

Follow these steps to get your own local AI assistant running.

### 1. Prerequisites

- An **NVIDIA GPU** with CUDA installed (CUDA 12.1+ recommended). The system should still work without a GPU, but it could be very slow (perhaps a performance speed up can be tested by using smaller quantised models)
- **Ollama** installed and running. Get it from [ollama.com](https://ollama.com).
- **Tesseract OCR Engine**. See installation instructions [here](https://tesseract-ocr.github.io/tessdoc/Installation.html).

### 2. Initial Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dronefreak/local_rag_pipeline.git
    cd local_rag_pipeline
    ```
2.  **Create the virtual environment (with pre-installed dependencies):**
    ```bash
    conda env create -f environment.yml
    ```
3.  **Install `llama-cpp-python` with CUDA support (if not already installed):**
    - **For Windows:**

      ```cmd
        $env:CMAKE_ARGS = "-DGGML_CUDA=on"
        pip install llama-cpp-python

      ```

    - **For Linux/macOS:**
      ```bash
        CMAKE_ARGS="-DGGML_CUDA=on" LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"  pip install llama-cpp-python
      ```
      Please note that `llama-cpp-python` is under heavy active development, which is why the code snippets and function calls might be outdated sooner than expected. Please adapt the calls according to the versions you are using in your projects.

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

1.  **Download a model with Ollama:** I personally recommend starting with a powerful model like Llama 3 8B or Mixtral. Please install Ollama first (it is fairly simple), now run the following command in a separate terminal:
    `bash
    ollama run llama3:8b # Or 'ollama run mixtral'
    `
    Please note that you can also create your custom ollama servings using a `Modelfile` that uses a local downloaded .gguf model. Look under the `modelfiles` folder for more information.
2.  **Update `rag.py`:** Change the `MODEL_NAME` global constant to match the model you downloaded (e.g., `MODEL_NAME = "llama3:8b"`).
3.  **Run the application:**
    ```bash
    python rag.py
    ```
4.  The first time you run it, the script will build the vector database. This may take some time, especially if you have many PDFs requiring OCR. Subsequent runs will be much faster.

## Future Improvements & Next Steps

This project provides a decent foundation. I am currently working on some updates for the future, please let me know in discussions if there is something you would like to see specifically:

- **Implement a Web UI:**
  - Wrap the application in a user-friendly web interface using **Streamlit** or **Gradio**. This would allow for easy interaction, displaying chat history, and visualizing source documents.

- **Upgrade to More Powerful Models:**
  - As new and better open-source models are released, you can easily test them. For example, moving from an 8B model to a 70B model (like `llama3:70b`) on a machine with more VRAM (e.g., RTX 4090 with 24GB) would provide a massive leap in reasoning capabilities. I am current running Ubuntu 24 on a PC with RTX 4070 Super 12GB VRAM, it works just fine for the 8B models. Consider changing the models to suit your needs and according to the hardware specifications on your paltform.

- **Advanced Agentic Workflow with Tools:**
  - Re-integrate the agent architecture to give the bot more tools, such as a **Calculator**, a **Python REPL** for data analysis, or the **Tavily Web Search** for real-time information. The current RAG Fusion pipeline could become one powerful tool that the agent can choose to use. There is a loose implementation of **Tavily Web Search** already in the pipeline, but I would not recommend using it for the moment, perhaps I will remove it in the future.

- **Systematic Evaluation with RAGAs:**
  - Implement the **RAGAs** (RAG Assessment) framework to quantitatively measure the performance of the pipeline. This allows you to objectively test changes (like different chunking strategies or embedding models) and see their impact on metrics like `Faithfulness` and `Answer Relevancy`.

- **Explore Alternative Embedding Models:**
  - While `all-MiniLM-L6-v2` is fast and efficient, you could experiment with larger, more powerful embedding models like `bge-large-en-v1.5` or Cohere's `embed-english-v3.0` to potentially improve retrieval accuracy, especially for nuanced or domain-specific language.

---
