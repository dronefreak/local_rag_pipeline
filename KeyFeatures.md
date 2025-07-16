## Key Strengths & Advanced Features

This project is built using a modern, state-of-the-art local RAG architecture, something that I did not find in other implementations:

- **Intelligent Retrieval with RAG Fusion:** Instead of a simple vector search, this pipeline uses RAG Fusion. It takes a single user query, generates multiple perspectives on it, searches for all of them in parallel, and uses a Reciprocal Rank Fusion algorithm to intelligently re-rank and fuse the results. This makes retrieval highly resilient to the user's phrasing and excellent at finding information buried deep within documents.

- **Hierarchical Data Ingestion:** The system is designed to handle complex, real-world document structures. It automatically walks through nested subdirectories, extracting and tagging data with hierarchical metadata (e.g., `topic`, `sub_topic`). This allows for highly organized, department-like data segregation. Please note that this can also be used for personal documents as well, even if they are segregated and nested deeply.

- **Robust & Resilient PDF Parsing:** The data ingestion pipeline uses a powerful fallback mechanism. It first attempts to parse PDFs with the extremely fast `PyMuPDFLoader`. If it detects a protected, image-based, or malformed PDF, it automatically falls back to a more powerful OCR (Optical Character Recognition) engine to ensure no document is left behind.

- **Fully GPU Accelerated:** Every computationally intensive part of the pipeline is optimized to run on a local NVIDIA GPU via CUDA (if available):
  - **Text Embeddings:** Document and query embeddings are generated on the GPU for a significant speed-up during data ingestion and runtime.
  - **LLM Inference:** The language model is served by Ollama, which ensures all possible layers are offloaded to your GPU's VRAM for near-instantaneous response times (e.g., 70+ tokens/second).

- **100% Local and Private:** Your documents and your questions never leave your machine. The entire pipeline, from the embedding model to the powerful LLM, runs locally, guaranteeing complete data privacy and security (especially requried in case of sensitive documents).

- **State-of-the-Art Local LLM:** The system is configured to use top-tier local models like Llama 3 or Mixtral via Ollama, providing high-quality reasoning and instruction-following capabilities without relying on external APIs. The model choice can be configured externally via using the `Modelfile` file.

- **Polished Final Output:** A final "formatter" chain ensures that all responses are clean, polite, and consistently structured, creating a professional and user-friendly experience.
