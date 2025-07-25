# Local RAG Fusion AI Assistant

Welcome to the **Local RAG Fusion AI Assistant**, a powerful, private, and highly advanced retrieval-augmented generation (RAG) pipeline that runs entirely on your local machine. This is a realtively simple Q&A bot but a few sophisticated implementations for for data ingestion, retrieval, and response generation allow for versatile use cases.

## Setup and Installation

Follow these steps to get your own local AI assistant up and running!

### 1. Prerequisites

- The system should still work with a CPU, but it could be slow (perhaps a performance speed up can be tested by using smaller quantised models), I would personally recommend to have a **NVIDIA GPU** with CUDA installed (CUDA 12.3+ preferably).
- **Ollama** installed and running. Get it from [ollama.com](https://ollama.com).

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
3.  **Install `llama-cpp-python` (if not already installed) and other requirements from the given .txt file:**

    ```bash
      pip install -r requirements.txt
    ```

    Please note that `llama-cpp-python` is under heavy active development, which is why the code snippets and function calls might be outdated sooner than expected. Please adapt the calls according to the versions you are using in your projects.
    It is totally possible that the installation fails for `llama-cpp-python` while trying to build it for CUDA, it is toally possible to use CPU (much easier build process) but with some speed limitations.

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
3.  You can find some data samples that I was using personally on my side [here](https://drive.google.com/drive/folders/14E74tuIjifEIZlxSCR2WUpSBUx3d8jMg?usp=sharing). These documents are available on the internet for free and there is no copyright issue, in fact most of them are datasheets of electronic equipments that are made public by the companies themselves. Just download and unzip the folder `data.zip` and you will have a pre-formatted data directory structure ready to use. For the moment, you can parse CSV, PDFs and .TXT files. Other file strucutres are to be added soon.

If you also want the pre

### 4. Configure and Run

1.  **Download a model with Ollama:** I personally recommend starting with a simpler model like Llama or Mistral. Please install Ollama first (it is fairly simple), now run the following command in a separate terminal:
    `bash
ollama run llama3 # Or 'ollama run mistral'
`
    Please note that you can also create your custom ollama servings using a `Modelfile` that uses a local downloaded .gguf model. Look under the `modelfiles` folder for more information.
2.  **Run the application:**
    ```bash
    python src/rag.py model.name=llama3.2:3b model.type=llama
    ```
3.  Please check the `configs/rag_pipeline.yaml` and other YAML files in the project for more configurable parameters.

The first time you run it, the script will build the vector database. This may take some time, especially if you have many PDFs with tables inside. Subsequent runs will be much faster.

**Important Note**

If you want, you can pre-create a vector store and datastore by simply running the following command:

```
python src/create_local_datastore.py vectorstore_path=vectorstore docstore_path=docstore.pkl dataset_path=data
```

Since pre-processing PDFs can be time consuming and tricky if the PDF contains scanned pages, perhaps it is better to pre-process the data using `llm-parser.py` by simply running the following command:

```
python src/llm_parser.py model.name=llama3.2:3 dataset_path=data
```

Make sure your dataset paths are correctly inside the `llm_parser.yaml` config.

## Usage Example

Once you run the `python src/rag.py` you can play around the app like this:

```
--- RAG Fusion Assistant is Ready ---
Ask questions about the provided documents. Type 'exit' to quit.
----------------------------------------------------------

You: What is the supply voltage range for the ADXL380 accelerometer?
⠹ Thinking...

Assistant: Thank you for your inquiry! According to the information provided by the system, the ADXL380 accelerometer
operates within a specific power supply range. Specifically, it requires a voltage between 2.25 V and 3.6 V, as
stated in Table 1 of its data sheet.

Additionally, this device also utilizes an independent supply voltage (VDDIO) and features an internal low dropout
regulator that can be bypassed if needed. If you have any further questions or concerns regarding the power
requirements for this accelerometer, please feel free to ask!


You: exit
Assistant: Goodbye!
```

This is an example created using the datasheet of ADXL380 accelerometer, if you want you can do the same process with other specific documents as well.

---

## Project structure

```
├── src
│   ├── chains.py
│   ├── data_ingestion.py
│   ├── rag.py
│   └── utils.py
```

- [configs/rag_pipeline.yaml](configs/rag_pipeline.yaml): In this YAML file you can set the desired parameters for the base LLM model, specify data paths etc.
- [configs/data_ingestion/data_ingestion.yaml](configs/data_ingestion/data_ingestion.yaml): In this YAML file you can set the desired parameters for the creation of local databases.
- [configs/data_ingestion/llm_parser.yaml](configs/data_ingestion/llm_parser.yaml): In this YAML file you can set the desired parameters for the `llm_parser.py` script.
- [configs/data_ingestion/model.yaml](configs/data_ingestion/model.yaml): In this YAML file you can set the desired parameters for the base LLM model.
- [src/chains.py](src/chains.py): The file that contains the chains required for the RAG pipeline.
- [src/create_local_datastore.py](src/create_local_datastore.py): The file that contains the `create_vector_store` function and its helper functions.
- [src/rag.py](src/rag.py): Main executable for the project.
- [src/utils.py](src/utils.py): Some utility functions required and used in this project.
- [src/llm_parser.py](src/llm_parser.py): An independent parsing technique using OCR + LLMs for a variety of file types.

If you like this project, leave a star so and please feel free to point out issues and open up discussions on what could be improved/changed or done better in this project.

As always, Hare Krishna and happy coding!
