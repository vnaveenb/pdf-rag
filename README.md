# PDF Question Answering with RAG

A Retrieval-Augmented Generation (RAG) system that allows users to upload multiple PDF documents and ask questions in natural language. The application extracts text from PDFs, processes it, and returns precise answers with source information.

## Features

- **PDF Upload:** Upload and process multiple PDF files simultaneously
- **Layout-Aware Text Extraction:** Handle both single-column and multi-column PDF layouts
- **Smart Text Chunking:** Divide text into meaningful chunks with configurable size and overlap
- **Embedding Generation:** Convert text chunks into vector embeddings for semantic search
- **Vector Storage:** Store embeddings in FAISS or ChromaDB for efficient retrieval
- **Natural Language Queries:** Ask questions in plain English about your documents
- **Source Attribution:** Get answers with exact source information (file name and page number)
- **Dual-Answer Format:** Get both refined answers and exact source text for each query
- **List Detection:** Automatically identify and process list-type questions with enhanced retrieval
- **Advanced Ranking:** Multi-factor scoring system to select the most relevant content
- **Image Recognition:** Extract and interpret text from images using OCR
- **Table Detection:** Identify and parse tabular data with special handling
- **Contextual Understanding:** Maintain conversation history for better follow-up questions
- **Semantic Boundaries:** Respect document structure with paragraph and sentence-aware chunking
- **Content Filtering:** Precisely identify relevant sections while excluding irrelevant content
- **Configurable:** Customize embedding models, chunk size, and retrieval parameters
- **Comprehensive Evaluation:** Assess system performance with detailed metrics
- **Architecture Documentation:** Detailed explanation of system design and components

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Tesseract OCR (for image text extraction)

### Setup Steps

1. Clone or download this repository
2. Navigate to the project directory
3. Install dependencies:

```bash
# Install all required packages
pip install -r pdf_rag_app/requirements.txt
```

4. Install Tesseract OCR for image text extraction:

```bash
# For Ubuntu/Debian
sudo apt-get install tesseract-ocr

# For macOS
brew install tesseract

# For Windows
# Download and install from https://github.com/UB-Mannheim/tesseract/wiki
```

5. Create a `.env` file in the pdf_rag_app directory:

```bash
# For Windows
copy pdf_rag_app\.env.example pdf_rag_app\.env

# For Linux/Mac
cp pdf_rag_app/.env.example pdf_rag_app/.env
```

6. Edit the `.env` file to add your Google API key:

```
GOOGLE_API_KEY=your_google_api_key_here
```

You can obtain a Google API key from the [Google AI Studio](https://makersuite.google.com/app/apikey).

## Usage

### Running the Application

You can run the application in two ways:

#### Option 1: Using the run.py script

```bash
python pdf_rag_app/run.py
```

#### Option 2: Using Streamlit directly

```bash
cd pdf_rag_app
streamlit run app.py
```

The application will automatically open in your default web browser at http://localhost:8501.

### Using the Application

The application has a streamlined interface with two main tabs:

1. In the **Upload & Process** tab:
   - Use the file uploader to select one or more PDF files
   - Click "Process Files" to extract, chunk, and index the content
   - View the processing statistics
   - The system will automatically:
     - Extract text with layout awareness
     - Recognize tables and preserve their structure
     - Perform OCR on images to extract embedded text
     - Create semantically meaningful chunks respecting document structure

2. In the **Exact Answers** tab:
   - Type your question in the text input field
   - Click "Ask" to retrieve relevant content from the documents
   - View both tabs:
     - **Refined Answer:** An AI-generated answer based on the retrieved content
     - **Exact Text Excerpts:** The verbatim text from source documents
   - For list-type questions (e.g., "top 3 credit card issuers"), the system:
     - Automatically identifies the question type
     - Retrieves multiple relevant excerpts from different sources
     - Synthesizes a comprehensive list answer in the "Refined Answer" tab
     - Shows all supporting excerpts in the "Exact Text Excerpts" tab
   - For table-related questions, the system provides specialized analysis
   - The system maintains conversational context for follow-up questions
   - All answers include source attribution (file name and page number)

## Advanced Features

### Layout-Aware Document Processing

The system handles complex document layouts with advanced processing:

- **Multi-column Detection:** Automatically identifies and correctly processes multi-column layouts
- **Table Recognition:** Detects tabular structures and preserves their formatting
- **Image Text Extraction:** Uses OCR to extract text from embedded images
- **Structure Preservation:** Maintains heading hierarchies, lists, and document structure
- **Mixed Content Handling:** Processes documents with mixed text, tables, and images

### Semantic Chunking Strategy

The system employs intelligent chunking that respects document semantics:

- **Boundary Awareness:** Preserves paragraph, sentence, and semantic boundaries
- **Content-Type Specific:** Different chunking strategies for text, tables, and image captions
- **Markdown Header Recognition:** Uses headings as natural chunk boundaries
- **Information Density:** Optimizes chunk size based on information content
- **Long Sentence Handling:** Special processing for extremely long sentences
- **Metadata Preservation:** Maintains document structure information in chunks

### Contextual Understanding

The system maintains context across different parts of the conversation:

- **Session History:** Tracks previous questions and answers
- **Reference Tracking:** Maintains awareness of previously cited document sources
- **Follow-up Question Handling:** Understands references to previous questions
- **Cross-Document Context:** Synthesizes information from multiple documents
- **Consistent Attribution:** Ensures consistent source referencing across a session

### Multi-Factor Ranking

The application uses a sophisticated ranking system to select the most relevant content:
- **Vector Similarity:** Base semantic relevance from embedding model
- **Keyword Matching:** Direct term overlap between query and text
- **Named Entity Recognition:** Prioritizes documents with matching entities
- **Position Analysis:** Preferential scoring for documents with matches early in the text
- **Length Appropriateness:** Optimal scoring for excerpts of appropriate length
- **Information Density:** Higher ranking for excerpts with factual content
- **Source Diversity:** Selection of excerpts from different sources for list questions
- **Contextual Relevance:** Matches multi-word phrases and concepts
- **Query Type Detection:** Adapts ranking based on question intent and type

### List Detection and Processing

The system automatically detects questions asking for lists, rankings, or aggregated information by:
- Identifying key terms like "top", "best", "most", etc.
- Recognizing numerical indicators (e.g., "3 largest", "top 5")
- Detecting comparison and aggregation patterns

For these questions, the system:
- Retrieves multiple relevant documents from different sources
- Prioritizes information-dense excerpts
- Uses specialized prompting to synthesize a coherent list answer
- Shows all relevant excerpts with full source attribution

## Configuration

The application offers several configuration options in the sidebar:

- **Embedding Model:** Choose the model for generating text embeddings
  - For Google Generative AI: models/text-embedding-004 or models/embedding-001
  - For Vertex AI: text-embedding-004 or textembedding-gecko
  
- **LLM Model:** Select the model for generating answers (gemini-1.5-flash, gemini-1.5-pro)
- **Vector Store Type:** Choose between ChromaDB or FAISS for vector storage
- **Chunk Size & Overlap:** Configure text chunking parameters
- **Number of Documents to Retrieve:** Adjust how many documents are retrieved for each query
- **Enable OCR:** Toggle OCR for image text extraction
- **Detect Tables:** Toggle table detection and parsing
- **Maintain Session Context:** Enable/disable conversation history tracking

## System Evaluation

The PDF RAG system includes a comprehensive evaluation framework to assess performance across multiple dimensions:

### Running an Evaluation

To evaluate the system performance:

1. Prepare a ground truth file with questions, reference answers, and source documents
2. Run the evaluation script:

```bash
python pdf_rag_app/evaluate_rag.py --verbose
```

Alternatively, you can customize evaluation parameters:

```bash
python pdf_rag_app/evaluate_rag.py --ground_truth path/to/your/ground_truth.json --embedding_model models/text-embedding-004 --top_k 5 --verbose
```

### Evaluation Metrics

The evaluation framework assesses:

1. **Retrieval Performance**
   - Precision, Recall, F1 Score
   - Mean Reciprocal Rank (MRR)
   - Retrieval time

2. **Answer Quality**
   - ROUGE-1, ROUGE-2, ROUGE-L scores
   - BLEU score
   - Answer generation time

3. **Factual Accuracy & Hallucination**
   - Factual accuracy
   - Unsupported information ratio

4. **Answer Fluency**
   - Sentence and word length metrics

### Evaluation Reports

The evaluation process generates:
- A JSON file with detailed metrics
- A markdown report with interpretation of results
- A summary in the console output

## Architecture Documentation

For a comprehensive understanding of the system architecture, design choices, and implementation details, refer to the [Architecture Documentation](ARCHITECTURE.md).

This document provides:
- Detailed component descriptions
- Data flow explanations
- System configuration options
- Performance considerations
- Evaluation methodology
- Future enhancement possibilities

## Troubleshooting

### Common Issues

- **API Key Error**: Ensure your Google API key is correctly set in the `.env` file
- **Module Import Error**: Verify that all dependencies are installed with `pip install -r pdf_rag_app/requirements.txt`
- **Model Selection Error**: If using Vertex AI, make sure to select a compatible embedding model
- **OCR Issues**: Ensure Tesseract is properly installed and in your system PATH

For Vertex AI users, ensure you have the necessary Google Cloud setup and permissions.

## Architecture

The application follows a standard RAG architecture:

1. **PDF Processing:** Extract text from PDFs using layout-aware extraction, OCR, and table detection
2. **Text Chunking:** Split text into manageable chunks while preserving semantic boundaries
3. **Embedding Generation:** Convert text chunks into vector embeddings
4. **Vector Storage:** Store embeddings in a vector database for retrieval
5. **Query Processing:** Convert user questions into embeddings and find similar text chunks
6. **Advanced Contextual Ranking:** Apply sophisticated multi-factor scoring to select relevant content
7. **Answer Generation:** Use retrieved chunks as context for generating precise answers with source attribution
8. **System Evaluation:** Measure performance against ground truth data

## Technologies Used

- **PDF Processing:** PyMuPDF (fitz), pdfplumber
- **OCR:** Tesseract, pytesseract
- **Text Processing:** LangChain text splitters, NLTK
- **Embeddings:** Google Generative AI / Vertex AI embeddings
- **Vector Storage:** ChromaDB, FAISS
- **Question Answering:** Google Gemini models
- **UI:** Streamlit
- **Evaluation:** NLTK, Rouge, BLEU score metrics

## Screenshots
