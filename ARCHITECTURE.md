# PDF RAG System Architecture Documentation

This document provides a comprehensive overview of the architecture, components, and approaches used in our PDF Retrieval-Augmented Generation (RAG) system.

## System Architecture

The PDF RAG system follows a modular architecture with several key components working together:

```
                    ┌─────────────────┐
                    │    PDF Files    │
                    └────────┬────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────┐
│            Enhanced PDF Processor                │
│  (Layout-aware extraction, OCR, Table detection) │
└─────────────────────────┬───────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────┐
│            Semantic Text Chunker                 │
│  (Boundary-aware, content-type specific chunks)  │
└─────────────────────────┬───────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────┐
│             Embeddings Generator                 │
│  (Convert text chunks to vector embeddings)      │
└─────────────────────────┬───────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────┐
│                Vector Store                      │
│  (Store and index vector embeddings)             │
└─────────────────────────┬───────────────────────┘
                          │
           ┌──────────────┴──────────────┐
           │                             │
           ▼                             ▼
┌────────────────────┐       ┌────────────────────┐
│  User Question     │       │   RAG Evaluator    │
└────────┬───────────┘       │  (Measure system   │
         │                   │   performance)     │
         ▼                   └────────────────────┘
┌────────────────────┐
│   Query Embedding  │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Vector Retrieval  │
└────────┬───────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│   Advanced Query Processing & Relevance Scoring  │
│ (List Detection, Multi-Doc Synthesis, Enhanced   │
│  Ranking using Multi-factor Scoring System)      │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────┐
│   Context-Aware LLM Generation                  │
│ (Session history, cross-document understanding) │
└────────┬───────────────────────────────────────┘
         │
         ▼
┌────────────────────┐
│  Response with     │
│  Source Attribution│
└────────────────────┘
```

## Component Details

### 1. Enhanced PDF Processor (`pdf_processor.py`)

The PDF Processor component is responsible for extracting text and rich content from PDF documents while preserving layout information and structure.

**Key Features:**
- Layout-aware text extraction using PyMuPDF (fitz) and pdfplumber
- Handles both single-column and multi-column PDF layouts
- OCR integration for extracting text from images
- Table detection and structured data extraction
- Layout element tracking and classification
- Preserves document metadata (source file, page number)
- Supports batch processing of multiple PDF files

**Implementation Details:**
- Uses PyMuPDF as the primary extraction engine with fallback to pdfplumber for complex layouts
- Integrates Tesseract OCR via pytesseract for image text extraction
- Implements heuristic table detection algorithms
- Preserves bounding box information for spatial analysis
- Creates a structured representation of document elements
- Extracts and classifies:
  - Regular text blocks
  - Tables
  - Images with OCR text
  - Font style and formatting information
- Maintains a mapping between text and source information

### 2. Semantic Text Chunker (`text_chunker.py`)

The Text Chunker divides extracted document text into smaller, semantically meaningful chunks while respecting document structure, content types, and natural boundaries.

**Key Features:**
- Boundary-aware chunking that respects semantic structure
- Content-type specific chunking strategies
- Different handling for text, tables, and image captions
- Configurable chunk size and overlap
- Preserves metadata and structure information
- Markdown header recognition
- Sentence and paragraph boundary preservation

**Implementation Details:**
- Uses advanced text splitting with semantic boundary detection
- Implements NLTK sentence tokenization for boundary recognition
- Integrates LangChain's RecursiveCharacterTextSplitter with custom separators
- Uses MarkdownHeaderTextSplitter for header-based chunking
- Preserves document layout information in chunk metadata
- Handles long sentences with specialized processing
- Maintains document hierarchy and source information
- Special handling for content types:
  - Regular text gets semantic chunking
  - Tables are preserved as individual chunks
  - Image text is kept together as a unit
- Tracks information density for optimal chunking

### 3. Embeddings Generator (`embeddings_generator.py`)

This component converts text chunks into vector embeddings that capture semantic meaning.

**Key Features:**
- Support for multiple embedding models:
  - Google's text-embedding-004 and embedding-001 models
  - Vertex AI embedding models
- Batched processing for efficiency
- Error handling and retry logic
- Content-aware embedding strategies

**Implementation Details:**
- Interfaces with Google Generative AI and/or Vertex AI APIs
- Configurable model selection
- Produces high-dimensional vectors (768-1536 dimensions depending on model)
- Caching mechanism to avoid redundant embedding generation
- Special handling for different content types

### 4. Vector Store (`vector_store.py`)

The Vector Store component indexes and stores embeddings for efficient similarity search.

**Key Features:**
- Support for multiple vector databases (ChromaDB, FAISS)
- Persistent storage of embeddings
- Efficient similarity search with source retrieval
- Collection management and metadata querying
- Content-type specific filtering

**Implementation Details:**
- Abstracts away differences between vector store implementations
- Handles vector normalization and indexing
- Provides efficient top-k retrieval
- Preserves and returns source metadata with search results
- Maintains connections to content types and structures

### 5. Context-Aware LLM Retriever (`llm_retriever.py`)

This component queries language models with retrieved context to generate answers, maintaining contextual awareness across user interactions.

**Key Features:**
- Context-aware question answering
- Conversation history tracking
- Source attribution in responses
- Specialized handling for list-type and aggregation questions
- Table data interpretation capabilities
- Cross-document context understanding
- Reference tracking for consistent attribution
- Dual prompt systems for different question types
- Configurable model selection (Gemini models)
- Temperature and top-p settings for response tuning

**Implementation Details:**
- Maintains conversation session history
- Tracks document references across multiple queries
- Crafts specialized prompts for different content types:
  - Standard RAG prompts for general queries
  - List synthesis prompts for aggregation questions 
  - Table interpretation prompts for tabular data
- Formats retrieved documents based on content type
- Implements intelligent synthesis for multi-document answers
- Uses conversation history for follow-up question understanding
- Handles fallback for zero-shot answering when no relevant context is found
- Processes and formats model responses with appropriate structure

### 6. RAG Pipeline (`rag_pipeline.py`)

The RAG Pipeline orchestrates the entire process from document processing to answering.

**Key Features:**
- End-to-end pipeline management
- Configuration management
- Performance statistics and logging
- Vector store maintenance
- Content-type awareness

**Implementation Details:**
- Coordinates all system components
- Handles document preprocessing, chunking, embedding, and storage
- Manages query processing flow
- Provides a simple interface for the application layer

### 7. RAG Evaluator (`evaluation.py`)

The evaluation component measures the performance of the RAG system along multiple dimensions.

**Key Features:**
- Comprehensive metrics for retrieval quality
- Answer quality assessment
- Hallucination detection
- Fluency evaluation
- Detailed reporting
- Content-type specific evaluation

**Implementation Details:**
- Uses ground truth data for comparative evaluation
- Calculates precision, recall, F1 for retrieval assessment
- Employs ROUGE and BLEU scores for answer quality
- Measures factual accuracy and unsupported information
- Generates human-readable reports with metric interpretation

## Data Flow

1. **Ingestion Phase:**
   - User uploads PDF files through the Streamlit interface
   - Enhanced PDF Processor extracts text with layout awareness, identifies tables, and performs OCR on images
   - Semantic Text Chunker divides content into optimal segments respecting boundaries and content types
   - Embeddings Generator converts chunks to vector embeddings
   - Vector Store indexes and stores the embeddings with source metadata

2. **Query Phase:**
   - User submits a natural language question in the Exact Answers tab
   - System checks if it's a follow-up question using conversation history
   - Question is analyzed to determine if it's a standard, list-type, or table-related query
   - Question is converted to a vector embedding
   - Vector Store performs similarity search to find relevant chunks
   - For standard questions:
     - System selects the most relevant document using advanced multi-factor scoring
     - LLM Retriever generates a refined answer using the selected document as context
   - For list-type questions:
     - System selects multiple relevant documents from diverse sources using specialized list scoring
     - LLM Retriever synthesizes information across documents using specialized prompting
     - System generates a comprehensive list answer with attribution to multiple sources
   - For table-related questions:
     - System identifies and prioritizes tabular data
     - Specialized table prompt extracts and analyzes structured information
     - System presents findings with appropriate formatting
   - Conversation history is updated to maintain context for future questions
   - Results are presented in dual-view format (Refined Answer and Exact Text Excerpts)

3. **Evaluation Phase:**
   - Ground truth data is compared with system outputs
   - Multiple metrics assess different aspects of performance
   - Results are aggregated and summarized
   - Comprehensive evaluation report is generated

## System Configuration

The system offers multiple configuration options to adapt to different requirements:

- **Embedding Models:**
  - Google Generative AI: models/text-embedding-004, models/embedding-001
  - Vertex AI: text-embedding-004, textembedding-gecko

- **LLM Models:**
  - Gemini 1.5 Flash, Gemini 1.5 Pro, Gemini 1.0 Pro

- **Vector Store Types:**
  - ChromaDB (persistent, metadata-rich)
  - FAISS (efficient for large collections)

- **Chunking Parameters:**
  - Chunk size: 100-2000 characters
  - Chunk overlap: 0-500 characters
  - Semantic boundary preservation
  - Markdown header recognition
  - Content-type specific strategies

- **OCR Options:**
  - Enable/disable OCR
  - Image resolution settings
  - Language selection

- **Table Detection:**
  - Enable/disable table detection
  - Structure recognition thresholds
  - Table formatting preferences

- **Session Context:**
  - Conversation history length
  - Reference tracking
  - Context window size

- **Retrieval Parameters:**
  - Number of documents to retrieve: 1-10
  - Content-type specific ranking

## Performance Considerations

1. **Scalability:**
   - The system handles multiple PDFs efficiently
   - Vector stores scale to thousands of documents
   - Batch processing reduces API costs
   - OCR processing optimized for performance

2. **Accuracy:**
   - Layout-aware extraction improves text quality
   - OCR enhances coverage of image-based content
   - Table detection preserves structured data
   - Semantic search finds relevant information even with paraphrased queries
   - Advanced multi-factor scoring increases relevance precision
   - List-question handling improves comprehensiveness for complex queries
   - Conversation history enables better follow-up question handling
   - Source attribution enhances trustworthiness

3. **Response Time:**
   - Optimized vector retrieval for fast results
   - Efficient embedding generation through batching
   - Caching mechanisms to prevent redundant processing
   - Content-type specific optimizations

## Evaluation Methodology

The evaluation system provides a comprehensive assessment framework:

1. **Retrieval Metrics:**
   - Precision: Percentage of retrieved documents that are relevant
   - Recall: Percentage of relevant documents that are retrieved
   - F1 Score: Harmonic mean of precision and recall
   - Mean Reciprocal Rank (MRR): Inverse of the rank of the first relevant document
   - Retrieval Time: Time taken to retrieve documents
   - Content Type Coverage: Evaluation of retrieval across different content types (text, tables, images)

2. **Answer Quality Metrics:**
   - ROUGE-1, ROUGE-2, ROUGE-L: Overlap of n-grams between generated and reference answers
   - BLEU: Precision of n-grams in generated answer compared to reference
   - Factual Accuracy: Percentage of facts in the generated answer that are correct
   - Unsupported Information Ratio: Percentage of generated content not supported by the sources
   - List Synthesis Quality: Correctness and comprehensiveness of synthesized list answers
   - Table Interpretation Accuracy: Correctness of data extracted from tables
   - Cross-Reference Consistency: Consistency of information across multiple sources

3. **System Metrics:**
   - End-to-end processing time
   - Query processing time
   - Resource utilization
   - Content extraction coverage

## User Interface

The application provides a streamlined Streamlit interface with two main tabs:

1. **Upload & Process Tab:**
   - File uploader for PDF documents
   - Processing controls and progress indicators
   - Processing statistics and visualizations
   - Vector store management
   - Content type breakdown (text, tables, images)

2. **Exact Answers Tab:**
   - Question input field
   - Dual-view answer presentation:
     - Refined Answer: AI-generated response based on retrieved content
     - Exact Text Excerpts: Original text from source documents
   - Source attribution for all information
   - Enhanced display for list-type questions with multiple sources
   - Special formatting for table data
   - Additional expandable view for all retrieved documents
   - Indicator for follow-up questions using conversation context

## Advanced Features

### Enhanced Document Processing

The system implements advanced document processing capabilities:

1. **Layout Analysis:**
   - Multi-column detection and handling
   - Reading order determination
   - Structural element identification (headers, footers, margins)
   - Font and style recognition for content classification

2. **Table Detection:**
   - Rule-based and heuristic table identification
   - Grid structure recognition
   - Header row detection
   - Cell content extraction
   - Markdown conversion for structured preservation

3. **Image Processing:**
   - Embedded image extraction
   - OCR text recognition
   - Caption association
   - Content classification (diagrams, charts, photos)
   - Visual element positioning

4. **Mixed Content Handling:**
   - Positional awareness between text and non-text elements
   - Context preservation across element types
   - Hierarchical document structure maintenance
   - Content relationship mapping

### Enhanced Relevance Scoring System

The system implements a sophisticated multi-factor relevance scoring system that significantly improves retrieval accuracy:

1. **Query Type Detection:**
   - Automatically identifies query intent (definition, comparison, factual, list)
   - Adapts scoring weights based on query type
   - Dynamically adjusts ideal answer length and structure
   - Recognizes follow-up questions and maintains context

2. **Advanced Scoring Factors:**
   - **Base Vector Similarity:** Foundation semantic relevance with minimum threshold
   - **Keyword Matching:** Direct term overlap between query and documents
   - **Named Entity Recognition:** Prioritizes documents containing query entities
   - **Contextual Relevance:** Matches n-grams and phrase patterns
   - **Position Analysis:** Preferential scoring for documents with matches early in the text
   - **Information Density:** Higher ranking for excerpts with factual content (numbers, dates)
   - **Length Appropriateness:** Optimal scoring for excerpts of query-appropriate length
   - **Source Diversity:** Selection of excerpts from different sources for list questions
   - **Information Uniqueness:** Prioritizes documents with novel information
   - **List Indicator Matching:** Detects and scores document sections with list-like structures
   - **Content Type Appropriateness:** Prioritizes tables for data questions, text for conceptual questions

3. **Score Optimization:**
   - Dynamic weighting based on query characteristics
   - Logarithmic score normalization for better separation
   - Minimum score thresholds for relevant content
   - Enhanced scoring for complete entity matches
   - Content-type weighting based on query

4. **List-Specific Scoring:**
   - Automatic detection of list size requirements (e.g., "top 3")
   - Pattern matching for list indicators (bullets, numbers, rankings)
   - Information uniqueness tracking across selected documents
   - Specialized scoring for structured content with list formats

### Semantic Chunking Strategy

The system implements sophisticated chunking strategies that preserve document semantics:

1. **Boundary Recognition:**
   - Paragraph boundary preservation
   - Sentence boundary detection using NLTK
   - Section and subsection recognition
   - List item grouping
   - Table integrity preservation

2. **Content-Specific Chunking:**
   - Text content: Semantic boundary chunking
   - Tables: Preserved as whole units
   - Image text: Kept as coherent blocks
   - Headers: Used as natural chunk boundaries
   - Lists: Preserved structure

3. **Chunk Optimization:**
   - Dynamic size adjustment based on content type
   - Information density assessment
   - Cross-reference preservation
   - Context window management
   - Metadata enrichment

### Contextual Understanding

The system provides sophisticated contextual understanding across multi-turn interactions:

1. **Conversation History:**
   - Tracks user questions and system answers
   - Maintains a rolling conversation window
   - Resolves references to previous content
   - Preserves context across session

2. **Document Reference Tracking:**
   - Records previously cited sources
   - Ensures consistent attribution
   - Handles cross-document references
   - Prioritizes previously referenced content for related questions

3. **Multi-Document Synthesis:**
   - Combines information across documents
   - Reconciles conflicting information
   - Identifies complementary content
   - Provides comprehensive answers

4. **Query Intent Memory:**
   - Recognizes follow-up questions
   - Maintains thematic consistency
   - Handles topic shifts appropriately
   - Preserves context when needed

## Future Enhancements

Planned improvements to the architecture include:

1. **Real-time Document Processing:**
   - Streaming processing for immediate availability of uploaded documents
   - Progressive chunking and indexing
   - Asynchronous OCR processing

2. **Enhanced Multi-modal Capabilities:**
   - Image content analysis beyond OCR
   - Chart and graph interpretation
   - Complex table analysis
   - Formula and equation extraction

3. **Conversational Context:**
   - Extended conversation history
   - More sophisticated reference resolution
   - Topic modeling for conversation segmentation
   - User intent tracking

4. **Advanced Evaluation:**
   - Expanded metrics for specialized content types
   - User feedback integration
   - A/B testing framework for component comparisons
   - Continuous monitoring and improvement

5. **Additional Vector Store Options:**
   - Support for Pinecone, Weaviate, and other vector databases
   - Hybrid vector search combining dense and sparse embeddings
   - Multi-vector representation for different content aspects

6. **Performance Optimizations:**
   - Further improvements to document processing speed
   - Enhanced caching mechanisms
   - Parallel processing pipelines

7. **Relevance Scoring Enhancements:**
   - Machine learning-based relevance model training
   - User feedback incorporation into ranking algorithm
   - Adaptive weighting based on historical performance
   - Content-type specific scoring models

## Conclusion

The PDF RAG system provides a robust solution for question answering over PDF documents with advanced capabilities for handling document layout variations, images, tables, and maintaining contextual understanding. Its modular architecture allows for flexibility and adaptation to different use cases, while the comprehensive evaluation framework ensures system quality and trustworthiness. The advanced relevance scoring system delivers highly accurate answers with detailed source attribution, making it an effective tool for information retrieval from document collections. 