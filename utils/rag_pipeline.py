import os
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm

# Import our utility classes
from .pdf_processor import PDFProcessor
from .text_chunker import TextChunker
from .embeddings_generator import EmbeddingsGenerator
from .vector_store import VectorStore
from .llm_retriever import LLMRetriever

class RAGPipeline:
    """A complete RAG pipeline for processing PDFs and answering questions."""
    
    def __init__(
        self,
        vector_store_dir: str = "./data/vectorstore",
        collection_name: str = "pdf_documents",
        vector_store_type: str = "chroma",
        embedding_model: str = "models/text-embedding-004",
        llm_model: str = "gemini-1.5-flash",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
        api_key: Optional[str] = None,
        use_vertex_ai: bool = False,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        verbose: bool = False
    ):
        """Initialize the RAG pipeline.
        
        Args:
            vector_store_dir: Directory to store vector database.
            collection_name: Name of the vector database collection.
            vector_store_type: Type of vector store ("chroma" or "faiss").
            embedding_model: Name of the embedding model.
            llm_model: Name of the LLM for question answering.
            chunk_size: Size of text chunks in characters.
            chunk_overlap: Overlap between text chunks in characters.
            top_k: Number of similar documents to retrieve.
            api_key: Google API key.
            use_vertex_ai: Whether to use Vertex AI for embeddings.
            project_id: GCP project ID for Vertex AI.
            location: GCP region for Vertex AI.
            verbose: Whether to print detailed logs.
        """
        self.vector_store_dir = vector_store_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.verbose = verbose
        
        # Initialize the PDF processor
        self.pdf_processor = PDFProcessor(verbose=verbose)
        
        # Initialize the text chunker
        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            verbose=verbose
        )
        
        # Initialize the embeddings generator
        self.embeddings_generator = EmbeddingsGenerator(
            model_name=embedding_model,
            use_vertex_ai=use_vertex_ai,
            api_key=api_key,
            project_id=project_id,
            location=location,
            verbose=verbose
        )
        
        # Initialize the vector store
        self.vector_store = VectorStore(
            store_type=vector_store_type,
            persist_directory=vector_store_dir,
            collection_name=collection_name,
            verbose=verbose
        )
        
        # Initialize the LLM retriever
        self.llm_retriever = LLMRetriever(
            model_name=llm_model,
            api_key=api_key,
            verbose=verbose
        )
    
    def process_pdf_files(self, pdf_files: List[str]) -> Dict[str, Any]:
        """Process PDF files through the entire RAG pipeline.
        
        Args:
            pdf_files: List of paths to PDF files.
            
        Returns:
            Dictionary with processing statistics.
        """
        start_time = time.time()
        stats = {"processed_files": 0, "pages": 0, "chunks": 0, "embedded_chunks": 0}
        
        if self.verbose:
            print(f"Processing {len(pdf_files)} PDF files...")
        
        # Step 1: Extract text from PDFs
        extracted_docs = self.pdf_processor.process_pdfs(pdf_files)
        stats["processed_files"] = len(pdf_files)
        stats["pages"] = len(extracted_docs)
        
        if not extracted_docs:
            print("No text extracted from PDF files")
            return stats
        
        if self.verbose:
            print(f"Extracted {len(extracted_docs)} pages of text")
        
        # Step 2: Chunk the text
        chunked_docs = self.text_chunker.chunk_documents(extracted_docs)
        stats["chunks"] = len(chunked_docs)
        
        if self.verbose:
            print(f"Created {len(chunked_docs)} text chunks")
        
        # Step 3: Generate embeddings
        embedded_docs = self.embeddings_generator.create_embeddings_for_chunks(chunked_docs)
        stats["embedded_chunks"] = sum(1 for doc in embedded_docs if 'embedding' in doc and doc['embedding'] is not None)
        
        if self.verbose:
            print(f"Generated embeddings for {stats['embedded_chunks']} chunks")
        
        # Step 4: Store in vector database
        self.vector_store.add_documents(embedded_docs)
        
        stats["processing_time"] = time.time() - start_time
        
        return stats
    
    def clear_vector_store(self) -> None:
        """Clear the vector store."""
        self.vector_store.delete_collection()
        if self.verbose:
            print(f"Cleared vector store collection: {self.collection_name}")
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the vector store.
        
        Returns:
            Dictionary with vector store information.
        """
        return self.vector_store.get_collection_info()
    
    def query(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Query the RAG pipeline with a question.
        
        Args:
            question: User's question.
            
        Returns:
            Tuple of (answer string, list of retrieved documents).
        """
        if self.verbose:
            print(f"Processing query: {question}")
        
        # Step 1: Generate embedding for the query
        query_embedding = self.embeddings_generator.generate_query_embedding(question)
        
        # Step 2: Retrieve similar documents
        retrieved_docs = self.vector_store.similar_search(
            query_embedding=query_embedding,
            k=self.top_k
        )
        
        if self.verbose:
            print(f"Retrieved {len(retrieved_docs)} documents")
        
        # Step 3: Query the LLM with the retrieved context
        answer = self.llm_retriever.query_with_context(
            question=question,
            retrieved_docs=retrieved_docs
        )
        
        return answer, retrieved_docs 