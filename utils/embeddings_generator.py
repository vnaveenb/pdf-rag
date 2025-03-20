import os
from typing import List, Dict, Any, Optional, Union
import time
from tqdm import tqdm
import numpy as np

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings

class EmbeddingsGenerator:
    """A class for generating embeddings from text chunks using Google models."""
    
    def __init__(
        self,
        model_name: str = "models/text-embedding-004",
        use_vertex_ai: bool = False,
        dimension: int = 768,
        task_type: str = "retrieval_document",
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        batch_size: int = 5,
        verbose: bool = False,
    ):
        """Initialize the EmbeddingsGenerator.
        
        Args:
            model_name: Name of the embedding model to use.
            use_vertex_ai: Whether to use Vertex AI (True) or Google Generative AI (False).
            dimension: Output dimension for the embedding vectors.
            task_type: Task type for the embeddings.
            api_key: Google API key (for Google Generative AI).
            project_id: GCP project ID (for Vertex AI).
            location: GCP region (for Vertex AI).
            batch_size: Number of texts to embed in a single API call.
            verbose: Whether to print additional information during processing.
        """
        self.model_name = model_name
        self.use_vertex_ai = use_vertex_ai
        self.dimension = dimension
        self.task_type = task_type
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Setup embedding model
        if use_vertex_ai:
            # Map the model name to Vertex AI format if needed
            vertex_model_name = self._get_vertex_model_name(model_name)
            
            if self.verbose:
                print(f"Using Vertex AI embeddings with model: {vertex_model_name}")
                
            if project_id:
                self.embeddings = VertexAIEmbeddings(
                    model_name=vertex_model_name,
                    project=project_id,
                    location=location
                )
            else:
                # Use environment variables
                self.embeddings = VertexAIEmbeddings(
                    model_name=vertex_model_name
                )
        else:
            # Use Google Generative AI embeddings
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
            
            if self.verbose:
                print(f"Using Google Generative AI embeddings with model: {model_name}")
                
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                task_type=task_type,
                dimension=dimension
            )
    
    def _get_vertex_model_name(self, model_name: str) -> str:
        """Convert model name to Vertex AI format if needed.
        
        Args:
            model_name: Original model name.
            
        Returns:
            Model name in Vertex AI format.
        """
        # Map of Google Generative AI model names to Vertex AI model names
        model_mapping = {
            "models/text-embedding-004": "text-embedding-004",
            "models/embedding-001": "textembedding-gecko",
            "text-embedding-004": "text-embedding-004"
        }
        
        return model_mapping.get(model_name, model_name)
    
    def create_embeddings_for_chunks(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of chunk dictionaries containing text and metadata.
            
        Returns:
            List of dictionaries with text, embeddings, and metadata.
        """
        texts = [chunk['text'] for chunk in chunks]
        
        # Process in batches to avoid API rate limits
        embedded_chunks = []
        
        if self.verbose:
            print(f"Generating embeddings for {len(texts)} chunks in batches of {self.batch_size}")
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch_chunks = chunks[i:i+self.batch_size]
            
            try:
                # Generate embeddings for the batch
                embeddings = self.embeddings.embed_documents(batch_texts)
                
                for j, embedding in enumerate(embeddings):
                    embedded_chunk = batch_chunks[j].copy()
                    embedded_chunk['embedding'] = embedding
                    embedded_chunks.append(embedded_chunk)
                
                if self.verbose and i + self.batch_size < len(texts):
                    print(f"Processed {i + len(batch_texts)}/{len(texts)} chunks")
                
                # Add a small delay to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error generating embeddings for batch starting at index {i}: {e}")
                # Still include the chunks without embeddings
                for chunk in batch_chunks:
                    chunk_copy = chunk.copy()
                    chunk_copy['embedding'] = None  # No embedding available
                    embedded_chunks.append(chunk_copy)
        
        if self.verbose:
            successful = sum(1 for chunk in embedded_chunks if chunk.get('embedding') is not None)
            print(f"Successfully embedded {successful}/{len(embedded_chunks)} chunks")
        
        return embedded_chunks
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query string.
        
        Args:
            query: Query text to embed.
            
        Returns:
            Embedding vector for the query.
        """
        try:
            embedding = self.embeddings.embed_query(query)
            return embedding
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * self.dimension 