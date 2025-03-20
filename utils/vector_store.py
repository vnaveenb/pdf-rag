import os
import json
import pickle
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from tqdm import tqdm

# For FAISS
import faiss

# For ChromaDB
import chromadb
from chromadb.utils import embedding_functions

class VectorStore:
    """A class for managing vector stores for document embeddings."""
    
    def __init__(
        self,
        store_type: str = "chroma",  # "faiss" or "chroma"
        persist_directory: str = "./data/vectorstore",
        collection_name: str = "pdf_documents",
        embedding_dimension: int = 768,
        distance_strategy: str = "cosine",  # "cosine", "euclidean", or "dot"
        verbose: bool = False
    ):
        """Initialize the VectorStore.
        
        Args:
            store_type: Type of vector store to use ("faiss" or "chroma").
            persist_directory: Directory to persist the vector store.
            collection_name: Name of the collection for ChromaDB.
            embedding_dimension: Dimension of embedding vectors.
            distance_strategy: Distance strategy for similarity search.
            verbose: Whether to print additional information.
        """
        self.store_type = store_type.lower()
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.distance_strategy = distance_strategy
        self.verbose = verbose
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the vector store
        if self.store_type == "faiss":
            self._init_faiss()
        elif self.store_type == "chroma":
            self._init_chroma()
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
    
    def _init_faiss(self):
        """Initialize a FAISS index."""
        # For cosine similarity, we need to normalize vectors
        if self.distance_strategy == "cosine":
            # L2 normalization + inner product = cosine similarity
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
        elif self.distance_strategy == "euclidean":
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
        else:  # "dot"
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
        
        # Storage for document metadata
        self.documents = []
        
        # Load existing index if available
        index_path = os.path.join(self.persist_directory, "faiss_index.bin")
        docs_path = os.path.join(self.persist_directory, "faiss_documents.pkl")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                if self.verbose:
                    print(f"Loaded FAISS index with {len(self.documents)} documents")
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                # Reinitialize if there's an error
                self.index = faiss.IndexFlatIP(self.embedding_dimension)
                self.documents = []
    
    def _init_chroma(self):
        """Initialize ChromaDB."""
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_strategy}
            )
            
            if self.verbose:
                collection_count = self.collection.count()
                print(f"Using ChromaDB collection '{self.collection_name}' with {collection_count} documents")
                
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents with embeddings to the vector store.
        
        Args:
            documents: List of document dictionaries with text, embeddings, and metadata.
        """
        if not documents:
            print("No documents to add to vector store")
            return
        
        if self.store_type == "faiss":
            self._add_to_faiss(documents)
        elif self.store_type == "chroma":
            self._add_to_chroma(documents)
    
    def _add_to_faiss(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to FAISS index.
        
        Args:
            documents: List of document dictionaries with embeddings.
        """
        # Extract embeddings
        embeddings = []
        doc_indices = []
        
        for i, doc in enumerate(documents):
            if 'embedding' in doc and doc['embedding'] is not None:
                embedding = doc['embedding']
                # Convert to numpy array if it's a list
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)
                
                # Normalize for cosine similarity
                if self.distance_strategy == "cosine":
                    embedding = embedding / np.linalg.norm(embedding)
                
                embeddings.append(embedding)
                doc_indices.append(i)
                
                # Remove embedding from document to save memory
                doc_copy = doc.copy()
                doc_copy.pop('embedding', None)
                self.documents.append(doc_copy)
        
        if embeddings:
            # Convert to the format FAISS requires
            embeddings_matrix = np.vstack(embeddings).astype(np.float32)
            
            # Add to FAISS index
            self.index.add(embeddings_matrix)
            
            if self.verbose:
                print(f"Added {len(embeddings)} documents to FAISS index")
            
            # Save index and documents
            self._persist_faiss()
    
    def _add_to_chroma(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to ChromaDB collection.
        
        Args:
            documents: List of document dictionaries with embeddings.
        """
        ids = []
        embeddings = []
        metadatas = []
        texts = []
        
        for i, doc in enumerate(documents):
            if 'embedding' not in doc or doc['embedding'] is None:
                if self.verbose:
                    print(f"Skipping document {i} with missing embedding")
                continue
            
            # Create a unique ID for the document
            doc_id = doc.get('id', f"doc_{len(ids)}_{hash(doc['text'][:100])}")
            ids.append(doc_id)
            
            # Extract the embedding
            embeddings.append(doc['embedding'])
            
            # Extract text
            texts.append(doc['text'])
            
            # Extract metadata (remove text and embedding)
            metadata = {k: v for k, v in doc.items() 
                      if k not in ['text', 'embedding'] and v is not None}
            
            # Convert non-string metadata to strings
            metadata = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v 
                      for k, v in metadata.items()}
            
            metadatas.append(metadata)
        
        if ids:
            # Add in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                end_idx = min(i + batch_size, len(ids))
                
                batch_ids = ids[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]
                batch_texts = texts[i:end_idx]
                
                try:
                    self.collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas,
                        documents=batch_texts
                    )
                    
                    if self.verbose and i + batch_size < len(ids):
                        print(f"Added batch {i//batch_size + 1}/{(len(ids) + batch_size - 1)//batch_size}")
                        
                except Exception as e:
                    print(f"Error adding batch to ChromaDB: {e}")
            
            if self.verbose:
                print(f"Added {len(ids)} documents to ChromaDB collection")
    
    def _persist_faiss(self) -> None:
        """Save FAISS index and documents to disk."""
        index_path = os.path.join(self.persist_directory, "faiss_index.bin")
        docs_path = os.path.join(self.persist_directory, "faiss_documents.pkl")
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save documents
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
                
            if self.verbose:
                print(f"Saved FAISS index with {len(self.documents)} documents")
                
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
    
    def similar_search(
        self, 
        query_embedding: List[float], 
        k: int = 5, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for documents similar to the query embedding.
        
        Args:
            query_embedding: Embedding vector for the query.
            k: Number of similar documents to return.
            filter_dict: Dictionary of metadata filters (ChromaDB only).
            
        Returns:
            List of documents similar to the query.
        """
        if self.store_type == "faiss":
            return self._search_faiss(query_embedding, k)
        elif self.store_type == "chroma":
            return self._search_chroma(query_embedding, k, filter_dict)
        
        return []
    
    def _search_faiss(self, query_embedding: List[float], k: int) -> List[Dict[str, Any]]:
        """Search for similar documents using FAISS.
        
        Args:
            query_embedding: Embedding vector for the query.
            k: Number of similar documents to return.
            
        Returns:
            List of similar documents with metadata and scores.
        """
        if not self.documents:
            return []
        
        # Convert to numpy array
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Normalize for cosine similarity
        if self.distance_strategy == "cosine":
            query_vector = query_vector / np.linalg.norm(query_vector)
        
        # Get top k similar vectors
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
                
            doc = self.documents[idx].copy()
            doc['score'] = float(distances[0][i])
            results.append(doc)
        
        return results
    
    def _search_chroma(
        self, 
        query_embedding: List[float], 
        k: int, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using ChromaDB.
        
        Args:
            query_embedding: Embedding vector for the query.
            k: Number of similar documents to return.
            filter_dict: Dictionary of metadata filters.
            
        Returns:
            List of similar documents with metadata and scores.
        """
        # Convert filter_dict to ChromaDB where filter if provided
        where_filter = None
        if filter_dict:
            where_filter = filter_dict
        
        try:
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter
            )
            
            # Format results
            formatted_results = []
            if results and results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    doc = {
                        'text': results['documents'][0][i],
                        'score': results['distances'][0][i] if 'distances' in results else None,
                        **results['metadatas'][0][i]
                    }
                    formatted_results.append(doc)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching ChromaDB: {e}")
            return []
    
    def delete_collection(self) -> None:
        """Delete the vector store collection/index."""
        if self.store_type == "faiss":
            # Delete FAISS files
            index_path = os.path.join(self.persist_directory, "faiss_index.bin")
            docs_path = os.path.join(self.persist_directory, "faiss_documents.pkl")
            
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(docs_path):
                os.remove(docs_path)
                
            # Reinitialize
            self._init_faiss()
            
        elif self.store_type == "chroma":
            try:
                # Delete ChromaDB collection
                self.chroma_client.delete_collection(self.collection_name)
                
                # Reinitialize
                self._init_chroma()
                
            except Exception as e:
                print(f"Error deleting ChromaDB collection: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection.
        
        Returns:
            Dictionary with collection information.
        """
        info = {
            "store_type": self.store_type,
            "persist_directory": self.persist_directory,
            "collection_name": self.collection_name,
            "embedding_dimension": self.embedding_dimension
        }
        
        if self.store_type == "faiss":
            info["document_count"] = len(self.documents)
            
        elif self.store_type == "chroma":
            try:
                info["document_count"] = self.collection.count()
            except Exception as e:
                info["document_count"] = "Error getting count"
                info["error"] = str(e)
        
        return info 