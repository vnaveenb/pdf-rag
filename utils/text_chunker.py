from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import re
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextChunker:
    """A class for splitting text into chunks with semantic awareness and layout preservation."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n",
        use_semantic_boundaries: bool = True,
        preserve_layout: bool = True,
        respect_markdown_headers: bool = True,
        verbose: bool = False
    ):
        """Initialize the TextChunker.
        
        Args:
            chunk_size: Target size for each text chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
            separator: String separator to use for splitting text.
            use_semantic_boundaries: Whether to try to break at sentence boundaries.
            preserve_layout: Whether to preserve document layout structures.
            respect_markdown_headers: Whether to use headers as chunk boundaries.
            verbose: Whether to print additional information during processing.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.use_semantic_boundaries = use_semantic_boundaries
        self.preserve_layout = preserve_layout
        self.respect_markdown_headers = respect_markdown_headers
        self.verbose = verbose
        
        # Initialize the text splitter with semantic awareness
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ":", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        # Initialize header-aware splitter for markdown content
        if self.respect_markdown_headers:
            self.markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "header_1"),
                    ("##", "header_2"),
                    ("###", "header_3"),
                    ("####", "header_4"),
                    ("#####", "header_5"),
                    ("######", "header_6"),
                ]
            )
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks with metadata and layout preservation.
        
        Args:
            documents: List of document dictionaries with text and metadata.
            
        Returns:
            List of chunk dictionaries with text and metadata.
        """
        chunked_documents = []
        
        for doc in documents:
            # Process document based on its content type and structure
            if self.preserve_layout and 'layout_elements' in doc:
                # Process layout-aware documents with mixed content types
                chunked_docs = self._chunk_layout_document(doc)
                chunked_documents.extend(chunked_docs)
            else:
                # Process regular text documents
                doc_text = doc['text']
                
                # Skip empty documents
                if not doc_text or not doc_text.strip():
                    continue
                
                # Get chunks using semantic boundaries if enabled
                if self.use_semantic_boundaries:
                    chunks = self._semantic_chunking(doc_text)
                else:
                    chunks = self.text_splitter.split_text(doc_text)
                
                if self.verbose:
                    print(f"Split document into {len(chunks)} chunks")
                
                # Create a new document for each chunk with metadata
                for i, chunk_text in enumerate(chunks):
                    if not chunk_text.strip():  # Skip empty chunks
                        continue
                        
                    chunked_doc = {
                        'text': chunk_text,
                        'filename': doc.get('filename', ''),
                        'page_num': doc.get('page_num', 0),
                        'total_pages': doc.get('total_pages', 0),
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'content_type': 'text'
                    }
                    
                    chunked_documents.append(chunked_doc)
        
        return chunked_documents
    
    def _chunk_layout_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a document with layout information.
        
        This handles documents with mixed content types (text blocks, tables, images).
        
        Args:
            doc: Document dictionary with layout_elements field.
            
        Returns:
            List of chunk dictionaries preserving layout information.
        """
        chunked_docs = []
        layout_elements = doc.get('layout_elements', [])
        
        if not layout_elements:
            # Fallback to regular chunking if no layout elements
            return self.chunk_text_with_metadata(doc.get('text', ''), {
                'filename': doc.get('filename', ''),
                'page_num': doc.get('page_num', 0),
                'total_pages': doc.get('total_pages', 0),
                'content_type': 'text'
            })
        
        # Group elements by type for efficient processing
        text_elements = []
        table_elements = []
        image_elements = []
        
        for elem in layout_elements:
            elem_type = elem.get('type', '')
            
            if elem_type in ['text', 'text_block']:
                text_elements.append(elem)
            elif elem_type == 'table':
                table_elements.append(elem)
            elif elem_type == 'image_text':
                image_elements.append(elem)
        
        # Process text elements - these might need further chunking
        for elem in text_elements:
            elem_text = elem.get('content', '')
            if not elem_text.strip():
                continue
                
            # Check if this is a large text block that needs chunking
            if len(elem_text) > self.chunk_size:
                # Further chunk this text element
                if self.use_semantic_boundaries:
                    elem_chunks = self._semantic_chunking(elem_text)
                else:
                    elem_chunks = self.text_splitter.split_text(elem_text)
                
                # Create a document for each chunk
                for i, chunk_text in enumerate(elem_chunks):
                    if not chunk_text.strip():
                        continue
                        
                    chunked_doc = {
                        'text': chunk_text,
                        'filename': doc.get('filename', ''),
                        'page_num': doc.get('page_num', 0),
                        'total_pages': doc.get('total_pages', 0),
                        'chunk_index': i,
                        'total_chunks': len(elem_chunks),
                        'content_type': 'text',
                        'bbox': elem.get('bbox', None),
                        'is_bold': elem.get('is_bold', False),
                        'font_size': elem.get('font_size', 0)
                    }
                    
                    chunked_docs.append(chunked_doc)
            else:
                # Small enough to keep as is
                chunked_doc = {
                    'text': elem_text,
                    'filename': doc.get('filename', ''),
                    'page_num': doc.get('page_num', 0),
                    'total_pages': doc.get('total_pages', 0),
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'content_type': 'text',
                    'bbox': elem.get('bbox', None),
                    'is_bold': elem.get('is_bold', False),
                    'font_size': elem.get('font_size', 0)
                }
                
                chunked_docs.append(chunked_doc)
        
        # Process tables - each table is a single chunk
        for elem in table_elements:
            table_content = elem.get('content', '')
            if not table_content.strip():
                continue
                
            chunked_doc = {
                'text': table_content,
                'filename': doc.get('filename', ''),
                'page_num': doc.get('page_num', 0),
                'total_pages': doc.get('total_pages', 0),
                'content_type': 'table',
                'bbox': elem.get('bbox', None)
            }
            
            chunked_docs.append(chunked_doc)
        
        # Process image text - each image text is a single chunk
        for elem in image_elements:
            img_text = elem.get('content', '')
            if not img_text.strip():
                continue
                
            chunked_doc = {
                'text': img_text,
                'filename': doc.get('filename', ''),
                'page_num': doc.get('page_num', 0),
                'total_pages': doc.get('total_pages', 0),
                'content_type': 'image_text',
                'bbox': elem.get('bbox', None)
            }
            
            chunked_docs.append(chunked_doc)
        
        return chunked_docs
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """Split text into chunks with respect to semantic boundaries.
        
        This tries to split at paragraph or sentence boundaries where possible.
        
        Args:
            text: Text to split into chunks.
            
        Returns:
            List of text chunks.
        """
        if not text:
            return []
        
        # First try to split by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        # If the paragraphs are short enough, just return them
        if all(len(p) <= self.chunk_size for p in paragraphs):
            return paragraphs
        
        # Check if we should use markdown header splitting
        if self.respect_markdown_headers and any(line.strip().startswith('#') for line in text.split('\n')):
            try:
                # Split by markdown headers
                splits = self.markdown_splitter.split_text(text)
                
                # Convert each split into pure text again (remove metadata)
                md_chunks = []
                for split in splits:
                    # Combine header information with content
                    content = split.page_content
                    metadata = split.metadata
                    
                    # Add headers to content
                    header_text = []
                    for level in range(1, 7):
                        header_key = f'header_{level}'
                        if header_key in metadata and metadata[header_key]:
                            header_prefix = '#' * level
                            header_text.append(f"{header_prefix} {metadata[header_key]}")
                    
                    if header_text:
                        content = '\n'.join(header_text) + '\n\n' + content
                    
                    md_chunks.append(content)
                
                # If chunks are still too large, apply sentence-based chunking
                result_chunks = []
                for chunk in md_chunks:
                    if len(chunk) <= self.chunk_size:
                        result_chunks.append(chunk)
                    else:
                        result_chunks.extend(self._split_by_sentences(chunk))
                
                return result_chunks
            except Exception as e:
                if self.verbose:
                    print(f"Error in markdown chunking: {e}")
                # Fall back to sentence-based chunking
        
        # Otherwise, go to sentence-level chunking
        return self._split_by_sentences(text)
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into chunks at sentence boundaries.
        
        Args:
            text: Text to split.
            
        Returns:
            List of text chunks.
        """
        # Get all sentences
        sentences = sent_tokenize(text)
        
        # Combine sentences into chunks of appropriate size
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Handle very long sentences by splitting them further
            if sentence_length > self.chunk_size:
                # If we have a current chunk, add it first
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split the long sentence at word boundaries
                words = sentence.split()
                current_sentence_part = []
                current_part_length = 0
                
                for word in words:
                    word_length = len(word) + 1  # +1 for the space
                    
                    if current_part_length + word_length <= self.chunk_size:
                        current_sentence_part.append(word)
                        current_part_length += word_length
                    else:
                        # Add the current part and start a new one
                        chunks.append(' '.join(current_sentence_part))
                        current_sentence_part = [word]
                        current_part_length = word_length
                
                # Add any remaining part
                if current_sentence_part:
                    chunks.append(' '.join(current_sentence_part))
            
            # Normal case - sentence fits or fits with current chunk
            elif current_length + sentence_length + 1 <= self.chunk_size:  # +1 for space
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            else:
                # Add the current chunk to results
                chunks.append(' '.join(current_chunk))
                # Start a new chunk with this sentence
                current_chunk = [sentence]
                current_length = sentence_length
        
        # Add the last chunk if there is one
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_text_with_metadata(
        self, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Split text into chunks and preserve metadata.
        
        Args:
            text: The text to split into chunks.
            metadata: Metadata to preserve for each chunk.
            
        Returns:
            List of chunk dictionaries with text and metadata.
        """
        # Skip empty text
        if not text or not text.strip():
            return []
        
        # Use semantic boundaries if enabled
        if self.use_semantic_boundaries:
            chunks = self._semantic_chunking(text)
        else:
            chunks = self.text_splitter.split_text(text)
        
        if self.verbose:
            print(f"Split text into {len(chunks)} chunks")
        
        chunked_documents = []
        
        # Create a new document for each chunk with metadata
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():  # Skip empty chunks
                continue
                
            # Create a copy of metadata to avoid modifying the original
            chunk_metadata = metadata.copy()
            
            # Add chunk-specific metadata
            chunk_metadata['chunk_index'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            
            chunked_doc = {
                'text': chunk_text,
                **chunk_metadata
            }
            
            chunked_documents.append(chunked_doc)
        
        return chunked_documents 