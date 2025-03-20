import os
import time
import tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd

# Import our RAG pipeline
from utils.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vectorstore")
TEMP_PDF_DIR = os.path.join(DATA_DIR, "temp_pdfs")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(TEMP_PDF_DIR, exist_ok=True)

# App configuration
st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialization
def initialize_rag_pipeline():
    """Initialize the RAG pipeline with settings from the session state."""
    api_key = os.getenv("GOOGLE_API_KEY")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    # Get parameters from session state
    embedding_model = st.session_state.get("embedding_model", "models/text-embedding-004")
    llm_model = st.session_state.get("llm_model", "gemini-1.5-flash")
    vector_store_type = st.session_state.get("vector_store_type", "chroma")
    chunk_size = st.session_state.get("chunk_size", 1000)
    chunk_overlap = st.session_state.get("chunk_overlap", 200)
    top_k = st.session_state.get("top_k", 5)
    use_vertex_ai = st.session_state.get("use_vertex_ai", False)
    
    return RAGPipeline(
        vector_store_dir=VECTOR_STORE_DIR,
        collection_name="pdf_documents",
        vector_store_type=vector_store_type,
        embedding_model=embedding_model,
        llm_model=llm_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        api_key=api_key,
        use_vertex_ai=use_vertex_ai,
        project_id=project_id,
        location=location,
        verbose=True
    )

# Sidebar
def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        
        # Use different embedding models based on API choice
        use_vertex_ai = st.checkbox(
            "Use Vertex AI",
            value=False,
            help="Use Vertex AI for embeddings (requires GCP project)"
        )
        st.session_state.use_vertex_ai = use_vertex_ai
        
        if use_vertex_ai:
            # Vertex AI embedding models
            embedding_options = [
                "text-embedding-004",
                "textembedding-gecko",
                "textembedding-gecko-multilingual"
            ]
            embedding_default_idx = 0
        else:
            # Google Generative AI embedding models
            embedding_options = [
                "models/text-embedding-004",
                "models/embedding-001"
            ]
            embedding_default_idx = 0
        
        st.session_state.embedding_model = st.selectbox(
            "Embedding Model",
            options=embedding_options,
            index=embedding_default_idx,
            help="Select the embedding model to use"
        )
        
        st.session_state.llm_model = st.selectbox(
            "LLM Model",
            options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
            index=0,
            help="Select the LLM model for question answering"
        )
        
        # Chunking settings
        st.subheader("Chunking Settings")
        
        st.session_state.chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="Size of text chunks in characters"
        )
        
        st.session_state.chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Overlap between text chunks in characters"
        )
        
        # Retrieval settings
        st.subheader("Retrieval Settings")
        
        st.session_state.top_k = st.slider(
            "Number of Documents to Retrieve",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Number of similar documents to retrieve"
        )
        
        st.session_state.vector_store_type = st.selectbox(
            "Vector Store Type",
            options=["chroma", "faiss"],
            index=0,
            help="Type of vector store to use"
        )
        
        # Reset vector store
        if st.button("Reset Vector Store", type="primary"):
            pipeline = initialize_rag_pipeline()
            pipeline.clear_vector_store()
            st.success("Vector store has been reset.")
            
            # Clear the processed files
            if "processed_files" in st.session_state:
                st.session_state.processed_files = []
                st.session_state.processing_stats = None
            
            st.session_state.vector_store_info = pipeline.get_vector_store_info()
            
        # Vector store info
        if "vector_store_info" in st.session_state:
            st.subheader("Vector Store Information")
            for key, value in st.session_state.vector_store_info.items():
                st.write(f"**{key}:** {value}")

# File uploader
def file_uploader_section():
    """Render the file uploader section."""
    st.header("📤 Upload PDF Files")
    
    # Initialize session state for processed files
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to process"
    )
    
    if uploaded_files:
        process_files_button = st.button("Process Files", type="primary")
        
        if process_files_button:
            # Process the files
            process_uploaded_files(uploaded_files)
    
    # Display processed files
    if st.session_state.processed_files:
        st.subheader("Processed Files")
        
        for file in st.session_state.processed_files:
            st.write(f"📄 {file}")
        
        # Display processing stats if available
        if "processing_stats" in st.session_state and st.session_state.processing_stats:
            display_processing_stats(st.session_state.processing_stats)

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files.
    
    Args:
        uploaded_files: List of uploaded PDF files.
    """
    # Initialize RAG pipeline
    pipeline = initialize_rag_pipeline()
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Save uploaded files to temporary directory
    temp_files = []
    for i, file in enumerate(uploaded_files):
        progress = (i / len(uploaded_files)) * 0.3
        progress_bar.progress(progress)
        status_text.text(f"Saving {file.name}...")
        
        # Create temporary file
        temp_file_path = os.path.join(TEMP_PDF_DIR, file.name)
        with open(temp_file_path, "wb") as f:
            f.write(file.getvalue())
        
        temp_files.append(temp_file_path)
        
        # Add to processed files if not already there
        if file.name not in st.session_state.processed_files:
            st.session_state.processed_files.append(file.name)
    
    # Process the PDF files
    status_text.text("Processing PDFs...")
    progress_bar.progress(0.3)
    
    try:
        # Process the files
        stats = pipeline.process_pdf_files(temp_files)
        
        # Update progress
        progress_bar.progress(0.9)
        status_text.text("Finalizing...")
        
        # Save processing stats
        st.session_state.processing_stats = stats
        
        # Get vector store info
        st.session_state.vector_store_info = pipeline.get_vector_store_info()
        
        # Complete progress bar
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Success message
        st.success(f"Successfully processed {len(temp_files)} PDF files with {stats['pages']} pages and {stats['chunks']} chunks.")
        
    except Exception as e:
        st.error(f"Error processing PDF files: {str(e)}")
    
    finally:
        # Clean up the progress elements
        progress_bar.empty()
        status_text.empty()

def display_processing_stats(stats):
    """Display processing statistics with a chart.
    
    Args:
        stats: Dictionary of processing statistics.
    """
    col1, col2 = st.columns(2)
    
    with col1:
        # Display stats as metrics
        st.metric("Files Processed", stats["processed_files"])
        st.metric("Pages Extracted", stats["pages"])
        st.metric("Chunks Created", stats["chunks"])
        st.metric("Chunks Embedded", stats["embedded_chunks"])
        
        if "processing_time" in stats:
            st.metric("Processing Time", f"{stats['processing_time']:.2f} seconds")
    
    with col2:
        # Create a bar chart
        chart_data = pd.DataFrame({
            "Metric": ["Pages", "Chunks", "Embedded Chunks"],
            "Count": [stats["pages"], stats["chunks"], stats["embedded_chunks"]]
        })
        
        fig = px.bar(
            chart_data,
            x="Metric",
            y="Count",
            title="Document Processing Statistics",
            color="Metric",
            color_discrete_sequence=["#0068C9", "#83C9FF", "#29B09D"]
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Query section
def query_section():
    """Render the query section."""
    st.header("❓ Ask Questions")
    
    # Check if any files have been processed
    if not st.session_state.get("processed_files", []):
        st.warning("Upload and process PDF files first.")
        return
    
    # Query input
    query = st.text_input("Enter your question:", placeholder="e.g., What are the health hazards mentioned in the documents?")
    
    # Query button
    col1, col2 = st.columns([1, 9])
    with col1:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    with col2:
        pass
    
    # Process query
    if query and ask_button:
        process_query(query)

def process_query(query):
    """Process a query and display the results.
    
    Args:
        query: User's query string.
    """
    # Initialize RAG pipeline
    pipeline = initialize_rag_pipeline()
    
    # Display spinner while processing
    with st.spinner("Processing your question..."):
        # Query the RAG pipeline
        answer, retrieved_docs = pipeline.query(query)
        
        # Save results to session state
        st.session_state.last_query = query
        st.session_state.last_answer = answer
        st.session_state.last_retrieved_docs = retrieved_docs
    
    # Display results
    display_query_results(query, answer, retrieved_docs)

def display_query_results(query, answer, retrieved_docs):
    """Display query results.
    
    Args:
        query: User's query string.
        answer: Answer generated by the LLM.
        retrieved_docs: List of retrieved documents.
    """
    # Display answer
    st.subheader("📝 Answer")
    st.markdown(answer)
    
    # Display retrieved documents
    with st.expander("View Retrieved Documents", expanded=False):
        st.subheader("📚 Retrieved Documents")
        
        for i, doc in enumerate(retrieved_docs):
            score = doc.get('score', 0.0)
            filename = doc.get('filename', 'Unknown')
            page_num = doc.get('page_num', 'Unknown')
            
            st.markdown(f"**Document {i+1}** - Score: {score:.4f}")
            st.markdown(f"**Source:** {filename}, page {page_num}")
            st.text_area(
                f"Content {i+1}",
                value=doc.get('text', 'No text available'),
                height=150,
                key=f"doc_{i}"
            )
            st.markdown("---")

# Exact Answers Section
def exact_answers_section():
    """Render the exact answers section."""
    st.header("📑 Exact Answers from Documents")
    
    # Check if any files have been processed
    if not st.session_state.get("processed_files", []):
        st.warning("Upload and process PDF files first.")
        return
    
    # Query input
    query = st.text_input("Enter your question:", 
                          placeholder="e.g., What are the health hazards mentioned in the documents?",
                          key="exact_query_input")
    
    # Query button
    col1, col2 = st.columns([1, 9])
    with col1:
        ask_button = st.button("Ask", type="primary", use_container_width=True, key="exact_ask_button")
    with col2:
        pass
    
    # Process query
    if query and ask_button:
        process_exact_query(query)

def process_exact_query(query):
    """Process a query and find relevant answer excerpts.
    
    Args:
        query: User's query string.
    """
    # Initialize RAG pipeline
    pipeline = initialize_rag_pipeline()
    
    # Display spinner while processing
    with st.spinner("Retrieving answers from documents..."):
        # Step 1: Retrieve relevant documents - no LLM processing
        query_embedding = pipeline.embeddings_generator.generate_query_embedding(query)
        
        # Retrieve more candidates initially for better selection
        k_retrieve = min(pipeline.top_k * 3, 10)  # Retrieve more candidates than we'll show
        retrieved_docs = pipeline.vector_store.similar_search(
            query_embedding=query_embedding,
            k=k_retrieve
        )
        
        if not retrieved_docs:
            st.warning("No relevant documents found for your query.")
            return
            
        # Step 2: Determine if this is a list-type or aggregation question
        is_list_question = detect_list_or_aggregation_question(query)
        
        # Step 3: Select best answers based on query type
        if is_list_question:
            # For list questions, get multiple relevant documents
            best_docs = select_multiple_relevant_excerpts(query, retrieved_docs)
            # Use all selected docs as context for the refined answer
            refined_answer = generate_synthesized_answer(query, best_docs, pipeline)
        else:
            # For standard questions, get the single best answer
            best_doc = rerank_and_select_best_answer(query, retrieved_docs)
            best_docs = [best_doc] if best_doc else []
            # Generate refined answer from the single best doc
            refined_answer = generate_refined_answer(query, best_doc, pipeline)
        
        # Save results to session state
        st.session_state.last_exact_query = query
        st.session_state.best_answer_docs = best_docs
        st.session_state.refined_answer = refined_answer
        st.session_state.all_retrieved_docs = retrieved_docs  # Keep all docs for reference
        st.session_state.is_list_question = is_list_question
    
    # Display the results
    display_best_exact_result(query, best_docs, refined_answer, retrieved_docs, is_list_question)

def detect_list_or_aggregation_question(query):
    """Detect if a query is asking for a list, ranking, or aggregated information.
    
    Args:
        query: User's query string.
        
    Returns:
        Boolean indicating if this is a list-type question.
    """
    query_lower = query.lower()
    
    # List of patterns that suggest a list-type or aggregation question
    list_patterns = [
        "top", "list", "best", "ranking", "ranked", "rank", 
        "most", "highest", "largest", "biggest",
        "major", "main", "key", "primary", "principal",
        "factors", "reasons", "causes", "effects",
        "compare", "comparison", "versus", "vs",
        "advantages", "disadvantages", "pros", "cons",
        "steps", "stages", "phases", "methods",
        "types", "categories", "classes", "groups",
        "examples", "instances", "cases",
        "statistics", "numbers", "figures", "data points"
    ]
    
    # Check for cardinal numbers indicating a list request (e.g., "top 3", "5 ways")
    number_patterns = [
        r"\b\d+\s*(st|nd|rd|th)?\b"  # Numbers like "3" or "3rd"
    ]
    
    # Check if query contains list patterns
    for pattern in list_patterns:
        if pattern in query_lower.split():
            return True
    
    # Check for cardinal numbers in the context of a list
    import re
    for pattern in number_patterns:
        matches = re.findall(pattern, query_lower)
        if matches and any(p in query_lower for p in ["top", "best", "ways", "methods", "reasons", "factors"]):
            return True
    
    return False

def select_multiple_relevant_excerpts(query, candidate_docs, max_excerpts=3):
    """Select multiple relevant excerpts for list-type questions.
    
    Args:
        query: User's query string.
        candidate_docs: List of candidate document excerpts.
        max_excerpts: Maximum number of excerpts to select.
        
    Returns:
        List of the most relevant document excerpts.
    """
    if not candidate_docs:
        return []
    
    # If only a few candidates, return all of them
    if len(candidate_docs) <= max_excerpts:
        return candidate_docs
    
    # Enhanced imports
    import re
    import math
    
    # Score all documents using the enhanced scoring method
    scored_docs = []
    query_words = set(query.lower().split())
    
    # Named entity detection for queries (similar to rerank_and_select_best_answer)
    potential_entities = re.findall(r'\b[A-Z][a-zA-Z]*\b|\b[A-Z]{2,}\b', query)
    entity_words = set([entity.lower() for entity in potential_entities])
    
    # Determine if we're looking for numeric values or dates
    is_numeric_query = any(x in query.lower() for x in ['how many', 'number of', 'amount', 'total', 'count', 'quantity'])
    is_date_query = any(x in query.lower() for x in ['when', 'date', 'year', 'month', 'time'])
    
    # Extract list indicators (top N, best X, etc.)
    list_count_match = re.search(r'(?:top|best|worst|leading|major|key|main|primary|important|significant)\s+(\d+)', query.lower())
    list_count = int(list_count_match.group(1)) if list_count_match else 3
    
    # Adjust max_excerpts based on detected list count but keep within reasonable bounds
    max_excerpts = min(max(list_count, 3), 5)
    
    # Extract query bigrams for contextual matching
    query_bigrams = []
    query_terms = query.lower().split()
    for i in range(len(query_terms) - 1):
        query_bigrams.append(f"{query_terms[i]} {query_terms[i+1]}")
    
    # Used to track uniqueness of information
    seen_key_points = set()
    seen_numbers = set()
    seen_dates = set()
    
    for doc in candidate_docs:
        text = doc.get('text', '').lower()
        
        # 1. Original semantic similarity score (already in doc)
        base_score = doc.get('score', 0.0)
        
        # Floor the base score to improve overall relevance
        if base_score > 0:
            base_score = 0.3 + (base_score * 0.7)  # Ensure minimum score if there's any match
        
        # 2. Keyword match score - how many query terms appear in the text
        text_words = set(text.split())
        keyword_match_score = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
        
        # 3. Entity match score - prioritize documents with relevant named entities
        entity_match_score = 0
        if entity_words:
            entity_matches = len(entity_words.intersection(text_words))
            entity_match_score = entity_matches / len(entity_words)
            if entity_matches == len(entity_words) and len(entity_words) > 0:
                entity_match_score *= 1.5  # Significant boost for complete entity match
        
        # 4. Information density score - prefer excerpts with numbers and specific data
        # More sophisticated for list queries to find data-rich passages
        numbers_in_text = re.findall(r'\b\d+(?:[\.,]\d+)?%?\b', text)
        dates_in_text = re.findall(r'\b\d{4}\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:\s*,\s*\d{4})?\b', text, re.IGNORECASE)
        
        # Extract unique numbers to avoid redundant excerpts
        unique_numbers = set(numbers_in_text) - seen_numbers
        unique_dates = set(dates_in_text) - seen_dates
        
        # Calculate density score with emphasis on the right type of data
        if is_numeric_query:
            density_factor = len(numbers_in_text) * 1.5 + len(dates_in_text) * 0.5
        elif is_date_query:
            density_factor = len(dates_in_text) * 1.5 + len(numbers_in_text) * 0.5
        else:
            density_factor = len(numbers_in_text) + len(dates_in_text)
        
        density_score = min(density_factor / 5, 1.0)
        
        # Information uniqueness - how much unique information this excerpt adds
        uniqueness_score = 0
        if len(seen_key_points) > 0:
            # Extract potential key points (sentences with numbers or entities)
            sentences = re.split(r'[.!?]+', text)
            key_sentences = [s for s in sentences if any(word in s for word in entity_words) or re.search(r'\b\d+\b', s)]
            
            # Check for unique key sentences
            unique_sentences = []
            for sent in key_sentences:
                sent_key = ' '.join(sorted(set(sent.split())))
                if sent_key not in seen_key_points and len(sent) > 20:  # Avoid trivial sentences
                    unique_sentences.append(sent)
                    seen_key_points.add(sent_key)
            
            uniqueness_score = min(len(unique_sentences) / 2, 1.0)
        else:
            uniqueness_score = 1.0  # First document is fully unique
        
        # 5. Contextual relevance - check for query term pairs
        bigram_matches = sum(1 for bg in query_bigrams if bg in text)
        context_score = min(1.0, bigram_matches / max(1, len(query_bigrams)))
        
        # 6. List indicator match - check for list patterns (bullets, numbers, "top X", etc.)
        has_list_marker = bool(re.search(r'(\d+\.\s|\•|\-\s|first|second|third|fourth|fifth|top|best|most)', text))
        list_marker_score = 0.3 if has_list_marker else 0
        
        # 7. Source diversity boost - based on filename to encourage different sources
        # This is handled later when selecting documents
        
        # 8. Calculate combined score with weights tailored for list questions
        combined_score = (
            base_score * 0.30 +           # Original semantic similarity
            keyword_match_score * 0.15 +   # Keyword matching
            entity_match_score * 0.20 +    # Entity matching
            density_score * 0.25 +         # Information density
            uniqueness_score * 0.25 +      # Information uniqueness
            context_score * 0.15 +         # Contextual relevance
            list_marker_score * 0.10       # List marker bonus
        )
        
        # Apply logarithmic boost to ensure there's more separation between scores
        combined_score = 0.5 + (0.5 * math.log(1 + combined_score))
        
        # Store scores for debugging/analysis
        doc['_list_scores'] = {
            'base': base_score,
            'keyword': keyword_match_score,
            'entity': entity_match_score,
            'density': density_score,
            'uniqueness': uniqueness_score,
            'context': context_score,
            'list_marker': list_marker_score,
            'combined': combined_score
        }
        
        # Update seen numbers and dates from this document
        seen_numbers.update(numbers_in_text)
        seen_dates.update(dates_in_text)
        
        scored_docs.append((doc, combined_score))
    
    # Sort by combined score (descending)
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Apply diversity selection - prefer excerpts from different sources with high scores
    selected_docs = []
    seen_sources = set()
    
    # First pass - get top docs from different sources
    for doc, score in scored_docs:
        source_key = f"{doc.get('filename', 'Unknown')}:{doc.get('page_num', 'Unknown')}"
        
        if source_key not in seen_sources:
            # Update the document's score with our enhanced score
            doc['score'] = score
            selected_docs.append(doc)
            seen_sources.add(source_key)
            
            if len(selected_docs) >= max_excerpts:
                break
    
    # If we don't have enough diverse sources, add remaining top docs
    if len(selected_docs) < max_excerpts:
        for doc, score in scored_docs:
            if doc not in selected_docs:
                # Update the document's score with our enhanced score
                doc['score'] = score
                selected_docs.append(doc)
                
                if len(selected_docs) >= max_excerpts:
                    break
    
    return selected_docs

def generate_synthesized_answer(query, docs, pipeline):
    """Generate a synthesized answer from multiple document excerpts.
    
    Args:
        query: User's query string.
        docs: List of relevant document excerpts.
        pipeline: The RAG pipeline.
        
    Returns:
        A synthesized answer string.
    """
    if not docs:
        return "No relevant information found to answer this question."
    
    # Prepare context from all documents
    context_docs = []
    for doc in docs:
        context_docs.append({
            'text': doc.get('text', ''),
            'filename': doc.get('filename', 'Unknown'),
            'page_num': doc.get('page_num', 'Unknown')
        })
    
    try:
        # Use the specialized list query method from the LLMRetriever
        refined_answer = pipeline.llm_retriever.query_list_with_context(
            question=query,
            retrieved_docs=context_docs
        )
        
        return refined_answer
    except Exception as e:
        return f"Could not generate a synthesized answer: {str(e)}"

def rerank_and_select_best_answer(query, candidate_docs):
    """Rerank candidate documents and select the best answer.
    
    This function implements a multi-factor scoring approach to select
    the best document excerpt that answers the query.
    
    Args:
        query: User's query string.
        candidate_docs: List of candidate document excerpts.
        
    Returns:
        The single best document excerpt.
    """
    if not candidate_docs:
        return None
    
    # If only one candidate, return it
    if len(candidate_docs) == 1:
        return candidate_docs[0]
    
    # Calculate enhanced scores for each candidate
    scored_docs = []
    query_words = set(query.lower().split())
    
    # Enhanced imports
    import re
    import math
    
    # Named entity detection for queries (simple pattern matching)
    # Look for potential named entities in the query (proper nouns, acronyms, etc.)
    potential_entities = re.findall(r'\b[A-Z][a-zA-Z]*\b|\b[A-Z]{2,}\b', query)
    entity_words = set([entity.lower() for entity in potential_entities])
    
    # Determine query type to adjust weights
    is_definition = any(x in query.lower() for x in ['what is', 'define', 'explain', 'describe', 'meaning of'])
    is_comparison = any(x in query.lower() for x in ['compare', 'difference', 'versus', 'vs', 'similarities', 'differences'])
    is_factual = any(x in query.lower() for x in ['how many', 'when did', 'where is', 'who is', 'which'])
    
    for doc in candidate_docs:
        text = doc.get('text', '').lower()
        
        # 1. Original semantic similarity score (already in doc)
        base_score = doc.get('score', 0.0)
        
        # Scale the base score to improve overall relevance
        # This helps when the initial vector similarity is low
        if base_score > 0:
            base_score = 0.2 + (base_score * 0.8)  # Minimum score of 0.2 if there's any match
        
        # 2. Keyword match score - how many query terms appear in the text
        text_words = set(text.split())
        keyword_match_score = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
        
        # 2.1 Named entity match - prioritize documents containing the same named entities
        entity_match_score = 0
        if entity_words:
            entity_matches = len(entity_words.intersection(text_words))
            entity_match_score = entity_matches / len(entity_words)
            # Boost if ALL entities match (exact relevant document)
            if entity_matches == len(entity_words) and len(entity_words) > 0:
                entity_match_score *= 1.5
        
        # 3. Position score - higher score for terms appearing early in the text
        position_score = 0
        for word in query_words:
            pos = text.find(word)
            if pos >= 0:
                # Normalize position (earlier is better)
                position_score += 1 - min(pos / 150, 0.9)  # Increased from 100 to 150 for better weighting
        position_score = position_score / len(query_words) if query_words else 0
        
        # 4. Text length score - prefer concise answers but not too short
        # Ideal length now dynamically adjusted based on query type
        text_len = len(text)
        ideal_min = 100 if is_definition else 150
        ideal_max = 300 if is_factual else 400
        
        if text_len < 50:
            length_score = 0.4  # Too short (decreased penalty)
        elif ideal_min <= text_len <= ideal_max:
            length_score = 1.0  # Ideal length
        else:
            length_score = 0.7  # Longer but still acceptable (reduced penalty)
        
        # 5. Contextual relevance - check if the text contains context-specific terms
        query_bigrams = []
        query_terms = query.lower().split()
        for i in range(len(query_terms) - 1):
            query_bigrams.append(f"{query_terms[i]} {query_terms[i+1]}")
        
        bigram_matches = sum(1 for bg in query_bigrams if bg in text)
        context_score = min(1.0, bigram_matches / max(1, len(query_bigrams)))
        
        # 6. Information density score - prefer excerpts with numbers and specific data
        numbers_in_text = len(re.findall(r'\b\d+\b', text))
        dates_in_text = len(re.findall(r'\b\d{4}\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b', text, re.IGNORECASE))
        
        # Prioritize numbers/dates more for factual questions
        density_score = min((numbers_in_text + dates_in_text) / (5 if is_factual else 8), 1.0)
        
        # 7. Adjust weights based on query type
        if is_definition:
            weights = {
                'base': 0.35,
                'keyword': 0.25,
                'entity': 0.20,
                'position': 0.25, 
                'length': 0.15,
                'context': 0.20,
                'density': 0.05
            }
        elif is_comparison:
            weights = {
                'base': 0.35,
                'keyword': 0.25,
                'entity': 0.20,
                'position': 0.15, 
                'length': 0.05,
                'context': 0.25,
                'density': 0.15
            }
        elif is_factual:
            weights = {
                'base': 0.30,
                'keyword': 0.20,
                'entity': 0.25,
                'position': 0.15, 
                'length': 0.05,
                'context': 0.15,
                'density': 0.25
            }
        else:
            # General/default weights
            weights = {
                'base': 0.35,
                'keyword': 0.20,
                'entity': 0.15,
                'position': 0.15, 
                'length': 0.05,
                'context': 0.20,
                'density': 0.10
            }
            
        # 8. Calculate combined score with adjusted weights
        combined_score = (
            base_score * weights['base'] +
            keyword_match_score * weights['keyword'] +
            entity_match_score * weights['entity'] +
            position_score * weights['position'] +
            length_score * weights['length'] +
            context_score * weights['context'] +
            density_score * weights['density']
        )
        
        # 9. Apply logarithmic boost to ensure there's more separation between scores
        # This helps distinguish the best document more clearly
        combined_score = 0.5 + (0.5 * math.log(1 + combined_score))
        
        # Store the scores for debugging/explanation if needed
        doc['_scores'] = {
            'base': base_score,
            'keyword': keyword_match_score,
            'entity': entity_match_score,
            'position': position_score,
            'length': length_score,
            'context': context_score,
            'density': density_score,
            'combined': combined_score
        }
        
        scored_docs.append((doc, combined_score))
    
    # Sort by combined score (descending)
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top document
    best_doc = scored_docs[0][0]
    
    # Store the enhanced relevance score in the original document
    best_doc['score'] = scored_docs[0][1]  # Update with the new higher score
    
    # Return the highest-scoring document
    return best_doc

def generate_refined_answer(query, doc, pipeline):
    """Generate a refined, concise answer from the selected document using the LLM.
    
    Args:
        query: User's query string.
        doc: The best document excerpt.
        pipeline: The RAG pipeline.
        
    Returns:
        A refined answer string.
    """
    if not doc:
        return "No relevant information found to answer this question."
    
    text = doc.get('text', '')
    filename = doc.get('filename', 'Unknown')
    page_num = doc.get('page_num', 'Unknown')
    
    # Create a single document list for the LLM retriever
    context_doc = [{
        'text': text,
        'filename': filename,
        'page_num': page_num
    }]
    
    try:
        # Use the LLM to generate a refined answer based on just this excerpt
        refined_answer = pipeline.llm_retriever.query_with_context(
            question=query,
            retrieved_docs=context_doc
        )
        return refined_answer
    except Exception as e:
        return f"Could not generate a refined answer: {str(e)}"

def display_best_exact_result(query, best_docs, refined_answer, all_docs, is_list_question):
    """Display both the exact text excerpts and a refined answer.
    
    Args:
        query: User's query string.
        best_docs: The best document excerpts selected.
        refined_answer: The refined answer generated from the excerpts.
        all_docs: All retrieved documents (for reference).
        is_list_question: Whether this is a list-type question.
    """
    if not best_docs:
        st.warning("No relevant answer found for your query.")
        return
    
    # Create tabs for different views of the answer
    answer_tab1, answer_tab2 = st.tabs(["Refined Answer", "Exact Text Excerpts"])
    
    with answer_tab1:
        # Display the refined answer
        if is_list_question:
            st.subheader("📝 Synthesized List Answer")
        else:
            st.subheader("📝 Refined Answer")
        
        # Add the answer content
        st.markdown(refined_answer)
        
        # Add source attribution for list questions
        if is_list_question:
            st.markdown("---")
            st.markdown("*Answer synthesized from multiple sources:*")
            for i, doc in enumerate(best_docs):
                filename = doc.get('filename', 'Unknown')
                page_num = doc.get('page_num', 'Unknown')
                st.markdown(f"*Source {i+1}: {filename}, page {page_num}*")
    
    with answer_tab2:
        # Display the exact text excerpts
        if is_list_question:
            st.subheader("📄 Relevant Text Excerpts")
            
            # Display multiple excerpts for list questions
            for i, doc in enumerate(best_docs):
                score = doc.get('score', 0.0)
                filename = doc.get('filename', 'Unknown')
                page_num = doc.get('page_num', 'Unknown')
                text = doc.get('text', 'No text available')
                
                # Display source information
                st.markdown(f"### Excerpt {i+1} (Relevance: {score:.4f})")
                st.markdown(f"**Source:** {filename}, page {page_num}")
                
                # Display the exact text
                st.markdown("""
                <style>
                .answer-box {
                    background-color: #f0f2f6;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-left: 4px solid #0068C9;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown(f'<div class="answer-box">{text}</div>', unsafe_allow_html=True)
                
                if i < len(best_docs) - 1:  # Don't add divider after the last excerpt
                    st.markdown("---")
        else:
            # For regular questions, display the single best excerpt
            st.subheader("📄 Exact Text Excerpt")
            
            # Extract information from the best document (first in the list)
            best_doc = best_docs[0]
            score = best_doc.get('score', 0.0)
            filename = best_doc.get('filename', 'Unknown')
            page_num = best_doc.get('page_num', 'Unknown')
            text = best_doc.get('text', 'No text available')
            
            # Display source information prominently
            st.markdown(f"**Source:** {filename}, page {page_num}")
            
            # Display the exact text in a box with a light background
            st.markdown("""
            <style>
            .answer-box {
                background-color: #f0f2f6;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
                border-left: 4px solid #0068C9;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f'<div class="answer-box">{text}</div>', unsafe_allow_html=True)
    
    # Add option to view all retrieved excerpts
    with st.expander("View All Retrieved Excerpts", expanded=False):
        st.subheader("📚 All Retrieved Excerpts")
        
        for i, doc in enumerate(all_docs):
            # Skip docs that are already shown in the main view
            if doc in best_docs:
                continue
                
            doc_score = doc.get('score', 0.0)
            doc_filename = doc.get('filename', 'Unknown')
            doc_page_num = doc.get('page_num', 'Unknown')
            doc_text = doc.get('text', 'No text available')
            
            st.markdown(f"**Excerpt {i+1}** - Relevance Score: {doc_score:.4f}")
            st.markdown(f"**Source:** {doc_filename}, page {doc_page_num}")
            st.text_area(
                f"Content {i+1}",
                value=doc_text,
                height=100,
                key=f"alt_doc_{i}"
            )
            st.markdown("---")

# Main app
def main():
    """Main application function."""
    st.title("📚 PDF Question Answering")
    st.markdown("""
    Upload PDF documents and ask questions to get accurate answers based on the content.
    This application uses Retrieval-Augmented Generation (RAG) to provide precise responses.
    """)
    
    # Check if API key is available
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error(
            "Google API key not found! Please create a .env file with your GOOGLE_API_KEY."
        )
        st.stop()
    
    # Initialize RAG pipeline and get vector store info if not already in session state
    if "vector_store_info" not in st.session_state:
        pipeline = initialize_rag_pipeline()
        st.session_state.vector_store_info = pipeline.get_vector_store_info()
    
    # Render sidebar
    render_sidebar()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Upload & Process", "Exact Answers"])
    
    with tab1:
        file_uploader_section()
    
    with tab2:
        exact_answers_section()

if __name__ == "__main__":
    main() 