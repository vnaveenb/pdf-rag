#!/usr/bin/env python
"""
Evaluation script for the PDF RAG system.

This script runs a comprehensive evaluation of the RAG system using a set of questions
and ground truth data. It generates evaluation metrics and produces a detailed report.
"""

import os
import argparse
import json
from dotenv import load_dotenv
from pathlib import Path

# Import RAG pipeline and evaluator
from utils.rag_pipeline import RAGPipeline
from utils.evaluation import RAGEvaluator

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vectorstore")
EVAL_DIR = os.path.join(DATA_DIR, "evaluation")

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

def create_ground_truth_template():
    """Create a template ground truth file if it doesn't exist."""
    template_path = os.path.join(EVAL_DIR, "ground_truth_template.json")
    if not os.path.exists(template_path):
        evaluator = RAGEvaluator(verbose=True)
        evaluator.create_ground_truth_template(template_path)
        print(f"Created ground truth template at {template_path}")
        print("Please edit this file with your actual ground truth data before running evaluation.")
        return False
    return True

def initialize_rag_pipeline(args):
    """Initialize the RAG pipeline with command line arguments."""
    api_key = os.getenv("GOOGLE_API_KEY")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    return RAGPipeline(
        vector_store_dir=VECTOR_STORE_DIR,
        collection_name="pdf_documents",
        vector_store_type=args.vector_store_type,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        api_key=api_key,
        use_vertex_ai=args.use_vertex_ai,
        project_id=project_id,
        location=location,
        verbose=args.verbose
    )

def run_evaluation(args):
    """Run the evaluation process."""
    # Create necessary directories
    setup_directories()
    
    # Check for ground truth data
    if not os.path.exists(args.ground_truth):
        if args.ground_truth == os.path.join(EVAL_DIR, "ground_truth.json"):
            # Try to create template
            if create_ground_truth_template():
                print(f"Using existing ground truth template: {args.ground_truth}")
            else:
                return
        else:
            print(f"Ground truth file not found: {args.ground_truth}")
            return
    
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    pipeline = initialize_rag_pipeline(args)
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = RAGEvaluator(
        rag_pipeline=pipeline,
        ground_truth_file=args.ground_truth,
        verbose=args.verbose
    )
    
    # Load ground truth data to get questions
    with open(args.ground_truth, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    questions = [item["question"] for item in ground_truth]
    print(f"Loaded {len(questions)} questions for evaluation")
    
    # Run comprehensive evaluation
    print("Running evaluation...")
    results = evaluator.run_comprehensive_evaluation(questions)
    
    # Save results
    results_file = os.path.join(EVAL_DIR, "evaluation_results.json")
    evaluator.save_results(results_file)
    print(f"Saved evaluation results to {results_file}")
    
    # Generate report
    report_file = os.path.join(EVAL_DIR, "evaluation_report.md")
    report = evaluator.generate_report(report_file)
    print(f"Generated evaluation report at {report_file}")
    
    # Print summary
    print("\n===== EVALUATION SUMMARY =====")
    if "retrieval_metrics" in results:
        print(f"Retrieval F1: {results['retrieval_metrics']['avg_f1']:.4f}")
    if "answer_quality_metrics" in results:
        print(f"ROUGE-L Score: {results['answer_quality_metrics']['avg_rougeL']:.4f}")
    if "hallucination_metrics" in results:
        print(f"Factual Accuracy: {results['hallucination_metrics']['avg_factual_accuracy']:.4f}")
    
    print(f"\nSee the full report at {report_file}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the RAG system")
    
    # Evaluation settings
    parser.add_argument("--ground_truth", type=str, 
                        default=os.path.join(EVAL_DIR, "ground_truth.json"),
                        help="Path to ground truth data file")
    
    # Model settings
    parser.add_argument("--embedding_model", type=str, 
                        default="models/text-embedding-004",
                        help="Embedding model to use")
    parser.add_argument("--llm_model", type=str, 
                        default="gemini-1.5-flash",
                        help="LLM model for question answering")
    parser.add_argument("--use_vertex_ai", action="store_true",
                        help="Use Vertex AI instead of Google Generative AI")
    
    # Vector store settings
    parser.add_argument("--vector_store_type", type=str, 
                        default="chroma", choices=["chroma", "faiss"],
                        help="Vector store type")
    
    # Chunking settings
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Size of text chunks in characters")
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="Overlap between text chunks in characters")
    
    # Retrieval settings
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of documents to retrieve")
    
    # Other settings
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args) 