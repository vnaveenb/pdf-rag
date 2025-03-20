import os
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class RAGEvaluator:
    """Evaluator for Retrieval-Augmented Generation systems.
    
    This class provides methods to evaluate the performance of a RAG system using
    various metrics including retrieval accuracy, answer relevance, and fluency.
    """
    
    def __init__(
        self,
        rag_pipeline: Any = None,
        ground_truth_file: Optional[str] = None,
        verbose: bool = False
    ):
        """Initialize the RAG evaluator.
        
        Args:
            rag_pipeline: The RAG pipeline to evaluate.
            ground_truth_file: Path to a JSON file with ground truth data.
            verbose: Whether to print detailed logs.
        """
        self.rag_pipeline = rag_pipeline
        self.verbose = verbose
        self.ground_truth = self._load_ground_truth(ground_truth_file) if ground_truth_file else None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.evaluation_results = {}
    
    def _load_ground_truth(self, file_path: str) -> List[Dict[str, Any]]:
        """Load ground truth data from a JSON file.
        
        The expected format is a list of dictionaries, each containing:
        - 'question': The question text
        - 'answer': The ground truth answer
        - 'sources': A list of source document IDs that contain the answer
        
        Args:
            file_path: Path to the ground truth JSON file.
            
        Returns:
            List of ground truth items.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Ground truth file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if self.verbose:
            print(f"Loaded {len(data)} ground truth items")
        
        return data
    
    def create_ground_truth_template(self, output_file: str) -> None:
        """Create a template ground truth file.
        
        Args:
            output_file: Path to save the template file.
        """
        template = [
            # Items from Credit_Cards_A_Sectoral_Analysis.pdf
            {
                "question": "What are the primary factors driving the growth of credit card usage in India?",
                "answer": "The report identifies increased digital adoption, evolving consumer behavior, and the expanding middle class as key factors driving the growth of credit card usage in India.",
                "sources": ["Credit_Cards_A_Sectoral_Analysis.pdf:Abstract", "Credit_Cards_A_Sectoral_Analysis.pdf:Introduction"]
            },
            {
                "question": "How does the paper conduct the sectoral analysis of the Indian credit card industry?",
                "answer": "It performs a SWOT analysis, evaluating the strengths, weaknesses, opportunities, and threats of major issuers such as HDFC, SBI, and ICICI, while also considering market trends.",
                "sources": ["Credit_Cards_A_Sectoral_Analysis.pdf:SWOT", "Credit_Cards_A_Sectoral_Analysis.pdf:Conclusion"]
            },
            # Items from Deloitte_C_Financial_Outlook.pdf
            {
                "question": "What are the key challenges for banks in 2024 as per Deloitte's outlook?",
                "answer": "Deloitte's outlook highlights challenges such as a slowing global economy, rising interest rates, increasing deposit costs, and evolving regulatory requirements.",
                "sources": ["Deloitte_C_Financial_Outlook.pdf:Key_Messages", "Deloitte_C_Financial_Outlook.pdf:Conclusion"]
            },
            {
                "question": "How do macroeconomic factors like inflation and interest rates impact bank profitability?",
                "answer": "High inflation and elevated interest rates lead to increased operating costs and squeezed net interest margins, ultimately reducing bank profitability.",
                "sources": ["Deloitte_C_Financial_Outlook.pdf:Macroeconomic_Factors", "Deloitte_C_Financial_Outlook.pdf:Analysis"]
            },
            # Items from HDFC_Credit_Cards_B_MITC.pdf
            {
                "question": "What are the main fees and charges detailed in the HDFC Credit Cards MITC document?",
                "answer": "The document lists various fees including joining fees, annual membership fees, cash advance fees, and foreign currency transaction charges, among other service charges.",
                "sources": ["HDFC_Credit_Cards_B_MITC.pdf:Fees_and_Charges"]
            },
            {
                "question": "What procedures are outlined for reporting lost or stolen credit cards?",
                "answer": "It details the process of immediate reporting via toll-free numbers or online platforms, followed by contacting the nearest branch to block the card and initiate further fraud protection measures.",
                "sources": ["HDFC_Credit_Cards_B_MITC.pdf:Loss_Theft_Misuse"]
            },
            # Items from VISA_D_Payments.pdf
            {
                "question": "What trends are driving the transformation of payments for SMBs according to the VISA report?",
                "answer": "The report emphasizes trends such as the adoption of real-time payments, digital wallets, QR code transactions, and integrated payment solutions as key drivers for transforming SMB payments.",
                "sources": ["VISA_D_Payments.pdf:Trends", "VISA_D_Payments.pdf:Overview"]
            },
            {
                "question": "How is the evolution of payment technologies expected to impact SMB operations?",
                "answer": "The evolution of payment technologies is expected to streamline payment processes, lower transaction costs, and improve client acquisition and retention for SMBs.",
                "sources": ["VISA_D_Payments.pdf:Impact", "VISA_D_Payments.pdf:Analysis"]
            }
        ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2)
        
        if self.verbose:
            print(f"Created ground truth template at {output_file}")
    
    def evaluate_retrieval(self, questions: List[str], top_k: int = 5) -> Dict[str, float]:
        """Evaluate the retrieval component of the RAG system.
        
        Args:
            questions: List of questions to evaluate.
            top_k: Number of documents to retrieve.
            
        Returns:
            Dictionary with retrieval metrics.
        """
        if not self.ground_truth:
            raise ValueError("Ground truth data required for retrieval evaluation")
        
        if not self.rag_pipeline:
            raise ValueError("RAG pipeline required for evaluation")
        
        metrics = {
            "precision": [],
            "recall": [],
            "f1": [],
            "mean_reciprocal_rank": [],
            "retrieval_time": []
        }
        
        # Map questions to ground truth items
        gt_map = {item["question"]: item for item in self.ground_truth if item["question"] in questions}
        
        for question in questions:
            if question not in gt_map:
                if self.verbose:
                    print(f"Skipping question not in ground truth: {question}")
                continue
            
            # Get ground truth sources
            gt_sources = gt_map[question]["sources"]
            
            # Measure retrieval time
            start_time = time.time()
            
            # Get retrieved documents
            _, retrieved_docs = self.rag_pipeline.query(question)
            
            retrieval_time = time.time() - start_time
            
            # Extract source identifiers from retrieved docs
            retrieved_sources = [f"{doc.get('filename', 'unknown')}:{doc.get('page_num', 0)}" 
                                for doc in retrieved_docs]
            
            # Calculate metrics
            relevant_docs = set(gt_sources).intersection(set(retrieved_sources))
            
            precision = len(relevant_docs) / len(retrieved_sources) if retrieved_sources else 0
            recall = len(relevant_docs) / len(gt_sources) if gt_sources else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate Mean Reciprocal Rank (MRR)
            mrr = 0
            for i, source in enumerate(retrieved_sources):
                if source in gt_sources:
                    mrr = 1.0 / (i + 1)
                    break
            
            # Store metrics
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1"].append(f1)
            metrics["mean_reciprocal_rank"].append(mrr)
            metrics["retrieval_time"].append(retrieval_time)
        
        # Calculate averages
        results = {
            "avg_precision": np.mean(metrics["precision"]) if metrics["precision"] else 0,
            "avg_recall": np.mean(metrics["recall"]) if metrics["recall"] else 0,
            "avg_f1": np.mean(metrics["f1"]) if metrics["f1"] else 0,
            "avg_mrr": np.mean(metrics["mean_reciprocal_rank"]) if metrics["mean_reciprocal_rank"] else 0,
            "avg_retrieval_time": np.mean(metrics["retrieval_time"]) if metrics["retrieval_time"] else 0,
            "num_questions_evaluated": len(metrics["precision"])
        }
        
        # Store results
        self.evaluation_results["retrieval_metrics"] = results
        
        return results
    
    def evaluate_answer_quality(self, questions: List[str]) -> Dict[str, float]:
        """Evaluate the quality of generated answers.
        
        Args:
            questions: List of questions to evaluate.
            
        Returns:
            Dictionary with answer quality metrics.
        """
        if not self.ground_truth:
            raise ValueError("Ground truth data required for answer quality evaluation")
        
        if not self.rag_pipeline:
            raise ValueError("RAG pipeline required for evaluation")
        
        metrics = {
            "rouge1_f": [],
            "rouge2_f": [],
            "rougeL_f": [],
            "bleu": [],
            "answer_time": []
        }
        
        # Map questions to ground truth items
        gt_map = {item["question"]: item for item in self.ground_truth if item["question"] in questions}
        
        for question in questions:
            if question not in gt_map:
                if self.verbose:
                    print(f"Skipping question not in ground truth: {question}")
                continue
            
            # Get ground truth answer
            gt_answer = gt_map[question]["answer"]
            
            # Measure answer generation time
            start_time = time.time()
            
            # Get generated answer
            answer, _ = self.rag_pipeline.query(question)
            
            answer_time = time.time() - start_time
            
            # Calculate ROUGE scores
            rouge_scores = self.rouge_scorer.score(answer, gt_answer)
            
            # Calculate BLEU score
            answer_tokens = word_tokenize(answer.lower())
            gt_tokens = word_tokenize(gt_answer.lower())
            bleu_score = sentence_bleu([gt_tokens], answer_tokens, weights=(0.25, 0.25, 0.25, 0.25))
            
            # Store metrics
            metrics["rouge1_f"].append(rouge_scores["rouge1"].fmeasure)
            metrics["rouge2_f"].append(rouge_scores["rouge2"].fmeasure)
            metrics["rougeL_f"].append(rouge_scores["rougeL"].fmeasure)
            metrics["bleu"].append(bleu_score)
            metrics["answer_time"].append(answer_time)
        
        # Calculate averages
        results = {
            "avg_rouge1": np.mean(metrics["rouge1_f"]) if metrics["rouge1_f"] else 0,
            "avg_rouge2": np.mean(metrics["rouge2_f"]) if metrics["rouge2_f"] else 0,
            "avg_rougeL": np.mean(metrics["rougeL_f"]) if metrics["rougeL_f"] else 0,
            "avg_bleu": np.mean(metrics["bleu"]) if metrics["bleu"] else 0,
            "avg_answer_time": np.mean(metrics["answer_time"]) if metrics["answer_time"] else 0,
            "num_questions_evaluated": len(metrics["rouge1_f"])
        }
        
        # Store results
        self.evaluation_results["answer_quality_metrics"] = results
        
        return results
    
    def evaluate_hallucination(self, questions: List[str]) -> Dict[str, float]:
        """Evaluate hallucination in generated answers.
        
        This function measures how much of the generated answer is 
        not supported by the retrieved context.
        
        Args:
            questions: List of questions to evaluate.
            
        Returns:
            Dictionary with hallucination metrics.
        """
        if not self.rag_pipeline:
            raise ValueError("RAG pipeline required for evaluation")
        
        metrics = {
            "factual_accuracy": [],
            "unsupported_info_ratio": []
        }
        
        for question in questions:
            # Get generated answer and retrieved documents
            answer, retrieved_docs = self.rag_pipeline.query(question)
            
            # Combine all retrieved text into a single context
            context = " ".join([doc.get("text", "") for doc in retrieved_docs])
            
            # Tokenize answer and context
            answer_tokens = set(word_tokenize(answer.lower()))
            context_tokens = set(word_tokenize(context.lower()))
            
            # Common stopwords to exclude from analysis
            stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "like", "through", "over", "before", "after", "between", "from", "up", "down", "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "should", "could", "may", "might", "must", "can", "of", "this", "that", "these", "those"}
            
            # Filter out stopwords
            answer_tokens = answer_tokens - stopwords
            context_tokens = context_tokens - stopwords
            
            # Calculate unsupported information ratio
            if len(answer_tokens) > 0:
                unsupported_tokens = answer_tokens - context_tokens
                unsupported_ratio = len(unsupported_tokens) / len(answer_tokens)
            else:
                unsupported_ratio = 0
            
            # Calculate factual accuracy (approximate)
            if len(answer_tokens) > 0:
                factual_accuracy = len(answer_tokens.intersection(context_tokens)) / len(answer_tokens)
            else:
                factual_accuracy = 0
            
            # Store metrics
            metrics["factual_accuracy"].append(factual_accuracy)
            metrics["unsupported_info_ratio"].append(unsupported_ratio)
        
        # Calculate averages
        results = {
            "avg_factual_accuracy": np.mean(metrics["factual_accuracy"]) if metrics["factual_accuracy"] else 0,
            "avg_unsupported_info_ratio": np.mean(metrics["unsupported_info_ratio"]) if metrics["unsupported_info_ratio"] else 0,
            "num_questions_evaluated": len(metrics["factual_accuracy"])
        }
        
        # Store results
        self.evaluation_results["hallucination_metrics"] = results
        
        return results
    
    def evaluate_fluency(self, questions: List[str]) -> Dict[str, float]:
        """Evaluate the fluency of generated answers.
        
        This uses simple metrics like average sentence length and
        presence of common grammatical elements.
        
        Args:
            questions: List of questions to evaluate.
            
        Returns:
            Dictionary with fluency metrics.
        """
        if not self.rag_pipeline:
            raise ValueError("RAG pipeline required for evaluation")
        
        metrics = {
            "avg_sentence_length": [],
            "avg_word_length": []
        }
        
        for question in questions:
            # Get generated answer
            answer, _ = self.rag_pipeline.query(question)
            
            # Tokenize into sentences and words
            try:
                sentences = nltk.sent_tokenize(answer)
                words = word_tokenize(answer)
                
                # Calculate average sentence length
                if sentences:
                    avg_sent_len = len(words) / len(sentences)
                else:
                    avg_sent_len = 0
                
                # Calculate average word length
                if words:
                    avg_word_len = sum(len(word) for word in words) / len(words)
                else:
                    avg_word_len = 0
                
                # Store metrics
                metrics["avg_sentence_length"].append(avg_sent_len)
                metrics["avg_word_length"].append(avg_word_len)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error analyzing fluency: {e}")
        
        # Calculate averages
        results = {
            "avg_sentence_length": np.mean(metrics["avg_sentence_length"]) if metrics["avg_sentence_length"] else 0,
            "avg_word_length": np.mean(metrics["avg_word_length"]) if metrics["avg_word_length"] else 0,
            "num_questions_evaluated": len(metrics["avg_sentence_length"])
        }
        
        # Store results
        self.evaluation_results["fluency_metrics"] = results
        
        return results
    
    def run_comprehensive_evaluation(self, questions: List[str]) -> Dict[str, Any]:
        """Run a comprehensive evaluation of the RAG system.
        
        Args:
            questions: List of questions to evaluate.
            
        Returns:
            Dictionary with all evaluation metrics.
        """
        start_time = time.time()
        
        # Run all evaluations
        retrieval_metrics = self.evaluate_retrieval(questions)
        answer_quality_metrics = self.evaluate_answer_quality(questions)
        hallucination_metrics = self.evaluate_hallucination(questions)
        fluency_metrics = self.evaluate_fluency(questions)
        
        total_time = time.time() - start_time
        
        # Combine all metrics
        comprehensive_results = {
            "retrieval_metrics": retrieval_metrics,
            "answer_quality_metrics": answer_quality_metrics,
            "hallucination_metrics": hallucination_metrics,
            "fluency_metrics": fluency_metrics,
            "total_evaluation_time": total_time,
            "num_questions_evaluated": len(questions)
        }
        
        self.evaluation_results = comprehensive_results
        
        return comprehensive_results
    
    def save_results(self, output_file: str) -> None:
        """Save evaluation results to a JSON file.
        
        Args:
            output_file: Path to save the results.
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        if self.verbose:
            print(f"Saved evaluation results to {output_file}")
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a human-readable evaluation report.
        
        Args:
            output_file: Optional path to save the report.
            
        Returns:
            Report text as a string.
        """
        if not self.evaluation_results:
            return "No evaluation results available"
        
        report_lines = []
        report_lines.append("# RAG System Evaluation Report")
        report_lines.append("\n## Summary")
        
        # Add summary information
        if "num_questions_evaluated" in self.evaluation_results:
            report_lines.append(f"- Questions evaluated: {self.evaluation_results['num_questions_evaluated']}")
        if "total_evaluation_time" in self.evaluation_results:
            report_lines.append(f"- Total evaluation time: {self.evaluation_results['total_evaluation_time']:.2f} seconds")
        
        # Add retrieval metrics
        if "retrieval_metrics" in self.evaluation_results:
            report_lines.append("\n## Retrieval Performance")
            metrics = self.evaluation_results["retrieval_metrics"]
            report_lines.append(f"- Precision: {metrics.get('avg_precision', 0):.4f}")
            report_lines.append(f"- Recall: {metrics.get('avg_recall', 0):.4f}")
            report_lines.append(f"- F1 Score: {metrics.get('avg_f1', 0):.4f}")
            report_lines.append(f"- Mean Reciprocal Rank: {metrics.get('avg_mrr', 0):.4f}")
            report_lines.append(f"- Average retrieval time: {metrics.get('avg_retrieval_time', 0):.4f} seconds")
        
        # Add answer quality metrics
        if "answer_quality_metrics" in self.evaluation_results:
            report_lines.append("\n## Answer Quality")
            metrics = self.evaluation_results["answer_quality_metrics"]
            report_lines.append(f"- ROUGE-1: {metrics.get('avg_rouge1', 0):.4f}")
            report_lines.append(f"- ROUGE-2: {metrics.get('avg_rouge2', 0):.4f}")
            report_lines.append(f"- ROUGE-L: {metrics.get('avg_rougeL', 0):.4f}")
            report_lines.append(f"- BLEU: {metrics.get('avg_bleu', 0):.4f}")
            report_lines.append(f"- Average answer generation time: {metrics.get('avg_answer_time', 0):.4f} seconds")
        
        # Add hallucination metrics
        if "hallucination_metrics" in self.evaluation_results:
            report_lines.append("\n## Factual Accuracy & Hallucination")
            metrics = self.evaluation_results["hallucination_metrics"]
            report_lines.append(f"- Factual accuracy: {metrics.get('avg_factual_accuracy', 0):.4f}")
            report_lines.append(f"- Unsupported information ratio: {metrics.get('avg_unsupported_info_ratio', 0):.4f}")
        
        # Add fluency metrics
        if "fluency_metrics" in self.evaluation_results:
            report_lines.append("\n## Answer Fluency")
            metrics = self.evaluation_results["fluency_metrics"]
            report_lines.append(f"- Average sentence length: {metrics.get('avg_sentence_length', 0):.2f}")
            report_lines.append(f"- Average word length: {metrics.get('avg_word_length', 0):.2f}")
        
        # Add interpretation
        report_lines.append("\n## Interpretation")
        report_lines.append("### Retrieval Performance")
        report_lines.append("- **Precision**: Percentage of retrieved documents that are relevant. Higher is better.")
        report_lines.append("- **Recall**: Percentage of relevant documents that are retrieved. Higher is better.")
        report_lines.append("- **F1 Score**: Harmonic mean of precision and recall. Higher is better.")
        report_lines.append("- **Mean Reciprocal Rank**: Position of the first relevant document. Higher is better.")
        
        report_lines.append("\n### Answer Quality")
        report_lines.append("- **ROUGE scores**: Measure overlap between generated and reference answers. Higher is better.")
        report_lines.append("- **BLEU score**: Measures translation quality between generated and reference answers. Higher is better.")
        
        report_lines.append("\n### Factual Accuracy & Hallucination")
        report_lines.append("- **Factual accuracy**: Proportion of answer tokens supported by retrieved context. Higher is better.")
        report_lines.append("- **Unsupported information ratio**: Proportion of answer not supported by context. Lower is better.")
        
        # Compile the report
        report_text = "\n".join(report_lines)
        
        # Save report if output file is provided
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            if self.verbose:
                print(f"Saved evaluation report to {output_file}")
        
        return report_text
