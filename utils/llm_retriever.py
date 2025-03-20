import os
from typing import List, Dict, Any, Optional, Union
import time
import re

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class LLMRetriever:
    """A class for querying an LLM with context from retrieved documents."""
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.2,
        top_p: float = 0.9,
        api_key: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        maintain_session_context: bool = True,
        verbose: bool = False
    ):
        """Initialize the LLMRetriever.
        
        Args:
            model_name: Name of the LLM model to use.
            temperature: Temperature parameter for the LLM.
            top_p: Top-p parameter for the LLM.
            api_key: Google API key.
            conversation_history: Previous conversation turns.
            maintain_session_context: Whether to maintain context across queries.
            verbose: Whether to print additional information.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose
        self.conversation_history = conversation_history or []
        self.maintain_session_context = maintain_session_context
        self.recent_context_references = set()  # Track document references for context awareness
        
        # Set API key if provided
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            convert_system_message_to_human=True
        )
        
        # Create the RAG prompt
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "{question}")
        ])
        
        # Create the list synthesis prompt
        self.list_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_list_system_prompt()),
            ("human", "{question}")
        ])
        
        # Create the table interpretation prompt
        self.table_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_table_system_prompt()),
            ("human", "{question}")
        ])
        
        # Create the standalone prompt
        self.standalone_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions accurately and concisely."),
            ("human", "{question}")
        ])
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for RAG.
        
        Returns:
            System prompt string.
        """
        # Add conversation history if available
        history_context = ""
        if self.maintain_session_context and self.conversation_history:
            history_context = "Previous conversation:\n"
            for turn in self.conversation_history[-3:]:  # Include up to 3 most recent turns
                history_context += f"User: {turn.get('user', '')}\n"
                history_context += f"Assistant: {turn.get('assistant', '')}\n"
            history_context += "\n"
        
        # Add recent document references if available
        document_context = ""
        if self.recent_context_references and self.maintain_session_context:
            document_context = "References mentioned in previous answers:\n"
            for ref in self.recent_context_references:
                document_context += f"- {ref}\n"
            document_context += "\nPlease maintain consistency with these references if relevant to the current question.\n\n"
                
        return f"""You are a helpful, accurate, and concise assistant. Your task is to answer the user's question
based on the retrieved context from PDF documents. Follow these guidelines:

1. Answer ONLY based on the context provided. Do not use prior knowledge.
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question."
3. Be concise and to the point.
4. Always include the source of your information in the format (Source: [filename], page [page_num]).
5. Where relevant, include direct quotes from the context.
6. If multiple sources provide information, cite each source.
7. Pay special attention to tables, images, and document layout elements in the provided context.
8. For tabular data, interpret the structure and extract relevant information.
9. If the user's question refers to previous conversation, maintain consistency in your answers.

{history_context}{document_context}Context:
{{context}}
"""

    def _get_list_system_prompt(self) -> str:
        """Get the specialized system prompt for list-type questions.
        
        Returns:
            System prompt string for list synthesis.
        """
        # Add conversation history if available
        history_context = ""
        if self.maintain_session_context and self.conversation_history:
            history_context = "Previous conversation:\n"
            for turn in self.conversation_history[-3:]:  # Include up to 3 most recent turns
                history_context += f"User: {turn.get('user', '')}\n"
                history_context += f"Assistant: {turn.get('assistant', '')}\n"
            history_context += "\n"
                
        return f"""You are a helpful, accurate, and concise assistant. Your task is to synthesize information 
from multiple document excerpts to create a comprehensive list-based answer to the user's question.

Follow these guidelines:
1. Identify list items, rankings, or key points from ALL provided excerpts
2. Combine and deduplicate information from different sources
3. Present your answer as a clear, numbered list wherever appropriate
4. For rankings, ensure you respect the order suggested in the sources
5. Include the source of each piece of information in the format (Source: [filename], page [page_num])
6. If different sources provide conflicting information, note the discrepancy
7. If the context doesn't contain enough information, say "I don't have complete information to answer this question."
8. Pay special attention to tables and structured data in the context
9. Consider images with OCR text as valid sources of information
10. Keep your answer focused on the most relevant information to the user's question

{history_context}
Context:
{{context}}
"""

    def _get_table_system_prompt(self) -> str:
        """Get the specialized system prompt for table interpretation.
        
        Returns:
            System prompt string for table data.
        """
        return """You are a data analysis assistant specializing in interpreting tabular information. 
Your task is to extract and analyze data from tables in the provided context to answer the user's question.

Follow these guidelines:
1. Carefully analyze the structure of tables in the context
2. Extract specific data points requested in the question
3. Perform calculations, comparisons, or trend analysis if needed
4. Present numeric data precisely as it appears in the source
5. Include the table source in your answer (Source: [filename], page [page_num])
6. If the tables contain incomplete information, clearly state the limitations
7. Format your answer in a clear, structured way that highlights the key data
8. If appropriate, suggest visualizations that would help interpret the data

Context:
{context}
"""
    
    def _format_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string.
        
        Args:
            docs: List of retrieved documents.
            
        Returns:
            Formatted context string.
        """
        formatted_docs = []
        
        for i, doc in enumerate(docs):
            # Get document metadata
            filename = doc.get('filename', 'Unknown document')
            page_num = doc.get('page_num', 'Unknown')
            content_type = doc.get('content_type', 'text')
            
            # Store reference for context tracking if maintaining session
            if self.maintain_session_context:
                ref_key = f"{filename}, page {page_num}"
                self.recent_context_references.add(ref_key)
            
            # Format the document text based on content type
            if content_type == 'table':
                doc_text = f"Table {i+1} (Source: {filename}, page {page_num}):\n{doc['text']}\n"
            elif content_type == 'image_text':
                doc_text = f"Image Text {i+1} (Source: {filename}, page {page_num}):\n{doc['text']}\n"
            else:
                doc_text = f"Document {i+1} (Source: {filename}, page {page_num}):\n{doc['text']}\n"
            
            formatted_docs.append(doc_text)
        
        # Join all documents with a separator
        return "\n---\n".join(formatted_docs)
    
    def _update_conversation_history(self, question: str, answer: str):
        """Update the conversation history with the latest exchange.
        
        Args:
            question: User's question.
            answer: Assistant's answer.
        """
        if self.maintain_session_context:
            # Prevent history from growing too large
            if len(self.conversation_history) >= 10:
                self.conversation_history.pop(0)
            
            # Add the current exchange
            self.conversation_history.append({
                'user': question,
                'assistant': answer
            })
    
    def _extract_document_references(self, text: str):
        """Extract document references from an answer to track context.
        
        Args:
            text: The answer text to extract references from.
        """
        if not self.maintain_session_context:
            return
            
        # Use regex to find document references in the format (Source: filename, page X)
        pattern = r'Source:\s*(.*?),\s*page\s*(\d+)'
        matches = re.findall(pattern, text)
        
        for match in matches:
            if len(match) >= 2:
                filename = match[0].strip()
                page_num = match[1].strip()
                self.recent_context_references.add(f"{filename}, page {page_num}")
    
    def query_with_context(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """Query the LLM with context from retrieved documents.
        
        Args:
            question: User's question.
            retrieved_docs: List of retrieved documents to use as context.
            
        Returns:
            LLM response string.
        """
        if not retrieved_docs:
            if self.verbose:
                print("No documents retrieved for context")
            return self.query_without_context(question)
        
        # Check if this is a table-heavy context
        table_docs = [doc for doc in retrieved_docs if doc.get('content_type') == 'table']
        
        # Format the documents into context
        context = self._format_docs(retrieved_docs)
        
        if self.verbose:
            print(f"Querying with {len(retrieved_docs)} documents as context")
            if table_docs:
                print(f"Context includes {len(table_docs)} tables")
        
        try:
            # Choose the appropriate prompt based on content
            if len(table_docs) > 0 and len(table_docs) / len(retrieved_docs) >= 0.5:
                # Use table prompt if tables make up at least half the context
                prompt_template = self.table_prompt
            else:
                # Use standard RAG prompt
                prompt_template = self.rag_prompt
            
            # Create and run the chain
            rag_chain = (
                {"context": lambda _: context, "question": RunnablePassthrough()}
                | prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            answer = rag_chain.invoke(question)
            
            # Update conversation history and extract references
            self._update_conversation_history(question, answer)
            self._extract_document_references(answer)
            
            return answer
            
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def query_list_with_context(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """Query the LLM with a specialized prompt for list-type questions.
        
        Args:
            question: User's question about a list, ranking, or aggregation.
            retrieved_docs: List of retrieved documents to use as context.
            
        Returns:
            A synthesized list answer.
        """
        if not retrieved_docs:
            if self.verbose:
                print("No documents retrieved for context")
            return "No relevant information found to answer this question."
        
        # Format the documents into context
        context = self._format_docs(retrieved_docs)
        
        if self.verbose:
            print(f"Querying for list synthesis with {len(retrieved_docs)} documents as context")
        
        try:
            # Create and run the chain with the list-specific prompt
            list_chain = (
                {"context": lambda _: context, "question": RunnablePassthrough()}
                | self.list_prompt
                | self.llm
                | StrOutputParser()
            )
            
            answer = list_chain.invoke(question)
            
            # Update conversation history and extract references
            self._update_conversation_history(question, answer)
            self._extract_document_references(answer)
            
            return answer
            
        except Exception as e:
            print(f"Error querying LLM for list synthesis: {e}")
            return f"Sorry, I encountered an error synthesizing a list answer: {str(e)}"
    
    def query_without_context(self, question: str) -> str:
        """Query the LLM without any context.
        
        Args:
            question: User's question.
            
        Returns:
            LLM response string.
        """
        try:
            # Create and run the chain
            standalone_chain = (
                {"question": RunnablePassthrough()}
                | self.standalone_prompt
                | self.llm
                | StrOutputParser()
            )
            
            answer = standalone_chain.invoke(question)
            
            # Update conversation history
            self._update_conversation_history(question, answer)
            
            return answer
            
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
            
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        self.recent_context_references = set() 