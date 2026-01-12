"""
Query pipeline: Query embedding → Chroma retrieval → prompt building → Gemini LLM.
"""
import os
import logging
from typing import Dict, List, Any
import google.generativeai as genai

from backend.embedder import get_embedder
from backend.chroma_client import get_chroma_client

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Build prompts according to PRD section 9 template."""
    
    TEMPLATE = """Answer the question using ONLY the information from the provided context. Be direct and concise.

Context:
{context}

Question:
{user_question}

Instructions:
- Provide a clear, direct answer without listing sources inline
- Use natural conversational language
- Be concise (2-4 sentences maximum)
- If the context doesn't contain the answer, respond with ONLY: "I don't know."
- Do NOT make up information or use knowledge outside the context
"""
    
    @staticmethod
    def build(query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Build prompt from query and retrieved chunks.
        
        Args:
            query: User question
            chunks: List of dicts with 'document' and 'metadata' keys
        
        Returns:
            Complete prompt string
        """
        context_parts = []
        
        for chunk in chunks:
            metadata = chunk["metadata"]
            document = chunk["document"]
            
            source = metadata.get("source_filename", "unknown")
            page = metadata.get("page_number", "N/A")
            chunk_idx = metadata.get("chunk_index", "N/A")
            
            context_parts.append(
                f"--- [source: {source} | page: {page} | chunk: {chunk_idx}] ---\n{document}"
            )
        
        context_str = "\n\n".join(context_parts)
        
        prompt = PromptBuilder.TEMPLATE.format(
            context=context_str,
            user_question=query,
            source_filename="{source_filename}",
            page_number="{page_number}",
            chunk_index="{chunk_index}"
        )
        
        return prompt


class QueryService:
    """Main query orchestrator with Gemini integration."""
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash-lite"):
        """
        Initialize query service with Gemini API.
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model_name: Gemini model name
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
        self.embedder = get_embedder()
        self.chroma_client = get_chroma_client()
        self.prompt_builder = PromptBuilder()
        
        logger.info(f"QueryService initialized with model: {model_name}")
    
    def answer_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Complete query pipeline: embed → retrieve → prompt → LLM.
        
        Args:
            query: User question
            k: Number of chunks to retrieve (default: 5)
        
        Returns:
            Dict with answer, citations, retrieved_chunks
        """
        logger.info(f"Processing query: {query[:100]}...")
        
        # Embed query
        query_embedding = self.embedder.encode([query], task_type="retrieval_query")[0].tolist()
        
        # Retrieve from Chroma
        results = self.chroma_client.query_similar(query_embedding, k=k)
        
        if not results["ids"]:
            logger.warning("No results found in vector DB")
            return {
                "answer": "I don't know.",
                "citations": [],
                "retrieved_chunks": []
            }
        
        # Build chunks list
        retrieved_chunks = []
        for i in range(len(results["ids"])):
            retrieved_chunks.append({
                "id": results["ids"][i],
                "document": results["documents"][i],
                "metadata": results["metadatas"][i],
                "distance": results["distances"][i]
            })
        
        # Build prompt
        prompt = self.prompt_builder.build(query, retrieved_chunks)
        
        logger.debug(f"Built prompt with {len(retrieved_chunks)} chunks")
        
        # Call Gemini
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=0.5,
                max_output_tokens=512
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            answer = response.text.strip()
            logger.info(f"Generated answer: {answer[:100]}...")
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            
            # Check for rate limiting
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                raise RuntimeError("Gemini API rate limit exceeded. Please try again later.") from e
            
            raise RuntimeError(f"Gemini API error: {e}") from e
        
        # Extract citations from metadata
        citations = []
        for chunk in retrieved_chunks:
            metadata = chunk["metadata"]
            citations.append({
                "source_filename": metadata.get("source_filename"),
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index")
            })
        
        result = {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks
        }
        
        return result


# Singleton instance
_query_service_instance = None


def get_query_service(api_key: str = None) -> QueryService:
    """
    Get or create the singleton query service instance.
    
    Args:
        api_key: Optional API key override
    
    Returns:
        QueryService instance
    """
    global _query_service_instance
    
    if _query_service_instance is None:
        _query_service_instance = QueryService(api_key=api_key)
    
    return _query_service_instance
