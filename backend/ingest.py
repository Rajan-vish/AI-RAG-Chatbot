"""
PDF ingestion pipeline: PDF → Markdown → chunking → embedding → Chroma storage.
"""
import os
import re
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import fitz  # PyMuPDF
from transformers import AutoTokenizer

from backend.embedder import get_embedder
from backend.chroma_client import get_chroma_client

logger = logging.getLogger(__name__)


class MarkdownConverter:
    """Simple rules-based PDF text to Markdown converter."""
    
    @staticmethod
    def convert(text: str) -> str:
        """
        Convert plain text to Markdown using simple heuristics.
        
        Args:
            text: Plain text from PDF
        
        Returns:
            Markdown-formatted text
        """
        lines = text.split('\n')
        markdown_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                markdown_lines.append('')
                continue
            
            # Detect headings (ALL CAPS or Title Case with short length)
            if len(stripped) < 100:
                if stripped.isupper() and len(stripped.split()) <= 10:
                    # ALL CAPS → Heading
                    markdown_lines.append(f"## {stripped.title()}")
                    continue
                elif stripped[0].isupper() and stripped.endswith(':'):
                    # Title with colon → Heading
                    markdown_lines.append(f"### {stripped[:-1]}")
                    continue
            
            # Detect lists (lines starting with -, *, •, or numbers)
            if re.match(r'^[-*•]\s+', stripped):
                # Bullet list
                markdown_lines.append(f"- {stripped[2:].strip()}")
                continue
            elif re.match(r'^\d+\.\s+', stripped):
                # Numbered list
                markdown_lines.append(stripped)
                continue
            
            # Detect code blocks (indented lines)
            if line.startswith('    ') or line.startswith('\t'):
                markdown_lines.append(f"```\n{stripped}\n```")
                continue
            
            # Regular paragraph
            markdown_lines.append(stripped)
        
        return '\n'.join(markdown_lines)


class TextChunker:
    """Token-based text chunker with fixed window and overlap."""
    
    def __init__(self, tokenizer_name: str = "bert-base-uncased", chunk_size: int = 400, overlap: int = 50):
        """
        Initialize chunker with tokenizer.
        
        Args:
            tokenizer_name: HuggingFace tokenizer name
            chunk_size: Maximum tokens per chunk
            overlap: Token overlap between chunks
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info(f"TextChunker initialized with {tokenizer_name}, chunk_size={chunk_size}, overlap={overlap}")
    
    def chunk(self, text: str) -> List[str]:
        """
        Chunk text into fixed-size token windows with overlap.
        
        Args:
            text: Text to chunk
        
        Returns:
            List of text chunks
        """
        # Tokenize text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) == 0:
            return []
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            # Get chunk tokens
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            
            # Move start position with overlap
            if end >= len(tokens):
                break
            start += self.chunk_size - self.overlap
        
        logger.debug(f"Chunked text into {len(chunks)} chunks (total tokens: {len(tokens)})")
        return chunks


def retry_with_backoff(func, max_retries: int = 3, initial_delay: float = 1.0):
    """
    Retry decorator with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
    
    Returns:
        Function result or raises last exception
    """
    import time
    
    last_exception = None
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
    
    raise last_exception


class PDFIngestor:
    """Main PDF ingestion orchestrator."""
    
    def __init__(self):
        """Initialize ingestor with embedder and Chroma client."""
        self.embedder = get_embedder()
        self.chroma_client = get_chroma_client()
        self.markdown_converter = MarkdownConverter()
        self.chunker = TextChunker()
    
    def extract_pdf_text(self, pdf_path: str) -> Tuple[List[Tuple[int, str]], List[int]]:
        """
        Extract text from PDF with page-level tracking.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Tuple of (page_texts, failed_pages)
            page_texts: List of (page_number, text) tuples
            failed_pages: List of page numbers that failed
        """
        page_texts = []
        failed_pages = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    text = page.get_text()
                    
                    # Check if page is image-only (very short text)
                    if len(text.strip()) < 10:
                        logger.warning(f"Page {page_num + 1} appears to be image-only (OCR not supported)")
                        failed_pages.append(page_num + 1)
                        continue
                    
                    page_texts.append((page_num + 1, text))
                    
                except Exception as e:
                    logger.error(f"Failed to extract page {page_num + 1}: {e}")
                    failed_pages.append(page_num + 1)
            
            doc.close()
            logger.info(f"Extracted {len(page_texts)} pages, {len(failed_pages)} failed")
            
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise ValueError(f"Cannot open PDF file: {e}")
        
        return page_texts, failed_pages
    
    def process_pdf(self, pdf_path: str, filename: str) -> Dict:
        """
        Complete PDF ingestion pipeline.
        
        Args:
            pdf_path: Path to PDF file
            filename: Original filename
        
        Returns:
            Dict with doc_id, status, chunks, failed_pages
        """
        doc_id = str(uuid.uuid4())
        logger.info(f"Starting ingestion for {filename} (doc_id: {doc_id})")
        
        # Extract text from PDF
        page_texts, failed_pages = self.extract_pdf_text(pdf_path)
        
        if not page_texts:
            raise ValueError("No text could be extracted from PDF. It may be image-only (OCR not supported).")
        
        # Process each page
        all_chunks = []
        chunk_index = 0
        
        for page_num, text in page_texts:
            # Convert to Markdown
            markdown_text = self.markdown_converter.convert(text)
            
            # Chunk text
            chunks = self.chunker.chunk(markdown_text)
            
            for chunk_text in chunks:
                all_chunks.append({
                    "text": chunk_text,
                    "page_number": page_num,
                    "chunk_index": chunk_index
                })
                chunk_index += 1
        
        logger.info(f"Generated {len(all_chunks)} chunks from {len(page_texts)} pages")
        
        # Embed chunks with retry logic
        chunk_texts = [c["text"] for c in all_chunks]
        
        def embed_func():
            return self.embedder.encode(chunk_texts, batch_size=32)
        
        embeddings = retry_with_backoff(embed_func, max_retries=3, initial_delay=1.0)
        
        # Prepare data for Chroma
        ingested_at = datetime.now().isoformat()
        ids = [f"{doc_id}___{c['chunk_index']}" for c in all_chunks]
        documents = chunk_texts
        embeddings_list = embeddings.tolist()
        model_name = self.embedder.active_model_name

        metadatas = [
            {
                "doc_id": doc_id,
                "source_filename": filename,
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
                "ingested_at": ingested_at,
                "embedding_model": model_name
            }
            for c in all_chunks
        ]
        
        # Store in Chroma
        self.chroma_client.add_chunks(
            ids=ids,
            documents=documents,
            embeddings=embeddings_list,
            metadatas=metadatas
        )
        
        status = "partial" if failed_pages else "ingested"
        
        result = {
            "doc_id": doc_id,
            "status": status,
            "chunks": len(all_chunks),
            "failed_pages": failed_pages
        }
        
        logger.info(f"Ingestion complete: {result}")
        return result


# Singleton instance
_ingestor_instance: Optional[PDFIngestor] = None


def get_ingestor() -> PDFIngestor:
    """Get or create the singleton ingestor instance."""
    global _ingestor_instance
    
    if _ingestor_instance is None:
        _ingestor_instance = PDFIngestor()
    
    return _ingestor_instance
