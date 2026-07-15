"""
PDF ingestion pipeline: PDF → Markdown → chunking → embedding → Chroma storage.
"""
import os
import re
import uuid
import gc
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Generator
import fitz  # PyMuPDF
from transformers import AutoTokenizer

from backend.embedder import get_embedder
from backend.chroma_client import get_chroma_client

logger = logging.getLogger(__name__)

# Minimum characters required to consider a PDF page as having extractable text.
# Pages with fewer characters are treated as image-only (OCR not supported).
MIN_TEXT_LENGTH = 10


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

    def __init__(self, tokenizer_name: str = "bert-base-uncased", chunk_size: int = 200, overlap: int = 20):
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
        self.chunker = None  # Lazy initialized on first use

    def extract_pdf_text(self, pdf_path: str, failed_pages: List[int]) -> Generator[Tuple[int, str], None, None]:
        """
        Extract text from PDF with page-level tracking, streaming one page at a time.

        This is a generator: it yields (page_number, text) tuples one page at a
        time instead of accumulating every page's text in memory. Failed or
        image-only pages are recorded into the caller-supplied `failed_pages`
        list (mutated in place) so callers retain the exact same failure
        reporting behavior without the generator needing to return a tuple.

        Args:
            pdf_path: Path to PDF file
            failed_pages: List that page numbers of failed/image-only pages
                will be appended to (mutated in place)

        Yields:
            (page_number, text) tuples for pages with extractable text
        """
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise ValueError(f"Cannot open PDF file: {e}")

        pages_extracted = 0

        try:
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    text = page.get_text()

                    # Check if page is image-only (very short text)
                    if len(text.strip()) < MIN_TEXT_LENGTH:
                        logger.warning(f"Page {page_num + 1} appears to be image-only (OCR not supported)")
                        failed_pages.append(page_num + 1)
                        continue

                    pages_extracted += 1
                    yield (page_num + 1, text)

                except Exception as e:
                    logger.error(f"Failed to extract page {page_num + 1}: {e}")
                    failed_pages.append(page_num + 1)

            logger.info(f"Extracted {pages_extracted} pages, {len(failed_pages)} failed")

        finally:
            doc.close()

    def process_pdf(self, pdf_path: str, filename: str) -> Dict:
        """
        Complete PDF ingestion pipeline.

        Streams the PDF page-by-page directly from extract_pdf_text() (a
        generator), and within each page streams chunk embedding/storage in
        batches. Only one page's text/chunks and one embedding batch are ever
        held in memory at a time. No global page-text or chunk list is
        accumulated across the whole document.

        Args:
            pdf_path: Path to PDF file
            filename: Original filename

        Returns:
            Dict with doc_id, status, chunks, failed_pages
        """
        doc_id = str(uuid.uuid4())
        logger.info(f"Starting ingestion for {filename} (doc_id: {doc_id})")

        # Lazy-initialize chunker on first use
        if self.chunker is None:
            self.chunker = TextChunker()

        ingested_at = datetime.now().isoformat()
        BATCH_SIZE = 32
        total_stored = 0
        chunk_index = 0
        pages_processed = 0
        failed_pages: List[int] = []

        # Stream pages directly from the generator; nothing from a finished
        # page is retained once we move on to the next one.
        for page_num, text in self.extract_pdf_text(pdf_path, failed_pages):
            pages_processed += 1

            # Convert to Markdown
            markdown_text = self.markdown_converter.convert(text)

            # Chunk this page's text
            chunks = self.chunker.chunk(markdown_text)

            # Embed and store this page's chunks in batches immediately
            for start in range(0, len(chunks), BATCH_SIZE):
                batch_chunks = chunks[start:start + BATCH_SIZE]

                batch_embeddings = retry_with_backoff(
                    lambda: self.embedder.encode(batch_chunks, batch_size=8),
                    max_retries=3,
                    initial_delay=1.0
                )

                embeddings_list = batch_embeddings.tolist()

                batch_metadatas = []
                batch_ids = []

                for _ in batch_chunks:
                    batch_metadatas.append({
                        "doc_id": doc_id,
                        "source_filename": filename,
                        "page_number": page_num,
                        "chunk_index": chunk_index,
                        "ingested_at": ingested_at,
                        "embedding_model": self.embedder.active_model_name
                    })

                    batch_ids.append(f"{doc_id}___{chunk_index}")
                    chunk_index += 1

                self.chroma_client.add_chunks(
                    ids=batch_ids,
                    documents=batch_chunks,
                    embeddings=embeddings_list,
                    metadatas=batch_metadatas
                )

                total_stored += len(batch_ids)

                del batch_embeddings
                del embeddings_list
                batch_metadatas.clear()
                batch_ids.clear()
                del batch_ids
                del batch_metadatas
                del batch_chunks
                gc.collect()

            # Free this page's memory before moving to the next page
            chunks.clear()
            del chunks
            del markdown_text
            del text
            gc.collect()

        if pages_processed == 0:
            raise ValueError("No text could be extracted from PDF. It may be image-only (OCR not supported).")

        logger.info(f"Generated {total_stored} chunks")

        status = "partial" if failed_pages else "ingested"

        result = {
            "doc_id": doc_id,
            "status": status,
            "chunks": total_stored,
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