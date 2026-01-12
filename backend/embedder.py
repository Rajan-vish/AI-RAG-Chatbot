"""
Embedding layer using nomic-embed-text-v1 (local) or Google Gemini (cloud).
Provides lazy loading, CPU/GPU detection (local), and normalized vector output.
"""
import os
import logging
from typing import List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

logger = logging.getLogger(__name__)


class Embedder:
    """Wrapper for embedding models (local or Gemini)."""
    
    LOCAL_MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
    GEMINI_MODEL_NAME = "models/text-embedding-004"
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize embedder with lazy loading.
        
        Args:
            model_path: Optional custom path for local model files.
        """
        self.provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        # Configure Gemini if selected
        if self.provider == "gemini":
            if not self.api_key:
                logger.warning("EMBEDDING_PROVIDER is 'gemini' but GOOGLE_API_KEY is missing. Falling back to local.")
                self.provider = "local"
            else:
                genai.configure(api_key=self.api_key)

        # Local model setup (default path)
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path = os.path.join(base_dir, "models", "nomic-embed-text-v1")
        else:
            self.model_path = model_path

        self._model: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Embedder initialized. Provider: {self.provider}, Device: {self._device if self.provider == 'local' else 'api'}")
    
    def _load_model(self):
        """Load the model on first use."""
        # Check if already loaded
        if self.provider == "gemini":
            if self._embedding_dim is not None:
                return
            logger.info(f"Using Gemini embedding model: {self.GEMINI_MODEL_NAME}")
            self._embedding_dim = 768  # text-embedding-004 is 768d
            return

        if self._model is not None:
            return
        
        try:
            logger.info(f"Loading local embedding model: {self.LOCAL_MODEL_NAME}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Embedding model not found at: {self.model_path}\n"
                    "Please run 'python download_model.py' to download the model first."
                )
            
            # Load from local path
            logger.info(f"Loading model from local path: {self.model_path}")
            self._model = SentenceTransformer(self.model_path, device=self._device, trust_remote_code=True)
            
            # Validate model by getting embedding dimension
            test_embedding = self._model.encode(["test"], normalize_embeddings=True)
            self._embedding_dim = test_embedding.shape[1]
            logger.info(f"Model loaded successfully. Embedding dimension: {self._embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(
                f"Model load failed: {e}\n\n"
                "Please ensure you have run 'python download_model.py' to download the model."
            ) from e
    
    def encode(self, texts: List[str], batch_size: int = 32, task_type: str = "retrieval_document") -> np.ndarray:
        """
        Encode texts into normalized embeddings.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding (default: 32)
            task_type: Task type for Gemini (retrieval_document/retrieval_query)
        
        Returns:
            Numpy array of shape (len(texts), embedding_dim) with L2-normalized vectors
        """
        if not texts:
            return np.array([])
        
        # Lazy load model
        self._load_model()
        
        try:
            if self.provider == "gemini":
                return self._encode_gemini(texts, batch_size, task_type)
            else:
                return self._encode_local(texts, batch_size)
            
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise

    def _encode_local(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode using local SentenceTransformer."""
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # L2 normalization
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
        logger.debug(f"Encoded {len(texts)} texts locally into shape {embeddings.shape}")
        return embeddings

    def _encode_gemini(self, texts: List[str], batch_size: int, task_type: str) -> np.ndarray:
        """Encode using Google Gemini API."""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                result = genai.embed_content(
                    model=self.GEMINI_MODEL_NAME,
                    content=batch,
                    task_type=task_type
                )
                
                # result['embedding'] is a list of lists
                batch_embeddings = result['embedding']
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Gemini encoding failed for batch {i}: {e}")
                raise
        
        # Convert to numpy and normalize
        np_embeddings = np.array(embeddings)
        
        # Normalize (Gemini output might not be strictly L2 normalized)
        norms = np.linalg.norm(np_embeddings, axis=1, keepdims=True)
        np_embeddings = np_embeddings / np.maximum(norms, 1e-12)
        
        logger.debug(f"Encoded {len(texts)} texts via Gemini into shape {np_embeddings.shape}")
        return np_embeddings
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension (loads model if not already loaded)."""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim
    
    @property
    def device(self) -> str:
        """Get device being used (cpu, cuda, or api)."""
        if self.provider == "gemini":
            return "api"
        return self._device
    
    @property
    def active_model_name(self) -> str:
        """Get the name of the active embedding model."""
        if self.provider == "gemini":
            return self.GEMINI_MODEL_NAME
        return self.LOCAL_MODEL_NAME


# Singleton instance
_embedder_instance: Optional[Embedder] = None


def get_embedder(model_path: Optional[str] = None) -> Embedder:
    """
    Get or create the singleton embedder instance.
    
    Args:
        model_path: Optional custom path for model files
    
    Returns:
        Embedder instance
    """
    global _embedder_instance
    
    if _embedder_instance is None:
        _embedder_instance = Embedder(model_path=model_path)
    
    return _embedder_instance