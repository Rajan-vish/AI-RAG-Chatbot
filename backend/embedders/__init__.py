"""Embedders package with factory function."""
import os
import logging
from typing import Optional

from backend.interfaces.embedder import EmbedderInterface, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


def create_embedder(provider: Optional[str] = None) -> EmbedderInterface:
    """
    Factory function to create configured embedder.
    
    Args:
        provider: Embedding provider selection
            - "auto" (default): Gemini primary with local fallback
            - "gemini": Gemini API only
            - "local": Local model only
            - None: Read from EMBEDDING_PROVIDER env var
    
    Returns:
        EmbedderInterface: Configured embedder instance
        All embedders produce exactly EMBEDDING_DIMENSION (768) dimensions.
    
    Raises:
        ValueError: If unknown provider specified
        ValueError: If gemini/auto requested but GOOGLE_API_KEY not set
        FileNotFoundError: If local requested but model not downloaded
    
    Environment Variables:
        EMBEDDING_PROVIDER: Default provider if not specified
        GOOGLE_API_KEY: Required for gemini/auto providers
        EMBEDDING_MODEL_PATH: Optional custom local model path
        GEMINI_EMBEDDING_MODEL: Optional custom Gemini model name
    
    Example:
        >>> embedder = create_embedder("auto")
        >>> embeddings = embedder.encode(["hello", "world"])
        >>> embeddings.shape
        (2, 768)
    """
    from .local_embedder import LocalEmbedder
    from .gemini_embedder import GeminiEmbedder
    from .fallback_embedder import FallbackEmbedder
    
    # Get provider from param or env
    provider = provider or os.getenv("EMBEDDING_PROVIDER", "auto")
    provider = provider.lower().strip()
    
    logger.info(f"Creating embedder: provider={provider}, dim={EMBEDDING_DIMENSION}")
    
    if provider == "local":
        logger.info("Using local-only embedder (nomic-embed-text-v1)")
        return LocalEmbedder()
    
    elif provider == "gemini":
        logger.info("Using Gemini-only embedder")
        return GeminiEmbedder()
    
    elif provider == "auto":
        logger.info("Using auto mode: Gemini primary with local fallback")
        
        # Try to create Gemini embedder
        try:
            primary = GeminiEmbedder()
        except ValueError as e:
            logger.warning(f"Cannot create Gemini embedder: {e}")
            logger.info("Falling back to local-only mode")
            return LocalEmbedder()
        
        # Create local fallback
        secondary = LocalEmbedder()
        
        return FallbackEmbedder(primary, secondary)
    
    else:
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            f"Valid options: 'auto', 'gemini', 'local'"
        )


# Re-export for convenience
__all__ = [
    "create_embedder",
    "LocalEmbedder", 
    "GeminiEmbedder", 
    "FallbackEmbedder"
]
