import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from backend.embedder import Embedder

@pytest.fixture
def mock_genai():
    with patch('backend.embedder.genai') as mock:
        yield mock

@pytest.fixture
def gemini_embedder(mock_genai):
    # Force provider to gemini
    with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "gemini", "GOOGLE_API_KEY": "test_key"}):
        embedder = Embedder()
        yield embedder

def test_initialization_gemini(gemini_embedder, mock_genai):
    assert gemini_embedder.provider == "gemini"
    mock_genai.configure.assert_called_with(api_key="test_key")

def test_encode_gemini(gemini_embedder, mock_genai):
    # Mock return value of embed_content
    # It returns a dict with 'embedding' key which is a list of lists (for list input)
    mock_genai.embed_content.return_value = {
        'embedding': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    }
    
    texts = ["hello", "world"]
    embeddings = gemini_embedder.encode(texts)
    
    assert embeddings.shape == (2, 3)
    mock_genai.embed_content.assert_called_once()
    args, kwargs = mock_genai.embed_content.call_args
    assert kwargs['model'] == "models/text-embedding-004"
    assert kwargs['content'] == texts
    assert kwargs['task_type'] == "retrieval_document" # default

def test_encode_gemini_query_task(gemini_embedder, mock_genai):
    mock_genai.embed_content.return_value = {
        'embedding': [[0.1, 0.2, 0.3]]
    }
    
    texts = ["query"]
    gemini_embedder.encode(texts, task_type="retrieval_query")
    
    args, kwargs = mock_genai.embed_content.call_args
    assert kwargs['task_type'] == "retrieval_query"

def test_fallback_if_no_api_key():
    with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "gemini"}):
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]
            
        embedder = Embedder()
        # Should fallback to local
        assert embedder.provider == "local"

def test_embedding_dim_gemini(gemini_embedder, mock_genai):
    # Should be 768
    assert gemini_embedder.embedding_dim == 768
