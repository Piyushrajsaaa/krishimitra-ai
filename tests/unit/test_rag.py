"""
KrishiMitra AI -- Unit Tests for RAG Service
Run: pytest tests/unit/test_rag.py -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from app.services.rag_service import RAGService


@pytest.fixture
def initialized_rag():
    """RAG service with mocked ChromaDB and embedder."""
    rag = RAGService()

    # Mock embedder — return numpy array so .tolist() works
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.array([0.1] * 384)
    rag.embedder = mock_embedder

    # Mock collection
    mock_collection = MagicMock()
    mock_collection.count.return_value = 8
    mock_collection.query.return_value = {
        "documents": [[
            "Crop: tomato. Disease: Early Blight. Apply Mancozeb 2g/L.",
            "Crop: tomato. Disease: Late Blight. Apply Ridomil 2.5g/L.",
        ]],
        "metadatas": [[
            {
                "title":    "Tomato Early Blight Detection and Management",
                "crop":     "tomato",
                "disease":  "Early Blight",
                "language": "en",
                "source":   "ICAR-IARI 2019"
            },
            {
                "title":    "Tomato Late Blight Detection and Management",
                "crop":     "tomato",
                "disease":  "Late Blight",
                "language": "en",
                "source":   "ICAR-NRCH 2020"
            },
        ]],
        "distances": [[0.15, 0.35]],
    }
    rag.collection = mock_collection
    rag.is_ready   = True
    return rag


# ── Initialization Tests ──────────────────────────────────

def test_rag_not_ready_by_default():
    rag = RAGService()
    assert rag.is_ready is False


def test_rag_retrieve_fails_when_not_initialized():
    rag = RAGService()
    result = rag.retrieve(query="tomato disease")
    assert result == []


def test_rag_initialized_flag(initialized_rag):
    assert initialized_rag.is_ready is True


# ── Retrieval Tests ───────────────────────────────────────

def test_retrieve_returns_list(initialized_rag):
    result = initialized_rag.retrieve(query="tomato brown spots")
    assert isinstance(result, list)


def test_retrieve_returns_correct_count(initialized_rag):
    result = initialized_rag.retrieve(query="tomato disease", top_k=2)
    assert len(result) == 2


def test_retrieve_doc_has_required_keys(initialized_rag):
    result = initialized_rag.retrieve(query="tomato disease")
    required = ["content", "title", "crop", "disease", "language", "source", "similarity"]
    for key in required:
        assert key in result[0], f"Missing key: {key}"


def test_retrieve_similarity_in_range(initialized_rag):
    result = initialized_rag.retrieve(query="tomato disease")
    for doc in result:
        assert -1.0 <= doc["similarity"] <= 1.0


def test_retrieve_enriches_query_with_disease(initialized_rag):
    initialized_rag.retrieve(
        query="brown spots",
        disease_name="Early Blight",
        crop_name="tomato"
    )
    # encode is called — check first positional arg of first call
    call_arg = initialized_rag.embedder.encode.call_args_list[0][0][0]
    assert "Early Blight" in call_arg
    assert "tomato" in call_arg


def test_retrieve_english_filter(initialized_rag):
    initialized_rag.retrieve(query="tomato", language="en")
    # get keyword args passed to collection.query
    call_kwargs = initialized_rag.collection.query.call_args.kwargs
    assert call_kwargs["where"] == {"language": "en"}


def test_retrieve_hindi_filter(initialized_rag):
    initialized_rag.retrieve(query="tomato", language="hi")
    call_kwargs = initialized_rag.collection.query.call_args.kwargs
    assert call_kwargs["where"] == {"language": "hi"}


# ── Context Text Tests ────────────────────────────────────

def test_get_context_text_returns_string(initialized_rag):
    context = initialized_rag.get_context_text(query="tomato disease")
    assert isinstance(context, str)


def test_get_context_text_not_empty(initialized_rag):
    context = initialized_rag.get_context_text(query="tomato disease")
    assert len(context) > 0


def test_get_context_text_contains_source(initialized_rag):
    context = initialized_rag.get_context_text(query="tomato disease")
    assert "Source" in context


def test_get_context_text_no_docs_returns_fallback():
    rag = RAGService()
    rag.is_ready = False
    context = rag.get_context_text(query="unknown query")
    assert "No specific knowledge" in context
