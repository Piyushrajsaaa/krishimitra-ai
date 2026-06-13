"""
KrishiMitra AI -- RAG (Retrieval Augmented Generation) Service
Retrieves relevant ICAR knowledge for a given disease/query.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from app.core.config import settings
from app.core.logger import logger


class RAGService:
    """
    Retrieves relevant agricultural knowledge from ChromaDB
    based on disease name and farmer query.
    """

    def __init__(self):
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self.embedder: Optional[SentenceTransformer] = None
        self.is_ready = False

    def initialize(self) -> bool:
        """Load ChromaDB client and embedding model."""
        try:
            chroma_path = settings.CHROMA_DB_PATH
            if not Path(chroma_path).exists():
                logger.error(f"ChromaDB not found at {chroma_path}. Run ingest_knowledge.py first.")
                return False

            self.client = chromadb.PersistentClient(path=chroma_path)
            self.collection = self.client.get_collection(
                name=settings.CHROMA_COLLECTION_NAME
            )
            self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
            self.is_ready = True
            count = self.collection.count()
            logger.info(f"RAG Service ready | {count} documents in knowledge base")
            return True

        except Exception as e:
            logger.error(f"RAG Service initialization failed: {e}")
            return False

    def retrieve(
        self,
        query: str,
        disease_name: Optional[str] = None,
        crop_name: Optional[str] = None,
        language: str = "en",
        top_k: int = 3
    ) -> List[Dict]:
        """
        Retrieves top-k relevant documents for a query.

        Args:
            query       : Farmer question or disease description
            disease_name: Detected disease name (optional filter)
            crop_name   : Crop name (optional filter)
            language    : 'en' or 'hi'
            top_k       : Number of documents to retrieve

        Returns:
            List of relevant document dicts with content + metadata
        """
        if not self.is_ready:
            logger.error("RAG Service not initialized. Call initialize() first.")
            return []

        try:
            # Build enriched query for better retrieval
            enriched_query = query
            if disease_name:
                enriched_query = disease_name + " " + enriched_query
            if crop_name:
                enriched_query = crop_name + " " + enriched_query

            # Embed the query
            query_embedding = self.embedder.encode(enriched_query).tolist()

            # Build metadata filter
            where_filter = None
            if language == "hi":
                where_filter = {"language": "hi"}
            else:
                where_filter = {"language": "en"}

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
                where=where_filter
            )

            # Format results
            retrieved = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                similarity = round(1 - dist, 4)
                retrieved.append({
                    "content":    doc,
                    "title":      meta.get("title", ""),
                    "crop":       meta.get("crop", ""),
                    "disease":    meta.get("disease", ""),
                    "language":   meta.get("language", "en"),
                    "source":     meta.get("source", "ICAR"),
                    "similarity": similarity,
                })

            logger.info(
                f"RAG retrieved {len(retrieved)} docs | "
                f"query='{enriched_query[:40]}' | lang={language}"
            )
            return retrieved

        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return []

    def get_context_text(
        self,
        query: str,
        disease_name: Optional[str] = None,
        crop_name: Optional[str] = None,
        language: str = "en",
        top_k: int = 2
    ) -> str:
        """
        Returns retrieved knowledge as a single context string
        ready to be injected into LLM prompt.
        """
        docs = self.retrieve(
            query=query,
            disease_name=disease_name,
            crop_name=crop_name,
            language=language,
            top_k=top_k
        )

        if not docs:
            return "No specific knowledge found. Provide general agricultural advice."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"[Source {i}: {doc['source']}]\n{doc['content']}"
            )

        return "\n\n".join(context_parts)


# Singleton instance
rag_service = RAGService()
