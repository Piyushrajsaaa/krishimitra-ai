"""
KrishiMitra AI -- Knowledge Base Ingestion Script
Loads ICAR documents into ChromaDB vector database.
Run: python scripts/ingest_knowledge.py
"""

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.logger import logger

KNOWLEDGE_FILE = Path("data/knowledge_base/icar_knowledge.json")
CHROMA_PATH    = Path(settings.CHROMA_DB_PATH)
COLLECTION     = settings.CHROMA_COLLECTION_NAME
EMBED_MODEL    = settings.EMBEDDING_MODEL


def load_documents():
    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(f"Knowledge file not found: {KNOWLEDGE_FILE}")
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)
    logger.info(f"Loaded {len(docs)} documents")
    return docs


def build_document_text(doc):
    recs = " ".join(doc.get("recommendations", []))
    return (
        "Crop: " + doc["crop"] + ". "
        "Disease: " + doc["disease"] + ". "
        "Pathogen: " + doc.get("pathogen", "unknown") + ". "
        + doc["content"] + " "
        "Recommendations: " + recs
    )


def ingest():
    print("\n KrishiMitra AI -- Knowledge Base Ingestion")
    print("=" * 50)

    docs = load_documents()
    print("Documents loaded:", len(docs))

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(name=COLLECTION)
        print("Cleared existing collection:", COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"description": "ICAR agricultural knowledge base"}
    )
    print("Collection created:", COLLECTION)

    print("\nLoading embedding model:", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL)
    print("Embedding model ready")

    print("\nEmbedding and ingesting documents...")

    ids        = []
    texts      = []
    embeddings = []
    metadatas  = []

    for doc in docs:
        text = build_document_text(doc)
        emb  = embedder.encode(text).tolist()
        ids.append(doc["id"])
        texts.append(text)
        embeddings.append(emb)
        metadatas.append({
            "title":    doc["title"],
            "crop":     doc["crop"],
            "disease":  doc["disease"],
            "language": doc["language"],
            "source":   doc.get("source", "ICAR"),
        })
        print("   Embedded:", doc["id"], "|", doc["title"][:45])

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    count = collection.count()
    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("   Documents in ChromaDB:", count)
    print("   Collection name      :", COLLECTION)
    print("   Storage path         :", CHROMA_PATH)
    print("=" * 50)

    print("\nRunning retrieval test...")
    test_query = "tomato brown spots fungicide treatment"
    test_emb   = embedder.encode(test_query).tolist()

    results = collection.query(
        query_embeddings=[test_emb],
        n_results=2,
        include=["metadatas", "distances"]
    )

    print("   Query:", test_query)
    print("   Top results:")
    for i, (meta, dist) in enumerate(zip(
        results["metadatas"][0],
        results["distances"][0]
    )):
        similarity = round(1 - dist, 4)
        print("  ", i+1, meta["title"][:50], "| similarity:", similarity)

    logger.info("Knowledge base ingestion complete | " + str(count) + " documents")
    print("\nChromaDB knowledge base is ready!\n")


if __name__ == "__main__":
    ingest()
