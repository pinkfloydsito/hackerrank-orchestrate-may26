"""Retriever agent for searching relevant chunks from FAISS index."""

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Optional

from config import (
    FAISS_INDEX_PATH, METADATA_PATH, EMBEDDING_MODEL, 
    EMBEDDING_DIM, TOP_K_RETRIEVAL
)
from models import Ticket, RetrievalResult, RetrievedChunk


class Retriever:
    """FAISS-based retriever with company filtering."""
    
    def __init__(self):
        """Initialize retriever with index and metadata."""
        print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        
        print(f"Loading metadata from {METADATA_PATH}...")
        with open(METADATA_PATH, "rb") as f:
            self.docs = pickle.load(f)
        
        print(f"Loading embedding model {EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        
        print(f"Retriever ready. Index size: {self.index.ntotal}")
    
    def retrieve(
        self, 
        ticket: Ticket, 
        product_area: Optional[str] = None,
        top_k: int = TOP_K_RETRIEVAL
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a ticket."""
        # Build query from ticket
        query = f"{ticket.subject} {ticket.issue}".strip()
        if not query:
            query = ticket.issue
        
        # Embed query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        
        # Search index
        distances, indices = self.index.search(query_embedding, top_k * 3)  # Get extra for filtering
        
        # Build results
        chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.docs):
                continue
            
            doc = self.docs[idx]
            
            # Filter by company if specified
            if ticket.company and ticket.company != "None":
                company_lower = ticket.company.lower()
                if doc["product_ecosystem"] != company_lower:
                    continue
            
            # Calculate similarity (convert L2 distance to similarity score)
            distance = distances[0][i]
            similarity = 1.0 / (1.0 + distance)  # Convert to 0-1 range
            
            chunks.append(RetrievedChunk(
                text=doc["text"],
                source_file=doc["source_file"],
                article_title=doc["article_title"],
                section_heading=doc.get("section_heading", ""),
                product_ecosystem=doc["product_ecosystem"],
                similarity=similarity,
            ))
            
            if len(chunks) >= top_k:
                break
        
        max_similarity = max([c.similarity for c in chunks]) if chunks else 0.0
        
        return RetrievalResult(
            chunks=chunks,
            max_similarity=max_similarity,
            query_used=query,
        )


async def retrieve_chunks(
    ticket: Ticket, 
    product_area: Optional[str] = None
) -> RetrievalResult:
    """Retrieve chunks for a ticket."""
    retriever = Retriever()
    return retriever.retrieve(ticket, product_area)
