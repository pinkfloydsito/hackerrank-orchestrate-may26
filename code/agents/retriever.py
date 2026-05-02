"""Advanced retriever agent with ChromaDB + multi-query + re-ranking."""

from typing import List, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from config import (
    EMBEDDING_MODEL, CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
    TOP_K_RETRIEVAL, TOP_K_FINAL
)
from models import Ticket, RetrievalResult, RetrievedChunk


def _resolve_product_area(ticket: Ticket, product_area: str) -> str:
    """Resolve clean product area for metadata filtering.
    
    Metadata stores clean product area names (e.g., 'Screen', not 'hackerrank:Screen').
    """
    if not product_area:
        return product_area
    
    # If has ecosystem prefix, strip it (metadata stores clean names)
    if ":" in product_area:
        return product_area.split(":", 1)[1].strip()
    
    return product_area


class Retriever:
    """Advanced retriever with metadata filtering, multi-query, and re-ranking."""
    
    _instance = None
    _vectorstore = None
    _embeddings = None
    _reranker = None
    
    def __new__(cls):
        """Singleton pattern to avoid reloading models."""
        if cls._instance is None:
            cls._instance = super(Retriever, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._vectorstore is not None:
            return
            
        print(f"Initializing ChromaDB retriever from {CHROMA_PERSIST_DIR}...")
        
        # Initialize embeddings (shared across all retrievals)
        self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Initialize vector store
        self._vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=self._embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        
        # Initialize cross-encoder re-ranker
        print("Loading cross-encoder re-ranker...")
        self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        print(f"Retriever ready. Collection size: {self._vectorstore._collection.count()}")
    
    def _generate_query_variants(self, query: str) -> List[str]:
        """Generate multiple query variants for better recall.
        
        Strategy: Original + paraphrased + keyword-focused variants.
        """
        variants = [query]
        
        # Add a keyword-focused variant (remove stop words)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                      'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                      'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
                      'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
                      'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
                      'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                      'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                      'because', 'until', 'while', 'what', 'which', 'who', 'whom', 'this',
                      'that', 'these', 'those', 'am', 'it', 'its', 'i', 'me', 'my', 'myself'}
        
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        if keywords:
            variants.append(" ".join(keywords))
        
        return variants
    
    def _build_metadata_filter(self, ticket: Ticket, product_area: Optional[str] = None) -> Optional[dict]:
        """Build metadata filter for ChromaDB query.
        
        ChromaDB requires exactly one operator at the top level.
        Use $and to combine multiple conditions.
        """
        conditions = []
        
        # Filter by company/ecosystem if specified
        if ticket.company and ticket.company != "None":
            conditions.append({"product_ecosystem": ticket.company.lower()})
        
        # Filter by product area if available (exact match on clean taxonomy)
        if product_area:
            conditions.append({"product_area": product_area})
        
        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
    
    def _multi_query_retrieval(
        self, 
        query: str, 
        filter_dict: Optional[dict],
        top_k: int = TOP_K_RETRIEVAL
    ) -> List[RetrievedChunk]:
        """Retrieve using multiple query variants and merge results."""
        all_results = []
        seen_ids = set()
        
        # Generate query variants
        variants = self._generate_query_variants(query)
        
        for variant in variants:
            # Search with each variant
            docs = self._vectorstore.similarity_search(
                query=variant,
                k=top_k,
                filter=filter_dict
            )
            
            for doc in docs:
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    all_results.append(doc)
        
        return all_results
    
    def _rerank_results(
        self, 
        query: str, 
        docs: List,
        top_k: int = TOP_K_FINAL
    ) -> List[RetrievedChunk]:
        """Re-rank results using cross-encoder."""
        if not docs:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, doc.page_content) for doc in docs]
        
        # Get scores
        scores = self._reranker.predict(pairs)
        
        # Sort by score (descending)
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to RetrievedChunk
        chunks = []
        for doc, score in scored_docs[:top_k]:
            # Normalize score to 0-1 range (cross-encoder scores can be negative)
            similarity = max(0.0, min(1.0, (score + 10) / 20))  # Rough normalization
            
            chunks.append(RetrievedChunk(
                text=doc.page_content,
                source_file=doc.metadata.get("source_file", ""),
                article_title=doc.metadata.get("article_title", ""),
                section_heading=doc.metadata.get("Header 2", doc.metadata.get("Header 1", "")),
                product_ecosystem=doc.metadata.get("product_ecosystem", ""),
                similarity=similarity,
            ))
        
        return chunks
    
    def retrieve(
        self, 
        ticket: Ticket, 
        product_area: Optional[str] = None,
        top_k: int = TOP_K_FINAL
    ) -> RetrievalResult:
        """Retrieve relevant chunks with advanced RAG techniques.
        
        Pipeline:
        1. Build metadata filter (resolve product area with ecosystem prefix)
        2. Multi-query retrieval (original + variants)
        3. Cross-encoder re-ranking
        4. Return top-k results
        """
        # Build query from ticket
        query = f"{ticket.subject} {ticket.issue}".strip()
        if not query:
            query = ticket.issue
        
        # Resolve product area to ecosystem-prefixed version
        resolved_product_area = _resolve_product_area(ticket, product_area) if product_area else None
        
        # Build metadata filter
        filter_dict = self._build_metadata_filter(ticket, resolved_product_area)
        
        # Step 1: Multi-query retrieval with filter
        docs = self._multi_query_retrieval(query, filter_dict, top_k=TOP_K_RETRIEVAL)
        
        # Fallback: if no results with product_area filter, try without it
        if not docs and resolved_product_area:
            # Try with just ecosystem filter
            ecosystem_only_filter = self._build_metadata_filter(ticket, None)
            docs = self._multi_query_retrieval(query, ecosystem_only_filter, top_k=TOP_K_RETRIEVAL)
        
        # Step 2: Re-rank
        chunks = self._rerank_results(query, docs, top_k=top_k)
        
        # Calculate max similarity
        max_similarity = max([c.similarity for c in chunks]) if chunks else 0.0
        
        return RetrievalResult(
            chunks=chunks,
            max_similarity=max_similarity,
            query_used=query,
        )


# Global retriever instance (lazy-loaded)
_retriever = None

def get_retriever() -> Retriever:
    """Get or create the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


async def retrieve_chunks(
    ticket: Ticket, 
    product_area: Optional[str] = None
) -> RetrievalResult:
    """Retrieve chunks for a ticket (async wrapper)."""
    retriever = get_retriever()
    return retriever.retrieve(ticket, product_area)
