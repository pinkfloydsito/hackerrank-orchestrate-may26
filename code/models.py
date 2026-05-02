"""Pydantic models for structured outputs throughout the pipeline."""

from typing import List, Literal
from pydantic import BaseModel, Field


class Ticket(BaseModel):
    """Input ticket parsed from CSV."""
    issue: str = Field(description="The main ticket body or question")
    subject: str = Field(default="", description="Ticket subject line")
    company: str = Field(default="None", description="Company/ecosystem: HackerRank, Claude, Visa, or None")


class ClassificationResult(BaseModel):
    """Output from classify_type node."""
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"] = Field(
        description="The type of request"
    )
    product_area: str = Field(
        description="The most relevant product area or support category"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score for this classification"
    )


class RetrievedChunk(BaseModel):
    """A single chunk retrieved from the corpus."""
    text: str = Field(description="The chunk text")
    source_file: str = Field(description="Path to source file")
    article_title: str = Field(description="Article title from metadata")
    section_heading: str = Field(default="", description="Section heading if available")
    product_ecosystem: str = Field(description="Which ecosystem this belongs to")
    similarity: float = Field(description="Similarity score")


class RetrievalResult(BaseModel):
    """Output from retrieve_chunks node."""
    chunks: List[RetrievedChunk] = Field(description="Top-k retrieved chunks")
    max_similarity: float = Field(description="Highest similarity score")
    query_used: str = Field(description="The query that was used for retrieval")


class EscalationCheck(BaseModel):
    """Output from check_sensitive node."""
    should_escalate: bool = Field(description="Whether this ticket should be escalated")
    reason: str = Field(description="Primary reason for escalation decision")
    details: str = Field(default="", description="Additional context")


class AgentOutput(BaseModel):
    """Final output for a ticket."""
    status: Literal["replied", "escalated"] = Field(description="Whether replied or escalated")
    product_area: str = Field(description="Product area classification")
    response: str = Field(description="User-facing response")
    justification: str = Field(description="Explanation of the decision")
    request_type: str = Field(description="Request type classification")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Classification confidence")
    max_similarity: float = Field(default=0.0, ge=0.0, le=1.0, description="Max retrieval similarity")
    
    def to_csv_row(self) -> dict:
        """Convert to dict for CSV output."""
        return {
            "status": self.status,
            "product_area": self.product_area,
            "response": self.response,
            "justification": self.justification,
            "request_type": self.request_type,
        }
