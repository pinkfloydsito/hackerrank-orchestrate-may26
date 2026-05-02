"""Configuration and constants for the support triage agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
SUPPORT_TICKETS_DIR = REPO_ROOT / "support_tickets"
CODE_DIR = REPO_ROOT / "code"

# ChromaDB Configuration
CHROMA_PERSIST_DIR = str(CODE_DIR / "chroma_data")
CHROMA_COLLECTION = "corpus_index"

# Product areas taxonomy
PRODUCT_AREAS_PATH = CODE_DIR / "product_areas.json"

# LLM Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# Embedding Model
# BGE-small-en-v1.5: Better retrieval quality than MiniLM, same 384d dimension
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Chunking
CHUNK_SIZE = 200  # Reduced for better granularity
CHUNK_OVERLAP = 30  # words

# Retrieval
TOP_K_RETRIEVAL = 10  # Retrieve more for re-ranking
TOP_K_FINAL = 5  # Final chunks after re-ranking
SIMILARITY_THRESHOLD = 0.25  # Lower threshold for more lenient matching
HIGH_CONFIDENCE_SIMILARITY = 0.5  # If above this, don't escalate for low classification confidence

# Classification
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.6

# Escalation
ESCALATION_TEMPLATES = {
    "sensitive_data": "This issue involves sensitive information that requires specialized handling. A human agent will contact you securely to assist with this matter.",
    "unsupported": "This request is outside the scope of our current support capabilities. A human agent will review your case and provide guidance.",
    "low_confidence": "We're unable to find sufficient information to address your request accurately. A human agent will assist you shortly.",
    "high_risk": "This issue requires immediate attention from a specialized team. A human agent will contact you as soon as possible.",
    "explicit_request": "As requested, this case has been escalated to a human agent who will contact you shortly.",
}

# Product ecosystems
ECOSYSTEMS = ["hackerrank", "claude", "visa"]

# Request types
REQUEST_TYPES = ["product_issue", "feature_request", "bug", "invalid"]

# CSV columns
INPUT_COLUMNS = ["Issue", "Subject", "Company"]
OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]
