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

# Index files
FAISS_INDEX_PATH = CODE_DIR / "corpus.index"
METADATA_PATH = CODE_DIR / "corpus_metadata.pkl"
PRODUCT_AREAS_PATH = CODE_DIR / "product_areas.json"

# LLM Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Chunking
CHUNK_SIZE = 300  # words
CHUNK_OVERLAP = 50  # words

# Retrieval
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.25

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
