# Support Triage Agent

Terminal-based AI agent for triaging support tickets across HackerRank, Claude, and Visa ecosystems.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Build the corpus index (one-time):
```bash
cd code
python -m pipeline.ingest
```

## Usage

Run on support tickets:
```bash
python main.py --input ../support_tickets/support_tickets.csv --output ../support_tickets/output.csv
```

Run on sample tickets (for testing):
```bash
python main.py --input ../support_tickets/sample_support_tickets.csv --output ../support_tickets/sample_output.csv --limit 5
```

## Architecture

Deterministic pipeline:
1. Parse ticket
2. Classify type (request_type, product_area)
3. Retrieve relevant chunks from FAISS index
4. Check for sensitive content / escalation triggers
5. Generate response or escalate
6. Write to CSV

## Project Structure

```
code/
├── agents/           # Pydantic-ai agents for each pipeline node
├── pipeline/         # Ingestion and processing
├── config.py         # Configuration
├── models.py         # Pydantic output models
└── main.py           # CLI entry point
```
