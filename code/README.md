# Support Triage Agent

Terminal-based AI agent for triaging support tickets across HackerRank, Claude, and Visa ecosystems.

## Prerequisites

- Python 3.10+
- At least one API key: DeepSeek (primary), OpenAI, or Anthropic

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `pydantic-ai` - Agent framework with structured outputs
- `langchain` + `langchain-chroma` - RAG pipeline and vector store
- `sentence-transformers` - Local embeddings (BGE-small-en-v1.5)
- `openai` - LLM client for DeepSeek/OpenAI/Anthropic APIs
- `pandas`, `numpy`, `tqdm` - Data processing

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Required: DeepSeek (recommended)
DEEPSEEK_API_KEY=your_deepseek_key_here

# Optional fallbacks
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

**Note:** DeepSeek is the primary LLM. OpenAI/Anthropic are used as fallbacks if DeepSeek fails.

### 3. Build the Corpus Index (One-Time)

This parses all support documentation and creates a ChromaDB vector index:

```bash
python -m pipeline.ingest
```

This will:
- Parse markdown files from `../data/hackerrank/`, `../data/claude/`, `../data/visa/`
- Split into chunks using markdown headers
- Generate embeddings using BGE-small-en-v1.5
- Store in ChromaDB at `./chroma_data/`
- Save product areas taxonomy to `./product_areas.json`

**Expected output:** ~15,000 chunks indexed (takes 5-10 minutes on first run).

## Usage

### Process All Support Tickets

```bash
python main.py --input ../support_issues/support_issues.csv --output ../support_issues/output.csv
```

### Process Sample Tickets (for Testing)

```bash
python main.py --input ../support_issues/sample_support_issues.csv --output ../support_issues/test_output.csv --limit 5
```

### Run with Ingestion

If you've updated the corpus data, rebuild the index and run:

```bash
python main.py --ingest --input ../support_issues/support_issues.csv --output ../support_issues/output.csv
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--input`, `-i` | Input CSV file with tickets | `../support_issues/support_issues.csv` |
| `--output`, `-o` | Output CSV file for results | `../support_issues/output.csv` |
| `--limit`, `-l` | Process only first N tickets (for testing) | None |
| `--ingest` | Rebuild corpus index before processing | False |

### Expected Output Format

The agent produces a CSV with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `status` | `replied` or `escalated` | `replied` |
| `product_area` | Most relevant support category | `Account Settings` |
| `response` | User-facing answer | `To reset your password...` |
| `justification` | Explanation of decision | `High confidence match...` |
| `request_type` | `product_issue`, `feature_request`, `bug`, `invalid` | `product_issue` |

## Architecture

Deterministic pipeline with async execution:

1. **Parse** - Read ticket from CSV (Issue, Subject, Company)
2. **Classify** - Determine request_type and product_area using LLM
3. **Retrieve** - Query ChromaDB for relevant documentation chunks
4. **Escalation Check** - Detect sensitive/high-risk cases requiring human review
5. **Generate** - Produce grounded response or escalation message
6. **Write** - Append results to output CSV

## Project Structure

```
code/
├── agents/
│   ├── classifier.py      # Ticket classification (type + product area)
│   ├── retriever.py       # ChromaDB semantic search
│   ├── escalation.py      # Risk/sensitivity detection
│   └── generator.py       # Response generation
├── pipeline/
│   ├── ingest.py          # Corpus parsing and indexing
│   └── processor.py       # Main processing orchestrator
├── config.py              # Configuration and constants
├── models.py              # Pydantic models for structured outputs
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
└── .env.example           # Environment variable template
```

## Troubleshooting

### "No module named 'pipeline'"
Run from the `code/` directory: `cd code && python main.py ...`

### "API key not found"
Ensure `.env` file exists in `code/` directory with valid API keys.

### "ChromaDB collection not found"
Run the ingestion step first: `python -m pipeline.ingest`

### Slow processing
- The agent makes multiple LLM calls per ticket
- Use `--limit` for quick testing
- First run downloads embedding model (~100MB)

## Development

### Running Smoke Tests

```bash
python smoke_test.py
```

This validates the pipeline on a small subset and checks output format.

### Adding New Product Areas

Update `product_areas.json` after ingestion, or modify the classifier agent logic in `agents/classifier.py`.

### Customizing Escalation Rules

Edit escalation triggers in `agents/escalation.py` to adjust sensitivity for your use case.
