"""Pipeline processor that orchestrates all agents."""

import pandas as pd
from tqdm import tqdm

from models import Ticket, AgentOutput
from agents.classifier import classify_ticket
from agents.retriever import retrieve_chunks
from agents.escalation import check_escalation
from agents.generator import generate_response, generate_escalation_response


async def process_ticket(ticket_row: dict) -> AgentOutput:
    """Process a single ticket through the pipeline.
    
    Pipeline:
    1. Parse ticket
    2. Classify type (request_type, product_area)
    3. Retrieve relevant chunks
    4. Check escalation
    5. Generate response or escalate
    """
    # Step 1: Parse
    ticket = Ticket(
        issue=ticket_row.get("Issue", ""),
        subject=ticket_row.get("Subject", ""),
        company=ticket_row.get("Company", "None"),
    )
    
    # Step 2: Classify
    classification = await classify_ticket(ticket)
    
    # Step 3: Retrieve
    retrieval = await retrieve_chunks(ticket, classification.product_area)
    
    # Step 4: Check escalation
    escalation = await check_escalation(ticket, classification, retrieval)
    
    # Step 5: Generate
    if escalation.should_escalate:
        output = await generate_escalation_response(ticket, classification, escalation, retrieval)
    else:
        output = await generate_response(ticket, classification, retrieval)
    
    return output


async def process_tickets(
    input_path: str,
    output_path: str,
    limit: int = None,
) -> None:
    """Process all tickets from input CSV and write to output CSV.
    
    Args:
        input_path: Path to input CSV with columns: Issue, Subject, Company
        output_path: Path to write output CSV
        limit: Optional limit on number of tickets to process
    """
    # Read input
    df = pd.read_csv(input_path)
    
    if limit:
        df = df.head(limit)
    
    print(f"Processing {len(df)} tickets...")
    
    # Process each ticket
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing tickets"):
        try:
            result = await process_ticket(row.to_dict())
            results.append(result.to_csv_row())
        except Exception as e:
            print(f"\nError processing ticket {idx}: {e}")
            # Fallback: escalate on error
            results.append({
                "status": "escalated",
                "product_area": "unknown",
                "response": "Error processing ticket. Escalated to human agent.",
                "justification": f"Pipeline error: {str(e)}",
                "request_type": "invalid",
            })
    
    # Write output
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)
    
    print(f"\nComplete! Output written to {output_path}")
    print("Summary:")
    print(f"  Total: {len(results)}")
    print(f"  Replied: {sum(1 for r in results if r['status'] == 'replied')}")
    print(f"  Escalated: {sum(1 for r in results if r['status'] == 'escalated')}")
