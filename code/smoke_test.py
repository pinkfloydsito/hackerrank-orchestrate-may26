"""Smoke test script for support triage agent - tests 20+ tickets."""

import sys
sys.path.insert(0, '/Users/chevass/Projects/hackerrank-orchestrate-may26/code')

import asyncio
import pandas as pd

from pipeline.processor import process_ticket


# Test cases covering different scenarios (as dicts for processor)
TEST_TICKETS = [
    # HackerRank - Screen
    {"Issue": "How do I extend the expiration date for a test invite?", "Subject": "Test expiration", "Company": "HackerRank"},
    {"Issue": "Candidates are not receiving email invitations for assessments", "Subject": "Email invites not working", "Company": "HackerRank"},
    {"Issue": "Can I create custom coding questions for my team?", "Subject": "Custom questions", "Company": "HackerRank"},
    
    # HackerRank - Interviews
    {"Issue": "The video interview keeps disconnecting during sessions", "Subject": "Interview connection issues", "Company": "HackerRank"},
    {"Issue": "How do I share interview feedback with my team?", "Subject": "Interview feedback sharing", "Company": "HackerRank"},
    
    # HackerRank - Library
    {"Issue": "I want to add Python questions to my test library", "Subject": "Add Python questions", "Company": "HackerRank"},
    {"Issue": "Can I duplicate existing questions in the library?", "Subject": "Duplicate questions", "Company": "HackerRank"},
    
    # HackerRank - Settings/Account
    {"Issue": "How do I reset my admin password?", "Subject": "Password reset", "Company": "HackerRank"},
    {"Issue": "I need to update my company billing information", "Subject": "Billing update", "Company": "HackerRank"},
    
    # HackerRank - Integrations
    {"Issue": "Greenhouse integration is not syncing candidate data", "Subject": "Greenhouse sync issue", "Company": "HackerRank"},
    
    # Claude - Claude Code
    {"Issue": "How do I install Claude Code on my Mac?", "Subject": "Claude Code installation", "Company": "None"},
    {"Issue": "Claude Code is not recognizing my API key", "Subject": "API key issue", "Company": "None"},
    {"Issue": "Can I use Claude Code with my Team plan?", "Subject": "Team plan access", "Company": "None"},
    
    # Claude - Desktop/Mobile
    {"Issue": "The desktop app crashes when I open large files", "Subject": "Desktop app crash", "Company": "None"},
    {"Issue": "How do I enable dark mode in Claude?", "Subject": "Dark mode", "Company": "None"},
    
    # Claude - API
    {"Issue": "Getting rate limit errors when calling the API", "Subject": "Rate limiting", "Company": "None"},
    {"Issue": "What models are available through the Claude API?", "Subject": "Available models", "Company": "None"},
    
    # Visa - Support
    {"Issue": "I lost my Visa card while traveling in Europe. What should I do?", "Subject": "Lost card abroad", "Company": "Visa"},
    {"Issue": "How do I report fraudulent charges on my Visa card?", "Subject": "Fraud report", "Company": "Visa"},
    {"Issue": "What are the foreign transaction fees for Visa?", "Subject": "Foreign transaction fees", "Company": "Visa"},
    {"Issue": "My Visa card was declined at a merchant even though I have funds", "Subject": "Card declined", "Company": "Visa"},
    
    # Edge cases
    {"Issue": "What is the weather like today?", "Subject": "Weather", "Company": "None"},
    {"Issue": "URGENT: Site is completely down!!!", "Subject": "Site down", "Company": "None"},
    {"Issue": "I want a refund for my subscription", "Subject": "Refund request", "Company": "HackerRank"},
]


async def run_smoke_tests():
    """Run smoke tests and save results."""
    results = []
    
    print("=" * 80)
    print("SMOKE TEST - Support Triage Agent")
    print("=" * 80)
    print(f"Testing {len(TEST_TICKETS)} tickets...\n")
    
    for i, ticket in enumerate(TEST_TICKETS, 1):
        print(f"\n--- Test {i}/{len(TEST_TICKETS)} ---")
        print(f"Issue: {ticket['Issue'][:80]}...")
        print(f"Company: {ticket['Company']}")
        
        try:
            result = await process_ticket(ticket)
            
            print(f"  Status: {result.status}")
            print(f"  Request Type: {result.request_type}")
            print(f"  Product Area: {result.product_area}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Max Similarity: {result.max_similarity:.3f}")
            print(f"  Response: {result.response[:100]}...")
            
            results.append({
                "ticket_id": f"ticket_{i}",
                "company": ticket['Company'],
                "issue": ticket['Issue'][:100],
                "status": result.status,
                "request_type": result.request_type,
                "product_area": result.product_area,
                "confidence": result.confidence,
                "max_similarity": result.max_similarity,
                "response_preview": result.response[:150],
                "justification": result.justification[:200] if result.justification else "",
                "error": ""
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "ticket_id": f"ticket_{i}",
                "company": ticket['Company'],
                "issue": ticket['Issue'][:100],
                "status": "ERROR",
                "request_type": "",
                "product_area": "",
                "confidence": 0,
                "chunks_count": 0,
                "max_similarity": 0,
                "response_preview": "",
                "justification": "",
                "error": str(e)
            })
    
    # Save results to CSV
    df = pd.DataFrame(results)
    output_path = "/Users/chevass/Projects/hackerrank-orchestrate-may26/code/smoke_test_results.csv"
    df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)
    print(f"Total tickets: {len(TEST_TICKETS)}")
    print(f"Successful: {len([r for r in results if r['status'] != 'ERROR'])}")
    print(f"Errors: {len([r for r in results if r['status'] == 'ERROR'])}")
    print("\nStatus distribution:")
    print(df['status'].value_counts())
    print("\nRequest type distribution:")
    print(df['request_type'].value_counts())
    print("\nProduct area distribution:")
    print(df['product_area'].value_counts())
    print(f"\nResults saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    df = asyncio.run(run_smoke_tests())
