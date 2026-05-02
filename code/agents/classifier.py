"""Classifier agent for request_type and product_area classification."""

import json
from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    PRODUCT_AREAS_PATH,
)
from models import Ticket, ClassificationResult


def load_product_areas() -> List[str]:
    """Load allowed product areas from corpus taxonomy.

    Returns clean product area names without ecosystem prefix.
    """
    try:
        with open(PRODUCT_AREAS_PATH, "r") as f:
            areas = json.load(f)

        # Extract clean product area names (strip ecosystem prefix)
        seen = set()
        filtered = []
        for area in areas:
            # remove ecosystem prefix
            if ":" in area:
                clean = area.split(":", 1)[1].strip()
            else:
                clean = area.strip()

            if clean and clean not in seen and len(clean) < 100:
                seen.add(clean)
                filtered.append(clean)

        return sorted(filtered)
    except Exception as e:
        print(f"Warning: Could not load product areas: {e}")
        return [
            "Account Settings",
            "Screen",
            "Interviews",
            "Library",
            "Settings",
            "Integrations",
            "General Help",
            "Claude Code",
            "Claude Desktop",
            "Support",
        ]


def load_few_shot_examples() -> str:
    """Load few-shot examples from sample tickets."""
    examples = """
Example 1:
Ticket: "I notice that people I assigned the test in October of 2025 have not received new tests. How long do the tests stay active in the system."
Company: HackerRank
Response: {"request_type": "product_issue", "product_area": "Screen", "confidence": 0.9}

Example 2:
Ticket: "site is down & none of the pages are accessible"
Company: None
Response: {"request_type": "bug", "product_area": "General Help", "confidence": 0.95}

Example 3:
Ticket: "What is the name of the actor in Iron Man?"
Company: None
Response: {"request_type": "invalid", "product_area": "General Help", "confidence": 0.95}

Example 4:
Ticket: "I bought Visa Traveller's Cheques from Citicorp and they were stolen in Lisbon last night. What do I do?"
Company: Visa
Response: {"request_type": "product_issue", "product_area": "Support", "confidence": 0.9}

Example 5:
Ticket: "Where can I report a lost or stolen Visa card from India?"
Company: Visa
Response: {"request_type": "product_issue", "product_area": "Support", "confidence": 0.9}

Example 6:
Ticket: "Can I create custom coding questions for my team?"
Company: HackerRank
Response: {"request_type": "feature_request", "product_area": "Library", "confidence": 0.85}

Example 7:
Ticket: "How do I add Python questions to my test library?"
Company: HackerRank
Response: {"request_type": "product_issue", "product_area": "Library", "confidence": 0.9}

Example 8:
Ticket: "The video interview keeps disconnecting during sessions"
Company: HackerRank
Response: {"request_type": "bug", "product_area": "Interviews", "confidence": 0.95}

Example 9:
Ticket: "How do I share interview feedback with my team?"
Company: HackerRank
Response: {"request_type": "product_issue", "product_area": "Interviews", "confidence": 0.85}

Example 10:
Ticket: "How do I install Claude Code on my Mac?"
Company: None
Response: {"request_type": "product_issue", "product_area": "Claude Code", "confidence": 0.95}

Example 11:
Ticket: "Claude Code is not recognizing my API key"
Company: None
Response: {"request_type": "bug", "product_area": "Claude Code", "confidence": 0.9}

Example 12:
Ticket: "Getting rate limit errors when calling the API"
Company: None
Response: {"request_type": "bug", "product_area": "Claude API and Console", "confidence": 0.9}

Example 13:
Ticket: "What models are available through the Claude API?"
Company: None
Response: {"request_type": "product_issue", "product_area": "Claude API and Console", "confidence": 0.95}

Example 14:
Ticket: "The desktop app crashes when I open large files"
Company: None
Response: {"request_type": "bug", "product_area": "Claude Desktop", "confidence": 0.9}

Example 15:
Ticket: "How do I enable dark mode in Claude?"
Company: None
Response: {"request_type": "feature_request", "product_area": "Claude", "confidence": 0.8}

Example 16:
Ticket: "I need to update my company billing information"
Company: HackerRank
Response: {"request_type": "product_issue", "product_area": "Subscriptions, Payments, and Billing", "confidence": 0.9}

Example 17:
Ticket: "Greenhouse integration is not syncing candidate data"
Company: HackerRank
Response: {"request_type": "bug", "product_area": "Integrations", "confidence": 0.95}

Example 18:
Ticket: "How do I reset my admin password?"
Company: HackerRank
Response: {"request_type": "product_issue", "product_area": "Account Settings", "confidence": 0.95}

Example 19:
Ticket: "What are the foreign transaction fees for Visa?"
Company: Visa
Response: {"request_type": "product_issue", "product_area": "Support", "confidence": 0.85}

Example 20:
Ticket: "My Visa card was declined at a merchant even though I have funds"
Company: Visa
Response: {"request_type": "product_issue", "product_area": "Support", "confidence": 0.9}

Example 21:
Ticket: "URGENT: Site is completely down!!!"
Company: None
Response: {"request_type": "bug", "product_area": "General Help", "confidence": 0.9}

Example 22:
Ticket: "I want a refund for my subscription"
Company: HackerRank
Response: {"request_type": "product_issue", "product_area": "Subscriptions, Payments, and Billing", "confidence": 0.9}
"""
    return examples


def create_classifier_agent() -> Agent:
    """Create the classification agent."""
    provider = OpenAIProvider(
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
    )

    model = OpenAIChatModel(
        DEEPSEEK_MODEL,
        provider=provider,
    )

    product_areas = load_product_areas()
    few_shot = load_few_shot_examples()

    system_prompt = f"""You are a support ticket classifier. Your job is to classify support tickets into request_type and product_area.

Allowed request_types: product_issue, feature_request, bug, invalid

Allowed product_areas (from corpus): {', '.join(product_areas)}

{few_shot}

Rules:
- product_issue: user has a problem or question about existing functionality
- feature_request: user wants new functionality or improvements
- bug: something is broken or not working as expected
- invalid: ticket is spam, off-topic, or not a real support request
- If company is specified, product_area should relate to that company's domain
- If company is None, infer from content
- Return confidence score 0.0-1.0

Always respond with valid JSON matching: {{"request_type": "...", "product_area": "...", "confidence": 0.0}}"""

    return Agent(
        model=model,
        system_prompt=system_prompt,
        output_type=ClassificationResult,
    )


async def classify_ticket(ticket: Ticket) -> ClassificationResult:
    """Classify a ticket using the classifier agent."""
    agent = create_classifier_agent()

    prompt = f"""Classify the following support ticket:

Issue: {ticket.issue}
Subject: {ticket.subject}
Company: {ticket.company}

Provide classification:"""

    result = await agent.run(prompt)
    return result.output
