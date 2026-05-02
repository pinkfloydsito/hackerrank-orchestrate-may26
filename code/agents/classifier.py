"""Classifier agent for request_type and product_area classification."""

import json
from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, PRODUCT_AREAS_PATH
from models import Ticket, ClassificationResult


def load_product_areas() -> List[str]:
    """Load allowed product areas from corpus taxonomy."""
    try:
        with open(PRODUCT_AREAS_PATH, "r") as f:
            areas = json.load(f)
        # Filter to reasonable length and deduplicate
        seen = set()
        filtered = []
        for area in areas:
            # Clean up area names
            clean = area.replace("/", " ").replace("-", " ").strip()
            if clean and clean not in seen and len(clean) < 100:
                seen.add(clean)
                filtered.append(clean)
        return filtered[:100]  # Limit to top 100
    except Exception as e:
        print(f"Warning: Could not load product areas: {e}")
        return [
            "account management", "billing", "api", "security",
            "interviews", "screen", "general support", "privacy",
            "travel support", "payments", "integrations"
        ]


def load_few_shot_examples() -> str:
    """Load few-shot examples from sample tickets."""
    examples = """
Example 1:
Ticket: "I notice that people I assigned the test in October of 2025 have not received new tests. How long do the tests stay active in the system."
Company: HackerRank
Response: {"request_type": "product_issue", "product_area": "screen", "confidence": 0.9}

Example 2:
Ticket: "site is down & none of the pages are accessible"
Company: None
Response: {"request_type": "bug", "product_area": "general support", "confidence": 0.95}

Example 3:
Ticket: "What is the name of the actor in Iron Man?"
Company: None
Response: {"request_type": "invalid", "product_area": "conversation management", "confidence": 0.95}

Example 4:
Ticket: "I bought Visa Traveller's Cheques from Citicorp and they were stolen in Lisbon last night. What do I do?"
Company: Visa
Response: {"request_type": "product_issue", "product_area": "travel support", "confidence": 0.9}

Example 5:
Ticket: "Where can I report a lost or stolen Visa card from India?"
Company: Visa
Response: {"request_type": "product_issue", "product_area": "general support", "confidence": 0.9}
"""
    return examples


def create_classifier_agent() -> Agent:
    """Create the classification agent."""
    provider = OpenAIProvider(
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
    )
    
    model = OpenAIModel(
        DEEPSEEK_MODEL,
        provider=provider,
    )
    
    product_areas = load_product_areas()
    few_shot = load_few_shot_examples()
    
    system_prompt = f"""You are a support ticket classifier. Your job is to classify support tickets into request_type and product_area.

Allowed request_types: product_issue, feature_request, bug, invalid

Allowed product_areas (from corpus): {', '.join(product_areas[:50])}

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
    return result.data
