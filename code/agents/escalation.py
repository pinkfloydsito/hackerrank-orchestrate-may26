"""Escalation agent for checking sensitive content and escalation triggers."""

import re
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    SIMILARITY_THRESHOLD, CLASSIFICATION_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_SIMILARITY, ESCALATION_TEMPLATES
)
from models import Ticket, ClassificationResult, RetrievalResult, EscalationCheck


# Regex patterns for PII and sensitive content
SENSITIVE_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "password": r"\b(password|passwd|pwd)\s*[:=]\s*\S+",
    "api_key": r"\b(api[_-]?key|token|secret)\s*[:=]\s*[a-zA-Z0-9]{10,}",
}

SENSITIVE_KEYWORDS = [
    "breach", "account takeover", "hacked", "stolen identity",
    "identity theft", "security vulnerability", "bug bounty", "legal action",
    "lawyer", "attorney", "lawsuit", "gdpr violation", "ccpa violation", "regulatory violation",
    "compliance violation", "data leak", "unauthorized access", "i committed fraud",
    "security incident", "data breach",
]

EXPLICIT_ESCALATION_KEYWORDS = [
    "escalate", "human agent", "speak to someone", "talk to a person",
    "manager", "supervisor", "complaint",
]


def check_regex_patterns(text: str) -> tuple:
    """Check text for sensitive patterns via regex."""
    text_lower = text.lower()
    
    # Check PII patterns
    for pattern_name, pattern in SENSITIVE_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return True, "sensitive_data", f"Detected {pattern_name} in ticket"
    
    # Check sensitive keywords
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in text_lower:
            return True, "high_risk", f"Detected sensitive keyword: {keyword}"
    
    # Check explicit escalation
    for keyword in EXPLICIT_ESCALATION_KEYWORDS:
        if keyword in text_lower:
            return True, "explicit_request", f"User explicitly requested escalation: {keyword}"
    
    return False, "", ""


def create_escalation_agent() -> Agent:
    """Create escalation check agent."""
    provider = OpenAIProvider(
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
    )
    
    model = OpenAIChatModel(
        DEEPSEEK_MODEL,
        provider=provider,
    )
    
    system_prompt = """You are an escalation checker for support tickets. Your job is to determine if a ticket requires escalation to a human agent.

Escalate if ANY of these apply:
1. Sensitive personal information exposed (SSN, credit card numbers, passwords, API keys)
2. Security incidents reported by user (breach, account takeover, identity theft, unauthorized access)
3. Legal or compliance issues (lawsuit, regulatory violation, GDPR/CCPA complaint)
4. Account takeover or security vulnerability reports
5. The ticket requests illegal actions or is dangerous/harmful
6. The ticket explicitly asks for a human agent or manager
7. The classification confidence is very low AND retrieval similarity is poor
8. No relevant documentation could be found for a legitimate question

Do NOT escalate:
- Simple how-to questions about products/services
- Feature requests
- Bug reports
- Account settings or password reset questions
- Billing or refund questions
- Lost card reports (standard support request)
- Questions about fees, rates, or policies
- Reporting fraud (standard support request, not admission of fraud)
- Questions about integrations or API usage

Always respond with valid JSON: {"should_escalate": true/false, "reason": "...", "details": "..."}"""

    return Agent(
        model=model,
        system_prompt=system_prompt,
        output_type=EscalationCheck,
    )


async def check_escalation(
    ticket: Ticket,
    classification: ClassificationResult,
    retrieval: RetrievalResult,
) -> EscalationCheck:
    """Check if ticket should be escalated."""
    # First, regex check
    text = f"{ticket.subject} {ticket.issue}"
    should_escalate_regex, reason_regex, details_regex = check_regex_patterns(text)
    
    if should_escalate_regex:
        return EscalationCheck(
            should_escalate=True,
            reason=reason_regex,
            details=details_regex,
        )
    
    # Check retrieval confidence
    if retrieval.max_similarity < SIMILARITY_THRESHOLD:
        return EscalationCheck(
            should_escalate=True,
            reason="low_confidence",
            details=f"Max retrieval similarity {retrieval.max_similarity:.3f} below threshold {SIMILARITY_THRESHOLD}",
        )
    
    # Check classification confidence - BUT allow override if retrieval is strong
    if classification.confidence < CLASSIFICATION_CONFIDENCE_THRESHOLD:
        # If we have high retrieval similarity and it's a valid request, don't escalate
        if retrieval.max_similarity >= HIGH_CONFIDENCE_SIMILARITY and classification.request_type != "invalid":
            # Continue to LLM check instead of auto-escalating
            pass
        else:
            return EscalationCheck(
                should_escalate=True,
                reason="low_confidence",
                details=f"Classification confidence {classification.confidence:.3f} below threshold {CLASSIFICATION_CONFIDENCE_THRESHOLD}",
            )
    
    # LLM check for edge cases
    agent = create_escalation_agent()
    
    prompt = f"""Check if this ticket requires escalation:

Issue: {ticket.issue}
Subject: {ticket.subject}
Company: {ticket.company}
Classification: {classification.request_type} in {classification.product_area} (confidence: {classification.confidence:.2f})
Retrieval: {retrieval.max_similarity:.3f} max similarity

Should this be escalated?"""
    
    result = await agent.run(prompt)
    return result.output


def get_escalation_response(escalation: EscalationCheck) -> str:
    """Get template escalation response."""
    template = ESCALATION_TEMPLATES.get(
        escalation.reason, 
        ESCALATION_TEMPLATES["unsupported"]
    )
    
    if escalation.details:
        return f"{template} Reason: {escalation.details}"
    
    return template
