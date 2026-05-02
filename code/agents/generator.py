"""Generator agent for creating responses and justifications."""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from models import Ticket, ClassificationResult, RetrievalResult, EscalationCheck, AgentOutput
from agents.escalation import get_escalation_response


def create_generator_agent() -> Agent:
    """Create the response generation agent."""
    provider = OpenAIProvider(
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
    )
    
    model = OpenAIModel(
        DEEPSEEK_MODEL,
        provider=provider,
    )
    
    system_prompt = """You are a support agent assistant. Your job is to generate helpful, accurate responses to support tickets using ONLY the provided context.

Rules:
- Answer ONLY using the provided help-center articles
- Do NOT invent policies, procedures, or facts not in the context
- Be concise and professional
- If the context doesn't fully answer the question, acknowledge limitations
- Always cite which article(s) you used in your justification
- Keep responses under 200 words when possible

For escalated tickets, provide a brief explanation of why escalation is needed.

Always respond with valid JSON: {"status": "replied|escalated", "product_area": "...", "response": "...", "justification": "...", "request_type": "..."}"""

    return Agent(
        model=model,
        system_prompt=system_prompt,
        result_type=AgentOutput,
    )


def format_context(retrieval: RetrievalResult) -> str:
    """Format retrieved chunks as context."""
    context_parts = []
    for i, chunk in enumerate(retrieval.chunks, 1):
        context_parts.append(
            f"\n--- Context {i} ---\n"
            f"Source: {chunk.article_title}\n"
            f"Section: {chunk.section_heading or 'General'}\n"
            f"Ecosystem: {chunk.product_ecosystem}\n"
            f"Similarity: {chunk.similarity:.3f}\n"
            f"Content: {chunk.text[:500]}..."
        )
    return "\n".join(context_parts)


async def generate_response(
    ticket: Ticket,
    classification: ClassificationResult,
    retrieval: RetrievalResult,
) -> AgentOutput:
    """Generate a response for a non-escalated ticket."""
    agent = create_generator_agent()
    
    context = format_context(retrieval)
    
    prompt = f"""Generate a response to this support ticket using ONLY the provided context.

Ticket:
- Issue: {ticket.issue}
- Subject: {ticket.subject}
- Company: {ticket.company}
- Classification: {classification.request_type} in {classification.product_area}

{context}

Generate response and justification:"""
    
    result = await agent.run(prompt)
    output = result.data
    
    # Ensure status is replied
    output.status = "replied"
    output.request_type = classification.request_type
    output.product_area = classification.product_area
    
    return output


async def generate_escalation_response(
    ticket: Ticket,
    classification: ClassificationResult,
    escalation: EscalationCheck,
) -> AgentOutput:
    """Generate a response for an escalated ticket."""
    agent = create_generator_agent()
    
    template_response = get_escalation_response(escalation)
    
    prompt = f"""Generate an escalation response for this support ticket.

Ticket:
- Issue: {ticket.issue}
- Subject: {ticket.subject}
- Company: {ticket.company}
- Classification: {classification.request_type} in {classification.product_area}
- Escalation reason: {escalation.reason}
- Details: {escalation.details}

Template response: {template_response}

Generate a professional escalation response and justification explaining why this was escalated:"""
    
    result = await agent.run(prompt)
    output = result.data
    
    # Ensure status is escalated
    output.status = "escalated"
    output.request_type = classification.request_type
    output.product_area = classification.product_area
    
    return output
