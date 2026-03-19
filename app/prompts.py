"""
Prompt Template Library for Activity 3 - 311 Triage Engine
AI-102: Implement generative AI solutions

This module contains reusable prompt templates for Memphis 311 request
classification. Students add their own templates alongside the example.
"""


# ---------------------------------------------------------------------------
# Example template (read-only reference)
# ---------------------------------------------------------------------------
def example_template(request_text: str) -> str:
    """Example: simple classification prompt (for reference only)."""
    return (
        f"Classify this Memphis 311 service request into one category.\n\n"
        f"Request: {request_text}\n\n"
        f"Respond with JSON: "
        f'{{"category": "<category>", "confidence": <0.0-1.0>, '
        f'"reasoning": "<one sentence>"}}'
    )


# ---------------------------------------------------------------------------
# TODO: Step 1 - Create your prompt templates
# ---------------------------------------------------------------------------

def classify_request(request_text: str) -> str:
    """Template 1: Basic classification prompt.

    Takes a request_text string and returns a formatted user message
    asking the model to classify it into one of the six 311 categories.

    Must include:
    - The request text
    - Output format instructions (JSON with category, confidence, reasoning)
    """
    return (
        f"Classify this Memphis 311 service request:\n\n"
        f"{request_text}\n\n"
        f"Respond with JSON containing: category, confidence (0.0-1.0), and reasoning."
    )


def classify_with_context(request_text: str, neighborhood: str) -> str:
    """Template 2: Classification with geographic context.

    Takes request_text and a neighborhood name to provide additional
    context that may help the model make better decisions.

    Must include:
    - The request text
    - The neighborhood name
    - Output format instructions
    """
    return (
        f"Classify this Memphis 311 service request from {neighborhood}:\n\n"
        f"{request_text}\n\n"
        f"Respond with JSON containing: category, confidence (0.0-1.0), and reasoning."
    )


def batch_classify(requests: list[str]) -> str:
    """Template 3: Batch classification prompt.

    Takes a list of request strings and asks the model to classify
    all of them in a single API call.

    Must include:
    - Numbered list of requests
    - Output format: JSON array of classification objects
    """
    requests_text = "\n".join([f"{i+1}. {req}" for i, req in enumerate(requests)])
    return (
        f"Classify these Memphis 311 service requests:\n\n"
        f"{requests_text}\n\n"
        f"Respond with a JSON array where each element has: category, confidence (0.0-1.0), and reasoning."
    )
