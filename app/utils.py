"""
Utility helpers for Activity 3 - 311 Triage Engine.

Provides input validation, data-loading functions, retry logic,
timing helpers, output formatting, and JSONL I/O used across the pipeline.
"""
import json
import os
import time
from contextlib import contextmanager


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def validate_input(text: str) -> str:
    """Validate and clean user input.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string, stripped and truncated to 1000 characters.

    Raises:
        ValueError: If input is empty or whitespace-only.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Input cannot be empty or whitespace-only")
    return cleaned[:1000]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_eval_set(path: str | None = None) -> list[dict]:
    """Load the labeled evaluation dataset.

    Args:
        path: Path to eval_set.json. Defaults to data/eval_set.json
              relative to the project root.

    Returns:
        List of dicts with 'id', 'input', 'expected_category', and
        'neighborhood' keys.

    Raises:
        FileNotFoundError: If the eval set file does not exist.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_set.json")
    with open(path) as f:
        return json.load(f)


def load_pricing(path: str | None = None) -> dict:
    """Load Azure OpenAI token pricing data.

    Args:
        path: Path to pricing.json. Defaults to data/pricing.json
              relative to the project root.

    Returns:
        Dict keyed by model name with input/output pricing per 1k tokens.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "pricing.json")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------
def retry_with_correction(
    call_fn,
    validation_fn,
    max_retries: int = 3,
    correction_prompt: str = "",
) -> dict:
    """Retry an API call when the response fails validation.

    This function implements a retry loop that:
    1. Calls call_fn() to get a model response
    2. Passes the response to validation_fn() to check validity
    3. If valid, returns the response immediately
    4. If invalid, calls call_fn(correction) with a correction prompt
    5. Repeats up to max_retries times

    Args:
        call_fn: Callable that takes an optional correction string and
                 returns a dict (the parsed model response).
                 - call_fn() for the first attempt (no correction)
                 - call_fn(correction_msg) for retries
        validation_fn: Callable that takes a dict and returns a dict
                       with keys: valid (bool), errors (list[str]).
        max_retries: Maximum number of retry attempts (default 3).
        correction_prompt: Base correction message to prepend error
                          details to on retry.

    Returns:
        dict with keys:
          - response: the validated response dict (or last attempt)
          - attempts: int, number of attempts made
          - valid: bool, whether the final response passed validation
          - errors: list of error strings from the last validation
    """
    # Step 3 - Implement the retry loop
    attempt = 0
    last_errors = []
    response = None

    while attempt < max_retries:
        attempt += 1

        # First attempt: no correction; retries: include correction
        if attempt == 1:
            response = call_fn()
        else:
            error_detail = "; ".join(last_errors)
            correction = f"{correction_prompt} Previous errors: {error_detail}"
            response = call_fn(correction)

        # Validate the response
        result = validation_fn(response)
        if result["valid"]:
            return {
                "response": response,
                "attempts": attempt,
                "valid": True,
                "errors": [],
            }

        last_errors = result["errors"]

    # All retries exhausted
    return {
        "response": response,
        "attempts": attempt,
        "valid": False,
        "errors": last_errors,
    }


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
@contextmanager
def timer():
    """Context manager that tracks elapsed wall-clock time in seconds.

    Usage:
        with timer() as t:
            do_something()
        print(f"Took {t.elapsed:.3f}s")
    """

    class _Timer:
        elapsed: float = 0.0

    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - start


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def append_jsonl(filepath: str, record: dict) -> None:
    """Append a single JSON record to a JSONL file.

    Args:
        filepath: Path to the .jsonl file (created if missing).
        record: Dict to serialize as one JSON line.
    """
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")


def write_json(filepath: str, data: dict | list) -> None:
    """Write data to a JSON file with pretty-printing.

    Args:
        filepath: Output file path.
        data: Serializable data to write.
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def format_routing_summary(routing: dict) -> str:
    """Format a routing decision as a human-readable summary.

    Args:
        routing: dict with department, category, priority, sla_hours.

    Returns:
        Formatted string summary.
    """
    return (
        f"Category: {routing.get('category', 'Unknown')}\n"
        f"Department: {routing.get('department', 'Unknown')}\n"
        f"Priority: {routing.get('priority', 'Unknown')}\n"
        f"SLA: {routing.get('sla_hours', 'N/A')} hours\n"
        f"Confidence: {routing.get('confidence', 0.0):.0%}\n"
        f"Reasoning: {routing.get('reasoning', 'N/A')}"
    )
