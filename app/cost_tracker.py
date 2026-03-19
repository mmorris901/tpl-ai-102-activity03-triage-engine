"""
Activity 3 - 311 Triage Engine: Token Usage & Cost Tracker
AI-102: Optimize and operationalize (2.3) - monitoring, tracing

Your task: Track token usage from API responses and calculate the
dollar cost of running the classifier at scale.
"""
from app.utils import load_pricing


# ---------------------------------------------------------------------------
# TODO: Step 5 - Implement token counting from a single API response
# ---------------------------------------------------------------------------
def extract_token_usage(response) -> dict:
    """Extract token counts from an Azure AI Inference API response.

    The response object has a `.usage` attribute with:
        - response.usage.prompt_tokens (int)
        - response.usage.completion_tokens (int)

    Args:
        response: The ChatCompletions response object from azure-ai-inference.

    Returns:
        Dict with keys: prompt_tokens, completion_tokens, total_tokens.

    Example:
        >>> usage = extract_token_usage(response)
        >>> usage
        {'prompt_tokens': 127, 'completion_tokens': 42, 'total_tokens': 169}
    """
    # Step 5 - Implement this function
    if not response or not response.usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    prompt_tokens = response.usage.prompt_tokens or 0
    completion_tokens = response.usage.completion_tokens or 0
    total_tokens = prompt_tokens + completion_tokens
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# TODO: Step 5 - Implement cost calculation
# ---------------------------------------------------------------------------
def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "gpt-4o",
    pricing: dict | None = None,
) -> float:
    """Calculate the dollar cost for a given number of tokens.

    Cost formula:
        cost = (prompt_tokens / 1000 * input_price_per_1k)
             + (completion_tokens / 1000 * output_price_per_1k)

    Args:
        prompt_tokens: Number of input/prompt tokens.
        completion_tokens: Number of output/completion tokens.
        model: Model name to look up in pricing data (default "gpt-4o").
        pricing: Optional pricing dict. If None, loads from data/pricing.json.

    Returns:
        Float representing the cost in US dollars.

    Raises:
        KeyError: If the model name is not found in pricing data.

    Example:
        >>> calculate_cost(1000, 500, model="gpt-4o")
        0.0075
    """
    # Step 5 - Implement this function
    if pricing is None:
        pricing = load_pricing()
    
    model_pricing = pricing[model]
    input_price = model_pricing["input_price_per_1k_tokens"]
    output_price = model_pricing["output_price_per_1k_tokens"]
    
    cost = (prompt_tokens / 1000 * input_price) + (completion_tokens / 1000 * output_price)
    return cost


# ---------------------------------------------------------------------------
# TODO: Step 5 - Implement cumulative cost tracking
# ---------------------------------------------------------------------------
class CostTracker:
    """Tracks cumulative token usage and cost across multiple API calls.

    Usage:
        tracker = CostTracker(model="gpt-4o")
        tracker.record(prompt_tokens=127, completion_tokens=42)
        tracker.record(prompt_tokens=130, completion_tokens=38)
        print(tracker.summary())
    """

    def __init__(self, model: str = "gpt-4o"):
        """Initialize the cost tracker.

        Args:
            model: The model name for pricing lookups.
        """
        self.model = model
        # Initialize tracking variables
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        self._pricing = load_pricing()

    def record(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Record token usage from one API call and return its cost.

        Args:
            prompt_tokens: Number of input tokens for this call.
            completion_tokens: Number of output tokens for this call.

        Returns:
            Float cost in dollars for this single call.
        """
        # Step 5 - Implement this method
        cost = calculate_cost(
            prompt_tokens, 
            completion_tokens, 
            model=self.model, 
            pricing=self._pricing
        )
        
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += cost
        self.call_count += 1
        
        return cost

    def summary(self) -> dict:
        """Return a summary of all tracked costs.

        Returns:
            Dict with keys: model, call_count, total_prompt_tokens,
            total_completion_tokens, total_tokens, total_cost,
            avg_prompt_tokens, avg_completion_tokens, avg_cost_per_call.
        """
        # Step 5 - Implement this method
        if self.call_count == 0:
            avg_prompt = avg_completion = avg_cost = 0.0
        else:
            avg_prompt = self.total_prompt_tokens / self.call_count
            avg_completion = self.total_completion_tokens / self.call_count
            avg_cost = self.total_cost / self.call_count
        
        return {
            "model": self.model,
            "call_count": self.call_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost": self.total_cost,
            "avg_prompt_tokens": avg_prompt,
            "avg_completion_tokens": avg_completion,
            "avg_cost_per_call": avg_cost,
        }

    def estimate_monthly_cost(self, calls_per_day: int) -> float:
        """Estimate monthly cost based on average cost per call.

        Args:
            calls_per_day: Expected number of API calls per day.

        Returns:
            Estimated monthly cost in dollars (assumes 30 days/month).
        """
        # Step 5 - Implement this method
        if self.call_count == 0:
            return 0.0
        
        avg_cost_per_call = self.total_cost / self.call_count
        monthly_cost = avg_cost_per_call * calls_per_day * 30
        return monthly_cost
