"""
Activity 3 - 311 Triage Engine: Metrics Calculator
AI-102: Evaluate models and flows (2.1), Optimize and operationalize (2.3)

Your task: Implement accuracy, per-category precision, and per-category
recall from a list of evaluation results.

Each result dict looks like:
    {"input": "...", "expected": "Pothole", "predicted": "Pothole", "correct": True}
"""
from app.schemas import VALID_CATEGORIES  # Single source of truth


# ---------------------------------------------------------------------------
# TODO: Step 5 - Implement overall accuracy
# ---------------------------------------------------------------------------
def accuracy(results: list[dict]) -> float:
    """Calculate overall classification accuracy.

    Args:
        results: List of eval result dicts, each with a 'correct' bool field.

    Returns:
        Float between 0.0 and 1.0 representing the fraction of correct
        predictions. Returns 0.0 if results is empty.

    Example:
        >>> accuracy([{"correct": True}, {"correct": False}, {"correct": True}])
        0.6666666666666666
    """
    # Step 5 - Implement this function
    if not results:
        return 0.0
    correct_count = sum(1 for r in results if r.get("correct", False))
    return correct_count / len(results)


# ---------------------------------------------------------------------------
# TODO: Step 5 - Implement per-category precision
# ---------------------------------------------------------------------------
def precision_per_category(results: list[dict]) -> dict[str, float]:
    """Calculate precision for each category.

    Precision = true positives / (true positives + false positives)
    i.e., of all times we *predicted* category X, how often was it correct?

    Args:
        results: List of eval result dicts, each with 'expected' and
                 'predicted' string fields.

    Returns:
        Dict mapping category name to precision float (0.0-1.0).
        Categories with zero predictions get precision 0.0.

    Example:
        >>> results = [
        ...     {"expected": "Pothole", "predicted": "Pothole"},
        ...     {"expected": "Noise Complaint", "predicted": "Pothole"},
        ... ]
        >>> precision_per_category(results)
        {'Pothole': 0.5, 'Noise Complaint': 0.0, ...}
    """
    # Step 5 - Implement this function
    precision_dict = {}
    
    for category in VALID_CATEGORIES:
        # Count true positives: predicted == category AND expected == category
        tp = sum(1 for r in results 
                 if r.get("predicted") == category and r.get("expected") == category)
        # Count false positives: predicted == category AND expected != category
        fp = sum(1 for r in results 
                 if r.get("predicted") == category and r.get("expected") != category)
        
        if tp + fp > 0:
            precision_dict[category] = tp / (tp + fp)
        else:
            precision_dict[category] = 0.0
    
    return precision_dict


# ---------------------------------------------------------------------------
# TODO: Step 5 - Implement per-category recall
# ---------------------------------------------------------------------------
def recall_per_category(results: list[dict]) -> dict[str, float]:
    """Calculate recall for each category.

    Recall = true positives / (true positives + false negatives)
    i.e., of all cases that *actually were* category X, how many did we catch?

    Args:
        results: List of eval result dicts, each with 'expected' and
                 'predicted' string fields.

    Returns:
        Dict mapping category name to recall float (0.0-1.0).
        Categories with zero actual instances get recall 0.0.

    Example:
        >>> results = [
        ...     {"expected": "Pothole", "predicted": "Pothole"},
        ...     {"expected": "Pothole", "predicted": "Other"},
        ... ]
        >>> recall_per_category(results)
        {'Pothole': 0.5, ...}
    """
    # Step 5 - Implement this function
    recall_dict = {}
    
    for category in VALID_CATEGORIES:
        # Count true positives: expected == category AND predicted == category
        tp = sum(1 for r in results 
                 if r.get("expected") == category and r.get("predicted") == category)
        # Count false negatives: expected == category AND predicted != category
        fn = sum(1 for r in results 
                 if r.get("expected") == category and r.get("predicted") != category)
        
        if tp + fn > 0:
            recall_dict[category] = tp / (tp + fn)
        else:
            recall_dict[category] = 0.0
    
    return recall_dict


# ---------------------------------------------------------------------------
# Provided: Summary helper (no TODO)
# ---------------------------------------------------------------------------
def summarize_metrics(results: list[dict]) -> dict:
    """Build a complete metrics summary from evaluation results.

    Calls accuracy(), precision_per_category(), and recall_per_category()
    and combines them into a single report dict.

    Args:
        results: List of eval result dicts.

    Returns:
        Dict with keys: overall_accuracy, per_category_precision,
        per_category_recall, total_evaluated, total_correct.
    """
    return {
        "overall_accuracy": accuracy(results),
        "per_category_precision": precision_per_category(results),
        "per_category_recall": recall_per_category(results),
        "total_evaluated": len(results),
        "total_correct": sum(1 for r in results if r.get("correct", False)),
    }
