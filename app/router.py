"""
Department Router for Activity 3 - 311 Triage Engine

Routes classified Memphis 311 requests to the correct city department
based on category and routing rules.
"""

import json
import os


def load_routing_rules(path: str | None = None) -> dict:
    """Load department routing rules from JSON file.

    Args:
        path: Path to routing_rules.json. Defaults to data/routing_rules.json
              relative to the project root.

    Returns:
        dict mapping category names to routing info (department, sla_hours, priority).
    """
    # Step 2.1 - Implement this function
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "routing_rules.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def route_request(classification: dict, rules: dict | None = None) -> dict:
    """Route a classified request to the correct Memphis department.

    Takes a classification dict (from the model) and looks up the routing
    info from the rules. Merges classification + routing into a single dict.

    Args:
        classification: dict with keys: category, confidence, reasoning
        rules: Optional routing rules dict. If None, loads from file.

    Returns:
        dict with keys: category, confidence, reasoning, department,
        sla_hours, priority. Falls back to General Services if category
        not found in rules.
    """
    # Step 2.2 - Implement routing logic
    if rules is None:
        rules = load_routing_rules()
    
    category = classification.get("category")
    
    # Look up routing info
    if category in rules:
        routing_info = rules[category]
    else:
        # Fallback to "Other"
        routing_info = rules.get("Other", {
            "department": "General Services",
            "sla_hours": 120,
            "priority": "low"
        })
    
    # Merge classification with routing info
    result = classification.copy()
    result.update({
        "department": routing_info.get("department", "General Services"),
        "sla_hours": routing_info.get("sla_hours", 120),
        "priority": routing_info.get("priority", "low")
    })
    
    return result


def escalate_priority(routing: dict, reason: str) -> dict:
    """Escalate a routed request's priority by one level.

    Priority levels (lowest to highest): low -> standard -> high -> critical

    Args:
        routing: A routing dict (output of route_request).
        reason: A string explaining why the priority was escalated.

    Returns:
        Updated routing dict with escalated priority and adjusted SLA.
        Adds an 'escalation_reason' field.
    """
    # TODO: Step 2 (Stretch) - Implement escalation logic
    #
    # Priority escalation ladder:
    #   low -> standard (SLA = SLA * 0.75)
    #   standard -> high (SLA = SLA * 0.5)
    #   high -> critical (SLA = SLA * 0.25)
    #   critical -> critical (no change, already max)
    #
    # 1. Determine the current priority
    # 2. Escalate to the next level
    # 3. Adjust SLA hours accordingly
    # 4. Add "escalation_reason" field
    # 5. Return updated dict
    raise NotImplementedError("Implement escalate_priority (Stretch)")
