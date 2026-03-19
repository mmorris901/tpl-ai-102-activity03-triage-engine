"""
JSON Schema Definitions for Activity 3 - 311 Triage Engine

This module defines the JSON schemas used to validate model outputs
for Memphis 311 request classification and routing.
"""

import jsonschema

# ---------------------------------------------------------------------------
# Valid categories
# ---------------------------------------------------------------------------
VALID_CATEGORIES = (
    "Pothole",
    "Noise Complaint",
    "Trash/Litter",
    "Street Light",
    "Water/Sewer",
    "Other",
)

VALID_PRIORITIES = {"low", "standard", "high", "critical"}


# ---------------------------------------------------------------------------
# TODO: Step 2 - Define the CLASSIFICATION_SCHEMA
# ---------------------------------------------------------------------------
# Define a JSON Schema (as a Python dict) that validates the model's
# classification output. The schema must require these fields:
#
#   - category: string, one of the VALID_CATEGORIES
#   - confidence: number between 0.0 and 1.0
#   - reasoning: string, at least 5 characters
#
# Use the jsonschema library format. Example structure:
#
# CLASSIFICATION_SCHEMA = {
#     "type": "object",
#     "properties": {
#         "category": {
#             "type": "string",
#             "enum": [...]
#         },
#         ...
#     },
#     "required": [...],
#     "additionalProperties": False
# }

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": list(VALID_CATEGORIES),
            "description": "The 311 request category"
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence level (0.0 to 1.0)"
        },
        "reasoning": {
            "type": "string",
            "minLength": 5,
            "description": "Brief explanation of the classification"
        }
    },
    "required": ["category", "confidence", "reasoning"],
    "additionalProperties": False
}


# ---------------------------------------------------------------------------
# TODO: Step 2 - Define the ROUTING_SCHEMA
# ---------------------------------------------------------------------------
# Define a JSON Schema that validates the complete routing decision.
# The schema must require these fields:
#
#   - category: string (same enum as classification)
#   - confidence: number between 0.0 and 1.0
#   - reasoning: string
#   - department: string (the routed department name)
#   - sla_hours: integer, minimum 1
#   - priority: string, one of "low", "standard", "high", "critical"
#
# ROUTING_SCHEMA = { ... }

ROUTING_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": list(VALID_CATEGORIES),
            "description": "The 311 request category"
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence level"
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation of the classification and routing"
        },
        "department": {
            "type": "string",
            "description": "Target Memphis city department"
        },
        "sla_hours": {
            "type": "integer",
            "minimum": 1,
            "description": "Service Level Agreement hours"
        },
        "priority": {
            "type": "string",
            "enum": list(VALID_PRIORITIES),
            "description": "Priority level"
        }
    },
    "required": ["category", "confidence", "reasoning", "department", "sla_hours", "priority"],
    "additionalProperties": False
}


# ---------------------------------------------------------------------------
# TODO: Step 2 - Implement validate_against_schema
# ---------------------------------------------------------------------------
def validate_against_schema(data: dict, schema: dict) -> dict:
    """Validate a data dict against a JSON schema.

    Args:
        data: The dictionary to validate.
        schema: The JSON schema to validate against.

    Returns:
        dict with keys:
          - valid: bool
          - errors: list of error message strings (empty if valid)
    """
    # Step 2.3 - Implement validation using jsonschema.validate()
    try:
        jsonschema.validate(data, schema)
        return {"valid": True, "errors": []}
    except jsonschema.ValidationError as e:
        return {"valid": False, "errors": [e.message]}
    except jsonschema.SchemaError as e:
        return {"valid": False, "errors": [f"Schema error: {e.message}"]}
    except Exception as e:
        return {"valid": False, "errors": [f"Validation error: {str(e)}"]}

