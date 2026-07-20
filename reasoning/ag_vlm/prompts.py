"""Prompt builders and guided-decoding JSON schemas for both passes."""

from __future__ import annotations

from typing import Dict, List, Optional


def BuildPredictPrompt(classList: List[str]) -> str:
    """Build the Stage A text prompt asking the model to predict the class."""
    
    classText: str = ", ".join(classList)
    return (
        "You are an expert agronomist. Look only at the image. From the following "
        "candidate classes, choose the single class that best matches what you see. "
        "Do not guess based on anything other than visual evidence.\n"
        f"Candidate classes: {classText}.\n"
        "Return JSON with: predicted_class (must be exactly one of the candidates), "
        "confidence (0-1), visual_evidence (short list of the features you used)."
    )


def BuildPredictSchema(classList: List[str]) -> Dict:
    """Build the JSON schema constraining the Stage A output."""
    
    return {
        "type": "object",
        "properties": {
            "predicted_class": {"type": "string", "enum": classList},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "visual_evidence": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["predicted_class", "confidence", "visual_evidence"],
        "additionalProperties": False,
    }


def BuildRationalizePrompt(
    trueClass: str,
    contrastive: bool = False,
    confusableClass: Optional[str] = None,
) -> str:
    """Build the Stage B text prompt asking the model to justify the given label."""
    
    basePrompt: str = (
        "You are an expert agronomist. This image is labeled "
        f"{trueClass}. Examine the image and explain, using only visible evidence, "
        "the features that are consistent with this label. Be specific and concrete.\n"
        "Return JSON with the fields below. If a field is not visible, set it to "
        '"not_visible".'
    )

    if contrastive and confusableClass is not None:
        basePrompt += (
            f"\nAlso explain why this is {trueClass} and not {confusableClass}."
        )
    return basePrompt


def BuildRationalizeSchema(evidenceFields: List[str]) -> Dict:
    """Build the JSON schema for Stage B, using configurable evidence fields."""
    
    evidenceProperties: Dict = {field: {"type": "string"} for field in evidenceFields}
    return {
        "type": "object",
        "properties": {
            "class": {"type": "string"},
            "summary": {"type": "string"},
            "visual_evidence": {
                "type": "object",
                "properties": evidenceProperties,
                "required": evidenceFields,
                "additionalProperties": False,
            },
            "distinguishing_features": {"type": "array", "items": {"type": "string"}},
            "uncertainty_notes": {"type": "string"},
        },
        "required": [
            "class",
            "summary",
            "visual_evidence",
            "distinguishing_features",
            "uncertainty_notes",
        ],
        "additionalProperties": False,
    }
