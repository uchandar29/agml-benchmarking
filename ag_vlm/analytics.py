"""Accuracy, confusion matrix and per-class metrics from the predict-pass."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional


def ComputeConfusion(records: List[Dict], classList: List[str]) -> Dict[str, Dict[str, int]]:
    """Build a true-class -> predicted-class count matrix."""
    matrix: Dict[str, Dict[str, int]] = {
        trueClass: {predClass: 0 for predClass in classList} for trueClass in classList
    }
    for record in records:
        trueLabel: str = record["true_label"]
        predicted: Optional[str] = record.get("predicted_class")
        if predicted is None or trueLabel not in matrix or predicted not in matrix[trueLabel]:
            continue
        matrix[trueLabel][predicted] += 1
    return matrix


def ComputePerClassMetrics(matrix: Dict[str, Dict[str, int]], classList: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall and support for each class from the confusion matrix."""
    metrics: Dict[str, Dict[str, float]] = {}
    for className in classList:
        truePositive: int = matrix[className][className]
        support: int = sum(matrix[className].values())
        predictedTotal: int = sum(matrix[other][className] for other in classList)
        precision: float = truePositive / predictedTotal if predictedTotal else 0.0
        recall: float = truePositive / support if support else 0.0
        metrics[className] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "support": float(support),
        }
    return metrics


def ComputeConfidenceCalibration(records: List[Dict]) -> Dict[str, Optional[float]]:
    """Compute mean confidence for correct vs incorrect predictions."""
    correctConfidences: List[float] = []
    wrongConfidences: List[float] = []
    for record in records:
        confidence: Optional[float] = record.get("confidence")
        if confidence is None:
            continue
        if record.get("correct"):
            correctConfidences.append(confidence)
        else:
            wrongConfidences.append(confidence)
    meanCorrect: Optional[float] = round(sum(correctConfidences) / len(correctConfidences), 4) if correctConfidences else None
    meanWrong: Optional[float] = round(sum(wrongConfidences) / len(wrongConfidences), 4) if wrongConfidences else None
    return {"mean_confidence_correct": meanCorrect, "mean_confidence_incorrect": meanWrong}


def BuildAnalytics(records: List[Dict], classList: List[str], malformedCount: int) -> Dict:
    """Assemble the full analytics summary for one dataset's predict-pass."""
    nSamples: int = len(records)
    nCorrect: int = sum(1 for record in records if record.get("correct"))
    accuracy: float = nCorrect / nSamples if nSamples else 0.0
    matrix: Dict[str, Dict[str, int]] = ComputeConfusion(records, classList)
    return {
        "n_samples": nSamples,
        "n_correct": nCorrect,
        "accuracy": round(accuracy, 4),
        "error_rate": round(1.0 - accuracy, 4),
        "per_class_metrics": ComputePerClassMetrics(matrix, classList),
        "confusion_matrix": matrix,
        "calibration": ComputeConfidenceCalibration(records),
        "malformed_outputs": malformedCount,
    }


def NearestConfusable(matrix: Dict[str, Dict[str, int]], trueClass: str) -> Optional[str]:
    """Find the wrong class most often predicted for a given true class (for contrastive mode)."""
    if trueClass not in matrix:
        return None
    bestClass: Optional[str] = None
    bestCount: int = 0
    for predClass, count in matrix[trueClass].items():
        if predClass == trueClass:
            continue
        if count > bestCount:
            bestCount = count
            bestClass = predClass
    return bestClass


def RenderAnalyticsMarkdown(analytics: Dict, datasetName: str, classList: List[str]) -> str:
    """Render a short human-readable markdown report of the analytics."""
    lines: List[str] = []
    lines.append(f"# Analytics — {datasetName}\n")
    lines.append(f"- **Samples:** {analytics['n_samples']}")
    lines.append(f"- **Correct:** {analytics['n_correct']}")
    lines.append(f"- **Accuracy:** {analytics['accuracy']}")
    lines.append(f"- **Error rate:** {analytics['error_rate']}")
    lines.append(f"- **Malformed outputs:** {analytics['malformed_outputs']}")
    calibration: Dict = analytics["calibration"]
    lines.append(f"- **Mean confidence (correct):** {calibration['mean_confidence_correct']}")
    lines.append(f"- **Mean confidence (incorrect):** {calibration['mean_confidence_incorrect']}\n")

    lines.append("## Per-class metrics\n")
    lines.append("| Class | Precision | Recall | Support |")
    lines.append("|---|---|---|---|")
    for className in classList:
        metric: Dict = analytics["per_class_metrics"][className]
        lines.append(f"| {className} | {metric['precision']} | {metric['recall']} | {int(metric['support'])} |")
    return "\n".join(lines) + "\n"
