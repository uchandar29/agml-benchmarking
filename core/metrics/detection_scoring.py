"""
Proposed 0 to 10 scores for object detection reports (10 = best).
The formulas are documented in SCORING_FORMULAS.md. They are proposals: weights and
cut-offs are hand-set and wait for calibration across the dataset library.

Same conventions as the classification scores: a metric is available when its key
is in report["metrics"] and not marked "skipped"; an axis needs all of its required
metrics or it is dropped; the overall score is a weighted mean over the axes present.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

AXIS_WEIGHTS = {"structural": 0.30, "difficulty": 0.25, "diversity": 0.25, "annotation": 0.20}

# Cut-offs, kept together so calibration only touches this block
IMBALANCE_RATIO_ZERO_AT = 100.0   # ratio at which the ratio term reaches 0
REDUNDANCY_ZERO_AT = 0.20
CROSS_SPLIT_MAX_PENALTY = 2.0
SMALL_SHARE_ZERO_AT = 50.0        # percent of small boxes that scores 0
HARD_SHARE_ZERO_AT = 50.0         # percent of hard boxes that scores 0
SPREAD_FACTOR_ZERO_AT = 3.0
CROP_DIVERSITY_FULL_AT = 0.5
NOISE_ZERO_AT = 0.10
MISSING_ZERO_AT = 0.25


def clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
	return max(low, min(high, value))


def _metric(report: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
	result = report.get("metrics", {}).get(name)
	if result is None or result.get("skipped"):
		return None
	return result


def _weighted_mean(parts: Dict[str, tuple]) -> float:
	"""parts: {name: (weight, score)}. Weights are rescaled to sum to 1."""
	total_weight = sum(weight for weight, _ in parts.values())
	return sum(weight * score for weight, score in parts.values()) / total_weight


def _structural(report: Dict[str, Any], sub_scores: Dict[str, float]) -> Optional[float]:
	imbalance = _metric(report, "instance_class_imbalance")
	exact = _metric(report, "exact_duplicate")
	near = _metric(report, "near_duplicate")
	size = _metric(report, "box_size_distribution")
	if None in (imbalance, exact, near, size):
		return None

	ratio_term = clamp(10 * (1 - math.log10(max(imbalance["imbalance_ratio"], 1.0)) / math.log10(IMBALANCE_RATIO_ZERO_AT)))
	balance = 0.5 * clamp(imbalance["normalized_entropy"] * 10) + 0.5 * ratio_term
	redundancy = clamp(10 * (1 - (exact["exact_duplicate_rate"] + near["near_duplicate_rate"]) / REDUNDANCY_ZERO_AT))

	shares = [size["coco"][f"pct_{bucket}"] / 100 for bucket in ("small", "medium", "large")]
	entropy = -sum(share * math.log(share) for share in shares if share > 0)
	size_coverage = clamp(entropy / math.log(3) * 10)

	total_images = exact["total_images"]
	leaked = exact["cross_split_duplicates"] + near["cross_split_near_duplicates"]
	penalty = clamp(leaked * 100 / total_images, 0, CROSS_SPLIT_MAX_PENALTY) if total_images else 0.0

	sub_scores.update({"instance_balance": balance, "redundancy": redundancy, "size_coverage": size_coverage, "cross_split_penalty": penalty})
	return clamp(0.40 * balance + 0.35 * redundancy + 0.25 * size_coverage - penalty)


def _difficulty(report: Dict[str, Any], sub_scores: Dict[str, float]) -> Optional[float]:
	size = _metric(report, "box_size_distribution")
	if size is None:
		return None
	crowding = _metric(report, "crowding")
	scale = _metric(report, "scale_variance")
	cartography = _metric(report, "dataset_cartography")

	# Nothing to crowd or compare when these are skipped, so they count as easy
	crowding_score = clamp(10 * crowding["mean_R"]) if crowding else 10.0
	scale_score = clamp(10 * (1 - (scale["mean_spread_factor"] - 1) / (SPREAD_FACTOR_ZERO_AT - 1))) if scale else 10.0
	small_score = clamp(10 * (1 - size["coco"]["pct_small"] / SMALL_SHARE_ZERO_AT))

	parts = {"crowding": (0.25, crowding_score), "small_objects": (0.25, small_score), "scale_spread": (0.20, scale_score)}
	sub_scores.update({"crowding": crowding_score, "small_objects": small_score, "scale_spread": scale_score})
	if cartography:
		hard_score = clamp(10 * (1 - cartography["pct_hard"] / HARD_SHARE_ZERO_AT))
		parts["cartography"] = (0.30, hard_score)
		sub_scores["cartography"] = hard_score
	return _weighted_mean(parts)


def _diversity(report: Dict[str, Any], sub_scores: Dict[str, float]) -> Optional[float]:
	crops = _metric(report, "crop_diversity")
	if crops is None:
		return None
	co_occurrence = _metric(report, "co_occurrence")
	crop_score = clamp(crops["mean_diversity"] / CROP_DIVERSITY_FULL_AT * 10)
	shortcut_score = clamp(10 * (1 - co_occurrence["shortcut_share"])) if co_occurrence else 10.0
	sub_scores.update({"crop_diversity": crop_score, "co_occurrence": shortcut_score})
	return 0.7 * crop_score + 0.3 * shortcut_score


def _annotation(report: Dict[str, Any], sub_scores: Dict[str, float]) -> Optional[float]:
	noise = _metric(report, "label_noise")
	missing = _metric(report, "missing_annotation")
	parts = {}
	if noise:
		noise_score = clamp(10 * clamp(1 - noise["estimated_noise_rate"] / NOISE_ZERO_AT, 0, 1) ** 1.5)
		parts["label_noise"] = (0.5, noise_score)
		sub_scores["label_noise"] = noise_score
	if missing:
		missing_score = clamp(10 * (1 - missing["missing_rate_estimate"] / MISSING_ZERO_AT))
		parts["missing_annotation"] = (0.5, missing_score)
		sub_scores["missing_annotation"] = missing_score
	return _weighted_mean(parts) if parts else None


def score_detection_report(report: Dict[str, Any]) -> Dict[str, Any]:
	"""Axis scores, overall score and the sub-scores behind them, all rounded to one decimal."""
	sub_scores: Dict[str, float] = {}
	axes = {
		"structural": _structural(report, sub_scores),
		"difficulty": _difficulty(report, sub_scores),
		"diversity": _diversity(report, sub_scores),
		"annotation": _annotation(report, sub_scores),
	}
	present = {name: (AXIS_WEIGHTS[name], score) for name, score in axes.items() if score is not None}
	result: Dict[str, Any] = {name: round(score, 1) for name, score in axes.items() if score is not None}
	result["overall"] = round(_weighted_mean(present), 1) if present else None
	result["sub_scores"] = {name: round(value, 2) for name, value in sub_scores.items()}
	result["status"] = "proposal, pending calibration"
	return result
