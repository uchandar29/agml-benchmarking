import math

import numpy as np

from core.metrics.common import imbalance_stats
from core.metrics.difficulty import CrowdingMetric, ScaleVarianceMetric
from core.metrics.diversity import CoOccurrenceMetric
from core.metrics.structural import BoxSizeDistributionMetric
from tests.detection.synthetic import make_box_table


def square(side: float, x: float = 0.0, y: float = 0.0):
	return [x, y, x + side, y + side]


def test_coco_buckets():
	sides = [10, 31, 32, 96, 97]  # areas 100, 961, 1024, 9216, 9409
	table = make_box_table([(1000, 1000)], [[square(side) for side in sides]], [[0] * 5], ["a"])
	result = BoxSizeDistributionMetric().run(box_table=table)["coco"]
	assert (result["n_small"], result["n_medium"], result["n_large"]) == (2, 2, 1)
	assert result["pct_small"] == 40.0 and result["pct_large"] == 20.0


def test_clark_evans_is_one_for_random_centers():
	rng = np.random.default_rng(0)
	centers = rng.uniform(0, 2000, size=(3000, 2))
	boxes = [[x - 1, y - 1, x + 1, y + 1] for x, y in centers]
	table = make_box_table([(2000, 2000)], [boxes], [[0] * len(boxes)], ["a"])
	result = CrowdingMetric().run(box_table=table)
	assert 0.85 <= result["mean_R"] <= 1.15


def test_clark_evans_low_for_clusters():
	rng = np.random.default_rng(1)
	centers = np.concatenate([rng.normal(200, 5, size=(50, 2)), rng.normal(800, 5, size=(50, 2))])
	boxes = [[x - 1, y - 1, x + 1, y + 1] for x, y in centers]
	table = make_box_table([(1000, 1000)], [boxes], [[0] * len(boxes)], ["a"])
	assert CrowdingMetric().run(box_table=table)["mean_R"] < 0.5


def test_scale_spread_factor():
	rng = np.random.default_rng(2)
	sides = rng.uniform(10, 80, size=200)
	table = make_box_table([(1000, 1000)], [[square(side) for side in sides]], [[0] * len(sides)], ["a", "unused"])
	result = ScaleVarianceMetric().run(box_table=table)
	expected_std = float(np.std(np.log(np.float32(sides) ** 2)))
	assert math.isclose(result["per_class"]["a"]["std_log_area"], expected_std, abs_tol=1e-3)
	assert math.isclose(result["per_class"]["a"]["spread_factor"], math.exp(expected_std), rel_tol=1e-3)
	assert "unused" not in result["per_class"]


def test_co_occurrence_lift():
	# 20 images: A in 0-5, B in 0-4, C in 10-19
	presence = {"A": range(0, 6), "B": range(0, 5), "C": range(10, 20)}
	boxes_by_image = [[] for _ in range(20)]
	classes_by_image = [[] for _ in range(20)]
	for class_id, name in enumerate(presence):
		for image in presence[name]:
			boxes_by_image[image].append(square(10, class_id * 20))
			classes_by_image[image].append(class_id)
	table = make_box_table([(100, 100)] * 20, boxes_by_image, classes_by_image, ["A", "B", "C"])
	result = CoOccurrenceMetric().run(box_table=table)
	pairs = {(pair["class_a"], pair["class_b"]): pair for pair in result["top_pairs"]}
	assert pairs[("A", "B")]["observed"] == 5
	assert math.isclose(pairs[("A", "B")]["expected"], 6 * 5 / 20)
	assert math.isclose(pairs[("A", "B")]["lift"], 5 / 1.5, rel_tol=1e-3)
	assert pairs[("A", "B")]["flagged"] is True
	assert pairs[("A", "C")]["lift"] == 0.0 and pairs[("A", "C")]["flagged"] is False
	assert result["flagged_pairs"] == 1 and result["pairs_evaluated"] == 3


def test_co_occurrence_skips_single_class():
	table = make_box_table([(100, 100)], [[square(10)]], [[0]], ["a"])
	assert CoOccurrenceMetric().run(box_table=table)["skipped"] is True


def test_imbalance_stats_values():
	stats = imbalance_stats({"a": 8, "b": 2}, num_classes=2)
	assert stats["imbalance_ratio"] == 4.0
	expected_entropy = -(0.8 * math.log(0.8) + 0.2 * math.log(0.2)) / math.log(2)
	assert stats["normalized_entropy"] == round(expected_entropy, 4)
	assert stats["most_frequent_class"] == "a" and stats["least_frequent_class"] == "b"
