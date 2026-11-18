from core.metrics.detection_scoring import score_detection_report


def good_report():
	return {"metrics": {
		"instance_class_imbalance": {"imbalance_ratio": 1.0, "normalized_entropy": 1.0},
		"exact_duplicate": {"exact_duplicate_rate": 0.0, "total_images": 100, "cross_split_duplicates": 0},
		"near_duplicate": {"near_duplicate_rate": 0.0, "cross_split_near_duplicates": 0},
		"box_size_distribution": {"coco": {"pct_small": 0.0, "pct_medium": 50.0, "pct_large": 50.0}},
		"crowding": {"mean_R": 1.1},
		"box_occlusion": {"mean_image_bor": 0.0},
		"scale_variance": {"mean_spread_factor": 1.0},
		"dataset_cartography": {"pct_hard": 0.0},
		"crop_diversity": {"mean_diversity": 0.6},
		"co_occurrence": {"shortcut_share": 0.0},
		"label_noise": {"estimated_noise_rate": 0.0},
		"missing_annotation": {"missing_rate_estimate": 0.0},
	}}


def test_good_report_scores():
	scores = score_detection_report(good_report())
	# structural loses a little: two size buckets cover less than three (ln 2 / ln 3 of 10)
	assert scores["structural"] == 9.1
	assert all(scores[axis] == 10.0 for axis in ("difficulty", "diversity", "annotation"))
	expected = (0.30 * 9.08 + 0.25 * 10 + 0.25 * 10 + 0.20 * 10)
	assert abs(scores["overall"] - expected) < 0.1


def test_bad_report_scores_low_and_in_range():
	report = good_report()
	report["metrics"]["instance_class_imbalance"] = {"imbalance_ratio": 500.0, "normalized_entropy": 0.2}
	report["metrics"]["dataset_cartography"] = {"pct_hard": 40.0}
	report["metrics"]["label_noise"] = {"estimated_noise_rate": 0.2}
	scores = score_detection_report(report)
	assert 0 <= scores["structural"] < 10 and 0 <= scores["difficulty"] < 10
	assert scores["annotation"] < 10 and scores["sub_scores"]["label_noise"] == 0.0


def test_missing_axis_is_dropped_and_skipped_metric_ignored():
	report = good_report()
	del report["metrics"]["near_duplicate"]
	report["metrics"]["label_noise"] = {"skipped": True, "reason": "one class"}
	scores = score_detection_report(report)
	assert "structural" not in scores
	assert "label_noise" not in scores["sub_scores"] and scores["annotation"] == 10.0
	# remaining weights rescale, so three perfect axes stay at 10
	assert scores["overall"] == 10.0


def test_high_box_occlusion_lowers_difficulty():
	report = good_report()
	report["metrics"]["box_occlusion"] = {"mean_image_bor": 0.4}
	scores = score_detection_report(report)
	assert scores["difficulty"] < 10.0 and scores["sub_scores"]["box_occlusion"] == 2.0
