import numpy as np
from datasets import Dataset, Features, Image, Sequence, Value

from core.data import DetectionDatasetAdapter, SplitManager, proxy_labels
from core.data.detection_adapter import boxes_to_xyxy, broadcast_categories, detect_bbox_format
from tests.detection.synthetic import make_box_table, make_detection_dataset


def test_yolo_and_coco_give_same_pixels():
	width, height = 640, 480
	coco = np.array([[64.0, 96.0, 128.0, 48.0]])
	yolo = np.array([[(64 + 64) / width, (96 + 24) / height, 128 / width, 48 / height]])
	assert detect_bbox_format([coco]) == "coco"
	assert detect_bbox_format([yolo]) == "yolo"
	from_coco = boxes_to_xyxy(coco, detect_bbox_format([coco]), width, height)
	from_yolo = boxes_to_xyxy(yolo, detect_bbox_format([yolo]), width, height)
	assert np.allclose(from_coco, from_yolo, atol=1e-3)
	assert np.allclose(from_coco, [[64, 96, 192, 144]])


def test_normalised_top_left_is_told_apart_from_yolo():
	width, height = 640, 640
	# Top-left boxes near the left and top edges would cross the edge if read as centres
	top_left = np.array([[0.02, 0.10, 0.20, 0.10], [0.50, 0.01, 0.10, 0.10], [0.75, 0.15, 0.247, 0.104]])
	assert detect_bbox_format([top_left]) == "coco_normalized"
	pixels = boxes_to_xyxy(top_left[2:], "coco_normalized", width, height)
	assert np.allclose(pixels, [[480, 96, 638.08, 162.56]], atol=1e-2)
	centre = np.array([[0.5, 0.5, 0.9, 0.9], [0.9, 0.9, 0.2, 0.2]])
	assert detect_bbox_format([centre]) == "yolo"


def test_voc_and_clipping():
	boxes = boxes_to_xyxy(np.array([[-5.0, 10.0, 50.0, 700.0]]), "voc", 100, 100)
	assert boxes.tolist() == [[0.0, 10.0, 50.0, 100.0]]


def test_scalar_categories_broadcast():
	assert broadcast_categories(0.0, 3).tolist() == [0, 0, 0]
	assert broadcast_categories([1.0, 0.0], 2).tolist() == [1, 0]


def test_build_box_table_yolo_scalar_categories(monkeypatch):
	features = Features({
		"image": Image(),
		"objects": {"bbox": Sequence(Sequence(Value("float32"))), "categories": Value("float32")},
	})
	from PIL import Image as PILImage
	images = [PILImage.new("RGB", (200, 100)), PILImage.new("RGB", (200, 100))]
	rows = [
		{"bbox": [[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.0, 0.0]], "categories": 0.0},
		{"bbox": [], "categories": 0.0},
	]
	dataset = Dataset.from_dict({"image": images, "objects": rows}, features=features)
	monkeypatch.setattr(DetectionDatasetAdapter, "_load_hf", lambda self: dataset)

	adapter = DetectionDatasetAdapter("fake/yolo")
	table = adapter.build_box_table(adapter.load())
	assert table.image_sizes.tolist() == [[200, 100], [200, 100]]
	assert table.num_boxes == 1 and table.dropped_boxes == 1
	assert np.allclose(table.boxes_xyxy[0], [80, 30, 120, 70])
	assert table.label_names == ["class_0"]
	assert adapter.schema().bbox_format == "yolo" and adapter.schema().num_classes == 1


def test_coco_dataset_with_class_labels(monkeypatch):
	dataset = make_detection_dataset(num_images=12)
	monkeypatch.setattr(DetectionDatasetAdapter, "_load_hf", lambda self: dataset)
	adapter = DetectionDatasetAdapter("fake/coco")
	table = adapter.build_box_table(adapter.load())
	assert adapter.schema().bbox_format == "coco"
	assert table.label_names == ["red", "blue"]
	assert table.image_sizes.tolist() == [[96, 96]] * 12
	assert (table.widths() > 0).all()


def test_proxy_labels_and_split_labels():
	table = make_box_table(
		[(10, 10)] * 4,
		[[[0, 0, 5, 5], [0, 0, 5, 5]], [[0, 0, 5, 5]], [], [[0, 0, 5, 5], [1, 1, 6, 6]]],
		[[0, 1], [0, 0], [], [0, 0]],
		["common", "rare"],
	)
	# class 1 has one box, so image 0 takes it; image 2 has no boxes
	assert proxy_labels(table).tolist() == [1, 0, 2, 0]

	dataset = make_detection_dataset(num_images=30)
	labels = np.arange(30) % 3
	manager = SplitManager()
	by_labels = manager.split_labels(dataset, labels)
	assert sum(by_labels.sizes.values()) == 30
	again = manager.split_labels(dataset, labels)
	assert all(np.array_equal(by_labels.indices[name], again.indices[name]) for name in by_labels.indices)
