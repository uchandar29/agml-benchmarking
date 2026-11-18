# Object Detection Benchmark

The detection benchmark scores object detection datasets from Hugging Face (`Project-AgML/...`). It follows the classification benchmark: the same splits, report writer and config, with box-level metrics. Scores are proposals, see [SCORING_FORMULAS.md](SCORING_FORMULAS.md#object-detection-scores-proposal).

Code: `benchmark/pipelines/object_detection.py`, started through `python -m benchmark.run --task object_detection`. Datasets: the `task: "object_detection"` entries of `benchmark/slurm/datasets.yaml`. Settings: the `object_detection` section of `benchmark/slurm/config.json`. Results: `benchmark/results/object_detection/<dataset>/<timestamp>/report.json`.

## Data handling

- The `objects` column holds `bbox` and `categories`. Categories can be one scalar that applies to every box.
- Box formats: `coco` (absolute x, y, w, h), `yolo` (normalised centre x, y, w, h), `voc` (absolute x1, y1, x2, y2). `auto` picks `yolo` when every coordinate is at most 1.0, otherwise `coco`.
- All boxes become absolute xyxy pixels, clipped to the image. Boxes with zero width or height after clipping are dropped and counted in `dataset_info.dropped_boxes`.
- Image sizes come from the file headers, so no image is decoded for this.
- Class names come from the ClassLabel feature, otherwise `class_<id>`.
- Splits are 70/15/15 with seed 42. Each image is stratified on the rarest class it contains (images without boxes form their own group). Splits are `select()` only.

## Metrics

Everything is under `metrics.<name>` in `report.json`. The report also has `task`, `dataset_info` and `scores`.

| Phase | Metric | What it tells you |
|---|---|---|
| 1 | `box_count_distribution` | Boxes per image (mean, median, std, min, max, p90), share of empty images, histogram |
| 1 | `box_size_distribution` | Share of small, medium and large boxes (COCO, absolute pixels), relative area percentiles, dataset terciles, median relative area per class |
| 1 | `aspect_ratio_distribution` | Width over height, overall and per class: median, p5, p95, spread of the log ratio, share of extreme ratios (above 3 or below 1/3) |
| 1 | `instance_class_imbalance` | Imbalance ratio and normalised entropy of box counts per class, on the training split |
| 1 | `scale_variance` | Spread of box area inside each class (`exp(std of log area)`) |
| 1 | `crowding` | Clark-Evans ratio of box centres per image. Below 1 means clustered, about 1 random, above 1 evenly spread |
| 1 | `box_occlusion` | Box Occlusion Ratio (BOR): per box, the share of its area covered by the union of the other boxes in the image. Image score is the mean over its boxes (0 for images with fewer than two boxes). Also reports box pairs with IoU of 0.5 or more |
| 1 | `co_occurrence` | Class pairs that appear together more than chance (lift above 2, at least 5 images). Skipped with one class |
| 1 | `exact_duplicate` | Identical images and cross-split leakage |
| 2 | `near_duplicate` | Near duplicate images with DINOv2 embeddings of whole images |
| 2 | `crop_diversity` | Intra-class diversity of DINOv2 embeddings of box crops (10% context) |
| 2 | `label_noise` | Confident learning on the crop embeddings. Skipped with a reason if fewer than 2 classes have enough crops. Classes with fewer crops than CV folds are dropped (`classes_dropped`) |
| 3 | `dataset_cartography` | Easy, ambiguous and hard **boxes**, from the confidence a reference detector gives each training box over the epochs |
| 3 | `missing_annotation` | Confident detector boxes (score 0.7 or more) on val and test images that overlap no ground truth (IoU below 0.5). Class is ignored. **Unverified**: a candidate may be a detector error. `review_sample` lists up to 50 to check by hand |
| 3 | `detector_eval` | AP50 of the reference detector on the test split, for context |

Crops are sampled with at most `crop_max_per_class` per class and `crop_max_total` overall. Crops with a box side under 8 pixels are skipped.

## Reference detector

Faster R-CNN with a MobileNetV3 FPN backbone (torchvision, COCO weights), new box head, 12 epochs of SGD. After each epoch it predicts on a fixed sample of up to 2000 training images. The confidence of a ground-truth box is the highest score of a same-class prediction with IoU of 0.5 or more, or 0.

Outputs are cached in `<run_dir>/phase3_detection/`.
