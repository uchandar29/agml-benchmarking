# AgML Dataset Scoring Formulas

All scores are on a **0 to 10 scale** (10 = best). They are computed only from a run's `report.json`, so no pipeline re-run is needed.

The reference implementation is `src/lib/scoring.ts` in the [project-agml.github.io](https://github.com/Project-AgML/project-agml.github.io) repository. This document matches it line for line.

`clamp(x, 0, 10)` limits a value to the range 0 to 10.

## Contents

1. [JSON field references](#json-field-references)
2. [Axis 1: Structural quality](#axis-1-structural-quality)
3. [Axis 2: Content difficulty](#axis-2-content-difficulty)
4. [Axis 3: Diversity and coverage](#axis-3-diversity-and-coverage)
5. [Axis 4: Annotation reliability](#axis-4-annotation-reliability)
6. [Overall score](#overall-score)
7. [Implementation](#implementation)
8. [Design notes](#design-notes)
9. [Known limitations](#known-limitations)
10. [Object detection scores (proposal)](#object-detection-scores-proposal)

## JSON field references

| Symbol | JSON path |
|--------|-----------|
| `entropy` | `metrics.class_imbalance.normalized_entropy` |
| `exact_rate` | `metrics.exact_duplicate.exact_duplicate_rate` |
| `near_rate` | `metrics.near_duplicate.near_duplicate_rate` |
| `cross_split_dups` | `metrics.exact_duplicate.cross_split_duplicates` |
| `cross_split_near` | `metrics.near_duplicate.cross_split_near_duplicates` |
| `total_images` | `metrics.exact_duplicate.total_images` |
| `area_cv` | `metrics.resolution_consistency.area_cv` |
| `silhouette` | `metrics.feature_separability.silhouette_score` |
| `davies_bouldin` | `metrics.feature_separability.davies_bouldin_index` |
| `mean_diversity` | `metrics.intra_class_diversity.mean_diversity` |
| `pct_hard` | `metrics.dataset_cartography.pct_hard` |
| `accuracy` | `metrics.class_confusability.accuracy` |
| `noise_rate` | `metrics.label_noise.estimated_noise_rate` |

A metric is **available** if its key exists in `metrics` and it does not contain `"skipped": true`.

**An axis needs every one of its metrics.** If any metric of an axis is missing, the whole axis is dropped. The one exception is content difficulty, which falls back to separability alone when the Phase 3 metrics are missing.

## Axis 1: Structural quality

> Needs class imbalance, exact duplicates, resolution consistency (Phase 1) and near duplicates (Phase 2).

### 1a. Class balance

$$S_{balance} = \text{clamp}(\text{entropy} \times 10,\ 0,\ 10)$$

A normalized entropy of 1 means all classes have the same size.

### 1b. Redundancy

Exact and near-duplicate rates are added together, so cleaning only one type cannot raise the score.

$$S_{redundancy} = \text{clamp}\!\left(10 \times \left(1 - \frac{\text{exact\_rate} + \text{near\_rate}}{0.20}\right),\ 0,\ 10\right)$$

A combined rate of 0% gives 10, and 20% or more gives 0.

### 1c. Cross-split penalty

Images that appear in more than one split make test results look better than they are. This penalty is subtracted from the axis score.

$$P_{cross} = \text{clamp}\!\left(\frac{(\text{cross\_split\_dups} + \text{cross\_split\_near}) \times 100}{\text{total\_images}},\ 0,\ 2\right)$$

Each 1% of images in cross-split groups costs 1 point, up to 2 points. If `total_images` is 0, the penalty is 0.

### 1d. Resolution consistency

$$S_{resolution} = \text{clamp}(10 \times (1 - \text{area\_cv}),\ 0,\ 10)$$

A CV of 0 (all images the same size) gives 10, and a CV of 1 or more gives 0.

### Structural quality score

$$\boxed{Q_{structural} = \text{clamp}\!\left(0.40 \cdot S_{balance} + 0.35 \cdot S_{redundancy} + 0.25 \cdot S_{resolution} - P_{cross},\ 0,\ 10\right)}$$

## Axis 2: Content difficulty

> Needs feature separability (Phase 2). Cartography and confusability (Phase 3) are used when available.

### 2a. Feature separability

Silhouette (higher is better) and Davies-Bouldin (lower is better) are both used, so optimising one alone does not help. Each term is clamped before averaging.

$$S_{separability} = \frac{\text{clamp}(\text{silhouette} \times 10,\ 0,\ 10) + \text{clamp}(10 - 3 \times \text{davies\_bouldin},\ 0,\ 10)}{2}$$

### 2b. Training difficulty *(Phase 3)*

Datasets with 50% or more hard samples score 0, and datasets with almost no hard samples score close to 10.

$$S_{cartography} = \text{clamp}\!\left(10 \times \left(1 - \frac{\text{pct\_hard}}{50}\right),\ 0,\ 10\right)$$

### 2c. Class confusability *(Phase 3)*

High accuracy of the reference ResNet-18 on the test split means the classes are easy to tell apart.

$$S_{confusability} = \text{clamp}(\text{accuracy} \times 10,\ 0,\ 10)$$

### Content difficulty score

$$\boxed{Q_{difficulty} = \begin{cases} 0.40 \cdot S_{separability} + 0.30 \cdot S_{cartography} + 0.30 \cdot S_{confusability} & \text{if Phase 3 is available} \\ S_{separability} & \text{otherwise} \end{cases}}$$

## Axis 3: Diversity and coverage

> Needs intra-class diversity (Phase 2).

### 3a. Intra-class diversity

`mean_diversity` is the mean Euclidean distance from each image's DINOv2 embedding to its class centroid, averaged over classes. A value of 0.4 or more gives the full score.

$$S_{diversity} = \text{clamp}\!\left(\frac{\text{mean\_diversity}}{0.4} \times 10,\ 0,\ 10\right)$$

### 3b. Metadata coverage

Metadata coverage is reported as frequency tables but is **not scored** yet.

### Diversity score

$$\boxed{Q_{diversity} = S_{diversity}}$$

## Axis 4: Annotation reliability

> Needs label noise (Phase 3).

### 4a. Label noise

The curve is steep near the bottom, so even a little noise lowers the score quickly. The base is clamped to the range 0 to 1 before the exponent, so the score is 0 once noise reaches 10%.

$$S_{noise} = \text{clamp}\!\left(10 \times \left(\text{clamp}\!\left(1 - \frac{\text{noise\_rate}}{0.10},\ 0,\ 1\right)\right)^{1.5},\ 0,\ 10\right)$$

0% noise gives 10, 5% gives about 3.5, and 10% or more gives 0.

### Annotation reliability score

$$\boxed{Q_{annotation} = S_{noise}}$$

## Overall score

The overall score is a weighted mean of the available axes. If an axis is missing, it is dropped and the remaining weights are scaled up in proportion, so the score is never pulled down by a missing axis.

$$\boxed{Q_{overall} = \frac{\sum_{i}\ w_i \cdot Q_i}{\sum_{i}\ w_i}}$$

| Axes available | Effective weights |
|----------------|-------------------|
| All four | **Structural 30%, Difficulty 25%, Diversity 25%, Annotation 20%** |
| Structural, Difficulty, Diversity (no Phase 3) | 37.5%, 31.25%, 31.25% |
| Structural, Diversity (single-class data, such as iNatAg) | 54.5%, 45.5% |

All scores are rounded to one decimal place. Overall scores of 7.5 and above are shown as good, 5.0 to 7.5 as fair, and below 5.0 as needing attention.

## Design notes

- **Duplicate cleaning:** exact and near-duplicate rates are added, so removing only exact duplicates does not raise the score.
- **Cross-split leakage:** penalised on its own, so it cannot be hidden by doing well elsewhere.
- **Diversity and duplication:** removing images to lower the duplicate rate also lowers diversity, which creates a natural trade-off.
- **Silhouette and Davies-Bouldin:** both are used, so optimising one alone moves the other the wrong way.
- **Label noise exponent:** the 1.5 exponent makes small changes near the 10% boundary worth very little.
- **Fixed split seed:** the split seed (42) is published, so the test set cannot be picked to raise confusability accuracy.

## Known limitations

These come from running the scores on 809 datasets and comparing them with model performance (see [IMAGE_CLASSIFICATION.md](IMAGE_CLASSIFICATION.md)).

- **Hand-set weights:** the axis weights and cut-offs were chosen by hand and were not fitted to any model results.
- **Diversity saturates:** the 0.4 ceiling is too low for DINOv2 embeddings. 73 of 82 curated datasets and all iNatAg subsets are above it, so the diversity axis is 10 for almost every dataset.
- **Almost no hard samples:** cartography uses the model's highest softmax probability rather than the probability of the true label, so very few images are marked hard. Only 1 of 82 curated datasets has any.
- **Structural axis:** on 20 test datasets it showed no link with fine-tuned performance, while content difficulty and annotation reliability did.

## Object detection scores (proposal)

> **Proposal, pending calibration across the library.** The weights and cut-offs below are hand-set and have not been checked against detector performance. The code is `core/metrics/detection_scoring.py` (`score_detection_report(report)`), and the result is written to `report.json` under `scores`. The metrics are described in [OBJECT_DETECTION.md](OBJECT_DETECTION.md).

The scale and rules are the same as above: 0 to 10 with 10 = best, a metric is available when it is in `metrics` and not skipped, an axis is dropped when a required metric is missing, and the overall score is the weighted mean of the axes that remain. As in the classification axes, a **higher score means an easier, cleaner dataset**, so on the difficulty axis a crowded dataset with many small objects scores low.

### Structural quality

Needs `instance_class_imbalance`, `exact_duplicate`, `near_duplicate`, `box_size_distribution`.

| Part | Formula |
|------|---------|
| Instance balance | `0.5 * clamp(entropy * 10) + 0.5 * clamp(10 * (1 - log10(ratio) / 2))`, with `entropy` and `ratio` from the box counts per class. A ratio of 1 gives 10, 100 or more gives 0 |
| Redundancy | `clamp(10 * (1 - (exact_rate + near_rate) / 0.20))` |
| Size coverage | `10 * H(pct_small, pct_medium, pct_large) / ln 3`, the normalised entropy of the three COCO size buckets |
| Cross-split penalty | `clamp((cross_split_dups + cross_split_near) * 100 / total_images, 0, 2)`, subtracted |

$$Q_{structural} = \text{clamp}(0.40 \cdot S_{balance} + 0.35 \cdot S_{redundancy} + 0.25 \cdot S_{size} - P_{cross},\ 0,\ 10)$$

### Content difficulty

Needs `box_size_distribution`. Crowding, box occlusion and scale variance count as 10 when they are skipped. Cartography is used when Phase 3 ran.

| Part | Formula | Weight |
|------|---------|--------|
| Crowding | `clamp(10 * mean_R)`, where `mean_R` is the mean Clark-Evans ratio (1 or more gives 10) | 0.25 |
| Box occlusion | `clamp(10 * (1 - mean_image_bor / 0.5))`, where `mean_image_bor` is the mean over images of the share of each box covered by the union of the other boxes (0 gives 10, 0.5 or more gives 0) | 0.25 |
| Small objects | `clamp(10 * (1 - pct_small / 50))`, COCO small share in percent | 0.25 |
| Scale spread | `clamp(10 * (1 - (mean_spread_factor - 1) / 2))`, a factor of 3 or more gives 0 | 0.20 |
| Cartography | `clamp(10 * (1 - pct_hard / 50))` on the tracked boxes of the reference detector | 0.30 |

The score is the weighted mean of the parts that are available. Weights are rescaled to sum to 1, so adding box occlusion lowers the share of every other part.

### Diversity

Needs `crop_diversity`.

$$Q_{diversity} = 0.7 \cdot \text{clamp}\!\left(\frac{\text{mean\_diversity}}{0.5} \times 10,\ 0,\ 10\right) + 0.3 \cdot \text{clamp}(10 \cdot (1 - \text{shortcut\_share}),\ 0,\ 10)$$

`mean_diversity` is the intra-class diversity of box crops. `shortcut_share` is the share of class pairs flagged by the co-occurrence metric. It counts as 0 (score 10) when co-occurrence is skipped.

### Annotation reliability

Needs `label_noise` or `missing_annotation`. Both are used when available, each with weight 0.5.

| Part | Formula |
|------|---------|
| Label noise | the same curve as classification, on the crop-level noise rate |
| Missing annotation | `clamp(10 * (1 - missing_rate_estimate / 0.25))` |

### Overall

Same weights as classification: **Structural 30%, Difficulty 25%, Diversity 25%, Annotation 20%**, rescaled over the axes present. Scores are rounded to one decimal.

### Open points

- The ceiling of 0.5 for crop diversity and all cut-offs above need calibration, since the classification diversity axis saturated with its first guess.
- Size coverage rewards a spread over small, medium and large boxes, which pulls against the small-object term in the difficulty axis.
- The missing annotation estimate is unverified (see OBJECT_DETECTION.md).
