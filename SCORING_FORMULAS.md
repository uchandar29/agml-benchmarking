# AgML Dataset Scoring Formulas

All scores are on a **0 – 10 scale** (10 = best).  
Scores are computed entirely from the `report.json` output — no pipeline re-run needed.  
All `clamp(x, 0, 10)` calls floor at 0 and ceil at 10.

---

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

A metric is considered **available** if its key exists in `metrics` and does not contain `"skipped": true`.

---

## Axis 1 — Structural Quality

> Requires Phase 1.

### 1a. Class Balance

$$S_{balance} = \text{clamp}(\text{entropy} \times 10,\ 0,\ 10)$$

A normalized entropy of 1.0 means perfectly equal class sizes.

### 1b. Redundancy

Combines exact and near-duplicate rates so cleaning only one type cannot inflate the score.

$$S_{redundancy} = \text{clamp}\!\left(10 \times \left(1 - \frac{\text{exact\_rate} + \text{near\_rate}}{0.20}\right),\ 0,\ 10\right)$$

Combined rate of 0% → 10. Combined rate ≥ 20% → 0.

### 1c. Cross-Split Contamination Penalty

Samples that appear in more than one split inflate test-set metrics.  
This penalty is subtracted from the final Structural axis score.

$$P_{cross} = \text{clamp}\!\left(\frac{(\text{cross\_split\_dups} + \text{cross\_split\_near}) \times 100}{\text{total\_images}},\ 0,\ 2\right)$$

Maximum penalty is 2 points. More than 2% cross-split contamination incurs the full penalty.

### 1d. Resolution Consistency

$$S_{resolution} = \text{clamp}(10 \times (1 - \text{area\_cv}),\ 0,\ 10)$$

CV = 0 → 10 (all images same resolution). CV ≥ 1.0 → 0.

### Structural Quality Score

$$\boxed{Q_{structural} = \text{clamp}\!\left(0.40 \cdot S_{balance} + 0.35 \cdot S_{redundancy} + 0.25 \cdot S_{resolution} - P_{cross},\ 0,\ 10\right)}$$

---

## Axis 2 — Content Difficulty

> Requires Phase 2. Phase 3 metrics improve precision if available.

### 2a. Feature Separability

Uses both silhouette (higher = better separated) and Davies-Bouldin (lower = better separated) to make it harder to optimise for one alone.

$$S_{separability} = \text{clamp}\!\left(\frac{\text{silhouette} \times 10 + \text{clamp}(10 - \text{davies\_bouldin} \times 3,\ 0,\ 10)}{2},\ 0,\ 10\right)$$

### 2b. Training Difficulty Balance *(Phase 3 only)*

A healthy dataset has a spread of easy, ambiguous, and hard samples. Datasets with more than 50% hard-to-learn samples score 0; datasets with fewer than 5% hard samples score close to 10.

$$S_{cartography} = \text{clamp}\!\left(10 \times \left(1 - \frac{\text{pct\_hard}}{50}\right),\ 0,\ 10\right)$$

### 2c. Class Confusability *(Phase 3 only)*

High model accuracy on held-out data means classes are genuinely distinct.

$$S_{confusability} = \text{clamp}(\text{accuracy} \times 10,\ 0,\ 10)$$

### Content Difficulty Score

$$\boxed{Q_{difficulty} = \begin{cases} S_{separability} & \text{if Phase 3 not available} \\ 0.40 \cdot S_{separability} + 0.30 \cdot S_{cartography} + 0.30 \cdot S_{confusability} & \text{if Phase 3 available} \end{cases}}$$

---

## Axis 3 — Diversity & Coverage

> Requires Phase 2.

### 3a. Intra-Class Visual Diversity

DINOv2 mean cosine distances in practice range 0.05 (highly repetitive) to 0.5+ (highly diverse). Normalised against 0.4 as the reference ceiling.

$$S_{diversity} = \text{clamp}\!\left(\frac{\text{mean\_diversity}}{0.4} \times 10,\ 0,\ 10\right)$$

### Diversity & Coverage Score

$$\boxed{Q_{diversity} = S_{diversity}}$$

---

## Axis 4 — Annotation Reliability

> Requires Phase 3.

### 4a. Label Noise

Non-linear decay — early noise matters more than marginal noise at the high end.

$$S_{noise} = \text{clamp}\!\left(10 \times \left(\max\!\left(0,\ 1 - \frac{\text{noise\_rate}}{0.10}\right)\right)^{1.5},\ 0,\ 10\right)$$

0% noise → 10. 5% noise → ~6.5. 10%+ noise → 0.  
The exponent 1.5 prevents marginal gaming around the 10% boundary.

### Annotation Reliability Score

$$\boxed{Q_{annotation} = S_{noise}}$$

---

## Overall Score

Axis weights depend on which phases were run. Axes that require unavailable phases are excluded and their weight is redistributed equally across the remaining axes.

| Phases run | Axes available | Effective weights (base / total) |
|------------|---------------|----------------------------------|
| Phase 1 only | Structural | 100% |
| Phases 1 + 2 | Structural, Difficulty, Diversity | 37.5% / 31.25% / 31.25%  (0.30 + 0.25 + 0.25 = 0.80) |
| Phases 1 + 2 + 3 | All four | **30% / 25% / 25% / 20%** |

$$\boxed{Q_{overall} = \frac{\sum_{i}\ w_i \cdot Q_i}{\sum_{i}\ w_i}}$$

where the sum runs only over axes whose required phases are present in `phases_completed`.

---

## Implementation notes

```js
function clamp(x, min, max) {
  return Math.min(Math.max(x, min), max);
}

function score(report) {
  const m = report.metrics;
  const phases = report.phases_completed;

  const has = (key) => key in m && !m[key]?.skipped;
  const p1 = phases.includes(1);
  const p2 = phases.includes(2);
  const p3 = phases.includes(3);

  const axes = {};

  // Structural Quality
  if (p1) {
    const balance    = clamp(m.class_imbalance.normalized_entropy * 10, 0, 10);
    const combined   = m.exact_duplicate.exact_duplicate_rate + m.near_duplicate.near_duplicate_rate;
    const redundancy = clamp(10 * (1 - combined / 0.20), 0, 10);
    const resolution = clamp(10 * (1 - m.resolution_consistency.area_cv), 0, 10);
    const crossSplit = m.exact_duplicate.cross_split_duplicates + m.near_duplicate.cross_split_near_duplicates;
    const penalty    = clamp((crossSplit * 100) / m.exact_duplicate.total_images, 0, 2);
    axes.structural  = clamp(0.40 * balance + 0.35 * redundancy + 0.25 * resolution - penalty, 0, 10);
  }

  // Content Difficulty
  if (p2) {
    const sil  = clamp(m.feature_separability.silhouette_score * 10, 0, 10);
    const db   = clamp(10 - m.feature_separability.davies_bouldin_index * 3, 0, 10);
    const sep  = (sil + db) / 2;
    if (p3) {
      const cart = clamp(10 * (1 - m.dataset_cartography.pct_hard / 50), 0, 10);
      const conf = clamp(m.class_confusability.accuracy * 10, 0, 10);
      axes.difficulty = 0.40 * sep + 0.30 * cart + 0.30 * conf;
    } else {
      axes.difficulty = sep;
    }
  }

  // Diversity & Coverage
  if (p2) {
    axes.diversity = clamp(m.intra_class_diversity.mean_diversity / 0.4 * 10, 0, 10);
  }

  // Annotation Reliability
  if (p3) {
    const base = Math.max(0, 1 - m.label_noise.estimated_noise_rate / 0.10);
    axes.annotation = clamp(10 * Math.pow(base, 1.5), 0, 10);
  }

  // Overall
  const weights = { structural: 0.30, difficulty: 0.25, diversity: 0.25, annotation: 0.20 };
  let weightedSum = 0, totalWeight = 0;
  for (const [axis, score] of Object.entries(axes)) {
    if (score == null || isNaN(score)) continue;
    weightedSum += score * weights[axis];
    totalWeight += weights[axis];
  }
  const overall = totalWeight > 0 ? weightedSum / totalWeight : null;

  return {
    overall:    overall !== null ? Math.round(overall * 10) / 10 : null,
    structural: axes.structural  != null ? Math.round(axes.structural  * 10) / 10 : null,
    difficulty: axes.difficulty  != null ? Math.round(axes.difficulty  * 10) / 10 : null,
    diversity:  axes.diversity   != null ? Math.round(axes.diversity   * 10) / 10 : null,
    annotation: axes.annotation  != null ? Math.round(axes.annotation  * 10) / 10 : null,
  };
}
```

---

## Anti-gaming notes

- **Duplicate cleaning** — exact and near-duplicate rates are summed, so removing only exact duplicates while keeping near-duplicates does not improve the score.
- **Cross-split contamination** — penalised separately and cannot be hidden by improving other components.
- **Diversity vs. duplication** — removing samples to lower the duplicate rate also reduces `mean_diversity`, creating a natural trade-off.
- **Silhouette + Davies-Bouldin** — both are included in the separability formula. Optimising for one alone moves the other in the wrong direction.
- **Noise rate exponent** — the 1.5 exponent in the annotation score makes marginal gaming near the 10% boundary worth very little.
- **Split seed is fixed and published** — the test set composition is locked, so accuracy on the confusability metric cannot be inflated by controlling which samples land in the test set.
