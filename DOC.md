# Stage 1 - Classification Baselines

## 1. Project context

The broader project investigates **Flow Matching as a layer for discriminative computer-vision systems**. Conventional Flow Matching is most familiar as a generative method: it learns a time-dependent velocity field that transports samples from a source distribution to a target distribution. The project presentation proposes adapting this idea to classification, first as a last layer and later before a conventional classifier.

The planned progression is:

1. **Stage 1 - baselines:** build reliable classifiers on frozen pretrained representations, with no Flow Matching component.
2. **Stage 2 - Flow Matching as the last layer:** compare an FM-based output mechanism with the selected prototype baseline, including standard versus rolled-out training.
3. **Stage 3 - Flow Matching before the last layer:** compare an FM transformation followed by a linear classifier with the Stage 1 linear probe.
4. **Stage 4 - extensions:** consider structured tasks such as segmentation and encoder fine-tuning.

Stage 1 is therefore the experimental foundation, not a preliminary Flow Matching implementation. Its job is to establish fair, reproducible reference points and reusable data/feature pipelines for the later FM comparisons.

## 2. Stage 1 goal

Build a reliable and reproducible classification pipeline using **frozen pretrained encoders** and **cached features**. Implement:

- the required **linear probe**; and
- exactly one prototype-based branch:
  - image-derived class prototypes, or
  - zero-shot CLIP text prototypes.

Only classifier/prototype logic operates on the cached representations. Encoder weights must never change. Stage 1 must make later claims about an FM layer credible: improvements or regressions should be attributable to the new layer rather than inconsistent data splits, preprocessing, feature extraction, or evaluation.

## 3. Decisions the group must make

The specification deliberately does not choose the following:

1. **Two datasets** from DTD, FGVC-Aircraft, and Oxford Flowers-102.
2. **One of those datasets** on which to run DINOv2 ViT-S/14 in addition to ResNet-18.
3. **One prototype branch:** image-derived prototypes or zero-shot CLIP.

These are decision gates, not implementation details. They should be fixed before running the experiment grid and recorded with a short rationale. The prototype choice determines the baseline carried into Stage 2; the linear-probe setting is carried into Stage 3.

## 4. Experimental protocol

### 4.1 Datasets and official splits

Choose two datasets:

| Dataset | Classes | Approximate images | Special rule |
|---|---:|---:|---|
| DTD | 47 | 5,640 | Use official partition 1 |
| FGVC-Aircraft | 100 | 10,000 | Use the `variant` annotation level |
| Oxford Flowers-102 | 102 | 8,000 | Use official splits |

For both selected datasets:

- use all classes;
- preserve the official training, validation, and test splits;
- do not merge training and validation;
- use the complete validation split for model selection when needed; and
- use the complete test split only for final evaluation.

Methods using labeled training examples must be evaluated at `K in {5, 10, full}`, where `K` is the number of training images per class. For 5-shot and 10-shot experiments, construct balanced subsets from the official training split with subset seeds 0, 1, and 2. The full setting uses the entire official training split.

This separation matters. Subset seeds measure sensitivity to which few examples are available; initialization seeds measure optimization variability when the data are fixed. They must not be mixed during aggregation.

### 4.2 Frozen encoders

Use the preprocessing associated with each public pretrained checkpoint.

| Encoder | Where used | Required representation |
|---|---|---|
| ImageNet-1K pretrained ResNet-18 | Both selected datasets | 512-dimensional feature before the final classification layer |
| DINOv2 ViT-S/14 | One selected dataset | Final class-token representation |
| CLIP RN50 | Only for zero-shot CLIP | Frozen image and text embeddings |

All encoder parameters remain frozen. Extract and cache training, validation, and test features before classifier training. Caching is a scientific-control measure as well as an efficiency measure: every downstream baseline for a dataset-encoder pair should see exactly the same representation of each example.

A useful cache should include features, labels, stable sample identifiers, split name, class-index mapping, checkpoint identity, preprocessing identity, feature dimensionality, and extraction configuration. Cache validation should check completeness, ordering, finite values, and consistency with the official split.

## 5. Required baseline: linear probing

Given a frozen feature vector `z`, train only a multiclass affine classifier:

```text
s = Wz + b
```

The logits `s` are optimized with softmax cross-entropy; only `W` and `b` are trainable.

The suggested starting configuration is:

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Batch size | 64 |
| Maximum epochs | 200 |
| Checkpoint selection | Highest validation accuracy |

An extensive hyperparameter search is not required. The target is a stable, reasonable baseline. If this configuration behaves poorly, changes may be selected from validation results and must be reported. Test results must not guide those changes.

Run the linear probe with ResNet-18 on both datasets and with DINOv2 ViT-S/14 on the chosen dataset. For every dataset-encoder combination:

- 5-shot: three runs corresponding to subset seeds 0, 1, and 2;
- 10-shot: three runs corresponding to subset seeds 0, 1, and 2; and
- full: three runs with different classifier-initialization seeds.

Save per-epoch metrics. For one representative 10-shot run per dataset-encoder combination, report training and validation loss curves.

## 6. Choose one prototype branch

### 6.1 Option A: image-derived class prototypes

Use the same frozen ResNet-18/DINOv2 features and dataset coverage as the linear probe. Let `S_c` be the selected training examples for class `c`. Construct a unit-norm prototype by normalizing each feature, averaging within the class, and normalizing the mean:

```text
mu_c = normalize(mean_{i in S_c}(normalize(z_i)))
```

Normalize an evaluation feature and choose the class with greatest cosine similarity:

```text
y_hat = argmax_c cos(z, mu_c)
```

Evaluate 5-shot and 10-shot with subset seeds 0, 1, and 2. Evaluate the full-data setting once. This branch is naturally compatible with Stage 2 because the image-derived prototypes provide a feature-space target structure for an FM-based last layer.

Important implementation detail for later: the required order is **normalize each feature, average, normalize the mean**. Averaging raw features first is not equivalent.

### 6.2 Option B: zero-shot CLIP

Use frozen CLIP RN50 image and text encoders. Build one text prototype for each class with the prescribed dataset prompt:

- DTD: `a photo of a {class} texture`
- FGVC-Aircraft: `a photo of a {class} aircraft`
- Flowers-102: `a photo of a {class} flower`

Normalize image and text embeddings and classify by maximum cosine similarity:

```text
y_hat = argmax_c cos(z, t_c)
```

This branch uses no labeled training images, so it produces one result per selected dataset rather than separate 5-shot, 10-shot, and full trained models. Its accuracy can be displayed as a horizontal reference in the training-size plot. Class-name normalization used in prompts must be deterministic and documented.

## 7. Evaluation and required outputs

The primary metric is **top-1 accuracy on the complete official test split**.

Aggregation rules:

- 5-shot and 10-shot: mean and standard deviation over the three subset seeds;
- full linear probe: mean and standard deviation over three classifier-initialization seeds;
- full image-prototype: one run; and
- zero-shot CLIP: one run per dataset.

Stage 1 must produce:

1. **Accuracy table** covering every implemented dataset, encoder, training-set size, and baseline.
2. **Accuracy versus training-set size plot** for 5-shot, 10-shot, and full, with error bars where repeated runs exist. Zero-shot CLIP may appear as a horizontal reference.
3. **Training and validation loss curves** for one representative 10-shot linear-probe run per dataset-encoder combination. The discussion should address stability and overfitting.
4. **One row-normalized confusion matrix per selected dataset** for a representative setting chosen to expose meaningful errors.
5. **Feature visualizations** for a readable subset of about 8-10 classes using PCA, t-SNE, or another justified 2D projection.

For comparable feature plots on a dataset, hold the selected classes, test examples, and colors fixed. Plot test features with image-derived prototypes for ResNet-18/DINOv2 or with text prototypes for CLIP. When prototypes are shown, fit the projection jointly to all displayed image features and prototypes. These plots are qualitative views of geometry; they are not substitutes for classification metrics.

## 8. Expected workflow

The intended dependency order is:

```text
Fix decisions and seeds
        -> prepare official splits and few-shot manifests
        -> extract and validate frozen feature caches
        -> train/evaluate linear probes on cached features
        -> evaluate the chosen prototype branch
        -> select configurations using validation only
        -> perform final full-test evaluation
        -> aggregate metrics and create required figures
        -> verify reproducibility and preserve Stage 2/3 artifacts
```

Test evaluation is deliberately late in this workflow. Model selection, debugging, and any modest hyperparameter adjustment should use training/validation evidence without repeatedly consulting the test set.

## 9. Flow Matching concepts relevant to the project

These concepts explain the later stages, although Stage 1 does not implement them.

### 9.1 Probability paths, flows, and velocity fields

A flow-based generative model transports a simple source distribution `p_0` to a target distribution `p_1 = q` along a probability path `(p_t)`, with `t in [0,1]`. Rather than directly learning the time-indexed flow map, Flow Matching learns a velocity field `u_t(x)`. Sampling follows the ordinary differential equation

```text
dX_t/dt = u_theta(t, X_t),    X_0 ~ p_0.
```

An ODE solver (for example midpoint or Euler integration) numerically follows the learned field to produce a target sample.

### 9.2 Conditional Flow Matching

For a linear conditional path between paired endpoints,

```text
X_t = (1 - t) X_0 + t X_1,
```

the conditional target velocity is constant:

```text
u_t(X_t | X_0, X_1) = X_1 - X_0.
```

Training can regress a neural velocity field toward this target:

```text
L_CFM = E[ ||u_theta(t, X_t) - (X_1 - X_0)||^2 ].
```

The marginalization result underlying Flow Matching says that the conditional regression objective has the same parameter gradient as matching the generally intractable marginal velocity. The learned marginal velocity at a point is a conditional average of the velocities of training paths passing through that point.

### 9.3 Couplings

A coupling is the joint distribution used to pair source and target samples while preserving the desired source and target marginals. Independent pairing is simple, but its straight conditional paths can cross or nearly cross. At a given position and time, an unconditioned velocity model cannot output multiple conflicting velocities, so regression produces a conditional average. As this average changes over space and time, the learned trajectories can become curved even though individual training interpolations were straight.

This is especially relevant to the lab project because discriminative tasks are typically **paired**: an input representation is associated with a label, class representation, or prototype. The way those pairs are defined will shape any later FM layer and must be compared against the Stage 1 prototype baselines.

### 9.4 Rectified Flow and rolled-out training

Curved trajectories require more numerical-solver steps and therefore more network evaluations. Rectified Flow (also called reflow in the iterative procedure) attempts to straighten trajectories:

1. train an initial Flow Matching model under an initial coupling;
2. sample a source point and integrate the trained model to obtain its generated endpoint;
3. treat each source point and its model-generated endpoint as a new, model-induced coupling; and
4. retrain a velocity model on these pairs, repeating if desired.

ODE uniqueness means deterministic flow trajectories do not cross at the same space-time point. Training on the induced coupling therefore removes many conflicting crossing velocities and tends to yield straighter trajectories. Straighter paths can be integrated accurately with fewer steps, though enforcing straightness can trade some model flexibility or quality.

The project presentation's Stage 2 comparison of **standard FM training versus rolled-out training** aligns with this idea. Stage 1 must remain clean and reproducible so that any later benefit from the FM/reflow mechanism can be measured against strong prototype and linear baselines.

## 10. Common failure modes to prevent

- Accidentally fine-tuning an encoder or leaving it in training mode.
- Using preprocessing that does not match the pretrained checkpoint.
- Merging training and validation, or choosing settings using test accuracy.
- Sampling few-shot examples globally rather than exactly `K` per class.
- Comparing methods with different few-shot manifests for the same seed.
- Treating subset-seed variation and classifier-initialization variation as the same source of randomness.
- Re-extracting features in ways that change sample order or representations across baselines.
- Computing prototypes with the wrong normalization order.
- Reporting error bars for single-run results or omitting them for three-run results.
- Fitting a 2D projection separately for images and prototypes, making their locations incomparable.
- Comparing feature plots with different examples/classes/colors and inferring geometry from presentation changes.
- Treating PCA/t-SNE plots as quantitative evidence of classifier quality.
- Adding Flow Matching during Stage 1, which would contaminate the baseline phase.

## 11. Definition of done

Stage 1 is done when:

- the two datasets, DINOv2 dataset, and prototype branch are fixed and documented;
- all encoders are demonstrably frozen and all downstream work uses validated cached features;
- the required linear-probe grid and selected prototype baseline follow the exact split/seed protocol;
- final top-1 accuracies are aggregated correctly;
- the accuracy table, training-size plot, training curves, confusion matrices, and feature visualizations are complete;
- a representative run can be reproduced from recorded configuration and cached artifacts; and
- the selected prototype artifacts are ready for Stage 2 while the linear-probe artifacts are ready for Stage 3.

## 12. Reviewed sources

- `ref/intro to flow matching.pdf` - Flow Matching tutorial covering flows, velocity fields, conditional Flow Matching, couplings, geometric variants, rectification, adaptation, and discrete flows.
- `ref/stage_1.pdf` - authoritative Stage 1 requirements and evaluation protocol.
- `ref/flow matching as a layer.pdf` - project roadmap connecting the Stage 1 baselines to FM last-layer and pre-classifier stages.
- [A Visual Introduction to Rectified Flows](https://alechelbling.com/blog/rectified-flow/) - visual explanation of coupling-induced curvature and the reflow procedure.
