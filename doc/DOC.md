# Project Documentation - Stages 1, 2 and 3

This document covers the protocol for all three stages. **Part I** is Stage 1: classification baselines on frozen pretrained encoders, with no Flow Matching component. **Part II** is Stage 2: the Flow Matching layer added on top of the selected prototype baseline. **Part III** is Stage 3: a Flow Matching transformation inserted *before* the frozen Stage 1 linear classifier.

Stages 1 and 2 are measured. Stage 3 is implemented and verified to run, but has not yet been run on the real feature caches - no Stage 3 number exists yet.

Measured results and their discussion are in `RESULTS.md`. Per-stage task tracking is in `TODO_stage1.md`, `TODO_stage2.md` and `TODO_stage3.md`.

---

# Part I - Stage 1: Classification Baselines

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
- both prototype-based branches:
  - image-derived class prototypes; and
  - zero-shot CLIP text prototypes.

Only classifier/prototype logic operates on the cached representations. Encoder weights must never change. Stage 1 must make later claims about an FM layer credible: improvements or regressions should be attributable to the new layer rather than inconsistent data splits, preprocessing, feature extraction, or evaluation.

## 3. Comprehensive experiment decisions

This implementation deliberately evaluates every choice exposed by the specification:

1. all three datasets: DTD, FGVC-Aircraft, and Oxford Flowers-102;
2. both ResNet-18 and DINOv2 ViT-S/14 on every dataset; and
3. both image-derived prototypes and zero-shot CLIP.

This is a superset of the minimum required grid. Image-derived prototypes remain the natural Stage 2 handoff, while the linear-probe settings remain the Stage 3 references.

## 4. Experimental protocol

### 4.1 Datasets and official splits

Evaluate all three datasets:

| Dataset | Classes | Approximate images | Special rule |
|---|---:|---:|---|
| DTD | 47 | 5,640 | Use official partition 1 |
| FGVC-Aircraft | 100 | 10,000 | Use the `variant` annotation level |
| Oxford Flowers-102 | 102 | 8,000 | Use official splits |

For every dataset:

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
| ImageNet-1K pretrained ResNet-18 | All three datasets | 512-dimensional feature before the final classification layer |
| DINOv2 ViT-S/14 | All three datasets | Final class-token representation |
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

Run the linear probe with both ResNet-18 and DINOv2 ViT-S/14 on all three datasets. For every dataset-encoder combination:

- 5-shot: three runs corresponding to subset seeds 0, 1, and 2;
- 10-shot: three runs corresponding to subset seeds 0, 1, and 2; and
- full: three runs with different classifier-initialization seeds.

Save per-epoch metrics. For one representative 10-shot run per dataset-encoder combination, report training and validation loss curves.

## 6. Prototype branches

Run both branches below. This exceeds the minimum requirement but provides complete coverage of the available choices.

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

This branch uses no labeled training images, so it produces one result per dataset rather than separate 5-shot, 10-shot, and full trained models. Its accuracy can be displayed as a horizontal reference in the training-size plot. Class-name normalization used in prompts must be deterministic and documented.

The implementation also reports a supplementary prompt-ensemble variant. It averages five individually normalized, dataset-specific text embeddings per class and then normalizes the resulting class prototype. This does not replace the prescribed single-prompt Stage 1 baseline and is never selected using test labels. OpenAI RN50 weights are loaded with QuickGELU, matching the pretrained checkpoint, and the cache path is versioned so incompatible embeddings cannot be reused.

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
4. **One row-normalized confusion matrix per dataset** for a representative setting chosen to expose meaningful errors.
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

These concepts explain the later stages, although Stage 1 does not implement them. Part II below is the concrete Stage 2 instantiation of everything in this section: the "coupling" becomes `(image feature, its class prototype)`, and the "rolled-out training" of section 9.4 becomes one of the two objectives being compared.

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

- all three datasets, both visual encoders, and both prototype branches are documented;
- all encoders are demonstrably frozen and all downstream work uses validated cached features;
- the complete linear-probe grid and both prototype baselines follow the exact split/seed protocol;
- final top-1 accuracies are aggregated correctly;
- the accuracy table, training-size plot, training curves, confusion matrices, and feature visualizations are complete;
- a representative run can be reproduced from recorded configuration and cached artifacts; and
- the selected prototype artifacts are ready for Stage 2 while the linear-probe artifacts are ready for Stage 3.

## 12. Reviewed sources

- `ref/intro to flow matching.pdf` - Flow Matching tutorial covering flows, velocity fields, conditional Flow Matching, couplings, geometric variants, rectification, adaptation, and discrete flows.
- `ref/stage_1.pdf` - authoritative Stage 1 requirements and evaluation protocol.
- `ref/stage_2.pdf` - authoritative Stage 2 requirements: standard vs. rolled-out FM training, inference, and required figures.
- `ref/flow matching as a layer.pdf` - project roadmap connecting the Stage 1 baselines to FM last-layer and pre-classifier stages.
- [A Visual Introduction to Rectified Flows](https://alechelbling.com/blog/rectified-flow/) - visual explanation of coupling-induced curvature and the reflow procedure.

---

# Part II - Stage 2: Flow Matching to Class Prototypes

## 13. Stage 2 goal

Add a Flow Matching layer on top of the frozen prototype-based classifier from Stage 1. Given a frozen image feature `z_i`, the FM model transports it toward the fixed prototype `p_{y_i}` of its class; the transported point is then classified by the same cosine rule Stage 1 used.

Three questions define the stage:

1. does the FM layer beat the Stage 1 prototype baseline;
2. does standard FM training beat rolled-out training; and
3. how does the learned transformation change the geometry of the feature space.

This is Flow Matching used discriminatively rather than generatively. The source distribution is the encoder's feature distribution, the target is the finite, fixed set of class prototypes, and the coupling is supervised: each training feature is paired with its own class prototype. Section 9.3 explains why that pairing matters - at test time the network cannot know the label, so it necessarily learns a conditional average of the velocities of all training paths passing through a point, which is `E[p_y | z]`. Improvement therefore comes from denoising toward the conditional-mean prototype, not from moving each point onto its correct prototype.

## 14. What must be inherited unchanged from Stage 1

The entire value of Stage 2 is a controlled comparison, so `ΔAcc = Acc_FM - Acc_baseline` is only interpretable if everything except the FM layer is held fixed. Stage 2 reuses, without recomputation:

- the same datasets, official splits, and test sets;
- the same frozen feature caches - no encoder is ever re-run;
- the same class prototypes, loaded from Stage 1's saved `prototypes.pt`, never rebuilt;
- the same K-shot subsets, re-derived with the identical seeded `balanced_indices` logic, so the FM training pairs come from exactly the images that produced Stage 1's prototype for that class and seed; and
- the same subset seeds and initialization seeds.

A missing Stage 1 artifact must fail loudly rather than trigger a silent recompute under possibly different conditions. This is the single most important implementation constraint in the stage.

## 15. Standard FM training

For each training feature `z_i` and its class prototype `p_{y_i}`, sample `t ~ U(0,1)` and form the interpolation and target velocity:

```text
z_t = (1 - t) z_i + t p_{y_i}
u_i = p_{y_i} - z_i
```

Train the velocity network by regression:

```text
L_FM = || v_theta(z_t, t) - u_i ||^2
```

Standard FM supervises the velocity prediction directly, at randomly sampled points along the ideal straight interpolation. Note the mismatch this creates with inference: at test time the network is repeatedly evaluated on states generated by its *own* previous predictions, which are not guaranteed to lie on any ideal path.

Because the objective contains no reference to a step count, **one standard network per setting** is trained and evaluated at every `T`.

## 16. Inference and rolled-out training

At inference, start from the test feature `z_hat_0 = z` and take `T` Euler steps:

```text
z_hat_{k+1} = z_hat_k + (1/T) * v_theta(z_hat_k, k/T),    k = 0, ..., T-1
```

Evaluate `T` in {4, 12}. Classify `z_hat_T` with the same prototypes and cosine rule as Stage 1.

Rolled-out training addresses the train/inference mismatch above. Starting from `z_hat_0 = z_i`, it applies the same full `T`-step sequence used at inference, with the same network, and optimizes only the endpoint:

```text
L_roll = || z_hat_T - p_{y_i} ||^2
```

Gradients propagate through the complete sequence of `T` velocity predictions. This exposes the network to exactly the self-generated intermediate states it will see at test time, but supervises the trajectory far more weakly - nothing constrains the individual per-step velocities.

Two consequences follow directly:

- **`T` is part of the objective**, so a rolled-out network must use the same `T` at training and inference, and **two rolled-out networks per setting** are needed, one per `T`. A `T=4` rolled-out network is never evaluated at `T=12`.
- **Cost scales with `T`**: backpropagating through `T` sequential network evaluations is substantially more expensive than standard FM's single random-`t` sample.

The architecture and all main training choices are held fixed between the two objectives; only the loss differs.

## 17. Velocity network

A small MLP suffices, and the specification explicitly discourages architecture search - the goal is stable training and a fair comparison, not a tuned model.

| Setting | Value |
|---|---|
| Hidden layers | 2 |
| Width | 512 |
| Activation | SiLU |
| Time conditioning | scalar `t` concatenated to the input feature |
| Output dimension | equal to the feature dimension (512 ResNet-18, 384 DINOv2, 1024 CLIP RN50) |
| Optimizer | AdamW, learning rate `1e-3`, weight decay `1e-4` |
| Batch size | 64 |
| Maximum epochs | 200 |
| Checkpoint selection | highest validation accuracy |

Checkpoint selection deserves care, because it is the one place Stage 2 uses a signal the Stage 1 prototype baseline never needed. Run the *full* Euler rollout on the validation split, classify by cosine similarity to the Stage 1 prototypes, and checkpoint on best validation accuracy - mirroring Stage 1's rule rather than inventing an unrelated criterion. For rolled-out training this is unambiguous: accuracy at that network's own `T`. For standard training, where one network is later judged at every `T`, checkpoint on the mean of validation accuracy across `T_values`, and record the per-`T` accuracies separately so a per-`T` result row never reports the averaged number as though it were that `T`'s own.

## 18. Stage 2 experimental protocol

Same training-set sizes, subsets, and seeds as Stage 1: `K in {5, 10, full}`. For every setting, compare five things:

- the Stage 1 prototype baseline;
- standard FM at `T=4` and `T=12`; and
- rolled-out FM at `T=4` and `T=12`.

This is three trained networks per (dataset, encoder, K, seed) - one standard, two rolled-out. Across 3 datasets x 2 encoders x 3 K values x 3 repetitions that is 162 trainings producing 216 result rows.

The repetition protocol follows Stage 1: three runs per 5-shot and 10-shot setting using subset seeds {0,1,2}, three runs per full setting using initialization seeds {0,1,2}. One deliberate deviation is worth recording. Stage 1's *image-prototype* full setting is a single closed-form run and therefore has no error bar, while Stage 2's full setting uses three initialization seeds, because an FM network - unlike a class mean - has an initialization. The baseline column in the comparison table consequently carries no standard deviation at `full`.

The primary metric is top-1 accuracy on the complete official test split, computed only after training and checkpoint selection are finished.

## 19. Required outputs

1. **Classification results.** Baseline vs. standard and rolled-out FM, both `T` values, all `K`. Report top-1 accuracy and `ΔAcc = Acc_FM - Acc_baseline` against the *matching* baseline (same dataset, encoder, K). A table and an accuracy-versus-K plot with error bars.
2. **Training curves.** Representative training-loss curves for both objectives, to establish that training is stable and both reach reasonable solutions. Standard-FM loss (velocity MSE at random `t`) and rolled-out loss (endpoint MSE after a full rollout) are different quantities on different scales and must not be compared by magnitude.
3. **Feature-space visualizations.** For a readable subset of classes, compare original features, features after standard FM, and features after rolled-out FM, together with the corresponding prototypes. Same test examples and class colors across the compared plots. The projection must be fit **jointly** over every feature set and prototype shown, so the views share one coordinate frame - and the panels must then also share axis limits, since autoscaling each panel after a joint fit reintroduces exactly the incomparability the joint fit removed. PCA or t-SNE.
4. **Flow trajectories.** For a few representative test examples, plot intermediate FM states with the original feature, final transported feature, and target prototype. PCA is recommended here specifically because a trajectory has to stay interpretable as a path, which t-SNE does not preserve.

5. **Optional extensions** (both implemented; see `TODO_stage2.md` for where):
   - **The flow in reverse.** Start from the prototypes at `t=1` and integrate the same field backward. Report it quantitatively as a *recovery rate* - does each reverse-flowed prototype land nearest to its own class's mean test feature? - against the untouched forward prototype's recovery rate as the **pre-transport reference**. That reference is additionally a *ceiling* only on the image branch, where the prototype already is the class image mean; on the CLIP branch reverse recovery legitimately exceeds it, because a text prototype does not start out inside its own class's image cloud. Two things must be said: the backward pass evaluates the field on a shifted time grid, so it is not the exact inverse even for a perfect field; and the forward flow is a deliberate contraction, so it is not invertible in principle. This recovers a plausible pre-image, not the original point.
   - **Samples and prototypes at intermediate flow times.** Classify and measure *every* intermediate Euler state rather than only the endpoints. The resulting accuracy-versus-`t` curve is self-anchoring at both ends - `t=0` is the untouched feature, so its accuracy is the baseline classifier, and `t=1` is the reported FM result - which makes it ΔAcc unrolled, and makes it checkable: assert the `t=0` value against the saved baseline rather than trusting it. Track the **margin** (cosine to the true prototype minus the best competing one) alongside the distance to the prototype: distance falling while margin stays flat means the flow is contracting everything indiscriminately rather than discriminating, which is the failure mode a pure contraction metric cannot see.

## 20. Design decisions the specification leaves implicit

Four choices are not fixed by `ref/stage_2.pdf` and must be stated explicitly, because each changes what the numbers mean.

- **FM operates on L2-normalized features.** Stage 1's prototypes are already unit-norm (`normalize -> mean -> normalize`), so interpolating between a raw-scale feature and a unit-norm prototype would make the straight-line path geometrically arbitrary. Both endpoints live on the same unit sphere.
- **One standard network, two rolled-out networks, per setting.** Derived in sections 15 and 16. Getting it backwards either doubles standard FM's cost for no benefit or, worse, evaluates a rolled-out network at a `T` it was never trained for.
- **Checkpoint selection uses the validation split**, which the Stage 1 prototype baseline never needed. This is permitted - Stage 1 allows the full validation split for model selection - but it does mean the FM side of the comparison has an advantage the baseline side does not.
- **`ΔAcc` conflates two things.** It measures "added an FM layer" together with "added a trained, validation-selected ~0.8M-parameter head", against a baseline with zero trained parameters. That is what the specification asks for, and it should be presented plainly rather than as a clean isolation of the FM mechanism.

## 21. Extension: the zero-shot CLIP branch

`ref/stage_1.pdf` asks each group to implement the linear probe plus **one** of the two prototype branches, and states that the selected branch carries into Stage 2. This project implemented both branches, so Stage 2 on the image-prototype branch is the required deliverable and the CLIP branch is an optional extension.

Applying FM to CLIP text prototypes requires switching *both* endpoints into CLIP's joint image-text space. CLIP text prototypes do not live in the ResNet-18/DINOv2 feature space that `z_i` belongs to, so the straight line between them is otherwise undefined. With CLIP RN50 supplying the image embeddings as well, the construction is identical to the image branch.

Two properties of that branch differ, and must be stated whenever its numbers are shown:

- **The FM layer is not zero-shot.** It trains on labeled `(z_i, t_{y_i})` pairs, converting zero-shot CLIP into a K-shot method. The zero-shot baseline is a single number per dataset with no K axis, so `ΔAcc` against it compares a K-shot trained model against one that used no labels - unlike the image branch, where baseline and FM see exactly the same images.
- **A K-shot control is therefore required** to interpret it: image-derived prototypes built from the same CLIP features and the same K-shot subsets, closed-form and with zero trained parameters. FM beating that control is evidence about the FM layer; FM beating zero-shot is mostly evidence about having labels.

Text prototypes are also K-independent - unlike image prototypes they do not change with the subset or the seed, so the transport target is identical across every setting.

The completed CLIP-branch rerun sharpens the interpretation. Against the K-shot control, the mean FM accuracy is higher in 8 of 36 cells, but paired seed-level 95% intervals support only 6 FM wins; 28 cells favor the control and 2 show no measurable difference. Validation-selected early stopping of the flow helps 17 of 18 tested `(dataset, K, objective)` conditions (mean +.0239, best +.0520), but it does not turn any control-relative loss into a win. The large gains over zero-shot therefore remain gains from adding labels, not evidence that FM beats the like-for-like control.

## 22. Stage 2 failure modes to prevent

In addition to every Stage 1 failure mode in section 10:

- Recomputing Stage 1 prototypes instead of loading them, or building FM training pairs from a different subset than the prototype was built from.
- Training standard FM separately per `T`, or evaluating a rolled-out network at a `T` it was not trained for.
- Detaching anywhere inside the rollout, which silently reduces rolled-out training to something that is not backpropagation through the full sequence.
- Comparing standard and rolled-out loss values by magnitude; they are different objectives.
- Reading an identical `T=4` / `T=12` column as a copy-paste error. The ideal velocity is constant along the path, so a well-learned field gives a `T`-independent endpoint. This is the expected result and should be explained proactively.
- Fitting the feature-space projection separately per view, or fitting it jointly and then letting each panel autoscale independently. Both destroy the comparability the figure exists to provide.
- Reporting a three-run mean and standard deviation for a setting where the three runs are identical by construction. Flowers-102 at `K=10` is the concrete case: the official train split holds exactly 10 images per class, so every subset seed selects the same images.
- Letting test accuracy influence training or checkpoint choices.

## 23. Stage 2 definition of done

Stage 2 is done when:

- the full grid has been trained and evaluated on the identical Stage 1 splits, seeds, and prototypes;
- `ΔAcc` against the matching Stage 1 prototype baseline is reported for every combination, with correct aggregation;
- the accuracy-versus-K plot, training curves, feature-space visualizations, and flow-trajectory visualizations are complete;
- the standard-versus-rolled-out comparison and the main observations are written up; and
- the best FM checkpoints and per-run configuration are preserved as the Stage 3 starting point.

Measured outcomes against these criteria are in `RESULTS.md`; remaining open items are tracked in `TODO_stage2.md`.

All five criteria are met. Two results from the non-required analyses have to travel with the required table rather than be filed separately, because they change what it may be claimed to show: the paired 95% CI excludes zero in 48 of 72 cells (24/24 Aircraft, 7/24 DTD, 17/24 Flowers-102), although 8 Flowers-102 K=10 intervals are degenerate because there is only one effective subset; excluding those leaves 40 non-degenerate effects. On the largest-gain setting (Aircraft/DINOv2), a plain supervised MLP with no flow reaches 97% of the FM gain over the baseline. `STAGE2_COMPLIANCE.md` maps every specification clause to where it is satisfied and lists everything that is deliberately *not* part of the required experiment.

---

# Part III - Stage 3: FM Before a Linear Classifier

## 24. Stage 3 goal

Stage 3 starts from the Stage 1 linear-probe setting and inserts an FM transformation *before* the
pretrained linear classifier:

```
z --FM--> z_hat --frozen linear classifier--> s
```

where `z` is the frozen image-encoder feature, `z_hat` the feature after the FM transformation, and
`s` the vector of classifier logits. The linear classifier is trained first, exactly as in Stage 1,
and then kept frozen. The FM is initialized close to the identity, so that before Stage 3 training
the complete system behaves like the original linear probe.

The question is whether FM can transform the frozen encoder features into a representation that is
better handled by the *existing* classifier - not by a better classifier.

This is a harder question than Stage 2 asked. Stage 2 replaced a closed-form prototype rule, which
had obvious headroom; the measured result was that the FM layer closed a median 59% of the gap to
the linear probe without surpassing it. Stage 3 targets that probe itself, which `RESULTS.md`
records as the strongest method in the project at every full-data setting except saturated
Flowers-102/DINOv2. There is no reason to expect a large gain, and the honest framing is set out in
section 34.

## 25. What must be inherited unchanged

From Stage 1: the official train/validation/test splits, the cached frozen features, the seeded
K-shot subset selection, the trained linear head, and the feature transform that head expects.

From Stage 2: the velocity network architecture, the Euler integrator
`z_hat_{k+1} = z_hat_k + (1/T) v(z_hat_k, k/T)`, the checkpoint-resume convention, and the
deterministic per-batch sampling of the flow time `t`.

Nothing is recomputed. `06_fm_before_classifier.ipynb` imports neither `torchvision` nor an encoder,
and it never calls the Stage 1 training routine.

## 26. The frozen classifier and the space the FM operates in

This is the detail most likely to be got wrong, and it has no analogue in Stage 2.

Stage 1 selected a feature transform per run - `l2` or `standardize` - by validation accuracy, and
trained the linear head on *transformed* features. The head therefore lives in the transformed
space, and the `z` in the Stage 3 diagram is a transformed feature, not a raw encoder output. The FM
operates entirely inside the classifier's own input space.

Reproducing that space requires the transform mode and, for `standardize`, the mean and standard
deviation computed on the K-shot training subset. Those statistics are saved in Stage 1's
`checkpoints/final.pt` and **not** in `best_linear_head.pt`, so `final.pt` is the artifact to load
even though both carry the same validation-selected weights.

Two guards protect this, and they are the reason a wrong reconstruction cannot be mistaken for a
Stage 3 result:

- The transform statistics are recomputed from the re-derived subset and compared against Stage 1's
  saved ones.
- The complete Stage 1 inference path - cache, subset, transform, head - is replayed on the test
  split and required to reproduce Stage 1's saved `test_accuracy` to within `1e-6`, for every
  (dataset, seed) cell, before anything is trained.

The tolerance is not decoration. Stage 1 stored accuracies as float32 means, so an exact comparison
against a float64 recomputation fails by roughly `1e-8` even when everything is correct.

## 27. Identity initialization, and what it costs

The specification asks for an FM initialized close to the identity. Zeroing the velocity network's
output layer makes `v(z, t) = 0` everywhere, so each Euler step adds exactly zero and the rollout is
the identity map bit for bit. The system does not start *close* to the linear probe; it starts *at*
it. This reuses the `zero_init_output` flag that existed in Stage 2 as a never-run pilot (`04` 6b).

Two consequences follow, and both must be stated wherever Stage 3 results appear.

**Optimization.** At initialization the gradient reaches only the output layer, because every
earlier layer's gradient is multiplied by the zeroed output weights; the hidden layers unblock after
the first update. This is standard zero-initialized-residual-branch behaviour, and it makes the
first several epochs nearly flat. Stage 3 therefore raises `early_stopping_patience` from Stage 2's
25 to 40 and `lr_patience` from 10 to 15. With Stage 2's values a run could stop before leaving the
identity.

**Model selection.** Epoch 0 - the untrained, identity FM - is evaluated and checkpointed like any
other epoch. Because that state *is* the linear probe, the selection rule becomes "keep the FM only
if it helps on validation". The price is that **validation `ΔAcc` is `>= 0` by construction and
therefore carries no information**; only test `ΔAcc` does. A run whose selected checkpoint is epoch 0
learned nothing usable, and the count of such runs is reported rather than absorbed into a mean.

## 28. Strategy 1 - end-to-end rolled-out classification training

For each training feature `z`, run the complete `T`-step rollout to obtain `z_hat`, pass `z_hat`
through the frozen classifier, and minimize

```
L = CE(W z_hat + b, y)  +  lambda_disp * ||z_hat - z||^2  +  (lambda_vel / T) * sum_k ||v(z_k, t_k)||^2
```

backpropagating through all `T` sequential steps and updating only the FM parameters. Both penalty
weights default to zero, so the required main result is the unregularized objective.

The penalties are not decoration either. When Stage 1 selected the `l2` transform, the head was only
ever fit on unit-norm features. An unregularized rollout is free to move `z_hat` off that sphere into
regions where the head was never fit and its logits are unconstrained - so the FM can win training
accuracy by exploiting the classifier rather than by improving the representation. Penalizing
displacement is the direct guard. Whether this actually happens is measured in the notebook's
Section 14, which reports accuracy and mean `||z_hat - z|| / ||z||` side by side, rather than argued.

## 29. Strategy 2 - classifier-guided targets and standard FM training

The frozen classifier is used to construct an explicit target, and the FM is then trained by ordinary
conditional flow matching. For each training feature `z`:

1. run `z` through the current FM to obtain `z_hat`;
2. pass `z_hat` through the frozen classifier and compute the classification loss;
3. take one or more gradient steps of that loss **with respect to `z_hat`** - in feature space, never
   in parameter space - to reach a nearby improved representation `z_hat'`;
4. treat `z` as the source and `z_hat'` as the target;
5. perform a standard FM update between them: `t ~ U(0,1)`, `z_t = (1-t) z + t z_hat'`, target
   velocity `z_hat' - z`, loss `||v(z_t, t) - (z_hat' - z)||^2`;
6. recompute the targets as the FM changes during training.

The scheme is bootstrapped: each refresh moves the target a little further downhill in classifier
loss and the FM chases it. Nothing pins the target to a fixed endpoint the way Stage 2's class
prototypes did, which is why the step constraint carries real weight here.

**The step size is a fraction of the mean training-feature norm, not an absolute distance.** Because
Stage 1 chooses the transform per run, an absolute step of 0.5 would be a 50% displacement under
`l2` (`||z|| = 1`) but roughly 2.5% under `standardize` (`||z|| ~ sqrt(384)`). One configured number
has to mean the same thing across datasets, so it is expressed relatively.

All three constraint modes the specification names are implemented, defaulting to `trust_region`:

| mode | behaviour | trade-off |
|---|---|---|
| `none` | `z_hat' = z_hat - eps * grad` | already-confident samples take naturally small steps, which is desirable, but raw gradient norms vary by orders of magnitude across samples |
| `unit` | each step has length exactly `eps` | scale-free, but moves already-correct samples as far as wrong ones |
| `trust_region` | raw steps, cumulative displacement clipped to `eps` | keeps the gradient's own scaling while capping the worst case |

## 30. Stage 3 experimental protocol

| | |
|---|---|
| Datasets | DTD, FGVC-Aircraft, Flowers-102 |
| Encoder | DINOv2 ViT-S/14 on every dataset (384-dim) |
| Training-set size | K = 10 |
| Euler steps | T = 12, fixed for training and inference in every method |
| Repetitions | Stage 1's subset seeds {0, 1, 2}, `init_seed = 0` |
| Methods compared | Stage 1 linear probe, end-to-end rollout, classifier-guided |

18 trained FM layers, plus 9 linear-probe baseline rows loaded from Stage 1 at zero training cost.

The specification asks for two datasets and one representative encoder per dataset. All three Stage 1
datasets are run because the extra cost is small and it keeps the Stage 3 table directly comparable
to the Stage 1 and Stage 2 tables; Flowers-102 doubles as a near-saturated control. DINOv2 is the
representative encoder everywhere because Stage 2 measured that encoder choice dominates every other
factor by +.14 to +.31 over ResNet-18.

Both strategies are checkpointed on the **same** quantity - validation top-1 accuracy of the complete
system `head(rollout(z_val))` - which is also the rule Stage 1 used to select the linear probe.
Selecting each method on its own training loss would compare a well-tuned method against a
badly-tuned one, since the two objectives are not commensurable.

## 31. Required outputs

- **Classification results.** Top-1 test accuracy for the linear probe and both Stage 3 methods on
  every dataset, mean +/- std over the three subset seeds, with `ΔAcc` against the matching probe.
  The comparison is paired at the seed level, so `ΔAcc` is a mean of per-seed differences rather
  than a difference of means computed over different data.
- **Training behaviour.** Representative training and validation curves for both methods.
- **Feature-space visualization.** Original `z` and transported `z_hat` under both methods, on the
  same test examples with the same class colors, with the 2D embedding fit jointly over all three
  feature sets and the panels sharing axis limits. Both PCA and t-SNE.
- **Paired significance** (house convention, beyond the specification): a percentile bootstrap CI
  over test examples and an exact McNemar test on the discordant pairs, per (dataset, method, seed).

## 32. Variant comparisons and the optional extension

The specification describes its two strategies as "structured starting points rather than fixed
recipes". Two sweeps, both on subset seed 0 only, so they supplement rather than replace the
three-seed result: the Strategy 1 regularization weights, reported jointly with the measured
displacement; and the four Strategy 2 knobs the specification names - step size, number of
target-improvement steps, constraint mode, and refresh period.

The optional extension unfreezes the classifier and optimizes it jointly with the FM. **It ships
with a control the specification does not ask for, because without it the comparison cannot support
a claim:** unfreezing adds the FM *and* extra training of the classifier at the same time, so a joint
run that beats the Stage 1 probe may just be a probe that trained longer. A `head_only` control
continues the same head for the same budget under the same optimizer and selection rule, with no FM
at all. Joint results are to be read against that control, not against Stage 1.

## 33. Stage 3 failure modes to prevent

In addition to the Stage 1 and Stage 2 failure modes in sections 10 and 22:

- Running the FM on raw encoder features when the frozen head was trained on transformed ones. The
  system would still run and produce plausible-looking numbers against a baseline that is not the
  Stage 1 probe.
- Loading `best_linear_head.pt` and reconstructing the `standardize` statistics from the wrong data,
  instead of loading `final.pt` where Stage 1 saved them.
- Comparing a replayed accuracy to Stage 1's saved value with exact equality. It will fail by about
  `1e-8` for correct code, because Stage 1 stored float32 means.
- Letting any gradient reach the classifier in the required experiments. Freezing at load time with
  `requires_grad_(False)` makes this structurally impossible rather than a matter of discipline.
- Quoting validation `ΔAcc` as evidence that Stage 3 helped. Epoch-0 checkpointing makes it
  non-negative by construction; only test `ΔAcc` is informative.
- Averaging away runs that selected epoch 0. Those FM layers are the identity and learned nothing;
  reporting the count is part of reporting the result.
- Reading a rising Strategy 2 training loss as divergence. The target moves at every refresh, so the
  loss is measured against a different quantity than it was an epoch earlier.
- Comparing Strategy 1 and Strategy 2 training losses by magnitude. One is a classification
  cross-entropy, the other a velocity regression MSE.
- Reporting the joint fine-tuning extension against the Stage 1 probe rather than against the
  head-only control.

## 34. Stage 3 definition of done

Stage 3 is done when:

- the required grid has been trained and evaluated on the identical Stage 1 splits, subsets, seeds
  and frozen classifiers, with both guards in section 26 passing;
- top-1 test accuracy and `ΔAcc` against the matching linear probe are reported for both methods on
  every dataset, together with the count of runs that selected epoch 0;
- training/validation curves and the jointly-embedded feature-space visualizations are complete;
- the two variant sweeps are reported; and
- the joint fine-tuning extension is reported against its head-only control.

**A negative result is a result.** The linear probe is the strongest method in this project, and
Stage 3 asks a frozen affine classifier to do better on a representation it was itself fit on. If
the FM layer does not beat it, that should be stated plainly with the mechanism explained, following
the Stage 2 precedent: "the FM layer is a real improvement over the prototype rule it replaces, not a
replacement for a discriminatively trained classifier." What would make the finding uninterpretable
is not a negative `ΔAcc` - it is a positive one obtained against a mis-reconstructed baseline, which
is what sections 26 and 27 exist to prevent.

Measured outcomes against these criteria will be recorded in `RESULTS.md`; remaining open items are
tracked in `TODO_stage3.md`.
