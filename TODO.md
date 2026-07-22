# Stage 1 TODO - Classification Baselines

Stage 1 establishes reliable, reproducible classification baselines on frozen pretrained features. **Do not add a Flow Matching component in this stage.**

## 0. Lock the experimental plan

- [ ] Choose **two** datasets from:
  - DTD (47 classes; use official partition 1)
  - FGVC-Aircraft (100 classes; use the `variant` annotation level)
  - Oxford Flowers-102 (102 classes)
- [ ] Choose which one of the two datasets will also use DINOv2 ViT-S/14.
- [ ] Choose exactly one prototype branch:
  - **Option A:** image-derived class prototypes; or
  - **Option B:** zero-shot CLIP RN50 with text-derived prototypes.
- [ ] Record the decisions and rationale in the experiment configuration/notes.
- [ ] Confirm dataset licenses, download locations, checkpoint identifiers, and required compute/storage.

## 1. Define reproducibility conventions

- [ ] Define a single experiment configuration format containing dataset, split, encoder, preprocessing, feature layer, baseline, shot count, subset seed, initialization seed, and training hyperparameters.
- [ ] Fix and document all random-number generators and deterministic settings used.
- [ ] Define stable class-index mappings from the official dataset metadata.
- [ ] Define run identifiers and output locations for cached features, checkpoints, metrics, and figures.
- [ ] Save environment/package versions and exact pretrained checkpoint identifiers.
- [ ] Add validation checks that prevent train/validation/test leakage.

## 2. Prepare datasets and splits

- [ ] Download and verify the two selected datasets.
- [ ] Preserve the official training, validation, and test splits; never merge training and validation.
- [ ] Apply dataset-specific rules:
  - [ ] DTD: official partition 1.
  - [ ] FGVC-Aircraft: `variant` annotation level.
- [ ] Use the full validation split only for model selection.
- [ ] Reserve the full test split for final evaluation only.
- [ ] For each selected dataset, create balanced per-class training subsets for:
  - [ ] 5-shot with subset seeds 0, 1, and 2.
  - [ ] 10-shot with subset seeds 0, 1, and 2.
  - [ ] Full training split.
- [ ] Validate subset balance, disjoint split membership, class coverage, and deterministic regeneration.

## 3. Extract and cache frozen features

- [ ] Implement/use checkpoint-specific preprocessing without alteration.
- [ ] Keep every encoder in evaluation mode with gradients disabled and parameters frozen.
- [ ] Extract and cache train/validation/test features once per dataset-encoder combination.
- [ ] ResNet-18:
  - [ ] Use the public torchvision ImageNet-1K pretrained checkpoint.
  - [ ] Extract the 512-dimensional representation before the final classifier.
  - [ ] Run on both selected datasets.
- [ ] DINOv2 ViT-S/14:
  - [ ] Use a public pretrained checkpoint.
  - [ ] Extract the final class-token representation.
  - [ ] Run on the one chosen dataset.
- [ ] If zero-shot CLIP is selected:
  - [ ] Use frozen CLIP RN50 image and text encoders.
  - [ ] Cache test image embeddings and class text embeddings.
- [ ] Store labels, sample identifiers, split, class mapping, preprocessing/checkpoint metadata, and feature dimensionality with each cache.
- [ ] Verify cache completeness, ordering, finite values, and reproducibility.

## 4. Required baseline - linear probe

- [ ] Train only a multiclass affine head, `s = Wz + b`, on cached features.
- [ ] Use softmax cross-entropy.
- [ ] Start with the suggested configuration:
  - AdamW
  - learning rate `1e-3`
  - weight decay `1e-4`
  - batch size 64
  - at most 200 epochs
  - select the checkpoint with highest validation accuracy
- [ ] If the suggested setup behaves poorly, adjust it using validation results only and document every change.
- [ ] Run ResNet-18 on both datasets at 5-shot, 10-shot, and full.
- [ ] Run DINOv2 ViT-S/14 on the chosen dataset at 5-shot, 10-shot, and full.
- [ ] Repetition protocol:
  - [ ] 5-shot: one run for each subset seed 0, 1, and 2.
  - [ ] 10-shot: one run for each subset seed 0, 1, and 2.
  - [ ] Full: three classifier-initialization seeds.
- [ ] Save per-epoch training/validation loss and accuracy.
- [ ] Select one representative 10-shot run for every dataset-encoder combination for loss-curve reporting.

## 5A. Prototype branch - Option A: image-derived prototypes

Complete this section only if Option A is selected.

- [ ] Use the same ResNet-18 and DINOv2 dataset-encoder combinations as the linear probe.
- [ ] L2-normalize every selected training feature.
- [ ] For each class, average its normalized features and L2-normalize the resulting class mean.
- [ ] L2-normalize evaluation features and classify by maximum cosine similarity.
- [ ] Evaluate 5-shot and 10-shot using subset seeds 0, 1, and 2.
- [ ] Evaluate full data once.
- [ ] Save class prototypes and their metadata for Stage 2.

## 5B. Prototype branch - Option B: zero-shot CLIP

Complete this section only if Option B is selected.

- [ ] Use frozen CLIP RN50 image and text encoders.
- [ ] Build one prompt per class using the required template:
  - DTD: `a photo of a {class} texture`
  - FGVC-Aircraft: `a photo of a {class} aircraft`
  - Flowers-102: `a photo of a {class} flower`
- [ ] Verify that official class names are converted to prompt text consistently and document any name normalization.
- [ ] L2-normalize image and text embeddings.
- [ ] Classify by maximum image-text cosine similarity.
- [ ] Produce one zero-shot result per selected dataset; no labeled training subset is used.
- [ ] Save text prototypes and their metadata for Stage 2.

## 6. Final evaluation and aggregation

- [ ] Evaluate top-1 accuracy on the complete official test split only after model/configuration selection is finished.
- [ ] Report 5-shot and 10-shot results as mean +/- standard deviation over the three subset-seed runs.
- [ ] Report full linear-probe results as mean +/- standard deviation over three initialization seeds.
- [ ] Report full image-prototype or zero-shot CLIP results as a single run, as applicable.
- [ ] Produce an accuracy table covering every implemented dataset, encoder, baseline, and training-set size.
- [ ] Sanity-check run counts and aggregation axes before reporting.

## 7. Required figures and analysis

- [ ] Accuracy versus training-set size:
  - [ ] Show 5-shot, 10-shot, and full results.
  - [ ] Include error bars where three runs exist.
  - [ ] If using zero-shot CLIP, optionally show it as a horizontal reference line.
- [ ] Training curves:
  - [ ] Plot training and validation loss for one representative 10-shot run per dataset-encoder combination.
  - [ ] Discuss stability and evidence of overfitting.
- [ ] Confusion matrices:
  - [ ] Produce one row-normalized matrix for a useful representative setting on each selected dataset.
  - [ ] Discuss the major class confusions.
- [ ] Feature visualizations:
  - [ ] Select a readable subset of approximately 8-10 classes.
  - [ ] Use PCA, t-SNE, or another justified 2D projection.
  - [ ] Keep the same classes, test examples, and colors across comparisons on a dataset.
  - [ ] Show test features with image prototypes for ResNet-18/DINOv2 when Option A is used.
  - [ ] Show test image embeddings with text prototypes for CLIP when Option B is used.
  - [ ] Fit the projection jointly to every image feature and prototype displayed in a plot.
  - [ ] Treat 2D projections as qualitative evidence, not performance metrics.

## 8. Reproducibility and Stage 1 handoff

- [ ] Re-run a representative experiment from cached features using only the recorded configuration.
- [ ] Verify that encoders never received gradient updates.
- [ ] Verify that all reported numbers trace to saved run-level metrics.
- [ ] Document the complete experimental protocol, deviations, and main observations.
- [ ] Preserve the selected prototype setup and artifacts for Stage 2.
- [ ] Preserve the linear-probe setup and artifacts for Stage 3.
- [ ] Prepare to explain the protocol, results, failure modes, and quantitative/qualitative observations.

## Completion criteria

Stage 1 is complete when the required linear-probe grid and the chosen prototype branch have been run under the official split/seed protocol; all required tables and figures exist; results are reproducible from frozen cached features; and the baseline artifacts needed by Stages 2 and 3 are preserved.
