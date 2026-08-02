# Stage 1 TODO - Classification Baselines

Stage 1 establishes reliable, reproducible classification baselines on frozen pretrained features. **Do not add a Flow Matching component in this stage.**

## 0. Lock the comprehensive experimental plan

- [x] Evaluate **all three** datasets:
  - DTD (47 classes; use official partition 1)
  - FGVC-Aircraft (100 classes; use the `variant` annotation level)
  - Oxford Flowers-102 (102 classes)
- [x] Use DINOv2 ViT-S/14 on all three datasets.
- [x] Evaluate both prototype branches:
  - **Option A:** image-derived class prototypes; and
  - **Option B:** zero-shot CLIP RN50 with text-derived prototypes.
- [x] Record the comprehensive-grid rationale in the experiment configuration/notes.
- [ ] Confirm dataset licenses, download locations, checkpoint identifiers, and required compute/storage.

## 1. Define reproducibility conventions

- [x] Define a single experiment configuration format containing dataset, split, encoder, preprocessing, feature layer, baseline, shot count, subset seed, initialization seed, and training hyperparameters. (`stage1_config.json` / `DEFAULT_STAGE1` + per-run `config.json`)
- [x] Fix and document all random-number generators and deterministic settings used. (`seed_everything`; subset seeds vs. init seeds kept separate)
- [x] Define stable class-index mappings from the official dataset metadata. (`classes` stored in every feature cache, sourced from torchvision's official ordering)
- [x] Define run identifiers and output locations for cached features, checkpoints, metrics, and figures. (`OUTPUT_ROOT/runs/<dataset>/<encoder>/<shot>/<run_tag>/...`)
- [ ] Save environment/package versions and exact pretrained checkpoint identifiers. (checkpoint identifiers only - e.g. `torchvision/resnet18/IMAGENET1K_V1`; no `torch`/package version pinning saved anywhere)
- [ ] Add validation checks that prevent train/validation/test leakage. (no explicit leakage assertion exists; relies entirely on torchvision's official split loaders)

## 2. Prepare datasets and splits

- [ ] Download and verify all three datasets. (torchvision handles download; no explicit integrity/verification step)
- [x] Preserve the official training, validation, and test splits; never merge training and validation. (`build_dataset` uses torchvision's official split loaders directly)
- [x] Apply dataset-specific rules:
  - [x] DTD: official partition 1.
  - [x] FGVC-Aircraft: `variant` annotation level.
- [x] Use the full validation split only for model selection.
- [x] Reserve the full test split for final evaluation only.
- [x] For each dataset, create balanced per-class training subsets for:
  - [x] 5-shot with subset seeds 0, 1, and 2.
  - [x] 10-shot with subset seeds 0, 1, and 2.
  - [x] Full training split.
- [ ] Validate subset balance, disjoint split membership, class coverage, and deterministic regeneration. (`balanced_indices` guarantees per-class balance and seed-determinism; no explicit disjoint-membership/class-coverage assertion exists)

## 3. Extract and cache frozen features

- [x] Implement/use checkpoint-specific preprocessing without alteration. (`weights.transforms()` for ResNet-18; standard DINOv2 recipe; OpenCLIP's bundled preprocess for CLIP)
- [x] Keep every encoder in evaluation mode with gradients disabled and parameters frozen. (`.eval()` + `requires_grad_(False)` + a hard `assert` in every notebook)
- [x] Extract and cache train/validation/test features once per dataset-encoder combination.
- [x] ResNet-18:
  - [x] Use the public torchvision ImageNet-1K pretrained checkpoint.
  - [x] Extract the 512-dimensional representation before the final classifier.
  - [x] Run on all three datasets.
- [x] DINOv2 ViT-S/14:
  - [x] Use a public pretrained checkpoint.
  - [x] Extract the final class-token representation.
  - [x] Run on all three datasets.
- [x] If zero-shot CLIP is selected:
  - [x] Use frozen CLIP RN50 image and text encoders.
  - [x] Cache test image embeddings and class text embeddings.
- [ ] Store labels, sample identifiers, split, class mapping, preprocessing/checkpoint metadata, and feature dimensionality with each cache. (everything is stored except stable per-image sample identifiers - caches key on split/index only)
- [ ] Verify cache completeness, ordering, finite values, and reproducibility. (length-match + `torch.isfinite` checks exist; no explicit ordering or re-extraction-reproducibility check)

## 4. Required baseline - linear probe

- [x] Train only a multiclass affine head, `s = Wz + b`, on cached features.
- [x] Use softmax cross-entropy.
- [x] Start with the suggested configuration:
  - AdamW
  - learning rate `1e-3`
  - weight decay `1e-4`
  - batch size 64
  - at most 200 epochs
  - select the checkpoint with highest validation accuracy
- [x] If the suggested setup behaves poorly, adjust it using validation results only and document every change. (automatic 6-candidate search per run, every trial logged to `validation_tuning.csv`)
- [x] Run ResNet-18 on all three datasets at 5-shot, 10-shot, and full.
- [x] Run DINOv2 ViT-S/14 on all three datasets at 5-shot, 10-shot, and full.
- [x] Repetition protocol:
  - [x] 5-shot: one run for each subset seed 0, 1, and 2.
  - [x] 10-shot: one run for each subset seed 0, 1, and 2.
  - [x] Full: three classifier-initialization seeds.
- [x] Save per-epoch training/validation loss and accuracy. (`history.csv` per run)
- [x] Select one representative 10-shot run for every dataset-encoder combination for loss-curve reporting. (seed 0, plotted for every dataset-encoder pair)

## 5A. Prototype branch - Option A: image-derived prototypes

Complete this section for the comprehensive grid.

- [x] Use the same ResNet-18 and DINOv2 dataset-encoder combinations as the linear probe.
- [x] L2-normalize every selected training feature.
- [x] For each class, average its normalized features and L2-normalize the resulting class mean. (`make_prototypes`: normalize -> mean -> normalize, in that order)
- [x] L2-normalize evaluation features and classify by maximum cosine similarity.
- [x] Evaluate 5-shot and 10-shot using subset seeds 0, 1, and 2.
- [x] Evaluate full data once. (`seeds = SUBSET_SEEDS if shot != 'full' else [0]`)
- [x] Save class prototypes and their metadata for Stage 2. (`prototypes.pt` + `checkpoints/best.pt`/`final.pt` per run, with classes/shot/seed/config/metrics)

## 5B. Prototype branch - Option B: zero-shot CLIP

Complete this section for the comprehensive grid.

- [x] Use frozen CLIP RN50 image and text encoders.
- [x] Build one prompt per class using the required template:
  - DTD: `a photo of a {class} texture`
  - FGVC-Aircraft: `a photo of a {class} aircraft`
  - Flowers-102: `a photo of a {class} flower`
- [x] Verify that official class names are converted to prompt text consistently and document any name normalization. (deterministic `class_names_of`/hardcoded Flowers-102 list; normalization approach recorded in `DOC.md` section 6.2)
- [x] L2-normalize image and text embeddings.
- [x] Classify by maximum image-text cosine similarity.
- [x] Produce one zero-shot result per dataset; no labeled training subset is used. (only the test split is ever loaded in this notebook)
- [x] Save text prototypes and their metadata for Stage 2. (`{dataset}__text_{variant}.pt` with prompts/variant/templates/encoder metadata)

## 6. Final evaluation and aggregation

- [x] Evaluate top-1 accuracy on the complete official test split only after model/configuration selection is finished.
- [x] Report 5-shot and 10-shot results as mean +/- standard deviation over the three subset-seed runs.
- [x] Report full linear-probe results as mean +/- standard deviation over three initialization seeds.
- [x] Report full image-prototype or zero-shot CLIP results as a single run, as applicable.
- [ ] Produce an accuracy table covering every implemented dataset, encoder, baseline, and training-set size. (each notebook produces its own per-baseline table; there is no single cross-baseline table yet - `README.md` already names a `04_combined_results.ipynb` for this, but that notebook does not exist in the repo)
- [ ] Sanity-check run counts and aggregation axes before reporting. (no explicit assertion on expected row/run counts before reporting summaries)

## 7. Required figures and analysis

- [ ] Accuracy versus training-set size:
  - [x] Show 5-shot, 10-shot, and full results.
  - [x] Include error bars where three runs exist.
  - [ ] If using zero-shot CLIP, optionally show it as a horizontal reference line. (optional; not currently plotted anywhere)
- [ ] Training curves:
  - [x] Plot training and validation loss for one representative 10-shot run per dataset-encoder combination.
  - [ ] Discuss stability and evidence of overfitting. (curves exist, but there is no written discussion anywhere yet of what they show, e.g. the Aircraft/ResNet-18 overfitting signature)
- [ ] Confusion matrices:
  - [x] Produce one row-normalized matrix for a useful representative setting on each dataset.
  - [ ] Discuss the major class confusions. (plots exist across all three notebooks; no written discussion of the error patterns yet)
- [x] Feature visualizations:
  - [x] Select a readable subset of approximately 8-10 classes.
  - [x] Use PCA, t-SNE, or another justified 2D projection.
  - [x] Keep the same classes, test examples, and colors across comparisons on a dataset.
  - [x] Show test features with image prototypes for ResNet-18/DINOv2 when Option A is used.
  - [x] Show test image embeddings with text prototypes for CLIP when Option B is used.
  - [x] Fit the projection jointly to every image feature and prototype displayed in a plot.
  - [x] Treat 2D projections as qualitative evidence, not performance metrics.

## 8. Reproducibility and Stage 1 handoff

- [ ] Re-run a representative experiment from cached features using only the recorded configuration. (not yet exercised as an explicit check - and see the note above about `01_linear_probe.ipynb` reverting mid-session, which is exactly the kind of drift this step would catch)
- [x] Verify that encoders never received gradient updates. (hard `assert` in every notebook's `load_encoder`)
- [ ] Verify that all reported numbers trace to saved run-level metrics. (every number is traceable via `metrics.json`/`run_metrics.csv`, but no explicit audit step re-derives the summary tables from those files to confirm it)
- [ ] Document the complete experimental protocol, deviations, and main observations. (`DOC.md` documents the protocol and the scope deviations thoroughly; it does not yet contain results/observations - that discussion still needs to be written)
- [x] Preserve the selected prototype setup and artifacts for Stage 2. (`prototypes.pt`/checkpoints per run)
- [x] Preserve the linear-probe setup and artifacts for Stage 3. (`best_linear_head.pt`, `config.json`, `metrics.json` per run)
- [ ] Prepare to explain the protocol, results, failure modes, and quantitative/qualitative observations. (protocol: yes, via `DOC.md`; results/failure-mode/observation write-up: not yet done)

## Completion criteria

Stage 1 is complete when the required linear-probe grid and the chosen prototype branch have been run under the official split/seed protocol; all required tables and figures exist; results are reproducible from frozen cached features; and the baseline artifacts needed by Stages 2 and 3 are preserved.
