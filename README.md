# Flow-Matching
Computer Vision LAB Summer Project

## Stage 1 recommended run

The experiment plan is centralized in `stage1_config.json`. The current plan is
the comprehensive grid requested by the project team: DTD, FGVC-Aircraft, and
Flowers-102; ResNet-18 and DINOv2 on every dataset; image-derived prototypes;
and zero-shot CLIP on every dataset.

Each notebook also contains the same default configuration, so VS Code sessions
connected to a Colab kernel work even when `stage1_config.json` has not been
copied to Google Drive. If the JSON file exists in the Drive project folder or
the Colab working directory, it overrides the embedded defaults.
The linear-probe notebook creates the JSON file in the mounted Drive project on
its first run when it is missing, allowing subsequent notebooks to load the
same configuration.

Run the notebooks in this order:

1. `01_linear_probe.ipynb` — creates and validates versioned frozen-feature
   caches, selects the feature transform and AdamW settings using validation
   accuracy only, then evaluates the selected model on test.
2. `02_image_prototypes.ipynb` — reuses the complete dataset/encoder grid and
   evaluates the required normalized class prototypes.
3. `03_zero_shot_clip.ipynb` — evaluates zero-shot CLIP RN50 on all three
   datasets with checkpoint-compatible QuickGELU, the prescribed prompts, and
   a separately reported five-template prompt ensemble.
Old unversioned feature caches and old checkpoint directories are not reused by
the enhanced pipeline. Few-shot repetitions vary only the subset seed; full-data
repetitions vary only the classifier-initialization seed.

## Stage 2 recommended run

After Stage 1 artifacts exist, run:

4. `04_flow_matching.ipynb` — the required Stage 2 image-prototype branch. It
   reuses the frozen Stage 1 feature caches and saved prototypes and produces
   216 result rows from 162 trained velocity networks.
5. `05_flow_matching_clip.ipynb` — the separate CLIP extension. It transports
   frozen CLIP RN50 image embeddings toward text prototypes and always reports
   a same-supervision K-shot image-prototype control alongside the zero-shot
   reference. It produces 108 result rows from 81 trained velocity networks.

Both notebooks were rerun end to end on 2026-08-24. The required grids and all
ungated post-hoc analyses are populated. The zero-initialization pilot and the
10-repetition supplements remain disabled and are not part of the required
Stage 2 result.

## Stage 3 recommended run

Stage 3 inserts an FM transformation *before* the frozen Stage 1 linear classifier
(`z → FM → ẑ → W ẑ + b`). It needs both the Stage 1 feature caches and the Stage 1
linear-probe run directories, so run `01_linear_probe.ipynb` first.

6. `06_fm_before_classifier.ipynb` — the Stage 3 notebook. It compares the Stage 1
   linear probe against two FM training strategies: end-to-end rolled-out
   classification training, and classifier-guided targets with standard FM training.
   The required grid is 3 datasets × DINOv2 ViT-S/14 × K=10 × T=12 × 3 subset seeds
   × 2 strategies = 18 trained velocity networks and 27 result rows.

The classifier is loaded, never retrained, and frozen with `requires_grad_(False)`.
The FM is initialized to the exact identity by zeroing the velocity network's output
layer, so the untrained system reproduces the linear probe bit for bit. Two guards
enforce this before any training starts, and both raise rather than warn: the
reloaded probe must reproduce its Stage 1 test accuracy to within 1e-6, and the
untrained rollout must be the identity.

The notebook also runs two variant sweeps and the optional joint-fine-tuning
extension. All three are flag-gated at the top of their sections
(`RUN_REGULARIZATION_SWEEP`, `RUN_GUIDANCE_SWEEP`, `RUN_JOINT_FINETUNE`), enabled by
default. Completed runs are skipped by checking saved outputs on Drive; set
`FORCE_RETRAIN = True` to rerun the grid from scratch.

**Status: implemented, not yet measured.** The notebook has been verified to execute
end to end against fabricated Stage 1 artifacts, but the Colab run over the real
feature caches has not happened. No number in `doc/RESULTS.md` comes from Stage 3.
Protocol is documented in `doc/DOC.md` Part III; task tracking in
`doc/TODO_stage3.md`.
