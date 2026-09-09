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

For the required comparison, the classifier is loaded, never retrained, and frozen
with `requires_grad_(False)`. The FM is initialized to the exact identity by zeroing
the velocity network's output layer, so the untrained system reproduces the linear
probe bit for bit. The notebook raises on any failed guard: the reloaded probe must
reproduce both Stage 1 validation and test accuracy within 1e-6, the untrained
rollout must be the identity, Strategy 1 must backpropagate through the rollout only
to the FM, and Strategy 2 must produce source-bounded monotone targets and an active
standard-FM gradient.

Classifier-guided targets use a per-sample trust-region radius centred on the original
source feature `z`, with an independent within-radius step fraction. Gradient
normalization and projection are separate choices; the default combines unit-gradient
steps with source-centred projection and retains the lowest-cross-entropy feasible
iterate. Proposal projection incidence and final-boundary occupancy are reported
separately. Strategy 1's displacement and velocity penalties are scale-free, so their
strength is comparable for `l2` and standardized Stage 1 feature spaces. The optional
guided joint-fine-tuning path uses a separate endpoint classification loss with the FM
endpoint detached; this genuinely trains the classifier without changing the FM
objective from standard flow matching.

Beyond the required grid the notebook runs two variant sweeps, the optional
joint-fine-tuning extension (with a head-only control, without which that comparison
cannot be read), and Stage 3 versions of Stage 2's two optional analyses — samples at
intermediate flow times, with a validation-selected stopping time, and the flow in
reverse. All are flag-gated at the top of their sections (`RUN_REGULARIZATION_SWEEP`,
`RUN_GUIDANCE_SWEEP`, `RUN_JOINT_FINETUNE`, `RUN_TSTAR_ABLATION`, `RUN_REVERSE_FLOW`),
enabled by default. A completed run is reused only when its full effective training
configuration, implementation revision, and content signatures of the Stage 1 probe
and feature caches all match. Changed code, hyperparameters, or upstream artifacts
therefore retrain automatically; set `FORCE_RETRAIN = True` to ignore even compatible
cached runs.

The intermediate-flow-time curve is anchored at both ends and asserts it: identity
initialization makes `t=0` accuracy exactly the Stage 1 linear probe, and `t=1` exactly
the reported Stage 3 result, so the curve is ΔAcc unrolled.

Sections 19-22 are diagnostics for reading a small or flat ΔAcc: which predictions the
flow fixed versus broke and what distinguishes those groups, per-class effects, flow
trajectories chosen by outcome, and an effect-size panel with paired slopes and
bootstrap CIs. Section 23 renders the rollout as video — the test cloud flowing from
`z` to `ẑ` with a live accuracy readout, and individual paths coloured by outcome so a
fixed and a broken path sit in the same frame. MP4 where ffmpeg is available, GIF
otherwise. Section 23c is a 2D simulation - two concentric rings run through the same
training code - where the frozen decision boundary and the learned velocity field
can be drawn directly instead of projected. Section 25 reloads every saved table,
figure and animation from Drive and renders them inline as one report.

**Status (2026-09-09): implementation revision 2 completed on Colab on 2026-09-08; revision 3 is
ready for a fresh run.** Revision 2 executed all 40 code cells without an error on a
T4 and passed the fidelity, identity, and strategy-gradient guards. Revision 3 keeps
the same required Stage 3 comparison but improves guided targets with independent
per-sample radius and step fractions, unit-gradient steps inside the source-centred
trust region, and corrected boundary/projection diagnostics. Its end-to-end main run
uses the specification-permitted scale-free displacement penalty at λ=1, while the
unregularized model remains an explicit control. The table below remains
the locked revision-2 result until revision 3 is rerun from the top.
The notebook replay saved on 2026-09-09 loaded revision-2 cached runs and reproduced that table;
because its executed source still declared revision 2, it is a reproducibility check rather than a
revision-3 measurement. The committed notebook has therefore been restored to revision 3 with stale
outputs cleared.

| Dataset | Frozen linear probe | End-to-end rollout | Classifier-guided |
|---|---:|---:|---:|
| DTD | .7167 ± .0081 | .7167 (+.0000) | **.7238 (+.0071)** |
| Aircraft | .5285 ± .0064 | **.5476 (+.0191)** | .5448 (+.0163) |
| Flowers-102 | .9935 ± .0000 | .9935 (+.0000) | .9935 (+.0000) |

Nine of the 18 selected FM checkpoints are epoch 0 and therefore exact identity maps:
all six Flowers-102 runs and all three DTD end-to-end runs. Across the paired per-seed
tests, end-to-end improves all three Aircraft splits significantly; guided FM improves
all six DTD/Aircraft splits, with five bootstrap CIs and five McNemar tests excluding
no change. Full results, controls, sweeps, and caveats are in `doc/RESULTS.md`.

These Stage 3 probe values are the exact heads loaded by `06`, not replacements for the
older Stage 1 aggregate table. The notebook replayed both validation and test accuracy
within 1e-6 and content-signed the upstream artifacts; the historical aggregate-table
discrepancy remains documented in `doc/TODO_stage3.md`.

The lightweight source and synthetic-behaviour checks live in
`tests/test_stage3_notebook.py`. Run them in an environment with PyTorch (Colab is
fine) with `python tests/test_stage3_notebook.py`.

Protocol is documented in `doc/DOC.md` Part III; every specification clause is
mapped to where it is implemented in `doc/STAGE3_COMPLIANCE.md`, which also lists
what is deliberately *not* part of the required experiment; task tracking in
`doc/TODO_stage3.md`.

The consolidated PDF report is `doc/Report.pdf` (legacy filename). It
now contains Stages 1, 2, and 3, including the Stage 3 protocol, required result table,
training-behavior analysis, optional joint extension, and the revision-2/revision-3
status distinction. Rebuild it with `python doc/build_report.py` after results change.
