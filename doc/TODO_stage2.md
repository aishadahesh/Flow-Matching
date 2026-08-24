# Stage 2 TODO - Flow Matching to Class Prototypes

Source of truth: `ref/stage_2.pdf`. Stage 2 adds a flow-matching (FM) layer on top of the **frozen** Stage 1 image-prototype baseline. **Do not modify the frozen encoders, do not recompute Stage 1 features/prototypes under different conditions, and do not change the Stage 1 datasets, splits, subset seeds, or K values.** The entire point of Stage 2 is a controlled, apples-to-apples comparison against Stage 1 - any deviation in data/splits/seeds would make `ΔAcc` uninterpretable.

Status: the required grid is **trained and evaluated** (`04_flow_matching.ipynb`, 162 trainings / 216 result rows). Results and their discussion are in `doc/RESULTS.md`. Remaining open items are marked `[ ]` below.

## 0. Lock the Stage 2 experimental plan

- [x] Confirm the FM layer builds on the **image-derived class prototypes** branch (`02_image_prototypes.ipynb`), not zero-shot CLIP. CLIP text prototypes live in CLIP's joint image-text space, not the frozen image-encoder feature space that `z_i` belongs to - Stage 2's FM operates entirely within the ResNet-18/DINOv2 feature space, so only the image-prototype branch is geometrically compatible as a transport target. (A CLIP-branch variant that switches *both* endpoints into CLIP space was added separately as an extension - see Section 8.)
- [x] Reuse, unchanged, from Stage 1: all 3 datasets, both encoders (ResNet-18 and DINOv2 on every dataset), the official train/val/test splits, K ∈ {5, 10, full} with subset seeds {0, 1, 2} / classifier-initialization seeds {0, 1, 2}, and the already-cached frozen features on Drive. (Plan read from `stage1_config.json`'s `plan` section; never re-declared.)
- [x] Record the Stage 2 plan (datasets, encoders, K values, T values, standard vs. rolled-out, velocity-network architecture) in a versioned config, following the `stage1_config.json` pattern - either a new `flow_matching` section in that file or a parallel `stage2_config.json`. (`flow_matching` section, `config_revision: 1`, written on first run.)
- [x] Create `04_flow_matching.ipynb` as an independent notebook, following the same Drive-mount / cache-reuse / checkpoint-resume / skip-if-already-saved conventions established in `01_linear_probe.ipynb` and `02_image_prototypes.ipynb`.

## 1. Reuse Stage 1 artifacts - no new feature extraction

- [x] Load cached frozen features directly from `feature_cache/v{schema}/...pt`; never re-run an encoder. (`load_cached_features`; the notebook imports neither `torchvision` nor an encoder.)
- [x] Load the exact class prototypes already saved by `02_image_prototypes.ipynb` (`outputs/image_prototypes/runs/<dataset>/<encoder>/<shot>/seed_<seed>/prototypes.pt`) for every (dataset, encoder, shot) combination needed. Do not recompute prototypes from a different subset than Stage 1 used for that seed. (`load_stage1_prototypes`; `full` maps to `seed_0`, matching Stage 1's `seeds = SUBSET_SEEDS if shot != 'full' else [0]`.)
- [x] Re-derive each K-shot training subset with the identical seeded `balanced_indices(labels, k, seed)` logic Stage 1 used, rather than re-sampling - the FM training pairs `(z_i, p_{y_i})` must be built from exactly the same images Stage 1's prototype for that class/seed was built from. (Copied verbatim from `02_image_prototypes.ipynb`.)
- [x] Add a loud failure (not a silent recompute) if a required Stage 1 feature cache or prototype file is missing for a (dataset, encoder, shot, seed) combination Stage 2 needs. (`FileNotFoundError` naming the producing notebook.)

## 2. Velocity network `v_theta(z, t)`

- [x] Implement a small MLP: 2 hidden layers, width ~512, SiLU activations, scalar `t` concatenated to the input feature, output dimension equal to the feature dimension (512 for ResNet-18, 384 for DINOv2). (`VelocityNet`; feature dim read from the data, not hardcoded.)
- [x] Fix one architecture/hyperparameter configuration and reuse it for both standard and rolled-out training - the PDF explicitly says no extensive search is needed and the comparison must hold the architecture and main training choices fixed. (Single `flow_matching` config block consumed by both modes.)
- [x] Decide and document a checkpoint-selection rule consistent with Stage 1's "highest validation accuracy" convention: periodically run the *full* Euler rollout on the validation split, classify via cosine similarity to the Stage 1 prototypes, and checkpoint on best validation accuracy (with early stopping), rather than inventing an unrelated ad hoc rule. (Rolled-out: accuracy at its own `T`. Standard: mean over every `T` in `T_values`, since one network is judged at all of them - documented in the notebook's title cell.)

## 3. Standard FM training

- [x] For each training feature `z_i` and its class prototype `p_{y_i}`, sample `t ~ U(0,1)`, define `z_t = (1-t) z_i + t*p_{y_i}`, target velocity `u_i = p_{y_i} - z_i`.
- [x] Train with `L_FM = ||v_theta(z_t, t) - u_i||^2`. (`F.mse_loss`, i.e. the squared norm divided by the feature dimension - a constant factor, applied identically to both objectives.)
- [x] **Train exactly one standard-FM network per (dataset, encoder, K, seed)** - standard training has no dependency on the inference step count `T`, so the same trained network is evaluated at both `T=4` and `T=12` in Section 5. Do not train separate standard-FM networks per `T`.
- [x] Save per-epoch training (and validation, if using the Section 2 checkpoint rule) loss. (`history.csv` per run, with `val_accuracy` and per-`T` `val_accuracy_T{4,12}`.)

## 4. Rolled-out FM training

- [x] For a fixed `T`, starting from each training feature `z_hat_0 = z_i`, apply the same `T`-step Euler sequence used at inference, with the same velocity network forward pass (`v_theta(z_hat_k, k/T)` at every step). (The same `euler_rollout` used at inference.)
- [x] Backpropagate through the complete `T`-step sequence using `L_roll = ||z_hat_T - p_{y_i}||^2` - this supervises only the final transported point, not the individual per-step velocities. (No `detach` anywhere in the training path.)
- [x] **Train two separate rolled-out networks per (dataset, encoder, K, seed): one for `T=4`, one for `T=12`.** Rolled-out training must use the same `T` at training and inference, so, unlike standard FM, `T=4` and `T=12` rolled-out models are not interchangeable. (Separate `rollout_T4`/`rollout_T12` run directories; `eval_T_list = [T]`.)
- [x] Note the extra compute cost: backprop through `T` sequential steps is more expensive than standard FM's single random-`t` sample, and grows with `T` - budget accordingly, especially for `T=12`. (Measured: `runtime_seconds` in every `metrics.json`; rolled-out `T=12` is roughly 2x rolled-out `T=4` and ~1.5-2.5x standard FM per run.)
- [x] Save per-epoch training (and validation) loss.

## 5. Inference and classification

- [x] Implement one shared Euler integrator used by every FM variant: `z_hat_{k+1} = z_hat_k + (1/T) * v_theta(z_hat_k, k/T)`, `k = 0, ..., T-1`, starting from `z_hat_0 = z` (the test feature). (`euler_rollout`, also reused by the trajectory figure.)
- [x] Evaluate `T ∈ {4, 12}` for both the standard-trained network (one network, two `T` values at inference) and the rolled-out-trained networks (the matching-`T` network only - do not run the `T=4` rolled-out network at `T=12` or vice versa).
- [x] Classify `z_hat_T` by maximum cosine similarity to the Stage 1 class prototypes, reusing the exact prototype vectors from Section 1 - the same rule as Stage 1.
- [x] Run the complete grid: 3 datasets × 2 encoders × K ∈ {5, 10, full} × {standard, rolled-out} × T ∈ {4, 12}. (4 FM result cells - standard-T4, standard-T12, rollout-T4, rollout-T12 - per dataset/encoder/K, on top of the Stage 1 baseline cell.) (162 trainings, 216 rows in `run_metrics.csv`.)
- [x] Repetition protocol: identical to Stage 1 - 3 runs per 5-shot/10-shot setting (subset seeds 0, 1, 2), 3 runs per full setting (initialization seeds 0, 1, 2). (Deliberate documented deviation: Stage 1's *image-prototype* `full` setting is a single closed-form run, so its baseline column has no standard deviation; Stage 2's `full` uses 3 initialization seeds, matching the linear probe's convention, because an FM network does have an initialization.)

## 6. Final evaluation and aggregation

- [x] Evaluate top-1 accuracy on the complete official test split only after training/checkpoint selection is finished - never let test accuracy influence training or checkpoint choices. (Test split is touched only in `evaluate_split`, after `net.load_state_dict(best_state)`.)
- [x] Report mean ± standard deviation over 3 runs for every (dataset, encoder, K, T, training-mode) combination, matching Stage 1's aggregation convention exactly.
- [x] Compute `ΔAcc = Acc_FM - Acc_baseline` against the *matching* Stage 1 image-prototype baseline (same dataset, encoder, K) for every FM variant - never against a different K, encoder, or dataset. (Joined on `(dataset, encoder, shot)` against `image_prototypes/accuracy_summary.csv`; baseline numbers are never recomputed.)
- [x] Produce a combined accuracy table: Stage 1 baseline vs. standard-FM(T=4), standard-FM(T=12), rolled-out-FM(T=4), rolled-out-FM(T=12), for every dataset/encoder/K, with `ΔAcc` shown alongside raw accuracy. (`accuracy_summary_with_baseline.csv`.)
- [x] Produce an accuracy-vs-K plot with error bars, one per dataset/encoder, comparable in style to Stage 1's training-size plot, with all 5 series (baseline + 4 FM variants) shown together. **The stored figure must be regenerated:** the plotting cell used `part.T`, which resolves to pandas' transpose property rather than the column named `T`, so all four FM series were silently dropped and only the baseline was drawn. Fixed to `part['T']`; re-run the notebook's Section 8 to refresh `accuracy_vs_training_size.png`.

## 7. Required figures and analysis

- [x] **Classification results** (Section 6 above): table and/or accuracy-vs-K plot with `ΔAcc` clearly indicated. (Table and figure both complete - `accuracy_vs_training_size.png` was regenerated in the 2026-08-24 run and shows all five series. Discussion in `doc/RESULTS.md`.)
- [x] **Training curves**: one representative training-loss curve for standard FM and one for rolled-out training (use the same representative dataset/encoder/K as Stage 1's 10-shot loss-curve figure, for comparability). Discuss whether both approaches train stably and converge to reasonable solutions. (10-shot / seed 0, all six dataset-encoder pairs; discussion in `doc/RESULTS.md`.)
- [x] **Feature-space visualizations**: for the same ~8-10 classes, test examples, and class colors already used in Stage 1's PCA/t-SNE plots, compare (a) original encoder features, (b) features after standard FM, (c) features after rolled-out FM, together with the matching class prototypes. Fit the 2D projection (PCA and/or t-SNE) **jointly** across every feature set and prototype shown in a given figure, so the three views are directly comparable positions in the same projection. (Both projections produced: PCA in Section 10, t-SNE in Section 10b, each fit once over the row-stacked three views plus prototypes. The three panels also share axis limits - a joint fit that is then autoscaled per panel would rescale each view independently and hide the contraction the figure exists to show.)
- [x] **Flow trajectories**: for a small number of representative test examples, plot the intermediate FM states `z_hat_0, ..., z_hat_T` together with the original feature, final transported feature, and target class prototype, in one jointly-fit PCA projection per example. PCA is recommended over t-SNE here specifically because trajectories need to remain geometrically interpretable as a path, which t-SNE does not preserve.

## 8. Optional extensions

- [x] Explore the learned flow in reverse: start from class prototypes and integrate backward toward the encoder feature distribution. (Section 12 in both notebooks, now **quantitative as well as visual**: a per-class *recovery rate* asks whether each reverse-flowed prototype lands nearest to its own class's mean test feature, reported against the untouched forward prototype's recovery rate as the control/ceiling, plus a five-panel strip showing where the prototypes travel at reverse `t` = 1, 0.75, 0.5, 0.25, 0. Caveats to state when presenting: the backward pass evaluates the field at `t` in {1, ..., 1/T} while the forward pass uses {0, ..., 1-1/T}, so it is not the exact inverse even for a perfectly learned field; and the forward flow is a deliberate contraction, so it is not invertible in principle - this recovers a plausible pre-image, not the original point.)
- [x] Compare samples and prototypes at intermediate flow times (not only `t=0` and `t=1`). (Section 13 in both notebooks.) Implemented both quantitatively and visually:
  - `flow_metrics_over_time` classifies **every** intermediate Euler state over the complete test split, giving accuracy, margin (`cos` to the true prototype minus the best competing one), cosine to the true prototype, and L2 distance to it, at each `t = k/T`.
  - The accuracy-versus-`t` curve is **self-anchoring at both ends**, which is what makes it trustworthy: `t=0` is the untouched feature, so its accuracy *is* the baseline classifier, and `t=1` is the reported FM result. Both notebooks assert the `t=0` end - `04` against Stage 1's saved per-run `metrics.json`, `05` against the zero-shot CLIP baseline recomputed in its Section 7. So the curve is ΔAcc unrolled.
  - A five-panel snapshot strip per (dataset, objective) shows the sample cloud and the prototypes at `t` = 0, 1/4, 1/2, 3/4, 1, in one jointly-fit PCA plane per row with shared axis limits.
  - Margin is the metric that distinguishes useful transport from indiscriminate contraction: distance to the prototype falling while margin stays flat means the flow is collapsing everything, not classifying.
- [x] **Apply the same FM layer to the zero-shot CLIP branch** (`05_flow_matching_clip.ipynb`): CLIP RN50 image embeddings transported toward the frozen text prototypes from `03_zero_shot_clip.ipynb`, reusing the identical `flow_matching` config so the two branches are comparable. Not a required Stage 2 deliverable - `ref/stage_1.pdf` asks for one prototype branch and `04_flow_matching.ipynb` is it - but this project implemented both Stage 1 branches, so the extension is available. Two things must be said when presenting it: (a) the FM layer trains on labeled pairs, so it converts zero-shot CLIP into a K-shot method and `ΔAcc` against the zero-shot baseline is not like-for-like; (b) that notebook therefore also reports a K-shot image-prototype control on the same CLIP features, which is the comparison that isolates the FM layer from the mere presence of labels.
- [x] Run `05_flow_matching_clip.ipynb` end to end. **Done** - 81 velocity networks trained (108 result rows: the single standard network is reported at both `T`); the CLIP RN50 train/val splits were encoded once into the Stage 1 cache directory. Results in `doc/RESULTS.md` (Model 3). Headline: FM beats zero-shot in 32/36 cells but loses to the same-supervision K-shot control in 28/36, so the large Δ vs zero-shot measures the labels rather than the layer.

- [x] **Animated flow trajectories** (beyond the specification). Section 21 in `04`, Section 19 in `05`: the two objectives side by side from an identical starting feature, one panel each, with an arrow showing the actual Euler update at every step, exported as embedded GIFs. The setting is selected programmatically (largest ΔAcc in `04`; largest gain over the K-shot control in `05`) rather than hardcoded, and the cell asserts `T+1` states, an identical start for both models, and no gradients. Read qualitatively - the transport happens in the full feature space and this is a 2-D projection of it.

## 9. Reproducibility and Stage 2 handoff

- [x] Save per-run config, velocity-network checkpoints, training history, and test predictions, following the same run-directory / `metrics.json` / checkpoint-resume / skip-if-already-saved conventions as `01_linear_probe.ipynb`. (`runs/<dataset>/<encoder>/<shot>/<mode>/<run_tag>/` with `config.json`, `history.csv`, `checkpoints/{best,latest}.pt`, and `eval*/metrics.json` + `test_predictions.npy`.)
- [x] Verify that frozen encoders and frozen Stage 1 prototypes are never updated during Stage 2 training - only `v_theta`'s parameters should ever receive gradients. (No encoder is loaded at all; the prototype tensor is only ever read/indexed and is never passed to an optimizer.)
- [x] Document the protocol, the standard-vs-rolled-out comparison, and the main quantitative/qualitative observations (which dataset/encoder/K/T combinations improve over the Stage 1 baseline, and whether the feature-space/trajectory plots explain why). (`doc/DOC.md` Part II for the protocol, `doc/RESULTS.md` for the observations.)
- [x] Preserve the best FM checkpoints and configuration per setting as a reference point for Stage 3 (FM before a linear classifier). (`checkpoints/best.pt` retained per setting.)

## Findings that came out of the optional extensions

- **The FM layer repairs a broken classifier rather than sharpening a working one.** Cosine to the true prototype rises in every setting, but the margin against the best competitor rises only where the baseline was mis-ranked (Aircraft, negative margin). Where the baseline already ranked correctly (DTD, Flowers-102) the margin slightly falls and accuracy moves by under a point. This is the measured mechanism behind "gains are largest where the baseline is weakest".
- **`t = 1` is a convention, not an optimum.** Accuracy strictly peaks before the endpoint in all 6 CLIP-branch curves and 6 of 12 image-branch curves; rolled-out models on the CLIP branch leave up to +4.2 points on the table. Now promoted to a validation-selected hyperparameter in `04` §16: it helps in 18 of 36 conditions, mean +0.0018, and converts the only real regression in the project into a gain.
- **Most of the largest gain is not specific to flow matching.** The `04` §19 controls hold architecture, optimizer, schedule, subsets and seeds fixed and remove the flow: on Aircraft/DINOv2 a plain `direct` MLP reaches .5783 against FM's .5852, i.e. 97% of the gain over the .3423 baseline. FM still beats both controls in 16 of 18 settings, and its margin over `direct` is largest exactly where `direct` overfits *below* the baseline (DTD and Flowers-102 on ResNet-18, +.042 to +.063). Say this before being asked about the +24 points.
- **Most of the sub-1-point gains are not measurable.** The paired 95% CI in `04` §14 leaves 39 of 72 cells distinguishable from zero - 24 of 24 on Aircraft, 7 of 24 on DTD. The two regressions are not significant either. McNemar is significant far more often, but its unit is the test image rather than the run, so it ignores subset variance; quote the CI.
- **The flow discriminates, it does not merely contract.** `W(t)/B(t)` falls in all 12 curves in `04` §17 (median -0.219). But falling `W/B` only converts into accuracy where the true prototype was not already ranked first, which is the class-structure version of the margin finding above.
- **Two Euler steps are enough.** `04` §18: median `frac_at_T2` = 1.00 of the `T=12` gain, while a single step delivers only ~62% and is negative in three settings. The layer costs two velocity evaluations, not twelve.
- **The reverse flow makes CLIP's modality gap measurable.** Backward integration lifts text-prototype recovery from .55/.25/.75 to 1.00/.75/1.00, while on the image branch the same operation barely moves anything (cosine distance .006-.044) - which is what a contraction should do.

## Known issues to fix on the next run

- [x] Regenerate `accuracy_vs_training_size.png` after the `part['T']` fix - **done**; the stored figure now shows all five series.
- [x] Re-run `04_flow_matching.ipynb` for the optional-extension outputs - **done**; both notebooks were executed top to bottom and every optional-extension artifact exists. Two findings came out of it: the FM layer only improves the *margin* where the baseline was already mis-ranked, and accuracy peaks before `t=1` in every CLIP-branch curve and 6 of 12 image-branch curves.
- [x] `best_val_accuracy` in the standard-FM rows of `run_metrics.csv` is the *mean* over `T ∈ {4, 12}` but is written into both per-`T` rows as though it were that `T`'s own validation accuracy. Test accuracy is unaffected. The per-`T` values already exist in `history.csv` as `val_accuracy_T4`/`val_accuracy_T12`; `05_flow_matching_clip.ipynb` already records them separately as `best_val_accuracy_at_T`, and `04_flow_matching.ipynb` should be brought in line. **Done** - Section 6 now keeps a per-`T` checkpoint (`best_T{T}.pt`) as well as `best.pt`, and Section 7 writes `best_val_accuracy_at_T`, `selected_epoch_at_T` and `test_accuracy_sel_T`. Requires one pass with `FORCE_RETRAIN_STANDARD = True` to populate for the already-saved runs.
- [x] `fm_histories` and the training-curve cell hardcode `rollout_T4`/`rollout_T12` while the rest of the notebook reads `FM['T_values']`; changing `T_values` in the config raises `KeyError`. **Done** - both now build their series from `FM['T_values']`.
- [x] `validation_accuracies` runs with the network in `train()` mode. Harmless for the current `VelocityNet` (no dropout or batch-norm), but it will silently change behaviour if the architecture ever grows. **Done** - saves and restores training mode, matching `05`.
- [ ] The feature-space cell derives its example indices from `feature_bank[(dataset, 'resnet18')]` even when plotting DINOv2. Correct only because both caches were built with `shuffle=False` over the same dataset object, so it is currently harmless, but it should use the current encoder's own labels. **Not done** - §10 still reads `base_labels` from the ResNet-18 bank. (§12b and §13b do use the current encoder.)

## Review-driven hardening (external review, 2026-08-23)

Guards and reporting - no retraining needed, all post-hoc on artifacts already on Drive:

- [x] **Prove** the K-shot subsets are Stage 1's, rather than asserting it in prose: rebuild each prototype from the re-derived subset with Stage 1's exact recipe and require agreement to `1e-5` (`04` §5b). Stage 1 never saved the indices, so reconstruction is the only available check.
- [x] Prototype row-order and unit-norm assertions on both transport endpoints (`04` §5b; `05` already had the norm asserts in Section 6). Note `prototypes.pt['classes']` holds class *name strings*, so it cannot be compared against `arange` - the reconstruction check is what establishes ordering.
- [x] `effective_repetitions` / `distinct_subsets` diagnostics ported from `05` into `04` §5b, so Flowers-102 `K=10` (all subset seeds select the same 1020 images; `K=10` *is* `K=full` there) is never read as low variance.
- [x] Paired per-seed `ΔAcc` with 95% CIs plus McNemar on the shared test images (`04` §14), replacing difference-of-means. Now on **both** branches - `05` §19 pairs against the K-shot control, with `05` §9 saving the per-run control predictions McNemar needs. `05` §19 is **written but not yet executed**. At `K=full` the Stage 1 prototype is a single deterministic run, so the spread there is FM initialization variance alone - stated in the table via `variance_source`.
- [x] Error bars labelled explicitly (`±1 SD across Stage 1-matched repetitions`) on the accuracy-vs-K figure.
- [x] Fixed/broken transition counts plus the pre-flow margin distribution of each group (`04` §15).
- [ ] Training-curve figure should gain a validation-accuracy panel next to the loss panel (`04` §9) - the losses are different quantities on different scales, validation accuracy is what is actually comparable. **Not done**: §9 still plots `train_loss` only. `val_accuracy` is already in every `history.csv`, so this is a plotting change, not a re-run.
- [ ] Trajectory examples selected by transition type (helped / harmed / unchanged) plus one random, instead of the first example of the first three classes (`04` §11). **Not done**: §11 still uses `chosen_classes = np.arange(3)` and the first test example of each. §15 now provides the fixed/broken masks this selection would need.
- [ ] Visualization subset promoted to saved constants, with a seeded neutral subset as a robustness check. **Not done**: `np.arange(10)` is still hardcoded in each visualization cell and no `visualization_selection.csv` is written.
- [x] Deterministic `t ~ U(0,1)` seeded per `(init_seed, epoch, batch)` (`04` §6), so an interrupted-and-resumed run reproduces an uninterrupted one. Section 7 prints how many saved runs ever actually resumed.
- [x] Reverse-flow wording corrected. The forward prototype is the **pre-transport reference**, not a ceiling: on the CLIP branch reverse recovery exceeds it (1.000/0.750/1.000 vs 0.553/0.250/0.745) because a text prompt does not start out inside its own image cloud. Continuous `cos`/`L2` to the class mean now accompany the discrete recovery rate in both notebooks.

Ablations and controls - clearly separated from the required table:

- [x] Validation-selected stopping time `t*` (`04` §16). Selected on validation only; the test-optimal `t` is reported as an unreachable bound. The required table still classifies `ẑ_T`. **Measured**: helps in 18 of 36 conditions, mean +0.0018, best +0.0303, and it turns the DTD/ResNet-18 `full` rolled-out regression (-.0206) into +.0097 over the baseline. Now on **both** branches - `05` §20 ports it, and additionally reports `t*` against the K-shot control, since that is the like-for-like reference there. `05` §20 is **written but not yet executed**. This is where overshoot is worst: all 6 CLIP curves peak early, up to +4.2 points.
- [x] Inference-step sweep `T ∈ {1,2,4,8,12}`, standard FM only (`04` §18). Required table keeps `{4,12}`.
- [x] Quantitative geometry `W(t)`, `B(t)`, `W/B`, nearest-competitor similarity (`04` §17) - the spec's third goal answered with numbers, not only projections.
- [x] Non-FM controls: direct MLP endpoint regression and repeated residual with no time conditioning, same architecture/optimizer/subsets/seeds, in a separate `controls/` tree (`04` §19). This is what decides whether the +24pt Aircraft/DINOv2 result is specific to flow matching.
- [x] Stage 1 linear-probe comparison as context (`04` §20), labelled as a different Stage 1 baseline rather than an FM variant.
- [x] `doc/STAGE2_COMPLIANCE.md` maps every specification clause to where it is satisfied and lists everything that is deliberately *not* part of the required experiment.

### Colab execution status (updated after the 2026-08-24 run)

- [x] Execute `04` top to bottom. **Done** - Sections 5b and 14-21 all ran; their numbers are now in `doc/RESULTS.md` under "Hardening, ablations and controls - measured".
- [x] Train the control networks (`04` §19). **Done** - 108 control runs (`direct` and `residual`) written to `outputs/flow_matching/controls/`. Result: FM beats both controls in 16 of 18 settings, but on Aircraft/DINOv2 the plain `direct` MLP already delivers 97% of FM's gain over the baseline.
- [x] Execute `05` §17-§19. **Done** - reverse flow, intermediate flow times and the animations all produced.
- [ ] **Run `05` top to bottom** to execute the new §9 / §19 / §20. No FM retraining - §19 and §20 are post-hoc on saved checkpoints, and the control is closed form.
- [ ] **`04` §6b with `RUN_ZERO_INIT_PILOT = True`** (~72 networks). Adopt zero-init only if the 3 regressions clear, the seed spread falls, and `mean_epochs` has not blown out against the early-stopping patience. Adoption means `zero_init_output = True`, one `FORCE_RETRAIN = True` pass, and rewriting every reported number.
- [ ] **`04` §14b and `05` §19b with `RUN_EXTENDED = True`** (~380 and ~190 networks, minutes). The fix for the resolution problem: 33 of 72 cells are currently unresolvable at `n=3`, and intervals should narrow ~3.4× at `n=10`. The required tables stay at `n=3`.
- [ ] **One pass with `FORCE_RETRAIN_STANDARD = True`** to populate the per-`T` selection columns (`best_val_accuracy_at_T`, `selected_epoch_at_T`, `test_accuracy_sel_T`). Still outstanding: the re-run loaded all 54 standard-FM runs from Drive, because the skip logic checks whether the output files exist rather than which config produced them, so no `best_T{T}.pt` was written. Retrained standard runs will not reproduce the stored numbers to the last digit - report both during the transition. Rolled-out runs are unaffected (`t = k/T` is deterministic and batch order was already seeded).

Two facts from the run that close out earlier caveats:

- `04` §7 printed `Result rows: 216 | runs that actually resumed from a checkpoint: 0`. No stored result was ever produced by a resumed run, so the pre-fix nondeterministic `t` sampling never affected any saved number. The "standard-FM numbers will shift" warning can be retired.
- `04` §5b verified subset identity across **42** (dataset, encoder, K, seed) combinations at a maximum prototype-reconstruction drift of **0.00e+00**. The K-shot subsets are Stage 1's exactly.

### Added 2026-08-24, not yet executed

Five additions came out of reviewing why the results read as weak. The distinction between *coded* and *measured* is kept explicit here deliberately - documenting unrun analyses as findings is the mistake this file previously made.

- `05` §9 saves per-run control predictions (blocker for §19's McNemar).
- `05` §19 - paired ΔAcc vs the control, 95% CIs, McNemar.
- `05` §20 - validation-selected `t*`, reported against the control as well as against `t=1`.
- `04` §14b / `05` §19b - extended-repetition supplement, seeds 0-9, separate `extended/` tree.
- `04` §6b - zero-init pilot. Verified locally that a zeroed output layer makes the rollout the exact identity (drift `0.00e+00`) at `T=4` and `T=12`; an untrained default-init field displaces a unit-norm feature by 0.54-0.86 in L2, which is the size of the problem it addresses.

`build_report.py` now addresses notebook cells by content rather than by index, so inserting sections can no longer make the report silently parse the wrong table.

## Completion criteria

Stage 2 is complete when the full grid (3 datasets × 2 encoders × K ∈ {5, 10, full} × {standard, rolled-out} × T ∈ {4, 12}) has been trained and evaluated on the identical Stage 1 splits/seeds/prototypes; `ΔAcc` against the matching Stage 1 image-prototype baseline is reported for every combination; the training curves, feature-space visualizations, and flow-trajectory visualizations are produced; and the artifacts needed for Stage 3 are preserved.

**Met.** The accuracy-vs-K figure has been regenerated, both optional extensions are implemented and executed in both notebooks, and the ablations and controls have run. What remains is not required by `ref/stage_2.pdf`: the `FORCE_RETRAIN_STANDARD` pass for the per-`T` selection columns, and the three presentation/robustness items reopened above (validation-accuracy panel, transition-type trajectory selection, per-encoder visualization indices).

One reporting change the run forced, which must be carried into any presentation: **"improves in 66 of 72 cells" is a difference of means, and the paired 95% CI (§14) leaves only 39 of 72 distinguishable from zero** - all 24 Aircraft cells but just 7 of 24 on DTD. The two regressions are not significant either. Quote both numbers, not only the first.
