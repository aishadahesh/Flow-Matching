# Stage 2 TODO - Flow Matching to Class Prototypes

Source of truth: `ref/stage_2.pdf`. Stage 2 adds a flow-matching (FM) layer on top of the **frozen** Stage 1 image-prototype baseline. **Do not modify the frozen encoders, do not recompute Stage 1 features/prototypes under different conditions, and do not change the Stage 1 datasets, splits, subset seeds, or K values.** The entire point of Stage 2 is a controlled, apples-to-apples comparison against Stage 1 - any deviation in data/splits/seeds would make `ΔAcc` uninterpretable.

## 0. Lock the Stage 2 experimental plan

- [ ] Confirm the FM layer builds on the **image-derived class prototypes** branch (`02_image_prototypes.ipynb`), not zero-shot CLIP. CLIP text prototypes live in CLIP's joint image-text space, not the frozen image-encoder feature space that `z_i` belongs to - Stage 2's FM operates entirely within the ResNet-18/DINOv2 feature space, so only the image-prototype branch is geometrically compatible as a transport target.
- [ ] Reuse, unchanged, from Stage 1: all 3 datasets, both encoders (ResNet-18 and DINOv2 on every dataset), the official train/val/test splits, K ∈ {5, 10, full} with subset seeds {0, 1, 2} / classifier-initialization seeds {0, 1, 2}, and the already-cached frozen features on Drive.
- [ ] Record the Stage 2 plan (datasets, encoders, K values, T values, standard vs. rolled-out, velocity-network architecture) in a versioned config, following the `stage1_config.json` pattern - either a new `flow_matching` section in that file or a parallel `stage2_config.json`.
- [ ] Create `04_flow_matching.ipynb` as an independent notebook, following the same Drive-mount / cache-reuse / checkpoint-resume / skip-if-already-saved conventions established in `01_linear_probe.ipynb` and `02_image_prototypes.ipynb`.

## 1. Reuse Stage 1 artifacts - no new feature extraction

- [ ] Load cached frozen features directly from `feature_cache/v{schema}/...pt`; never re-run an encoder.
- [ ] Load the exact class prototypes already saved by `02_image_prototypes.ipynb` (`outputs/image_prototypes/runs/<dataset>/<encoder>/<shot>/seed_<seed>/prototypes.pt`) for every (dataset, encoder, shot) combination needed. Do not recompute prototypes from a different subset than Stage 1 used for that seed.
- [ ] Re-derive each K-shot training subset with the identical seeded `balanced_indices(labels, k, seed)` logic Stage 1 used, rather than re-sampling - the FM training pairs `(z_i, p_{y_i})` must be built from exactly the same images Stage 1's prototype for that class/seed was built from.
- [ ] Add a loud failure (not a silent recompute) if a required Stage 1 feature cache or prototype file is missing for a (dataset, encoder, shot, seed) combination Stage 2 needs.

## 2. Velocity network `v_theta(z, t)`

- [ ] Implement a small MLP: 2 hidden layers, width ~512, SiLU activations, scalar `t` concatenated to the input feature, output dimension equal to the feature dimension (512 for ResNet-18, 384 for DINOv2).
- [ ] Fix one architecture/hyperparameter configuration and reuse it for both standard and rolled-out training - the PDF explicitly says no extensive search is needed and the comparison must hold the architecture and main training choices fixed.
- [ ] Decide and document a checkpoint-selection rule consistent with Stage 1's "highest validation accuracy" convention: periodically run the *full* Euler rollout on the validation split, classify via cosine similarity to the Stage 1 prototypes, and checkpoint on best validation accuracy (with early stopping), rather than inventing an unrelated ad hoc rule.

## 3. Standard FM training

- [ ] For each training feature `z_i` and its class prototype `p_{y_i}`, sample `t ~ U(0,1)`, define `z_t = (1-t) z_i + t*p_{y_i}`, target velocity `u_i = p_{y_i} - z_i`.
- [ ] Train with `L_FM = ||v_theta(z_t, t) - u_i||^2`.
- [ ] **Train exactly one standard-FM network per (dataset, encoder, K, seed)** - standard training has no dependency on the inference step count `T`, so the same trained network is evaluated at both `T=4` and `T=12` in Section 5. Do not train separate standard-FM networks per `T`.
- [ ] Save per-epoch training (and validation, if using the Section 2 checkpoint rule) loss.

## 4. Rolled-out FM training

- [ ] For a fixed `T`, starting from each training feature `z_hat_0 = z_i`, apply the same `T`-step Euler sequence used at inference, with the same velocity network forward pass (`v_theta(z_hat_k, k/T)` at every step).
- [ ] Backpropagate through the complete `T`-step sequence using `L_roll = ||z_hat_T - p_{y_i}||^2` - this supervises only the final transported point, not the individual per-step velocities.
- [ ] **Train two separate rolled-out networks per (dataset, encoder, K, seed): one for `T=4`, one for `T=12`.** Rolled-out training must use the same `T` at training and inference, so, unlike standard FM, `T=4` and `T=12` rolled-out models are not interchangeable.
- [ ] Note the extra compute cost: backprop through `T` sequential steps is more expensive than standard FM's single random-`t` sample, and grows with `T` - budget accordingly, especially for `T=12`.
- [ ] Save per-epoch training (and validation) loss.

## 5. Inference and classification

- [ ] Implement one shared Euler integrator used by every FM variant: `z_hat_{k+1} = z_hat_k + (1/T) * v_theta(z_hat_k, k/T)`, `k = 0, ..., T-1`, starting from `z_hat_0 = z` (the test feature).
- [ ] Evaluate `T ∈ {4, 12}` for both the standard-trained network (one network, two `T` values at inference) and the rolled-out-trained networks (the matching-`T` network only - do not run the `T=4` rolled-out network at `T=12` or vice versa).
- [ ] Classify `z_hat_T` by maximum cosine similarity to the Stage 1 class prototypes, reusing the exact prototype vectors from Section 1 - the same rule as Stage 1.
- [ ] Run the complete grid: 3 datasets × 2 encoders × K ∈ {5, 10, full} × {standard, rolled-out} × T ∈ {4, 12}. (4 FM result cells - standard-T4, standard-T12, rollout-T4, rollout-T12 - per dataset/encoder/K, on top of the Stage 1 baseline cell.)
- [ ] Repetition protocol: identical to Stage 1 - 3 runs per 5-shot/10-shot setting (subset seeds 0, 1, 2), 3 runs per full setting (initialization seeds 0, 1, 2).

## 6. Final evaluation and aggregation

- [ ] Evaluate top-1 accuracy on the complete official test split only after training/checkpoint selection is finished - never let test accuracy influence training or checkpoint choices.
- [ ] Report mean ± standard deviation over 3 runs for every (dataset, encoder, K, T, training-mode) combination, matching Stage 1's aggregation convention exactly.
- [ ] Compute `ΔAcc = Acc_FM - Acc_baseline` against the *matching* Stage 1 image-prototype baseline (same dataset, encoder, K) for every FM variant - never against a different K, encoder, or dataset.
- [ ] Produce a combined accuracy table: Stage 1 baseline vs. standard-FM(T=4), standard-FM(T=12), rolled-out-FM(T=4), rolled-out-FM(T=12), for every dataset/encoder/K, with `ΔAcc` shown alongside raw accuracy.
- [ ] Produce an accuracy-vs-K plot with error bars, one per dataset/encoder, comparable in style to Stage 1's training-size plot, with all 5 series (baseline + 4 FM variants) shown together.

## 7. Required figures and analysis

- [ ] **Classification results** (Section 6 above): table and/or accuracy-vs-K plot with `ΔAcc` clearly indicated.
- [ ] **Training curves**: one representative training-loss curve for standard FM and one for rolled-out training (use the same representative dataset/encoder/K as Stage 1's 10-shot loss-curve figure, for comparability). Discuss whether both approaches train stably and converge to reasonable solutions.
- [ ] **Feature-space visualizations**: for the same ~8-10 classes, test examples, and class colors already used in Stage 1's PCA/t-SNE plots, compare (a) original encoder features, (b) features after standard FM, (c) features after rolled-out FM, together with the matching class prototypes. Fit the 2D projection (PCA and/or t-SNE) **jointly** across every feature set and prototype shown in a given figure, so the three views are directly comparable positions in the same projection.
- [ ] **Flow trajectories**: for a small number of representative test examples, plot the intermediate FM states `z_hat_0, ..., z_hat_T` together with the original feature, final transported feature, and target class prototype, in one jointly-fit PCA projection per example. PCA is recommended over t-SNE here specifically because trajectories need to remain geometrically interpretable as a path, which t-SNE does not preserve.

## 8. Optional extensions

- [ ] Explore the learned flow in reverse: start from class prototypes and integrate backward toward the encoder feature distribution.
- [ ] Compare samples and prototypes at intermediate flow times (not only `t=0` and `t=1`).

## 9. Reproducibility and Stage 2 handoff

- [ ] Save per-run config, velocity-network checkpoints, training history, and test predictions, following the same run-directory / `metrics.json` / checkpoint-resume / skip-if-already-saved conventions as `01_linear_probe.ipynb`.
- [ ] Verify that frozen encoders and frozen Stage 1 prototypes are never updated during Stage 2 training - only `v_theta`'s parameters should ever receive gradients.
- [ ] Document the protocol, the standard-vs-rolled-out comparison, and the main quantitative/qualitative observations (which dataset/encoder/K/T combinations improve over the Stage 1 baseline, and whether the feature-space/trajectory plots explain why).
- [ ] Preserve the best FM checkpoints and configuration per setting as a reference point for Stage 3 (FM before a linear classifier).

## Completion criteria

Stage 2 is complete when the full grid (3 datasets × 2 encoders × K ∈ {5, 10, full} × {standard, rolled-out} × T ∈ {4, 12}) has been trained and evaluated on the identical Stage 1 splits/seeds/prototypes; `ΔAcc` against the matching Stage 1 image-prototype baseline is reported for every combination; the training curves, feature-space visualizations, and flow-trajectory visualizations are produced; and the artifacts needed for Stage 3 are preserved.
