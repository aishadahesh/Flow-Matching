# Stage 3 TODO - FM Before a Linear Classifier

Source of truth: `ref/part_3.pdf`. Stage 3 inserts an FM transformation between the **frozen** image-encoder feature and the **frozen** Stage 1 linear probe: `z -> FM -> z_hat -> (W z_hat + b) -> s`. **Do not retrain the linear classifier for the required experiments, do not recompute Stage 1 features, and do not change the Stage 1 splits, subset seeds, or K.** The direct baseline is the Stage 1 linear probe on the same dataset, encoder, subset and seed - any deviation makes `ΔAcc` uninterpretable, exactly as in Stage 2.

Status: **implemented and verified to run end to end; not yet measured on real data.** `06_fm_before_classifier.ipynb` executes all 25 code cells against a fabricated Stage 1 world (see `## Verification` below), but the Colab run over the real feature caches has not happened yet. No number in `doc/RESULTS.md` comes from Stage 3 so far.

## 0. Lock the Stage 3 experimental plan

- [x] Confirm the branch: Stage 3 builds on the **linear probe** (`01_linear_probe.ipynb`), not on the image-prototype or CLIP branches. Stage 2 transported features toward class prototypes and classified by cosine similarity; Stage 3 transports them into whatever representation a *fixed affine classifier* handles better. The two stages share the velocity network and integrator but nothing about the target or the decision rule.
- [x] Choose the datasets. The specification asks for the two chosen in Stage 1; Stage 1 was expanded to all three at the project team's request, so Stage 3 runs **all three** (DTD, FGVC-Aircraft, Flowers-102). The extra dataset costs one third more compute and keeps the Stage 3 table directly comparable to the Stage 1 and Stage 2 tables. Flowers-102 doubles as a near-saturated control where no method has room to move.
- [x] Choose one representative encoder per dataset: **DINOv2 ViT-S/14 everywhere**. Stage 2 measured that encoder choice dominates every other factor (+.14 to +.31 over ResNet-18), so the representative encoder is the one worth deploying. Feature dimension 384.
- [x] Choose one training-set size: **K = 10**, the specification's suggested default, which is also a Stage 1 grid point so the baseline already exists on Drive.
- [x] Choose a single number of Euler steps and use it throughout: **T = 12**, matching the `T` in Stage 2's headline figures. Training and inference use the same `T` for both strategies.
- [x] Record the plan in a versioned config, following the `stage1_config.json` pattern. (`stage3` section, `config_revision: 1`, written on first run.)
- [x] Create `06_fm_before_classifier.ipynb` as an independent notebook following the Drive-mount / cache-reuse / checkpoint-resume / skip-if-already-saved conventions of `01`, `02`, `04` and `05`.

## 1. Reuse Stage 1 artifacts - no new feature extraction, no classifier retraining

- [x] Load cached frozen features from `feature_cache/v{schema}/...pt`; never re-run an encoder. (`load_cached_features`; the notebook imports neither `torchvision` nor an encoder.)
- [x] Load the trained linear head from Stage 1's saved run directory. **Load `checkpoints/final.pt`, not `best_linear_head.pt`.** Both carry the same validation-selected weights - Stage 1 wrote `final.pt` after `head.load_state_dict(best_state)` - but only `final.pt` also carries `feature_mean` / `feature_std`, which are required to rebuild the `standardize` transform the head expects at its input. (`load_stage1_probe`.)
- [x] Freeze the classifier at load time with `requires_grad_(False)`, so no code path can update it by accident. Verified by a test asserting the frozen head accumulates no gradients while a joint copy does.
- [x] Re-derive each K-shot subset with the identical seeded `balanced_indices(labels, k, seed)` logic Stage 1 used, rather than re-sampling.
- [x] **Operate in the classifier's input space, not raw feature space.** Stage 1 selected a feature transform per run (`l2` or `standardize`) and trained the head on transformed features, so `z` in the Stage 3 diagram is a *transformed* feature. (`stage1_transform`, replayed from the saved mode plus statistics.)
- [x] Guard the replay: recompute the transform statistics from the re-derived subset and compare against Stage 1's saved ones; raise if they drift by more than `1e-4`. (`build_stage3_inputs`.)
- [x] **Fidelity guard (Section 4b).** Replay the complete Stage 1 inference path - cache, subset, transform, head - on the test split and require the result to match Stage 1's saved `test_accuracy` to within `1e-6` for every one of the 9 (dataset, seed) cells. This is the most important guard in the notebook: if the input space were reconstructed even slightly wrongly, every `ΔAcc` would be measured against the wrong baseline and the error would read as a Stage 3 result. Compare with a tolerance, never exactly - Stage 1 stored accuracies as float32 means.
- [x] Add a loud failure (not a silent recompute) if a required Stage 1 feature cache or linear-probe run is missing, naming the notebook that produces it.

## 2. Velocity network initialized at the identity

- [x] Reuse Stage 2's `VelocityNet` unchanged in architecture: `(feature_dim + 1) -> 512 -> 512 -> feature_dim`, SiLU, scalar `t` concatenated to the input.
- [x] **Zero the output layer.** The specification requires the FM to start close to the identity so the complete system initially behaves like the original linear probe. With `v(z, t) = 0` everywhere the Euler rollout is exactly the identity, so the system does not merely start *close* to the probe, it starts *at* it. This reuses the `zero_init_output` flag that existed in Stage 2 as a never-run pilot (`04` §6b); Stage 3 makes it required rather than optional.
- [x] **Identity guard (Section 5b).** Assert `rollout(z) == z` bit for bit and that the untrained system's test accuracy equals the linear probe's, for all 9 cells. Also assert that a *default*-initialized network would fail the guard, so the guard is not vacuous.
- [x] Note the optimization consequence in the notebook: with a zeroed output layer only that last layer receives gradient on the first step, because every earlier layer's gradient is multiplied by the zeroed output weights. Hidden layers unblock after the first update. This is why `early_stopping_patience` rises 25 -> 40 and `lr_patience` 10 -> 15 relative to Stage 2 - Stage 2's patience would risk stopping a run before it left the identity. Measured in the local smoke test: roughly 7 flat epochs before validation accuracy moved.

## 3. Strategy 1 - end-to-end rolled-out classification training

- [x] For each training feature `z`, run the complete `T`-step rollout to obtain `z_hat`, pass `z_hat` through the frozen classifier, and compute `L_cls = CE(W z_hat + b, y)`.
- [x] Backpropagate through the complete rollout and update **only** the FM parameters. (No `detach` in the training path; the head carries `requires_grad=False`.)
- [x] Implement the two regularizers the specification suggests - a penalty on the displacement `||z_hat - z||^2` and one on the mean predicted velocity magnitude - as configurable weights, defaulting to **zero**, so the required main result is the unregularized objective.
- [x] Document *why* the displacement penalty is worth having, beyond the specification suggesting it: when Stage 1 selected the `l2` transform, the head was only ever fit on unit-norm features, so an unregularized rollout can move `z_hat` off that sphere into regions where the head's logits are unconstrained and win training accuracy by exploiting the classifier rather than by improving the representation. Section 14 measures whether that happens instead of assuming it.

## 4. Strategy 2 - classifier-guided targets and standard FM training

- [x] Run `z` through the current FM to obtain `z_hat`.
- [x] Pass `z_hat` through the frozen classifier and compute the classification loss.
- [x] Take one or more gradient steps of that loss **with respect to `z_hat`** - in feature space, never in parameter space - to construct a nearby improved representation `z_hat'`. (`classifier_guided_targets`, via `torch.autograd.grad`.)
- [x] Treat `z` as the source and `z_hat'` as the target, and perform a standard conditional-FM update between them: `t ~ U(0,1)`, `z_t = (1-t) z + t z_hat'`, target velocity `z_hat' - z`, loss `||v(z_t, t) - (z_hat' - z)||^2`. Reuse Stage 2's deterministic per-batch `t` sampling so an interrupted-and-resumed run draws the same `t` values.
- [x] Recompute the targets as the FM changes during training, every `target_refresh_every` epochs over the whole training set.
- [x] **Express the feature-space step size as a fraction of the mean training-feature norm, not as an absolute distance.** Stage 1 chooses the transform per run, so an absolute step of 0.5 is a 50% displacement under `l2` (`||z|| = 1`) but roughly 2.5% under `standardize` (`||z|| ~ sqrt(384)`). Making it relative keeps one configured number comparable across datasets and transforms. This was caught by the local test, not by reading the specification.
- [x] Implement all three constraint modes the specification asks to experiment with - `none`, `unit`, `trust_region` - and default to `trust_region`. Rationale recorded in the notebook: under `none` already-confident samples take naturally small steps (their CE gradient is small), which is desirable, but raw gradient norms vary by orders of magnitude across samples; under `unit` every sample moves the same distance including correctly classified ones; `trust_region` keeps the gradient's own scaling while capping the worst case.
- [x] Verify the guidance step actually does what it claims: a unit test asserts that the constructed target lowers the classification loss under all three constraint modes and both feature transforms.

## 5. Training, checkpointing and model selection

- [x] Use **one** checkpoint-selection rule for both strategies: highest validation top-1 accuracy of the complete system `head(rollout(z_val))`. Two objectives that minimize different losses are still selected by the same rule, and that rule is the one Stage 1 used to select the linear probe. Selecting each method on its own training loss would compare a well-tuned method against a badly-tuned one.
- [x] **Evaluate and checkpoint epoch 0**, before any update. Because the FM is exactly the identity there, epoch 0 *is* the linear probe, and seeding the checkpoint with it makes the selection rule "keep the FM only if it helps on validation". This was added after the local test showed a run that only ever degraded still saved a degraded network.
- [x] **State the consequence wherever results are reported:** validation `ΔAcc` is `>= 0` by construction, so only **test** `ΔAcc` carries information about whether Stage 3 helped. (Stated in the notebook title cell, in Section 8, and below in Section 8 of this file.)
- [x] Report how many runs selected epoch 0 - i.e. learned nothing usable - rather than letting them disappear into a mean. (Section 10 prints the count and lists them.)
- [x] Reuse Stage 2's checkpoint-resume convention: `latest.pt` / `best.pt`, resume only when the saved `config` matches exactly, and skip runs whose outputs already exist on Drive. Verified: re-running the grid cell loads all 18 runs from disk and retrains none.
- [x] Record per-epoch training loss, validation loss and validation accuracy to `history.csv` for every run.

## 6. Final evaluation and aggregation

- [x] Touch the test split only after training and checkpoint selection are finished.
- [x] Run the required grid: 3 datasets x DINOv2 x K=10 x 3 subset seeds x 2 strategies = **18 trained FM layers**, plus 9 loaded linear-probe baseline rows at zero training cost, for 27 rows in `run_metrics.csv`.
- [x] Report mean +/- standard deviation over the three subset seeds, matching the Stage 1 and Stage 2 aggregation convention.
- [x] Compute `ΔAcc` against the *matching* Stage 1 linear probe - same dataset, same encoder, same subset, same seed - so the comparison is paired at the seed level and the reported `ΔAcc` is a mean of per-seed differences, not a difference of means over different data.
- [x] Record `relative_displacement = ||z_hat - z|| / ||z||` on the test split for every run, so "how far did the FM actually move the representation" is a measured quantity rather than an impression from a scatter plot.

## 7. Required figures and analysis

- [x] **Classification results**: top-1 test accuracy for the Stage 1 linear probe and both Stage 3 methods on every dataset, plus the change relative to the linear-probe baseline. (Table in Section 10, `accuracy_and_delta.png`, `accuracy_summary.csv`.)
- [x] **Training behaviour**: representative training and validation curves for both Stage 3 methods. Plotted with two reading notes stated on the figure - epoch 0 is the identity FM so its train loss is `NaN` by construction and is omitted, and the two strategies minimize different quantities (endpoint cross-entropy vs. velocity MSE against a moving target) so their loss magnitudes must not be compared. A rising Strategy 2 loss is usually a refreshed target, not divergence.
- [x] **Feature-space visualization**: for a readable subset of classes, the original features `z` and the transported `z_hat` under both Stage 3 methods, on the same test examples with the same class colors across all panels. The embedding is fit **jointly** over all three feature sets and the panels **share axis limits**, following the Stage 2 convention - a joint fit that is then autoscaled per panel rescales each view independently and hides the movement the figure exists to show. Both projections produced: PCA (Section 12) and t-SNE (Section 12b).
- [x] **Paired significance** (house convention, not required by the specification): percentile bootstrap CI over test examples and an exact McNemar test on the discordant pairs, per (dataset, method, seed), against the linear probe on the same split. A `ΔAcc` whose CI spans zero is to be reported as "no measurable difference", not as a small win.

## 8. Variant comparisons the specification invites

The specification calls its two strategies "structured starting points rather than fixed recipes" and encourages modifying details and comparing variants. Both sweeps run on **subset seed 0 only** - they are variant comparisons, not replacements for the three-seed main result.

- [x] **Strategy 1 regularization** (Section 14, `RUN_REGULARIZATION_SWEEP`): unregularized vs. displacement penalty at 1e-2 and 1e-1 vs. velocity penalty at 1e-3, reporting accuracy *and* `relative_displacement` together, so the off-manifold hypothesis in Section 3 above is testable rather than rhetorical.
- [x] **Strategy 2 guidance knobs** (Section 15, `RUN_GUIDANCE_SWEEP`): step size (0.02 / 0.1 / 0.30 of the feature norm), number of target-improvement steps (1 / 3 / 10), constraint mode (`trust_region` / `none` / `unit`), and refresh period (every epoch vs. every 10). One knob varied at a time from the default - these are the four choices the specification names.

## 9. Stage 2's optional analyses, adapted to Stage 3

`ref/stage_2.pdf` ends with an optional invitation to explore the learned flow in reverse starting
from the class prototypes, and to compare samples and prototypes at intermediate flow times. Both
were done for Stage 2 (`04` §12-§13, measured in `RESULTS.md`). The Stage 3 versions are **not the
same experiment**, because Stage 3 has no class prototypes - the flow's destination is a classifier,
not a point.

- [x] **Intermediate flow times** (Section 17). Classify every intermediate Euler state `z_hat_k`
  through the frozen head over the complete test split, tracking accuracy, cross-entropy, logit
  margin, and relative displacement. The margin is the informative one: it moves continuously where
  accuracy moves in jumps, showing whether the flow pushes samples across the decision boundary or
  merely deeper into the region they already occupied.
- [x] **Anchor the curve at both ends and assert it.** Under identity initialization `z_hat_0 = z`
  exactly, so `t=0` accuracy *is* the Stage 1 linear probe, and `t=1` is the Section 10 result. The
  notebook raises if either endpoint drifts. The curve is `ΔAcc` unrolled.
- [x] **Point-cloud snapshots** at `t` in {0, .25, .5, .75, 1} in a jointly-fit PCA per dataset with
  shared axes, on the same class subset as Section 12.
- [x] **Validation-selected stopping time `t*`** (Section 17c). `t*` is chosen on validation only;
  the test curve is computed for reporting but the index was fixed before test was touched.
  `oracle_best` is the price of choosing honestly. Ties break toward the earliest step, so an
  identity-selected run reports `t* = 0` from a flat curve - documented so it is not misread as a
  finding.
- [x] **Reverse flow from two anchors** (Section 18), since there are no prototypes to start from:
  - *classifier class templates* - the frozen head's weight rows `w_c`, rescaled to the mean feature
    norm, integrated backward. Asks what feature the flow thinks maps toward each class direction.
    The untouched template's recovery is the **pre-transport reference, not a ceiling**: `w_c` need
    not sit inside its class cloud, so reverse flow can beat it. This is precisely the error
    `RESULTS.md` records as a correction to the Stage 2 write-up, and it is available to repeat here.
  - *a round trip on the class means* - forward-flow each class mean, integrate back, measure the
    error. This anchor has an unambiguous ideal of zero, which the local test asserts for an identity
    field, so it is the one to lean on when interpreting.
- [x] Record the caveat in the notebook: the backward pass evaluates the field at `t` in
  {1, ..., 1/T} while the forward pass uses {0, ..., 1-1/T}, so neither is an exact inverse even for
  a perfectly learned field; and a flow trained to make a classifier's job easier has every reason to
  be contractive, which is not invertible in principle. These recover a plausible pre-image, not the
  original point.
- [x] **Five-panel reverse strip** (Section 18b) showing where the templates travel, drawn over the
  real test features.

## 10. Optional extension - jointly fine-tuning the classifier

- [x] After the frozen-classifier experiments, unfreeze the pretrained linear classifier and optimize it jointly with the FM, at its own smaller learning rate, with `unfreeze_epoch` available for delayed unfreezing. (Section 16, `RUN_JOINT_FINETUNE`.)
- [x] **Add the control that makes the comparison interpretable.** Unfreezing adds two things at once - the FM *and* extra training of the classifier itself - so a joint run that beats the Stage 1 probe may simply be a probe that trained longer. A `head_only` control continues the same Stage 1 head for the same epoch budget under the same optimizer and the same validation-checkpointing rule, with no FM at all. **The number that means anything is joint versus that control, not joint versus Stage 1.** The specification does not ask for this control; without it the extension cannot support a claim.

## Verification performed before the Colab run

Stage 3 was developed against a fabricated Stage 1 world - cached features plus linear-probe runs written in exactly the layout `01_linear_probe.ipynb` produces - so the code was known to run before consuming Colab time. Two suites, neither of which ships in the repository:

- **Unit/behaviour suite**, 58 checks over both feature transforms: probe loading, freezing, the fidelity and identity guards (including a negative control proving the identity guard has teeth), all four training variants, the resume path, gradient flow to a joint head and its absence from a frozen one, the bootstrap/McNemar helpers, and that the guidance target lowers classification loss under all three constraint modes. Plus a learnability check: against a deliberately undertrained frozen head, both strategies must beat epoch 0 - measured at validation .117 -> .433 and test .056 -> .300 for Strategy 1, .117 -> .383 / .278 for Strategy 2.
- **Notebook integration run**: every one of the 25 code cells parsed and executed against three fabricated datasets and three seeds, producing all 21 expected CSV and PNG artifacts and a 27-row `run_metrics.csv`; re-running the grid cell then loaded all 18 runs from disk and retrained none.
- **Optional-analysis suite**, 40 checks over both feature transforms: both curve endpoints self-anchoring, displacement starting at exactly zero and never decreasing, `accuracy_by_step` agreeing with the full metrics table, `t*` never beating the test-curve oracle, every recovery rate inside [0,1], and an identity field round-tripping with exactly zero error and a flat accuracy curve. It also confirms the training margin rises (-0.23 -> 3.59) and training cross-entropy collapses (1.76 -> 0.11) while *test* accuracy falls - overfitting at K=10, which is what motivates the `t*` ablation.

Three defects were found and fixed this way rather than on Colab: the missing epoch-0 checkpoint described in Section 5, the transform-dependent guidance step size described in Section 4, and float32-vs-float64 comparison against Stage 1's saved accuracies described in Section 1.

## Open items

- [ ] **Run `06_fm_before_classifier.ipynb` end to end on Colab** over the real Stage 1 caches. Nothing below can be filled in until this happens.
- [ ] Record the measured Stage 3 results in `doc/RESULTS.md`, including the epoch-0 count from Section 10 and the paired-significance summary from Section 13.
- [ ] Report the two variant sweeps: whether the displacement penalty changes anything, and which guidance knobs matter.
- [ ] Report the joint fine-tuning extension **against the `head_only` control**, not against the Stage 1 probe.
- [ ] Report the two optional analyses: the accuracy/margin curves over flow time with both endpoints anchored, whether `t*` beats `t=1`, and the two reverse-flow anchors with the pre-transport reference stated correctly.
- [ ] If Stage 3 turns out not to beat the linear probe, say so plainly and explain the mechanism. The Stage 2 precedent is the honest framing to follow: "the FM layer is a real improvement over the prototype rule it replaces, not a replacement for a discriminatively trained classifier." Stage 3 asks a harder question than Stage 2 did - the linear probe was already the strongest method in the project at every full-data setting, and a frozen affine classifier applied to a representation it was itself fit on leaves little obvious headroom.
