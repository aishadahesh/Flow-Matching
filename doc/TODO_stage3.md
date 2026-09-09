# Stage 3 TODO - FM Before a Linear Classifier

Source of truth: `ref/part_3.pdf`. Stage 3 inserts an FM transformation between the **frozen** image-encoder feature and the **frozen** Stage 1 linear probe: `z -> FM -> z_hat -> (W z_hat + b) -> s`. **Do not retrain the linear classifier for the required experiments, do not recompute Stage 1 features, and do not change the Stage 1 splits, subset seeds, or K.** The direct baseline is the Stage 1 linear probe on the same dataset, encoder, subset and seed - any deviation makes `ΔAcc` uninterpretable, exactly as in Stage 2.

Status (2026-09-09): **implementation revision 2 completed on Colab on 2026-09-08 and its results are transcribed; revision 3 is code-complete and awaiting a fresh run.** Revision 3 preserves the required Stage 3 comparison while separating the guided radius, step fraction, gradient normalization, and constraint; it implements the documented per-sample radius, corrects the trust-region diagnostics, and promotes the specification-permitted λ=1 displacement penalty to the end-to-end main run while retaining λ=0 as a control. Its configuration revision invalidates the affected caches automatically.

The notebook replay saved on 2026-09-09 still declared revision 2 and loaded all existing revision-2
main-grid caches. It reproduced the locked result table but did not close the revision-3 run item;
the revision-3 source has been restored and the mismatched outputs removed.

**Remaining provenance discrepancy:** the current K=10 DINOv2 artifacts give DTD .7167, Aircraft .5285, Flowers-102 .9935, while the older Stage 1 aggregate table records .7181, .5096, .9932. Revision 2 passed validation and test replay within 1e-6 and bound every result to content signatures of the exact probe and feature caches, so the new paired Stage 3 deltas are valid. The older Stage 1 aggregate row still needs its artifact provenance reconciled.

## 0. Lock the Stage 3 experimental plan

- [x] Confirm the branch: Stage 3 builds on the **linear probe** (`01_linear_probe.ipynb`), not on the image-prototype or CLIP branches. Stage 2 transported features toward class prototypes and classified by cosine similarity; Stage 3 transports them into whatever representation a *fixed affine classifier* handles better. The two stages share the velocity network and integrator but nothing about the target or the decision rule.
- [x] Choose the datasets. The specification asks for the two chosen in Stage 1; Stage 1 was expanded to all three at the project team's request, so Stage 3 runs **all three** (DTD, FGVC-Aircraft, Flowers-102). The extra dataset costs one third more compute and keeps the Stage 3 table directly comparable to the Stage 1 and Stage 2 tables. Flowers-102 doubles as a near-saturated control where no method has room to move.
- [x] Choose one representative encoder per dataset: **DINOv2 ViT-S/14 everywhere**. Stage 2 measured that encoder choice dominates every other factor (+.14 to +.31 over ResNet-18), so the representative encoder is the one worth deploying. Feature dimension 384.
- [x] Choose one training-set size: **K = 10**, the specification's suggested default, which is also a Stage 1 grid point so the baseline already exists on Drive.
- [x] Choose a single number of Euler steps and use it throughout: **T = 12**, matching the `T` in Stage 2's headline figures. Training and inference use the same `T` for both strategies.
- [x] Record the plan in a versioned config, following the `stage1_config.json` pattern. (`stage3` section, `config_revision: 2`, committed in the shared config.)
- [x] Create `06_fm_before_classifier.ipynb` as an independent notebook following the Drive-mount / cache-reuse / checkpoint-resume / skip-if-already-saved conventions of `01`, `02`, `04` and `05`.

## 1. Reuse Stage 1 artifacts - no new feature extraction, no classifier retraining

- [x] Load cached frozen features from `feature_cache/v{schema}/...pt`; never re-run an encoder. (`load_cached_features`; the notebook imports neither `torchvision` nor an encoder.)
- [x] Load the trained linear head from Stage 1's saved run directory. **Load `checkpoints/final.pt`, not `best_linear_head.pt`.** Both carry the same validation-selected weights - Stage 1 wrote `final.pt` after `head.load_state_dict(best_state)` - but only `final.pt` also carries `feature_mean` / `feature_std`, which are required to rebuild the `standardize` transform the head expects at its input. (`load_stage1_probe`.)
- [x] Freeze the classifier at load time with `requires_grad_(False)`, so no code path can update it by accident. Verified by a test asserting the frozen head accumulates no gradients while a joint copy does.
- [x] Re-derive each K-shot subset with the identical seeded `balanced_indices(labels, k, seed)` logic Stage 1 used, rather than re-sampling.
- [x] **Operate in the classifier's input space, not raw feature space.** Stage 1 selected a feature transform per run (`l2` or `standardize`) and trained the head on transformed features, so `z` in the Stage 3 diagram is a *transformed* feature. (`stage1_transform`, replayed from the saved mode plus statistics.)
- [x] Guard the replay: recompute the transform statistics from the re-derived subset and compare against Stage 1's saved ones; raise if they drift by more than `1e-4`. (`build_stage3_inputs`.)
- [x] **Fidelity guard (Section 4b).** Replay the complete Stage 1 inference path on validation and test and require both accuracies to match Stage 1 within `1e-6` for every cell. After the guard, use exact integer-count ratios so an identity FM reports exactly zero `ΔAcc` rather than float32 serialization noise.
- [x] Content-sign every loaded feature split and frozen probe, and include the combined signature in every Stage 3 run config so regenerated Stage 1 artifacts cannot silently reuse stale Stage 3 metrics.
- [x] Add a loud failure (not a silent recompute) if a required Stage 1 feature cache or linear-probe run is missing, naming the notebook that produces it.

## 2. Velocity network initialized at the identity

- [x] Reuse Stage 2's `VelocityNet` unchanged in architecture: `(feature_dim + 1) -> 512 -> 512 -> feature_dim`, SiLU, scalar `t` concatenated to the input.
- [x] **Zero the output layer.** The specification requires the FM to start close to the identity so the complete system initially behaves like the original linear probe. With `v(z, t) = 0` everywhere the Euler rollout is exactly the identity, so the system does not merely start *close* to the probe, it starts *at* it. This reuses the `zero_init_output` flag that existed in Stage 2 as a never-run pilot (`04` §6b); Stage 3 makes it required rather than optional.
- [x] **Identity guard (Section 5b).** Assert `rollout(z) == z` bit for bit and that the untrained system's test accuracy equals the linear probe's, for all 9 cells. Also assert that a *default*-initialized network would fail the guard, so the guard is not vacuous.
- [x] Note the optimization consequence in the notebook: with a zeroed output layer only that last layer receives gradient on the first step, because every earlier layer's gradient is multiplied by the zeroed output weights. Hidden layers unblock after the first update. This is why `early_stopping_patience` rises 25 -> 40 and `lr_patience` 10 -> 15 relative to Stage 2 - Stage 2's patience would risk stopping a run before it left the identity. Measured in the local smoke test: roughly 7 flat epochs before validation accuracy moved.

## 3. Strategy 1 - end-to-end rolled-out classification training

- [x] For each training feature `z`, run the complete `T`-step rollout to obtain `z_hat`, pass `z_hat` through the frozen classifier, and compute `L_cls = CE(W z_hat + b, y)`.
- [x] Backpropagate through the complete rollout and update **only** the FM parameters. (No `detach` in the training path; the head carries `requires_grad=False`.)
- [x] Implement the two regularizers the specification suggests as scale-free penalties, dividing squared displacement and velocity magnitudes by `||z||^2`. Both weights default to **zero**, so the required main result remains unregularized.
- [x] Document *why* the displacement penalty is worth having, beyond the specification suggesting it: when Stage 1 selected the `l2` transform, the head was only ever fit on unit-norm features, so an unregularized rollout can move `z_hat` off that sphere into regions where the head's logits are unconstrained and win training accuracy by exploiting the classifier rather than by improving the representation. Section 14 measures whether that happens instead of assuming it.

## 4. Strategy 2 - classifier-guided targets and standard FM training

- [x] Run `z` through the current FM to obtain `z_hat`.
- [x] Pass `z_hat` through the frozen classifier and compute the classification loss.
- [x] Take one or more gradient steps of that loss **with respect to `z_hat`** - in feature space, never in parameter space - to construct a nearby improved representation `z_hat'`. (`classifier_guided_targets`, via `torch.autograd.grad`.)
- [x] Treat `z` as the source and `z_hat'` as the target, and perform a standard conditional-FM update between them: `t ~ U(0,1)`, `z_t = (1-t) z + t z_hat'`, target velocity `z_hat' - z`, loss `||v(z_t, t) - (z_hat' - z)||^2`. Reuse Stage 2's deterministic per-batch `t` sampling so an interrupted-and-resumed run draws the same `t` values.
- [x] Recompute the targets as the FM changes during training, every `target_refresh_every` epochs over the whole training set.
- [x] **Separate radius from step length and make both per-sample relative quantities.** Revision 3 uses `rho_i = radius_fraction * ||z_i||` and a normalized step of `step_fraction * rho_i`, so changing the step no longer silently changes the constraint radius. This keeps one configuration comparable across `l2` and standardized feature spaces.
- [x] Make gradient normalization and projection independent. Revision 3 defaults to unit-gradient steps inside a source-centred trust region, while the sweep can disable either component separately. The old `none` / `unit` / `trust_region` mode conflated two scientific choices and could not test the strongest combination.
- [x] Source-anchor the default trust region so repeated refreshes cannot accumulate unbounded target drift; retain the lowest-CE feasible iterate and record the trust-region hit rate. Keep current-anchor and non-monotone variants in the sweep.
- [x] Correct the diagnostics: `guidance_trust_region_hit_rate` now measures final selected targets on the boundary, `guidance_projection_rate` measures proposals that required projection, and target CE is recorded before projection, after projection, and after guidance.
- [x] Add an executable guard asserting source-bounded monotone targets and an active standard-FM gradient before the main grid starts.

## 5. Training, checkpointing and model selection

- [x] Use **one** checkpoint-selection rule for both strategies: highest validation top-1 accuracy of the complete system `head(rollout(z_val))`. Two objectives that minimize different losses are still selected by the same rule, and that rule is the one Stage 1 used to select the linear probe. Selecting each method on its own training loss would compare a well-tuned method against a badly-tuned one.
- [x] **Evaluate and checkpoint epoch 0**, before any update. Because the FM is exactly the identity there, epoch 0 *is* the linear probe, and seeding the checkpoint with it makes the selection rule "keep the FM only if it helps on validation". This was added after the local test showed a run that only ever degraded still saved a degraded network.
- [x] **State the consequence wherever results are reported:** validation `ΔAcc` is `>= 0` by construction, so only **test** `ΔAcc` carries information about whether Stage 3 helped. (Stated in the notebook title cell, in Section 8, and below in Section 8 of this file.)
- [x] Report how many runs selected epoch 0 - i.e. learned nothing usable - rather than letting them disappear into a mean. (Section 10 prints the count and lists them.)
- [x] Reuse Stage 2's checkpoint-resume convention, but strengthen compatibility: the complete effective optimization config, implementation revision, and upstream artifact signature must match exactly. Revision 1 caches cannot be reused by revision 2, and revision 2 guided caches cannot be reused by revision 3.
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

- [x] **Strategy 1 regularization** (Section 14): unregularized vs. relative displacement penalties at 1, 10 and 100 vs. relative velocity penalty at 1e-3, reporting accuracy and measured displacement together.
- [x] **Strategy 2 guidance knobs** (Section 15): the four choices named by the specification plus current-anchor and non-monotone ablations, with trust-region hit rate reported.

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

## 11. Diagnostic figures beyond the required set

The required figures answer "did it work". These answer "what did it do", which is what a flat or
small `ΔAcc` actually needs. All were added after the first Colab run.

- [x] **Which predictions changed** (Section 19). Every test sample sorted into `fixed` / `broken` /
  `both_right` / `both_wrong` against the probe on the same split. `fixed - broken` is exactly the
  numerator of `ΔAcc`, and `fixed + broken` is the churn underneath it - a +.004 mean is consistent
  with "fixed 8, broke 0" and with "fixed 200, broke 192", and those are different findings. Verified
  by a test asserting the four categories are exhaustive and that `fixed - broken` reproduces `ΔAcc`.
- [x] Alongside it, the two panels that say *what distinguishes* those groups: how far the flow moved
  each one, and what the probe's own logit margin was on them. A flow that rescues samples the probe
  was unsure about is behaving sensibly; one that breaks confidently-correct samples is not, and no
  aggregate would show it.
- [x] **Translation versus transport** (Section 19b), added after the animation showed the cloud
  appearing to slide bodily across the frame. A large *common* displacement is the cheapest way for
  an FM in front of an affine classifier to buy accuracy: adding a fixed vector `m` to every feature
  shifts the logits by `W m`, a per-class constant, which is exactly a re-fit of the classifier's
  bias. The displacement is therefore split into a common translation and a per-sample residual and
  each is scored alone (`shift_only`, `residual_only`), with `m` estimated on the training subset so
  the control never sees test displacements. If `shift_only` recovers most of the gain, the result is
  a bias correction a single vector could deliver and must be reported as such, not as flow matching
  doing the work. Verified against synthetic fields: a pure translation is attributed entirely to
  `shift_only`, a pure contraction entirely to `residual_only`.
- [x] Record why direction relative to the class clusters is **not** evidence of a bug in Stage 3,
  unlike Stage 2. Stage 2's flow had an explicit prototype target, so features visibly moved toward
  their own class and anything else would have been wrong. Stage 3's objective is the frozen
  classifier's logits, and nothing in it rewards staying near the class cloud or near `z`; a flow
  that leaves the data manifold entirely is a valid minimum. The animation was separately verified
  faithful - frame 0 asserted equal to `z`, the final frame asserted equal to the endpoint Section 10
  scores, and cosine `+1.0` against a field with known constant velocity.
- [x] **Per-class effects** (Section 20). Per-class accuracy before vs after as a scatter about the
  diagonal, plus the largest movers by name. Guards against a flat mean that hides equal numbers of
  helped and hurt classes.
- [x] **Flow trajectories** (Section 21). The Stage 2 §11 figure that Stage 3 was missing: paths
  through the learned field in a jointly-fit PCA. Examples are chosen **by outcome** rather than at
  random, so a successful and an unsuccessful transport appear side by side instead of averaged.
- [x] **Effect sizes at a glance** (Section 22). Paired slope plot (one line per dataset x seed,
  which is the honest view of a paired comparison), a forest plot of every cell's bootstrap CI with
  McNemar significance marked, and the `ΔAcc` heatmap. The Section 10 bar chart is the least
  informative of the four and should not be the figure anyone quotes.
- [x] **The rollout as video** (Section 23). Two animations of the same `T`-step Euler rollout the
  rest of the notebook reports: the whole test cloud moving from `z` to `ẑ` with a live accuracy
  readout at each Euler state, and individual paths coloured by **outcome** rather than by class, so
  a `fixed` path and a `broken` path appear in the same frame. MP4 via `ffmpeg` where available (it
  is on Colab), animated GIF otherwise; both are saved and embedded inline. The dataset animated is
  the one with the largest `|ΔAcc|`, ranked on the absolute value so a flow that clearly makes things
  worse is just as likely to be shown.
- [x] Two honesty constraints on that video, both stated on the figure: frames between Euler states
  are **linear interpolation added only for legibility** - the model produces `T + 1` states and
  nothing between them, the caption reports the true Euler step, and the accuracy readout changes
  only at a real state; and the PCA is fit once over all states and held fixed, so points move
  because the features move, not because the projection is being refitted per frame.
- [x] **A 2D simulation** (Section 23c), because every other figure in this notebook is either a
  number or a projection of something that is not two-dimensional. Two concentric rings - an inner
  disc inside an annulus - run through the *same* `train_stage3` / `euler_rollout` /
  `evaluate_system` functions, so it is a demonstration and an end-to-end check at once. The frozen
  decision regions are drawn directly, the learned velocity field is shown as a quiver at three flow
  times (in 384 dimensions it can only be inferred from summary statistics), and the whole thing is
  animated.
- [x] Two mistakes fixed in that toy before it was worth keeping. The first design used a bimodal
  class with lobes on opposite sides, which *looks* non-linear but is not - a classifier can give
  that class near-zero weight and hand it every region the others do not claim, so the probe reached
  .92 and there was nothing to demonstrate. The second was training the toy probe without validation
  selection, which on a problem with no linear signal landed it *below* chance at .276 and inflated
  the flow's apparent gain; it now uses Stage 1's rule and sits at .458, where a line belongs.
- [x] Report the three-way toy result rather than a flattering two-way one. Revision 2 measured:
  probe .4583, end-to-end .9948 (+.5365), classifier-guided at both the default step and step 0.5
  .4583 (+.0000), with both guided runs selecting epoch 0. The earlier step-0.5 guided gain belonged
  to revision 1's current-endpoint anchoring; it does not survive the source-anchored trust region.
- [x] **Inline report** (Section 25). Reloads every saved CSV and PNG from Drive and renders them in
  the notebook. This exists because the skip-if-already-saved logic means a rerun does not re-execute
  the plotting cells' upstream work, so on a resumed run the figures would otherwise not reappear -
  only a list of filenames. Animations are re-embedded as base64, which grows the saved `.ipynb` - noted in the section itself.

## 10. Optional extension - jointly fine-tuning the classifier

> *"After completing the frozen-classifier experiments, you may also unfreeze the pretrained linear classifier and jointly optimize the FM transformation and classifier. Compare this with the frozen-classifier setting and with the original Stage 1 linear probe. You may experiment with choices such as different learning rates for the FM and classifier, delayed unfreezing, or additional regularization."*

- [x] Unfreeze the classifier and optimize it jointly with the FM, at its own learning rate, through the same `train_stage3` loop and the same validation-checkpointing rule.
- [x] **Compare against the frozen-classifier setting**, which the specification asks for explicitly. An earlier version of Section 16 compared only against the Stage 1 probe and the head-only control and silently omitted the frozen Stage 3 rows - the comparison that sentence actually names. The table now carries all six: `linear_probe`, `e2e__frozen`, `guided__frozen`, `head_only`, `e2e__joint`, `guided__joint`.
- [x] **Compare against the original Stage 1 linear probe**, also as asked.
- [x] **Add the `head_only` control the specification does not ask for.** Unfreezing adds the FM *and* extra classifier training at once, so a joint run that beats Stage 1 may just be a probe that trained longer. The control continues the same head for the same budget with no FM at all. Any joint gain smaller than the control's is not evidence for flow matching.
- [x] **Attribute every joint run four ways** on the same test split: `probe` = `W0 z + b0`, `fm_only` = `W0 z_hat + b0` (the FM judged by the *original* classifier), `head_only` = `W1 z + b1` (the tuned classifier with no FM), `full` = `W1 z_hat + b1`. Without this, "unfreeze everything and accuracy rose" does not say which half did it. Verified exact: with an identity FM, `fm_only == probe` and `full == head_only`.
- [x] Record `head_drift` per run - absolute and relative weight drift, bias drift, per-class weight cosine - so "how far did the classifier travel to buy that gain" is measured rather than assumed.
- [x] **Sweep the three things the specification names** (Section 16b; subset seed 0, end-to-end only, one knob at a time from the default):
  - *learning rates for the FM and classifier*: `head_lr` at `0.01x`, `0.1x` (default) and `1.0x` the FM learning rate. The head starts at a good solution and the FM starts at nothing, so an equal rate lets the head move fastest exactly when the FM has no signal yet.
  - *delayed unfreezing*: `unfreeze_epoch` in {1, 10, 30}. With a zero-initialised output the FM's first epochs are nearly flat, and that is precisely the interval in which an unfrozen head would absorb all the gradient.
  - *additional regularization*: an anchor `lambda * (||W - W0||^2 + ||b - b0||^2)` at {0, 1e-2, 1}. This rather than weight decay: weight decay pulls the head toward zero, which is not where it started, whereas the anchor pulls it toward the pretrained solution and so directly controls how much of any gain is allowed to come from moving the classifier. Measured to reduce drift.
- [x] Figure (Section 16c): four-way attribution bars, a drift-versus-gain scatter, and the sweep. The scatter is the one to read - high and to the left is a real gain with the classifier barely moved, which is what Stage 3 set out to test; high and to the right is a fine-tuned classifier wearing a flow-matching hat.
- [x] Print an explicit per-cell verdict ("the FM did the work" / "the classifier did the work - the flow is decoration here" / "both contributed") rather than leaving the reader to derive it.

## 11. Diagnostic figures beyond the required set

The required figures answer "did it work". These answer "what did it do", which is what a flat or
small `ΔAcc` actually needs. All were added after the first Colab run.

- [x] **Which predictions changed** (Section 19). Every test sample sorted into `fixed` / `broken` /
  `both_right` / `both_wrong` against the probe on the same split. `fixed - broken` is exactly the
  numerator of `ΔAcc`, and `fixed + broken` is the churn underneath it - a +.004 mean is consistent
  with "fixed 8, broke 0" and with "fixed 200, broke 192", and those are different findings. Verified
  by a test asserting the four categories are exhaustive and that `fixed - broken` reproduces `ΔAcc`.
- [x] Alongside it, the two panels that say *what distinguishes* those groups: how far the flow moved
  each one, and what the probe's own logit margin was on them. A flow that rescues samples the probe
  was unsure about is behaving sensibly; one that breaks confidently-correct samples is not, and no
  aggregate would show it.
- [x] **Translation versus transport** (Section 19b), added after the animation showed the cloud
  appearing to slide bodily across the frame. A large *common* displacement is the cheapest way for
  an FM in front of an affine classifier to buy accuracy: adding a fixed vector `m` to every feature
  shifts the logits by `W m`, a per-class constant, which is exactly a re-fit of the classifier's
  bias. The displacement is therefore split into a common translation and a per-sample residual and
  each is scored alone (`shift_only`, `residual_only`), with `m` estimated on the training subset so
  the control never sees test displacements. If `shift_only` recovers most of the gain, the result is
  a bias correction a single vector could deliver and must be reported as such, not as flow matching
  doing the work. Verified against synthetic fields: a pure translation is attributed entirely to
  `shift_only`, a pure contraction entirely to `residual_only`.
- [x] Record why direction relative to the class clusters is **not** evidence of a bug in Stage 3,
  unlike Stage 2. Stage 2's flow had an explicit prototype target, so features visibly moved toward
  their own class and anything else would have been wrong. Stage 3's objective is the frozen
  classifier's logits, and nothing in it rewards staying near the class cloud or near `z`; a flow
  that leaves the data manifold entirely is a valid minimum. The animation was separately verified
  faithful - frame 0 asserted equal to `z`, the final frame asserted equal to the endpoint Section 10
  scores, and cosine `+1.0` against a field with known constant velocity.
- [x] **Per-class effects** (Section 20). Per-class accuracy before vs after as a scatter about the
  diagonal, plus the largest movers by name. Guards against a flat mean that hides equal numbers of
  helped and hurt classes.
- [x] **Flow trajectories** (Section 21). The Stage 2 §11 figure that Stage 3 was missing: paths
  through the learned field in a jointly-fit PCA. Examples are chosen **by outcome** rather than at
  random, so a successful and an unsuccessful transport appear side by side instead of averaged.
- [x] **Effect sizes at a glance** (Section 22). Paired slope plot (one line per dataset x seed,
  which is the honest view of a paired comparison), a forest plot of every cell's bootstrap CI with
  McNemar significance marked, and the `ΔAcc` heatmap. The Section 10 bar chart is the least
  informative of the four and should not be the figure anyone quotes.
- [x] **The rollout as video** (Section 23). Two animations of the same `T`-step Euler rollout the
  rest of the notebook reports: the whole test cloud moving from `z` to `ẑ` with a live accuracy
  readout at each Euler state, and individual paths coloured by **outcome** rather than by class, so
  a `fixed` path and a `broken` path appear in the same frame. MP4 via `ffmpeg` where available (it
  is on Colab), animated GIF otherwise; both are saved and embedded inline. The dataset animated is
  the one with the largest `|ΔAcc|`, ranked on the absolute value so a flow that clearly makes things
  worse is just as likely to be shown.
- [x] Two honesty constraints on that video, both stated on the figure: frames between Euler states
  are **linear interpolation added only for legibility** - the model produces `T + 1` states and
  nothing between them, the caption reports the true Euler step, and the accuracy readout changes
  only at a real state; and the PCA is fit once over all states and held fixed, so points move
  because the features move, not because the projection is being refitted per frame.
- [x] **A 2D simulation** (Section 23c), because every other figure in this notebook is either a
  number or a projection of something that is not two-dimensional. Two concentric rings - an inner
  disc inside an annulus - run through the *same* `train_stage3` / `euler_rollout` /
  `evaluate_system` functions, so it is a demonstration and an end-to-end check at once. The frozen
  decision regions are drawn directly, the learned velocity field is shown as a quiver at three flow
  times (in 384 dimensions it can only be inferred from summary statistics), and the whole thing is
  animated.
- [x] Two mistakes fixed in that toy before it was worth keeping. The first design used a bimodal
  class with lobes on opposite sides, which *looks* non-linear but is not - a classifier can give
  that class near-zero weight and hand it every region the others do not claim, so the probe reached
  .92 and there was nothing to demonstrate. The second was training the toy probe without validation
  selection, which on a problem with no linear signal landed it *below* chance at .276 and inflated
  the flow's apparent gain; it now uses Stage 1's rule and sits at .458, where a line belongs.
- [x] Report the three-way toy result rather than a flattering two-way one. Revision 2 measured:
  probe .4583, end-to-end .9948 (+.5365), classifier-guided at both the default step and step 0.5
  .4583 (+.0000), with both guided runs selecting epoch 0. The earlier step-0.5 guided gain belonged
  to revision 1's current-endpoint anchoring; it does not survive the source-anchored trust region.
- [x] **Inline report** (Section 25). Reloads every saved CSV and PNG from Drive and renders them in
  the notebook. This exists because the skip-if-already-saved logic means a rerun does not re-execute
  the plotting cells' upstream work, so on a resumed run the figures would otherwise not reappear -
  only a list of filenames. Animations are re-embedded as base64, which grows the saved `.ipynb` - noted in the section itself.

## 10. Optional extension - jointly fine-tuning the classifier

- [x] After the frozen-classifier experiments, unfreeze the pretrained linear classifier and optimize it jointly with the FM, at its own smaller learning rate, with `unfreeze_epoch` available for delayed unfreezing. (Section 16, `RUN_JOINT_FINETUNE`.)
- [x] **Add the control that makes the comparison interpretable.** Unfreezing adds two things at once - the FM *and* extra training of the classifier itself - so a joint run that beats the Stage 1 probe may simply be a probe that trained longer. A `head_only` control continues the same Stage 1 head for the same epoch budget under the same optimizer and the same validation-checkpointing rule, with no FM at all. **The number that means anything is joint versus that control, not joint versus Stage 1.** The specification does not ask for this control; without it the extension cannot support a claim.

## 12. Adopted after reviewing another group's Stage 3 notebook

A peer group's Stage 3 write-up was read as reference. Four of their choices were better than ours
and were adopted; the reasoning is recorded here so the changes are traceable to evidence rather
than to imitation.

- [x] **Source-anchored trust region** (their idea, clearly the most valuable). Their Strategy 2
  projects each target iterate into a ball around the *source* `z`; ours projected around the
  current `z_hat`. Theirs is right: anchoring at `z_hat` bounds each refresh but not the
  accumulation, so over ~200 refreshes the target can drift arbitrarily far from the feature it is
  meant to be a nearby improvement of. Measured before adopting - against a field that had already
  carried features away, source anchoring held total displacement at the radius 0.10 while `z_hat`
  anchoring reached 2.17, a 21x overshoot. Now the default; `guidance_anchor='current'` keeps the
  old behaviour as a swept variant so the difference is measured, not asserted.
- [x] **Monotone target acceptance.** Take the lowest-CE iterate among the projected candidates
  rather than the last one, since a projected gradient step can be undone by the projection.
  Verified it is not dead code: at conservative settings it changes nothing, and at radius 0.5 with
  beta 5 it rescues 2.2% of samples and lowers mean target CE from 1.0456 to 1.0052.
- [x] **Scale-free displacement penalty.** Theirs is `||z_hat - z||^2 / ||z||^2`; ours was absolute.
  This is the same reasoning already applied to the guidance step size, so not applying it to the
  penalty was an inconsistency on our side. The lambda grid moves from {1e-2, 1e-1} to {1, 10, 100}
  accordingly.
- [x] **Trust-region hit-rate diagnostic.** They report it and flag in their limitations that the
  radius was binding for nearly all targets, so results may depend materially on it. That is a real
  caveat we had no way to detect. Now recorded per refresh and surfaced in the sweep table.
- [x] **The nonlinear-capacity limitation**, from their limitations list: freezing `W, b` does not
  make the system linear, because the FM warps the space. Our own Section 23c demonstrates it
  vividly (rings, .458 -> .990 with a frozen line), so it is now stated in `DOC.md` §33 and in
  `STAGE3_COMPLIANCE.md`.
- [x] **A compliance map** (`doc/STAGE3_COMPLIANCE.md`), matching the Stage 2 one and their
  deviations table. We had no Stage 3 equivalent.

Deliberately **not** adopted, with reasons:

- *Selecting hyperparameters on seed-0 validation and propagating the winners to seeds 1-2.* They do
  this and disclose it as a limitation ("seed 0 is partly a development run"). Our main table uses
  fixed defaults for every seed and confines sweeps to seed 0 without propagating, so we do not have
  the leakage and should not introduce it.
- *Their pre-registration and ADR process.* Sound practice, but a process artefact rather than a
  method, and not reconstructible after the fact.
- *T = 4 and two datasets.* Both are valid readings of the specification; ours are T = 12 and three
  datasets, already justified in Section 0.

Where our implementation is stronger, for the record: we compute paired bootstrap CIs and exact
McNemar per cell (they make no significance claims); our joint-extension attribution has four legs
including `fm_only`, where theirs has the classifier-only control but not the FM-under-the-original-
head evaluation; and we carry the outcome breakdown, per-class effects, displacement decomposition,
flow-time curves, reverse flow and the 2D simulation, which have no counterpart there.

## Verification performed before and during the Colab run

Stage 3 was developed against a fabricated Stage 1 world - cached features plus linear-probe runs written in exactly the layout `01_linear_probe.ipynb` produces - so the code was known to run before consuming Colab time. Two suites, neither of which ships in the repository:

- **Unit/behaviour suite**, 58 checks over both feature transforms: probe loading, freezing, the fidelity and identity guards (including a negative control proving the identity guard has teeth), all four training variants, the resume path, gradient flow to a joint head and its absence from a frozen one, the bootstrap/McNemar helpers, and that guided targets lower classification loss under the tested constraints. Revision 3 adds source features with deliberately varied norms, a zero-norm edge case, per-sample radius checks, and an executable check that unit-gradient normalization and projection are independent. Plus a learnability check: against a deliberately undertrained frozen head, both strategies must beat epoch 0 - measured at validation .117 -> .433 and test .056 -> .300 for Strategy 1, .117 -> .383 / .278 for Strategy 2.
- **Notebook integration run**: every one of the 40 code cells parsed and executed against three fabricated datasets and three seeds, producing all 27 expected CSV and PNG artifacts and a 27-row `run_metrics.csv`; re-running the grid cell then loaded all 18 runs from disk and retrained none. Because every fabricated run keeps the identity, the diagnostic sections are additionally exercised on their degenerate path (no changed predictions), and `outcome_labels` is tested directly on crafted predictions. The joint attribution is verified exact against an identity FM, and the toy simulation - being fully synthetic - produces a real measured result locally rather than a degenerate one. The animation frame plan is checked to cover every Euler step in order and end at `t=1`, and - since an identity field makes the videos static by construction - a non-identity field is confirmed to move features across the rollout, so a passing animation test is not vacuous.
- **Optional-analysis suite**, 40 checks over both feature transforms: both curve endpoints self-anchoring, displacement starting at exactly zero and never decreasing, `accuracy_by_step` agreeing with the full metrics table, `t*` never beating the test-curve oracle, every recovery rate inside [0,1], and an identity field round-tripping with exactly zero error and a flat accuracy curve. It also confirms the training margin rises (-0.23 -> 3.59) and training cross-entropy collapses (1.76 -> 0.11) while *test* accuracy falls - overfitting at K=10, which is what motivates the `t*` ablation.

Three defects were found and fixed this way rather than on Colab: the missing epoch-0 checkpoint described in Section 5, the transform-dependent guidance step size described in Section 4, and float32-vs-float64 comparison against Stage 1's saved accuracies described in Section 1.

## Open items

- [x] Run implementation revision 2 end to end on Colab over the real Stage 1 caches.
- [ ] Run implementation revision 3 from the top on Colab. The configuration and implementation revision are both 3, and the end-to-end default is now regularized, so the complete main grid, sweeps, joint extension, toy, and dependent analyses must regenerate before any revision-3 result is reported.
- [ ] **Reconcile the older Stage 1 aggregate table with the current signed artifacts.** This does not invalidate Stage 3's paired deltas: the run replayed the exact loaded heads within 1e-6.
- [x] Transcribe the required results, epoch-0 count, and paired-significance summary into `doc/RESULTS.md`.
- [x] Report both variant sweeps with displacement and trust-region binding diagnostics.
- [x] Report joint fine-tuning against the `head_only` control with four-way attribution.
- [x] Report flow-time, validation-selected `t*`, reverse-flow, and mechanism diagnostics.
