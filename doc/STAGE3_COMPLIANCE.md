# Stage 3 compliance map

Every clause of `ref/part_3.pdf` mapped to where it is implemented, plus an explicit list of the work that is **not** part of the required experiment. The purpose of this file is to make it impossible to mistake an added analysis for a change to the specified protocol.

`ref/part_3.pdf` is authoritative. Where this file or `doc/DOC.md` disagrees with it, the PDF wins.

The deliverable is **`06_fm_before_classifier.ipynb`**. Section numbers below refer to that notebook.

---

## Required protocol

| Specification clause | Where | Status |
|---|---|---|
| Insert an FM transformation before the pretrained linear classifier: `z → FM → ẑ → W ẑ + b → s` | §5–§8 | ✅ |
| Train the linear classifier first, exactly as in Stage 1 | §4 | ✅ loaded from Stage 1, never retrained |
| Keep the classifier frozen | §4 | ✅ `requires_grad_(False)` at load; tested that it accumulates no gradients |
| Initialize the FM close to identity, so the untrained system behaves like the original probe | §5 | ✅ **exact**, not approximate: zeroed output layer ⇒ `rollout(z) == z` bit for bit |
| Use the same data splits and sampled training subsets as Stage 1 | §4 | ✅ re-derived with Stage 1's seeded `balanced_indices`; transform statistics cross-checked |
| One representative image encoder per dataset | §3 | ✅ DINOv2 ViT-S/14 on all three |
| One training-set size, K = 10 suggested | §3 | ✅ K = 10 |
| Choose a single number of Euler steps `T` and use it throughout | §3 | ✅ T = 12, training and inference, both strategies |
| The corresponding Stage 1 linear probe is the direct baseline | §4, §10 | ✅ paired per (dataset, subset seed) |
| Same velocity-network design and Euler procedure as Stage 2 | §5 | ✅ `(d+1)→512→512→d`, SiLU, `t` concatenated; `ẑ_{k+1} = ẑ_k + (1/T)v(ẑ_k, k/T)` |
| From `z`, apply the FM for `T` Euler steps to obtain `ẑ`, then pass `ẑ` through the frozen classifier | §5, §8 | ✅ one shared integrator |
| **Strategy 1**: run the complete rollout, compute `L_cls = CE(W ẑ + b, y)` | §6 | ✅ literal |
| Backpropagate through the complete rollout, update only the FM parameters | §6 | ✅ no `detach` in the path; head frozen |
| Optional: regularization penalizing displacement `z→ẑ` or velocity magnitude | §6, §14 | ✅ implemented, default 0, swept separately |
| **Strategy 2** step 1: run `z` through the current FM to obtain `ẑ` | §7 | ✅ |
| step 2: pass `ẑ` through the frozen classifier, compute the classification loss | §7 | ✅ |
| step 3: use the gradient w.r.t. `ẑ` to construct a nearby improved `ẑ'` | §7 | ✅ `torch.autograd.grad` in feature space, never parameter space |
| step 4: treat `z` as source and `ẑ'` as target | §7 | ✅ |
| step 5: perform a standard FM update between `z` and `ẑ'` | §7 | ✅ `t~U(0,1)`, `z_t=(1−t)z+tẑ'`, target `ẑ'−z`, MSE on `v` |
| step 6: recompute targets as the FM changes | §7, §8 | ✅ every `target_refresh_every` epochs over the whole training set |
| Experiment with step size, number of target steps, refresh frequency, normalization/constraint | §15 | ✅ all four, plus two more |
| Same encoder, training subset and pretrained classifier for all methods within each dataset | §4, §9 | ✅ one `PROBES` / `STAGE3_DATA` entry shared by every method |
| **Main comparison**: Stage 1 probe vs end-to-end vs classifier-guided | §10 | ✅ |

### Results to present

| Clause | Where | Status |
|---|---|---|
| Top-1 test accuracy for the probe and both Stage 3 methods | §10 | ✅ mean ± std over 3 subset seeds |
| Change relative to the corresponding linear-probe baseline | §10 | ✅ ΔAcc paired per seed |
| Representative training and validation curves for both methods | §11 | ✅ both, on separate axes |
| Feature-space visualization: original `z` and transported `ẑ` for both methods | §12, §12b | ✅ |
| Same test examples and class colors across all comparisons | §12 | ✅ one sample, one palette |
| Embedding computed **jointly** over the before/after feature sets | §12 | ✅ joint fit, shared axis limits |
| PCA or t-SNE | §12, §12b | ✅ both |

### Optional extension

| Clause | Where | Status |
|---|---|---|
| Unfreeze the pretrained linear classifier and jointly optimize FM + classifier | §16 | ✅ |
| Compare with the **frozen-classifier setting** | §16 | ✅ `e2e__frozen`, `guided__frozen` rows |
| Compare with the **original Stage 1 linear probe** | §16 | ✅ `linear_probe` row |
| Experiment with different learning rates for the FM and classifier | §16b | ✅ head lr at 0.01× / 0.1× / 1.0× the FM lr |
| Experiment with delayed unfreezing | §16b | ✅ `unfreeze_epoch ∈ {1, 10, 30}` |
| Experiment with additional regularization | §16b | ✅ anchor `λ(‖W−W₀‖² + ‖b−b₀‖²)`, `λ ∈ {0, 1e-2, 1}` |

---

## Decisions the specification leaves open

| Decision | Choice | Why |
|---|---|---|
| Datasets | all three, not two | The spec asks for "the same two datasets chosen in Stage 1"; Stage 1 was expanded to three. Running all three costs a third more and keeps the Stage 3 table comparable to the Stage 1 and Stage 2 tables. Flowers-102 doubles as a near-saturated control. |
| Feature space the FM operates in | the **classifier's input space** | Stage 1 selected `l2` or `standardize` per run and trained the head on transformed features, so `z` in the spec's diagram is a transformed feature. The FM operates in the space its downstream classifier acts in — which in Stage 2, where the classifier was cosine-based, meant the unit sphere instead. |
| Checkpoint-selection rule | validation top-1 of the complete system, identical for both strategies | Spec is silent. This is the rule Stage 1 used to select the probe, so baseline and both methods share one rule; selecting each method on its own loss would compare a tuned method against an untuned one. |
| Epoch 0 checkpointed | yes | With exact identity initialization, epoch 0 *is* the linear probe, so the rule becomes "keep the FM only if it helps". **Consequence: validation ΔAcc ≥ 0 by construction and carries no information; only test ΔAcc does.** |
| Repetition protocol | Stage 1's subset seeds {0,1,2}, `init_seed = 0` | Matches the Stage 1 and Stage 2 convention, keeps ΔAcc paired. |
| Early-stopping patience | 40, vs Stage 2's 25 | Zero-initialized output means only the last layer has a gradient on the first step, so the first several epochs are nearly flat; Stage 2's patience risks stopping before the FM leaves the identity. |
| Strategy 1 penalties | **relative**, `‖ẑ−z‖²/‖z‖²` | Stage 1 picks the transform per run, so an absolute λ would mean two different strengths on two datasets in the same table. |
| Strategy 2 trust region | centred on the **source** `z` | Anchoring at `ẑ` bounds each refresh but not their accumulation, so across ~200 refreshes the target can drift arbitrarily far. Measured: source anchoring held total displacement at the radius 0.10, `ẑ` anchoring reached 2.17. |
| Strategy 2 target | lowest-CE iterate, not the last | A *projected* gradient step can be undone by the projection, so the final iterate may be worse than the start. |

---

## Additions that are NOT part of the required experiment

Everything below is diagnostic or exploratory. None of it changes the required table in §10, and none of it is required by the specification.

| Where | What | Why it exists |
|---|---|---|
| §13 | Paired bootstrap CI and exact McNemar against the probe | A three-seed mean has no usable error bar; a ΔAcc whose CI spans zero is "no measurable difference", not a small win. House convention from Stage 2. |
| §14 | Strategy 1 regularization sweep | The spec invites it; reported with measured displacement so the off-manifold hypothesis is testable. |
| §15 | Strategy 2 guidance sweep | The four knobs the spec names, plus `anchor_current` and `no_monotone` so those two design choices are measured rather than argued. |
| §16c | Four-way attribution of the joint runs, plus a `head_only` control | Unfreezing adds the FM *and* extra classifier training at once; without the control a joint gain cannot be attributed. |
| §17 | Accuracy/margin/displacement along the flow, and a validation-selected `t*` | The curve is self-anchored at both ends (`t=0` is the probe, `t=1` is the reported result) so it is ΔAcc unrolled. |
| §18 | Reverse flow from classifier templates and a class-mean round trip | The Stage 2 optional analysis, restated for a setting with no prototypes. |
| §19, §19b | Fixed/broken outcome breakdown; translation-vs-transport decomposition | A mean cannot distinguish "fixed 8, broke 0" from "fixed 200, broke 192". §19b separates a genuine per-sample transport from a classifier-bias re-fit. |
| §20–§22 | Per-class effects, flow trajectories, effect-size panel | Diagnostics for reading a small or flat ΔAcc. |
| §23 | The rollout as video | Interpolated frames are presentation only; the caption reports the true Euler step. |
| §23c | A 2D simulation on concentric rings | The only place the frozen decision regions and the learned velocity field can be drawn directly rather than projected. Uses the same training functions, so it doubles as an end-to-end check. |
| §25 | Inline report of every saved artifact | The skip-if-saved convention means a rerun does not re-execute the plotting cells. |

---

## Limitations to state when presenting

- **n = 3 subset seeds.** Spreads are sample standard deviations over subset sampling only; `init_seed` is fixed at 0. Significance claims come from the paired per-split tests in §13, not from the three-seed spread.
- **One K, one T, one encoder per dataset**, by the specification's own scoping. Conclusions are about this operating point.
- **Validation ΔAcc is non-negative by construction** (epoch-0 checkpointing). Only test ΔAcc is informative.
- **Runs that kept the identity must be counted, not averaged away.** §10 prints how many of the 18 selected epoch 0.
- **Flowers-102 at K=10 has one effective subset**, since the official train split is exactly 10 images per class; its `std = .0000` is by construction.
- **The trust-region radius is chosen, not fitted.** §15 reports the hit rate; if it is near 1.0 the radius binds for essentially every sample and the Strategy 2 result depends materially on that number.
- **Freezing the classifier does not freeze the model's capacity.** `W F_θ(z) + b` can represent a nonlinear decision boundary even with `W, b` fixed, because the FM warps the feature space. §23c demonstrates this directly: on concentric rings, which no line can separate, the frozen linear probe sits at .458 and the same frozen probe behind a trained FM reaches .990. Conclusions are therefore about this FM architecture at this operating point, not about "what a frozen linear classifier can do".
- **The Stage 1 baselines on Drive currently disagree with the Stage 1 table in `doc/RESULTS.md`** (Aircraft .5285 vs .5096). The fidelity guard passes, so Stage 3 faithfully reproduces what is on Drive; the discrepancy means the Stage 1 runs were regenerated after `RESULTS.md` was written. This must be resolved before the two tables appear in one document — the gap is the same size as the entire Stage 3 end-to-end gain.
