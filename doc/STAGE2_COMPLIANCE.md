# Stage 2 compliance map

Every clause of `ref/stage_2.pdf` mapped to where it is implemented, plus an explicit list of the work that is **not** part of the required experiment. The purpose of this file is to make it impossible to mistake an added analysis for a change to the specified protocol.

`ref/stage_2.pdf` is authoritative. Where this file or `doc/DOC.md` disagrees with it, the PDF wins.

The required deliverable is **`04_flow_matching.ipynb`** (image-derived class prototypes). `05_flow_matching_clip.ipynb` is an extension on the other Stage 1 prototype branch and is never merged into the required table — see the last section.

---

## Required protocol

| Specification clause | Where | Status |
|---|---|---|
| FM layer transports frozen feature `z_i` toward the fixed prototype `p_{y_i}` | §6 | ✅ |
| Same datasets, encoders, class prototypes, training subsets, seeds and test sets as Stage 1 | §2, §4, §5, **§5b** | ✅ asserted, not assumed |
| Standard FM: `t ~ U(0,1)`, `z_t = (1−t)z_i + t·p_{y_i}`, target `u_i = p_{y_i} − z_i` | §6 | ✅ literal |
| Standard FM loss `‖v_θ(z_t,t) − u_i‖²` | §6 | ✅ `F.mse_loss` |
| Inference: `ẑ_{k+1} = ẑ_k + (1/T)·v_θ(ẑ_k, k/T)`, `k = 0..T−1`, from `ẑ_0 = z` | §5 `euler_rollout` | ✅ one shared integrator |
| Evaluate `T ∈ {4, 12}` | §7 | ✅ |
| Classify `ẑ_T` by cosine similarity to the same class prototypes as Stage 1 | §5 `classify_cosine` | ✅ |
| Rolled-out: same `T`-step sequence as inference, network unchanged, predicts `v_θ(ẑ_k, k/T)` at every step | §6 | ✅ reuses `euler_rollout` |
| Rolled-out loss `‖ẑ_T − p_{y_i}‖²`, backpropagated through the complete `T`-step sequence | §6 | ✅ no `detach` in the path |
| Rolled-out uses the same `T` at training and inference | §6, §7 | ✅ separate `rollout_T4` / `rollout_T12` networks |
| Architecture and main training choices fixed across the two objectives | §3, §6 | ✅ one `flow_matching` config consumed by both |
| Velocity net: small MLP, 2 hidden layers ≈512, SiLU, scalar `t` concatenated, output = feature dim | §6 `VelocityNet` | ✅ exactly the suggested config |
| No extensive architecture/hyperparameter search | §3 | ✅ single fixed config |
| `K ∈ {5, 10, full}` with the same sampled subsets and seeds | §5, §7, **§5b** | ✅ subset identity proved by prototype reconstruction |
| Compare baseline, standard `T=4`/`T=12`, rolled-out `T=4`/`T=12` | §8 | ✅ five series |
| Top-1 accuracy on the complete official test split | §7 | ✅ full split, touched once after selection |
| Same repetition protocol as Stage 1 | §7, **§5b** | ✅ with `effective_repetitions` reported |

### Results to present

| Required result | Where | Notes |
|---|---|---|
| **1. Classification results** — top-1 plus `ΔAcc` vs the matching baseline; table and/or accuracy-vs-`K` plot with error bars | §8 | Error bars now labelled `±1 SD across Stage 1-matched repetitions`. §14 adds paired per-seed `ΔAcc` with 95% CIs. |
| **2. Training curves** — representative standard and rolled-out losses; verify stable convergence | §9 | Training loss for standard and both rolled-out objectives, 10-shot / seed 0, all six pairs. The two losses are different quantities on different scales and are not compared in magnitude; the shared log axis is for stability checking only. A companion validation-accuracy panel is *not* implemented — `val_accuracy` is in every `history.csv` if it is wanted. |
| **3. Feature-space visualizations** — original vs standard-FM vs rolled-out-FM, same examples/colours/prototypes, projection fit **jointly** | §10 (PCA), §10b (t-SNE) | One fit over the row-stacked `[z ∣ z_std ∣ z_roll ∣ prototypes]`; all panels share axis limits. The subset is the first 10 classes and up to 30 test examples each, hardcoded per cell rather than saved to a manifest. |
| **4. Flow trajectories** — intermediate states with original feature, final feature and prototype; PCA recommended | §11 | Three examples per dataset at its strongest encoder: the first test example of each of the first three classes, one jointly-fit PCA per trajectory. Selection is by class index, not by transition type. |
| *Optional:* explore the flow in reverse from the prototypes | §12, §12b | Quantitative (recovery rate + continuous `cos`/`L2` to class means) as well as visual. |
| *Optional:* compare samples and prototypes at intermediate flow times | §13, §13b | `t=0` is asserted equal to Stage 1's saved accuracy to `1e-6`, which anchors every `ΔAcc` claim. |

---

## Additions that are NOT part of the required experiment

These exist to make the required result trustworthy or interpretable. None of them changes the specified protocol, and none feeds the §8 table.

| Section | Addition | Why it is not a protocol change |
|---|---|---|
| §5b | Subset-identity, prototype-order and unit-norm assertions | Verifies the protocol; changes nothing about it. Fails loudly if the reused Stage 1 artifacts are not what they claim to be. |
| §5b | `effective_repetitions` diagnostics | Reporting metadata. Flowers-102 has exactly 10 train images per class, so at `K=10` all subset seeds pick the same images and `std = 0` by construction. |
| §14 | Paired per-seed `ΔAcc`, 95% CIs, McNemar | A better estimator of the same quantity the spec asks for (`ΔAcc = Acc_FM − Acc_baseline`), exploiting the shared subsets. |
| §15 | Fixed/broken transition counts and pre-flow margins | Decomposes the top-1 number the spec already requires. |
| §16 | Validation-selected stopping time `t*` | **Ablation.** The required table always classifies `ẑ_T`. `t*` is chosen on validation only; the test-optimal `t` is reported as an unreachable bound, never as a result. |
| §17 | `W(t)`, `B(t)`, `W/B`, nearest-competitor similarity | Directly addresses the spec's third goal (how the transformation changes feature-space geometry) with numbers rather than only projections. |
| §18 | `T ∈ {1,2,4,8,12}` inference sweep | **Ablation**, standard FM only. The required table keeps `T ∈ {4,12}`. Rolled-out networks are excluded because `T` is part of their objective. |
| §19 | Direct-MLP and repeated-residual controls | **Control experiment** in a separate `controls/` tree. Answers whether the gain is specific to flow matching or follows from adding a supervised MLP of the same size. |
| §20 | Stage 1 linear-probe comparison | **Context.** A different Stage 1 baseline, not an FM variant. Shows what fraction of the prototype → probe gap the layer recovers. |
| §21 | Animated trajectories | Presentation only. Setting chosen programmatically as the largest-ΔAcc one (Aircraft / DINOv2, `full`, standard `T=4`, +0.2428). |
| §14b | Extended-repetition supplement, seeds 0–9 | **Supplement.** The required table keeps Stage 1's 3 repetitions, per the spec's "same repetition protocol as in Stage 1". This re-runs the identical protocol at `n=10` in a separate `extended/` tree purely to narrow the intervals (~3.4×); it reports no new accuracy, only better resolution of the same effect. Gated behind `RUN_EXTENDED`. |
| §6b | Zero-init pilot for the velocity net's output layer | **Pilot.** Initialization only — the protocol, loss, integrator, `T` and classification rule are untouched, and the same init applies to both objectives. Would be legitimate for the required table if adopted, which is what the pilot decides. Gated behind `RUN_ZERO_INIT_PILOT`. |

### Execution status (2026-08-24 run)

Every section in the tables above has now been executed. What the non-required sections returned, in one line each,
because two of them qualify the required result rather than merely decorating it:

| Section | Outcome |
|---|---|
| §5b | Subset identity verified for 42 (dataset, encoder, K, seed) combinations at max drift `0.00e+00`. The reused subsets are Stage 1's exactly. |
| §14 | **Qualifies §8.** 66 of 72 cells have a positive paired ΔAcc, but only 39 have a 95% CI excluding zero (24/24 Aircraft, 7/24 DTD). The two regressions are not significant either. |
| §15 | In every setting, samples the layer fixes had negative pre-flow margin and samples it breaks had positive pre-flow margin. |
| §16 | Validation-selected `t*` helps in 18 of 36 conditions, mean +0.0018, best +0.0303; it converts DTD/ResNet-18 `full` rolled-out from −.0206 to +.0097 over the baseline. |
| §17 | `W(t)/B(t)` falls in all 12 curves (median −0.219): the flow discriminates, not merely contracts. |
| §18 | `T = 2` already yields the full `T = 12` gain (median 100%); `T = 1` yields ~62% and is negative in 3 settings. |
| §19 | **Qualifies §8.** FM beats both controls in 16 of 18 settings, but on Aircraft/DINOv2 a plain `direct` MLP delivers 97% of FM's gain over the baseline. |
| §20 | FM closes a median 59% (range 10–87%) of the prototype → probe gap where the probe leads; it beats the probe in 4 of 18 settings, all near-saturated. |

§14 and §19 are the two that must travel with the §8 table. Neither changes the protocol or the reported
`test_accuracy`; both change what the number may be claimed to show.

One required-protocol item is still outstanding, and it is a reporting gap rather than a protocol deviation:
`test_accuracy_sel_T` is unpopulated, because the run loaded all 54 standard-FM runs from Drive instead of
retraining them. It needs one pass with `FORCE_RETRAIN_STANDARD = True`. The headline `test_accuracy` column
is unaffected. Relatedly, §7 reported `runs that actually resumed from a checkpoint: 0`, so the deterministic
`t`-sampling fix changes no stored number retroactively.

### Selection-rule note

Standard FM now saves two checkpoints from a single training run: `best.pt` (highest **mean** validation accuracy across both `T` — the original rule, and what `test_accuracy` reports) and `best_T{T}.pt` (highest validation accuracy at that `T` alone, reported as `test_accuracy_sel_T`). The second exists so that `standard T=4` and `rollout T=4` are selected under the same criterion. Neither requires an extra training run, and the headline column is unchanged.

---

## The CLIP branch is separate

`05_flow_matching_clip.ipynb` applies the same layer to CLIP RN50 image embeddings transported toward frozen **text** prototypes. `ref/stage_1.pdf` asks each group to pick one prototype branch; this project implemented both, and the image-prototype branch is the Stage 2 deliverable.

Its numbers must never be merged into the Stage 2 table, for one specific reason: the FM layer trains on labelled pairs, so it converts zero-shot CLIP into a K-shot method, and `ΔAcc` against the zero-shot baseline measures the *labels* as much as the layer. That notebook therefore always reports three numbers together — zero-shot CLIP, the K-shot image-prototype control on the same CLIP features, and FM — and `delta_vs_control` is the only like-for-like comparison on that branch. Both of the analyses it used to lack are now written: §19 adds the paired per-seed CI and McNemar against the control, and §20 adds the validation-selected `t*`; the animation section moved to §21. **Neither has been executed yet**, so "loses to the control in 28 of 36 cells" is still a difference of means until `05` is re-run, and should be quoted as such in the meantime.

One finding is specific to it and worth stating: reverse flow **exceeds** the pre-transport reference there (recovery 1.000 / 0.750 / 1.000 against 0.553 / 0.250 / 0.745), because a CLIP text prototype does not start out inside its own image cloud and backward integration carries it toward that cloud. On the image branch no such improvement is possible — the forward prototype already *is* the class image mean. Earlier wording in both notebooks called the forward reference a "ceiling"; that was correct only for the image branch and has been corrected.
