# Results - Stages 1, 2 and 3

Top-1 accuracy on the complete official test split, every number traced to a saved `metrics.json` / `run_metrics.csv`. 5-shot and 10-shot are mean ± std over subset seeds {0,1,2}; `full` is mean ± std over initialization seeds {0,1,2} for trained methods and a single run for the closed-form image prototypes.

Three baselines were built in Stage 1. Stage 2 adds a flow-matching layer to one of them.

| | Model | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|---|
| 1 | Linear probe | done | not applicable | **implemented, not yet measured** - `06_fm_before_classifier.ipynb` |
| 2 | Image-derived class prototypes | done | **done** - `04_flow_matching.ipynb` | not applicable |
| 3 | Zero-shot CLIP RN50 text prototypes | done | **done** - `05_flow_matching_clip.ipynb` | not applicable |

Stage 3 puts the FM layer in front of the frozen Stage 1 linear probe, so the linear probe stops
being only a reference point and becomes the thing being improved on. Its notebook runs end to end
against fabricated Stage 1 artifacts but has not been run on the real feature caches; **every number
in this file is still Stage 1 or Stage 2**. The Stage 3 section at the end records what will be
filled in and what to be careful about when reading it.

---

## Model 1 - Linear probe

Trained affine head on frozen features, softmax cross-entropy, checkpoint on best validation accuracy.

| Dataset | Encoder | K=5 | K=10 | full |
|---|---|---|---|---|
| DTD | ResNet-18 | .4571 ± .0109 | .5539 ± .0057 | .6379 ± .0016 |
| DTD | DINOv2 ViT-S/14 | .6553 ± .0066 | .7181 ± .0111 | **.7764** ± .0027 |
| Aircraft | ResNet-18 | .1927 ± .0124 | .2621 ± .0065 | .3654 ± .0032 |
| Aircraft | DINOv2 ViT-S/14 | .3651 ± .0115 | .5096 ± .0138 | **.6752** ± .0020 |
| Flowers-102 | ResNet-18 | .7363 ± .0095 | .8235 ± .0000 | .8252 ± .0018 |
| Flowers-102 | DINOv2 ViT-S/14 | .9890 ± .0020 | .9932 ± .0000 | **.9937** ± .0005 |

**The linear probe is the strongest method in the project at every full-data setting except saturated Flowers/DINOv2.** It is the only baseline that keeps scaling with K on Aircraft: DINOv2 goes .3651 → .5096 → .6752, nearly doubling from 5-shot to full, while the prototype baseline on the same features stalls at .3423.

Encoder choice dominates everything else. DINOv2 beats ResNet-18 by +14 to +31 points on every dataset and every K. The gap is largest on Aircraft (+.31 at full), which is the hardest task here — 100 fine-grained variants where ImageNet-pretrained ResNet-18 features simply do not separate the classes.

The Flowers-102 K=10 rows have `std = .0000` by construction, not by stability: the official train split is exactly 1020 images = 10 per class, so all three subset seeds select the identical set, and non-`full` runs share `init_seed=0`. Those are one effective run. The `full` rows differ from K=10 only through initialization-seed variation on the same data.

---

## Model 2 - Image-derived class prototypes, and Stage 2 flow matching on top

### Stage 1 baseline

`normalize → class mean → normalize`, classified by cosine similarity. Closed form, zero trained parameters.

| Dataset | Encoder | K=5 | K=10 | full |
|---|---|---|---|---|
| DTD | ResNet-18 | .4651 ± .0064 | .5337 ± .0104 | .5878 |
| DTD | DINOv2 | .6426 ± .0051 | .6938 ± .0019 | .7266 |
| Aircraft | ResNet-18 | .1604 ± .0083 | .1985 ± .0047 | .2520 |
| Aircraft | DINOv2 | .2326 ± .0127 | .2781 ± .0127 | .3423 |
| Flowers-102 | ResNet-18 | .7019 ± .0014 | .7522 ± .0000 | .7522 |
| Flowers-102 | DINOv2 | .9904 ± .0016 | .9940 ± .0000 | .9940 |

At K=5 prototypes are competitive with the linear probe and sometimes beat it (DTD/ResNet-18 .4651 vs .4571) — with five examples per class, a closed-form class mean is a better use of the data than fitting a head. The ordering reverses as K grows: by `full` the probe leads everywhere, by +.05 (DTD/ResNet-18) to +.33 (Aircraft/DINOv2). A class mean cannot exploit more data beyond a point, because it discards within-class structure.

Flowers-102 K=10 and `full` are identical (.7522, .9940) — same reason as above, the train split *is* the 10-shot subset.

### Stage 2 - FM layer, best variant per cell

Best of the four FM variants (standard/rolled-out × T ∈ {4,12}), with ΔAcc against the matching prototype baseline.

| Dataset | Encoder | K=5 | K=10 | full |
|---|---|---|---|---|
| DTD | ResNet-18 | .4723 (+.0073) | .5417 (+.0080) | .5926 (+.0048) |
| DTD | DINOv2 | .6480 (+.0054) | .7083 (+.0145) | .7592 (+.0326) |
| Aircraft | ResNet-18 | .1886 (+.0282) | .2431 (+.0446) | .3180 (+.0660) |
| Aircraft | DINOv2 | .3430 (+.1104) | .4351 (+.1570) | **.5852 (+.2428)** |
| Flowers-102 | ResNet-18 | .7170 (+.0151) | .7852 (+.0330) | .7859 (+.0337) |
| Flowers-102 | DINOv2 | .9917 (+.0013) | .9941 (+.0002) | .9944 (+.0004) |

**The FM layer improves on the prototype baseline in 66 of 72 result cells**, is flat in 3 (ΔAcc = .0000 to four decimals, all Flowers-102/DINOv2 at K=10, where the baseline is already .9940), and regresses in 3. All three regressions are rolled-out training: DTD/DINOv2/K=5/T=4 (−.0002) and DTD/ResNet-18/full at both T (−.0213, −.0206).

**Four findings that matter:**

**1. The gain grows with K, and is largest where the baseline is weakest.** Aircraft/DINOv2 goes +.11 → +.16 → +.24. This is the signature of a *trained* layer: the prototype baseline saturates because a class mean stops absorbing information, while `v_theta` keeps using the extra labels. The gain is smallest exactly where the baseline was already near ceiling (Flowers/DINOv2, +.0004) or where prototypes already suit the geometry (DTD/ResNet-18, +.005).

**2. FM closes most of the prototype-to-probe gap but does not surpass the probe.** At `full`, comparing best FM against the linear probe on the same features:

| Dataset / Encoder | Prototypes | + FM | Linear probe |
|---|---|---|---|
| DTD / ResNet-18 | .5878 | .5926 | **.6379** |
| DTD / DINOv2 | .7266 | .7592 | **.7764** |
| Aircraft / ResNet-18 | .2520 | .3180 | **.3654** |
| Aircraft / DINOv2 | .3423 | .5852 | **.6752** |
| Flowers-102 / ResNet-18 | .7522 | .7859 | **.8252** |
| Flowers-102 / DINOv2 | .9940 | **.9944** | .9937 |

Measured in `04` §20, the notebook's signed median over all 18 rows is **46%**. Restricting the fraction to the 14 settings where the linear probe actually leads the prototype baseline gives the more interpretable **59% median, range 10–87%**. In the other 4 settings the probe sits *below* the prototype baseline, so "gap closed" is not a meaningful fraction. FM overtakes the probe in those same 4 settings — saturated Flowers-102 at all three K, plus DTD/ResNet-18 at K=5 — where the differences are inside seed noise. Honest framing: the FM layer is a real improvement over the prototype rule it replaces, not a replacement for a discriminatively trained classifier.

**3. Standard FM beats rolled-out training in most settings.** Comparing the better `T` of each objective across the 18 (dataset, encoder, K) cells: standard wins 14, rolled-out wins 3 (Aircraft/DINOv2 at K=5 and K=10, plus Flowers-102/DINOv2 at K=10 by .0001), one tie (Flowers-102/DINOv2 at K=5). The largest single divergence is Aircraft/DINOv2 at `full`: standard .5852 vs rolled-out .5392, a .046 gap in standard's favour. Rolled-out training is also the only variant that ever loses to the baseline (DTD/ResNet-18 full, −.021), and it costs more — backprop through T sequential steps, roughly 1.5–2.5× standard FM per run, scaling with T.

The intuition: `L_roll` supervises only the endpoint, so it constrains the field far more weakly than per-step velocity regression, and the extra freedom does not pay off except where the task is hard enough that endpoint-only supervision acts as useful slack (Aircraft/DINOv2, few-shot).

**4. T barely matters.** Standard FM at T=4 and T=12 differ by less than .01 almost everywhere, and are occasionally identical to six decimals. This is expected, not a bug: the target velocity `u = p − z` is *constant* along the ideal path, so for a well-learned field each Euler step adds `(p − z)/T` and the endpoint `z_T = z + (p − z)` is the same for any T. **Be ready to state this before being asked** — an identical-looking T=4/T=12 column otherwise reads as a copy-paste error.

### Training behaviour

Both objectives train stably; neither diverged in 162 runs. 161 of 162 early-stopped, one reached the 200-epoch cap. In 16 runs (~10%) the best validation accuracy was at epoch 1 and training stopped at epoch 26 — concentrated in the saturated settings (Flowers/DINOv2) where there is nothing left to gain. Those FM layers are effectively untrained and should not be described otherwise.

Standard-FM loss (velocity MSE at random `t`) and rolled-out loss (endpoint MSE after a full rollout) are different quantities on different scales; the shared log axis in the training-curve figure is for stability checking only, not magnitude comparison.

### Geometry

The joint PCA/t-SNE views show the mechanism directly: after FM, each class's test features contract toward their prototype, and previously overlapping clouds separate. Read the PCA panels for the magnitude of that contraction — t-SNE normalizes local density and deliberately re-expands the collapsed clusters, so it understates the effect. The trajectories are near-straight in PCA, consistent with finding 4.

The caveat worth stating: FM cannot know the true label at test time, so it learns `E[p_y | z]` and transports toward the *conditional mean* prototype. Improvement comes from denoising, not from moving each point to its correct prototype. That is also why the gains are bounded well below the linear probe.

---

## Model 3 - Zero-shot CLIP RN50 text prototypes

Frozen CLIP RN50, one text prototype per class from the prescribed prompt, cosine similarity. No labeled training images, so one result per dataset and no K axis.

| Dataset | Prescribed (Stage 1) | Prompt ensemble (5 templates) | Δ |
|---|---|---|---|
| DTD | .3979 | .4207 | +2.29 pp |
| Aircraft | .1704 | .1692 | −0.12 pp |
| Flowers-102 | .6360 | .6560 | +2.00 pp |

The ensemble is a supplementary variant, never selected on test labels; the prescribed single prompt remains the Stage 1 baseline.

**Where zero-shot lands relative to the supervised baselines:**

- **Aircraft**: .1704 zero-shot actually *beats* 5-shot ResNet-18 prototypes (.1604) and sits close to 10-shot (.1985). With 100 fine-grained variants, five labeled images per class buys less than a good text prompt.
- **DTD**: .3979 is below every supervised setting, including 5-shot ResNet-18 prototypes (.4651). Texture class names transfer poorly to prompts.
- **Flowers-102**: .6360 is well below 5-shot ResNet-18 prototypes (.7019) and far below DINOv2 (.9904).

Prompt ensembling helps where class names are natural-language-friendly (textures, flowers) and does nothing for aircraft variant codes — consistent with the prompt being the bottleneck on the first two and the visual features being the bottleneck on the third.

### Stage 2 on this branch — measured

`05_flow_matching_clip.ipynb` applies the identical FM layer here: CLIP RN50 image embeddings transported toward the frozen text prototypes, reusing the same `flow_matching` config so the two Stage 2 branches are directly comparable. 81 velocity networks (108 result rows, since the one standard network is reported at both `T`), same protocol.

Best variant per cell, with **(Δ vs zero-shot / Δ vs the K-shot control)**:

| Dataset | K=5 | K=10 | full |
|---|---|---|---|
| DTD | 0.3679 (-0.0300 / -0.1395) | 0.5922 (+0.1943 / +0.0211) | 0.6454 (+0.2475 / +0.0348) |
| Aircraft | 0.1950 (+0.0246 / -0.0396) | 0.2330 (+0.0626 / -0.0417) | 0.2822 (+0.1118 / -0.0442) |
| Flowers-102 | 0.7414 (+0.1054 / -0.0896) | 0.8216 (+0.1856 / -0.0608) | 0.8155 (+0.1794 / -0.0669) |

The two reference lines it is measured against:

| Dataset | Zero-shot CLIP | Control K=5 | Control K=10 | Control full |
|---|---|---|---|---|
| DTD | 0.3979 | 0.5074 | 0.5711 | 0.6106 |
| Aircraft | 0.1704 | 0.2346 | 0.2747 | 0.3264 |
| Flowers-102 | 0.6360 | 0.8310 | 0.8824 | 0.8824 |

**This is the result the control was added for.** Against zero-shot, FM looks decisive: it wins 32 of 36 cells, by as much as +24.8 points on DTD at full data. Against the *same-supervision* control — image prototypes built from the same CLIP features and the same K-shot subsets, zero trained parameters — it wins only 8 of 36, and only on DTD at K=10 and K=full. Everywhere else, simply averaging the labelled CLIP features beats transporting toward text prototypes.

The rerun now makes that control comparison paired rather than only a difference of means. Across the 36 `(dataset, K, objective, T)` cells, seed-level 95% intervals exclude zero in 34: **6 favor FM, 28 favor the control, and 2 show no measurable difference**. The two unresolved cells are DTD K=10 rolled-out at `T=4` and `T=12`; the six supported FM wins are DTD K=10 standard at both `T` values and all four DTD full-data cells. Four of the 34 excluding-zero intervals are degenerate Flowers-102 K=10 results with one effective subset, so their intervals collapse to points by construction.

Validation-selected stopping at `t*` also ran. It improves 17 of 18 CLIP conditions (mean +.0239, best +.0520), but **none of the 18 tested conditions crosses from below the K-shot control at `t=1` to above it at `t*`**. Early stopping reduces overshoot; it does not overturn the like-for-like conclusion.

Two further points worth stating plainly:

- **On DTD at K=5, FM is worse than zero-shot itself** (−.0300). Five images per class are not enough to fit a transport that beats the prompt it is aiming at.
- **The large Δ vs zero-shot measures the labels, not the layer.** Quoting it without the control would misrepresent the result.

---

## Optional extensions — measured

Both of `ref/stage_2.pdf`'s optional extensions are implemented and now executed in **both** notebooks. They turned out to be the most informative diagnostics in the project.

### Samples and prototypes at intermediate flow times

Every intermediate Euler state classified over the complete test split. The `t=0` end is asserted, not assumed: in `04` against Stage 1's saved per-run `metrics.json`, in `05` against the zero-shot baseline. Both assertions passed, so the accuracy-versus-`t` curve provably starts at the baseline and ends at the FM result.

**Finding A — cosine rises everywhere, but margin only rises where the baseline was broken.** This is the mechanism behind "gains are largest where the baseline is weakest", now measured rather than inferred:

| Setting (10-shot, seed 0, standard FM) | cos to true prototype | margin | accuracy |
|---|---|---|---|
| DTD / ResNet-18 | .726 → .819 | +.0051 → **+.0035** | .5239 → .5330 |
| DTD / DINOv2 | .544 → .639 | +.1135 → **+.1110** | .6936 → .7027 |
| Flowers-102 / DINOv2 | .815 → .821 | +.3142 → **+.3061** | .9940 → .9940 |
| Aircraft / ResNet-18 | .889 → .942 | −.0160 → **−.0072** | .2019 → .2514 |
| Aircraft / DINOv2 | .709 → .883 | −.0542 → **−.0088** | .2919 → .4257 |

Where the prototype baseline already ranked classes correctly (positive margin: DTD, Flowers), the flow pulls everything closer to every prototype — cosine to the true prototype rises, but the margin against the best competitor slightly *falls*, and accuracy barely moves. Where the baseline was mis-ranked (negative margin: Aircraft), the flow repairs the ranking — margin improves by +.045 on Aircraft/DINOv2 — and accuracy jumps +13.4 points. **The FM layer does not sharpen a working classifier; it repairs a broken one.**

**Finding B — the flow overshoots its own optimum.** Accuracy strictly peaks before `t=1` in 6 of 12 image-branch curves and in **all 6** CLIP-branch curves. (The remaining 6 image-branch curves peak *at* `t=1`. Counting the argmax index instead of a strict improvement gives 8 of 12, because Flowers-102/DINOv2 is flat to four decimals and its argmax lands on `t=0` by tie-break — the strict count is the honest one.) Rolled-out models overshoot hardest:

| Setting | best t | accuracy at best t | accuracy at t=1 | left on the table |
|---|---|---|---|---|
| CLIP / Flowers-102, rolled-out | 0.58 | .8551 | .8133 | **+4.2 pts** |
| CLIP / Aircraft, rolled-out | 0.58 | .2637 | .2367 | +2.7 pts |
| CLIP / DTD, rolled-out | 0.58 | .5941 | .5681 | +2.6 pts |
| DTD / ResNet-18, rolled-out | 0.42 | .5399 | .5282 | +1.2 pts |

Standard FM overshoots far less (+0.0 to +0.9 points). Integrating to `t=1` is a convention, not an optimum — treating the stopping time as a validation-selected hyperparameter is the obvious follow-up, and on the CLIP branch it is worth more than the choice between the two objectives.

### The flow in reverse

Recovery rate: the fraction of classes whose prototype, after backward integration, is nearest to its *own* class's mean test feature. The untouched forward prototype is the control and the ceiling.

| Branch | Dataset / encoder | forward | reverse | cosine distance moved |
|---|---|---|---|---|
| image | DTD / ResNet-18 | .957 | .957 | .033 |
| image | DTD / DINOv2 | 1.000 | 1.000 | .026 |
| image | Aircraft / ResNet-18 | .660 | **.530** | .026 |
| image | Aircraft / DINOv2 | .690 | **.620** | .044 |
| image | Flowers-102 / both | 1.000 | 1.000 | .014 / .006 |
| **CLIP** | **DTD** | **.553** | **1.000** | **.718** |
| **CLIP** | **Aircraft** | **.250** | **.750** | **.697** |
| **CLIP** | **Flowers-102** | **.745** | **1.000** | **.677** |

Two opposite behaviours, and the contrast is the point.

**On the image branch the reverse flow barely moves anything** — cosine distances of .006 to .044. The learned field is essentially zero near `t=1`, so backward integration does not travel back toward the feature distribution; it sits where it started. Recovery is preserved except on Aircraft, where it degrades (.66 → .53, .69 → .62). Read this as: the field is well-behaved but not usefully invertible, which is expected for a contraction.

**On the CLIP branch the reverse flow does something striking.** The prototypes move a long way (cosine distance ≈ .70, close to orthogonal) and recovery *improves dramatically*: text prototypes that originally identified their own class's image cloud only 55% / 25% / 75% of the time land nearest their own class 100% / 75% / 100% of the time after backward integration. **That is direct quantitative evidence of CLIP's image–text modality gap, and that the FM field learned to bridge it** — the forward recovery rate measures the gap, and the reverse pass closes it. The same story appears in the forward direction as cosine-to-prototype rising from ≈.23 to ≈.89.

The re-run added the continuous companions to that discrete test, and they are more emphatic than the recovery rate: cosine from the text prototype to its own class's mean **image** embedding rises from .271 / .276 / .289 to **.960 / .974 / .986** (DTD / Aircraft / Flowers-102), and the L2 distance falls from ≈1.20 to .279 / .226 / .166. A recovery rate saturates at 1.000 and cannot distinguish "just barely closest" from "landed on top of it"; these say the reverse-flowed prompt lands essentially *on* its visual class centroid.

Caveats unchanged: the backward pass uses a shifted time grid so it is not the exact inverse, and a contraction is not invertible in principle — this recovers a plausible pre-image, not the original point.

### Animations

Standard and rolled-out side by side from an identical starting feature, with the actual Euler step drawn at each frame, exported as embedded GIFs. The setting is selected programmatically: Aircraft/DINOv2 in `04` (largest ΔAcc, +.2428) and DTD in `05` (largest gain over the control, +.0348).

## Summary

Points 1–10 are the headline results; the section after this one is the evidence that qualifies points 2 and 3, and it should be read with them rather than after them.

1. **Encoder choice dominates.** DINOv2 ViT-S/14 over ResNet-18 is worth +14 to +31 points — larger than any method difference measured here.
2. **The FM layer works, but on fewer cells than the raw table suggests.** It improves on the prototype baseline in 66 of 72 cells (3 flat, 3 regressions), and the gain grows with K up to +.24 on Aircraft/DINOv2 at full data. Under the **paired** 95% CI (§14), 48 of 72 intervals exclude zero: all 24 Aircraft cells, 7 of 24 on DTD, and 17 of 24 on Flowers-102. Eight Flowers-102 K=10 intervals are degenerate because there is one effective subset, leaving 40 non-degenerate effects. None of the three mean regressions is significant.
3. **It does not beat a linear probe, and most of the largest gain is not specific to flow matching.** The notebook reports a 46% signed median gap closed over all 18 rows; restricted to the 14 rows where the probe leads the prototype, the median is 59% (range 10–87%). FM overtakes the probe in 4 of 18 settings, all near-saturated. The controls (§19) sharpen this: FM beats both a plain MLP and a time-free residual stack in 16 of 18 settings, but on the +24-point Aircraft/DINOv2 headline a plain `direct` MLP already delivers 97% of the gain. Flow matching's *specific* contribution is largest where the plain MLP overfits below the baseline — DTD and Flowers-102 on ResNet-18, +.042 to +.063.
4. **Standard FM ≥ rolled-out training**, on accuracy (14 of 18 settings) and on compute. Rolled-out training is also the only variant that ever regresses below the baseline, the one that overshoots hardest (point 8), and the one that improves `W/B` least (§17).
5. **T is nearly irrelevant** for standard FM, for a structural reason: the ideal velocity is constant along the path, so the Euler endpoint is T-independent. The step ablation (§18) puts a floor on that — two Euler steps deliver the full `T=12` gain (median 100%), one step delivers only ~62% and can be worse than not running the flow at all.
6. **On the CLIP branch the fair control reverses the headline.** FM beats zero-shot in 32 of 36 cells. Raw means beat the same-supervision control in only 8 of 36; the paired analysis supports 6 FM wins, 28 control wins, and 2 no-effect cells. The large Δ vs zero-shot measures the labels, not the layer.
7. **The FM layer repairs a broken classifier rather than sharpening a working one.** Cosine to the true prototype rises everywhere, but the *margin* against the best competitor only rises where the baseline was already mis-ranked (Aircraft). §15 makes the mechanism concrete: in every setting, samples the layer fixes had negative pre-flow margin and samples it breaks had positive pre-flow margin. §17 adds the class-structure view — `W/B` falls in all 12 curves, so the flow really does discriminate rather than merely contract, but falling `W/B` converts into accuracy only where the true prototype was not already ranked first.
8. **`t=1` is a convention, not an optimum.** Accuracy peaks before the endpoint in all 6 representative CLIP-branch curves and 6 of 12 image-branch curves. Validation-selected `t*` helps in 18 of 36 image-branch conditions (mean +.0018, best +.0303) and converts the largest regression (DTD/ResNet-18 `full`, rolled-out) from −.021 into +.010. On CLIP it helps in 17 of 18 conditions (mean +.0239, best +.0520), but no tested condition crosses from below the K-shot control to above it.
9. **The reverse flow makes CLIP's modality gap measurable.** Backward-integrating the text prototypes raises their recovery rate from .55/.25/.75 to 1.00/.75/1.00 — the forward rate measures the gap, the reverse pass closes it. On the image branch the same operation barely moves anything, which is what a contraction should do.
10. **Two reporting caveats that must not be silently dropped**: Flowers-102 at K=10 has zero variance by construction (the train split is exactly 10/class), which makes 8 of the 48 excluding-zero image-branch intervals degenerate; and the Stage 1 image-prototype `full` baseline is a single run, so it has no error bar and the full-data deltas are paired only to a repeated constant baseline.

## Hardening, ablations and controls — measured

The analyses added after the external review (2026-08-23) have now been executed. `04_flow_matching.ipynb`
ran Sections 5b and 14–21 and `05_flow_matching_clip.ipynb` ran Section 9 and Sections 17–21; every number below comes
from that run. Two things about the run itself, reported rather than quietly absorbed:

- **The stored accuracies did not shift.** Section 7 printed `Result rows: 216 | runs that actually
  resumed from a checkpoint: 0`. No saved run had ever resumed, so the pre-fix nondeterministic
  `t ~ U(0,1)` sampling never had the chance to affect a stored result, and the new deterministic
  per-`(init_seed, epoch, batch)` sampling changes nothing retroactively. Every Model 2 number above is
  the same before and after that fix — the transition caveat can be dropped rather than carried.
- **The per-`T` selection columns are still empty.** `test_accuracy_sel_T` needs one pass with
  `FORCE_RETRAIN_STANDARD = True`; the skip logic checks whether output files exist, so all 54
  standard-FM runs were loaded from Drive and no `best_T{T}.pt` was written. This is the one review item
  that remains genuinely outstanding.

Section 5b's guards all passed: subset identity verified for **42** (dataset, encoder, K, seed)
combinations at a maximum prototype-reconstruction drift of **0.00e+00** — the K-shot subsets are
bit-identical to Stage 1's, not merely statistically equivalent — plus prototype row-order and unit-norm
assertions on both transport endpoints.

### The paired confidence intervals cut the headline claim down (§14)

Section 8's "improves in 66 of 72 cells" is a difference of two independently-averaged means. Section 14
redoes it as a paired per-seed difference — FM run *s* and baseline run *s* saw the same K images — with
a 95% t-interval over the 3 repetitions. The two statements are very different:

| | count of 72 cells |
|---|---|
| positive mean paired ΔAcc | 66 |
| 95% CI excludes zero | 48 |
| …of which are degenerate (`std = 0` by construction, all Flowers-102 K=10) | 8 |
| **non-degenerate intervals excluding zero** | **40** |

Broken down by dataset, this is not uniform at all:

| Dataset | CI excludes zero |
|---|---|
| FGVC-Aircraft | **24 of 24** |
| Flowers-102 | 17 of 24 (8 of them degenerate) |
| DTD | **7 of 24** |

**The sub-1-point DTD gains do not survive.** Of DTD's 24 cells only the four DINOv2 `full` cells (+.030
to +.033), the two DINOv2 K=10 standard cells (+.0144, +.0145) and DTD/ResNet-18 `full` standard `T=12`
(+.0048, CI [+.0025, +.0071]) clear zero. Everything else on DTD — including every DTD/ResNet-18 K=5 and
K=10 cell — has a CI straddling zero and must be restated as **no measurable effect**, not as a small
gain. The `n=3` interval is wide, which is honest rather than unfortunate.

Symmetrically, **none of the three mean regressions is significant**. The two material regressions are DTD/ResNet-18 `full` rolled-out:
−.0213 (CI [−.0522, +.0096]) at `T=4` and −.0206 (CI [−.0499, +.0088]) at `T=12`. They are the largest
regressions measured and they still cannot be distinguished from zero at `n=3`.

McNemar, computed on the shared test images, is significant in all 3 seeds in 37 of 72 cells. Where it
disagrees with the CI it does so in the anti-conservative direction, as expected — 6 cells are
CI-negative but McNemar-significant in ≥2 seeds (both DTD/ResNet-18 `full` rolled-out regressions and
four Flowers-102 K=5 cells). McNemar's unit is the test image, not the run, so it ignores subset variance
entirely. **Where the two disagree, quote the CI.**

### The controls: flow matching, or just another MLP? (§19)

This is the experiment the accuracy table could not settle. Two non-FM controls, with identical hidden
architecture, optimizer, schedule, early stopping, subsets and seeds: `direct`, a plain MLP trained on
`‖g(z) − p_y‖²` with no time input and no integration; and `residual`, the same shared network applied
the same `T = 12` times, trained only on the endpoint, with no time conditioning and no per-`t` velocity
supervision.

**FM beats both controls in 16 of 18 settings** (median advantage over `direct` +.0255, over `residual`
+.0053). So the gain is not merely "a supervised nonlinear map of the same size". But the breakdown
qualifies the headline sharply:

| Setting (`full`) | prototype | direct MLP | residual | best FM | direct's share of FM's gain |
|---|---|---|---|---|---|
| Aircraft / DINOv2 | .3423 | .5783 | .5432 | .5852 | **97%** |
| Aircraft / ResNet-18 | .2520 | .2588 | .3089 | .3180 | 10% |
| DTD / DINOv2 | .7266 | .7551 | .7537 | .7592 | 88% |
| DTD / ResNet-18 | .5878 | .5617 | .5631 | .5926 | *below baseline* |
| Flowers-102 / ResNet-18 | .7522 | .7287 | .7760 | .7859 | *below baseline* |
| Flowers-102 / DINOv2 | .9940 | .9940 | .9941 | .9944 | 0% |

**The +24.3-point Aircraft/DINOv2 headline is almost entirely "a supervised MLP into prototype space",
not flow matching specifically.** The plain `direct` control reaches .5783 against FM's .5852 — 97% of
the gain over the .3423 baseline, with FM adding only +.0069 on top. The same holds at K=5 and K=10
(93%, 96%). Summary point 2 has to be read with this: the layer works, but in the setting that produces
the largest number, most of the work is done by adding a trained nonlinear head at all.

The mirror image is just as informative. On DTD/ResNet-18, Flowers-102/ResNet-18, and DTD/DINOv2 at low
`K`, the `direct` MLP lands *below* the closed-form prototype baseline — it overfits — while FM stays
above it. FM's advantage over `direct` is largest exactly there (+.042 to +.063). And `residual` sits
much closer to FM than `direct` does almost everywhere, which localizes the credit: **most of what FM has
over a plain MLP comes from iterative shared-weight computation, and the time conditioning plus per-`t`
velocity supervision adds a smaller increment on top** — with one clear exception, Aircraft/DINOv2 at
`full`, where FM beats `residual` by +.0420.

The two settings where FM does not beat both controls are Aircraft/DINOv2 at K=5 (`residual` .3442 vs FM
.3430) and Flowers-102/DINOv2 at K=5 (`direct` .9922 vs FM .9917) — both inside seed noise.

### Validation-selected stopping time (§16)

Finding B above is now measured honestly: `t*` is the argmax of **validation** accuracy over the `T = 12`
Euler grid, frozen, and only then evaluated once on test. Early stopping the flow helps in **18 of 36
conditions, mean gain +0.0018, best +0.0303**. The `oracle_best` column — the best any `t` could have
reached on test — is reported alongside as an unreachable bound; the gap to it is typically .001–.005, so
honest selection costs little.

The single most useful case is the notebook's only real regression:

| Setting | `t*` | test at `t*` | test at `t=1` | prototype baseline |
|---|---|---|---|---|
| DTD / ResNet-18 / `full`, rolled-out | 0.56 | **.5975** | .5672 | .5878 |

Validation-selected early stopping turns a −.021 regression into a +.010 gain over the baseline. That is
the strongest argument in the project for treating the stopping time as a hyperparameter rather than a
convention. Note the other half of the result though: it helps in only half the conditions and the mean
gain is +0.0018, so it is a fix for the overshooting cases, not a free improvement everywhere.

### Contraction or discrimination? (§17)

Section 13's metrics all improve whenever the flow contracts toward the prototype set, whether or not the
classes become easier to separate. Section 17 measures the class structure directly: within-class scatter
`W(t)`, between-class separation `B(t)`, and the discriminative ratio `W/B`.

**`W/B` falls in all 12 curves** (median change −0.219), so the flow is not merely dragging everything
inward — it does tighten classes relative to their separation. The magnitude tracks the accuracy gain
where the gain is large:

| Setting (10-shot, standard FM) | `W/B`, `t=0` → `t=1` | accuracy gain |
|---|---|---|
| Aircraft / DINOv2 | 0.698 → **0.185** | +.1338 |
| Aircraft / ResNet-18 | 2.510 → 1.794 | +.0495 |
| Flowers-102 / ResNet-18 | 0.774 → 0.562 | +.0330 |
| DTD / ResNet-18 | 1.605 → 1.290 | +.0090 |
| DTD / DINOv2 | 1.045 → 0.739 | +.0090 |
| Flowers-102 / DINOv2 | 0.236 → 0.228 | +.0000 |

So `W/B` falling is **necessary but not sufficient**: DTD sees a large ratio improvement (−0.31) for under
a point of accuracy. The reason is visible in the companion columns — on Aircraft, cosine to the *nearest
competitor* stays above cosine to the true prototype (.949 vs .942 on ResNet-18, .892 vs .883 on DINOv2),
which is exactly the negative margin of Finding A. Improving `W/B` reorganizes the cloud; whether that
converts into accuracy depends on whether the true prototype was ranked first to begin with. Same
"repairs a broken classifier" story, measured on class structure rather than per-sample margins.

Rolled-out FM improves `W/B` much less than standard FM on DTD, Aircraft/ResNet-18 and
Flowers-102/ResNet-18 (−0.086 vs −0.716 on Aircraft/ResNet-18, for instance) — consistent with
endpoint-only supervision constraining the field more weakly.

### Which samples the flow flips (§15)

Top-1 accuracy nets out two opposite effects; Section 15 separates them at 10-shot, seed 0, `T=12`, and
records each flipped sample's margin **before** the flow ran.

| Setting | fixed | broken | net | margin of fixed | margin of broken |
|---|---|---|---|---|---|
| Aircraft / DINOv2 | 562 | 116 | **+446** (+13.4 pp) | −.0444 | +.0231 |
| Aircraft / ResNet-18 | 285 | 120 | +165 (+4.95 pp) | −.0080 | +.0051 |
| Flowers-102 / ResNet-18 | 381 | 178 | +203 (+3.30 pp) | −.0082 | +.0091 |
| DTD / ResNet-18 | 85 | 68 | +17 (+0.90 pp) | −.0144 | +.0149 |
| DTD / DINOv2 | 56 | 39 | +17 (+0.90 pp) | −.0258 | +.0248 |
| Flowers-102 / DINOv2 | 1 | 1 | 0 (+0.00 pp) | −.0060 | +.0001 |

The pattern is identical everywhere and it constrains how the layer may be described: **fixed samples had
negative pre-flow margin and broken samples had positive pre-flow margin, in every single setting.** The
layer moves samples near the decision boundary in both directions; it wins by fixing more than it breaks,
not by leaving correct predictions alone. On Aircraft/DINOv2 the fixed group's mean margin is −.0444 —
genuinely confidently misclassified, not marginal — which is what makes that gain a restructuring rather
than a nudge. On DTD the two groups are near mirror images (−.0144 / +.0149, 85 fixed against 68 broken),
which is the same conclusion §14's confidence intervals reach by a different route: on DTD the layer is
churning, not improving.

### How many Euler steps the gain needs (§18)

Inference-only sweep over `T ∈ {1, 2, 4, 8, 12}`, standard FM only — a rolled-out network cannot
legitimately be run at a `T` it was not trained for. The required table keeps `T ∈ {4, 12}`.

**Two Euler steps already deliver the whole `T=12` gain**: median `frac_at_T2` = 1.00 across the 18
settings, reaching at least the full gain in 11 of them and strictly exceeding it in 9, so `T=2`
sometimes beats `T=12`. **One step does not**: median
`frac_at_T1` = 0.62, and it is *negative* in three settings (DTD/ResNet-18 `full` −7.07, DTD/ResNet-18
K=10 −0.80, Aircraft/ResNet-18 `full` −0.56), where a single large Euler step overshoots to somewhere
worse than the untouched feature.

This extends the T-independence result (finding 4) downward rather than contradicting it: the endpoint is
`T`-independent for `T ≥ 2`, and `T = 1` is simply too coarse a discretization of the path. The practical
reading is that the layer costs two velocity evaluations, not twelve.

## Correction

Earlier versions of both notebooks described the forward prototype's recovery rate in the reverse-flow
section as "the control and the ceiling". That is wrong on the CLIP branch and the measured numbers in
this file already contradict it: reverse recovery reaches 1.00/.75/1.00 against forward references of
.55/.25/.75. A CLIP text prototype does not sit inside its own class's image cloud, so integrating
backward through a field trained to carry images *to* text prototypes moves it toward that cloud and
can beat where it started. The forward value is a **pre-transport reference**; it is additionally a
ceiling only on the image branch, where the prototype already is the class image mean. Summary point 9
was always stated correctly - it was the notebook prose that mislabelled it.

## Gated analyses not run

The rerun completed the `05` control-prediction, paired-CI, validation-selected stopping, and animation sections. Two opt-in analyses remain deliberately disabled and do not contribute any result above:

| Where | Status | Purpose |
|---|---|---|
| `04` §14b, `05` §19b | `RUN_EXTENDED = False` | Ten-repetition supplement; the notebooks report about 540 and 270 velocity networks respectively. It narrows intervals without changing the required `n=3` table. |
| `04` §6b | `RUN_ZERO_INIT_PILOT = False` | Pilot for a zero-initialized output layer. The identity-at-initialization guard passed, but the comparative pilot did not run. |

The separate `04` pass with `FORCE_RETRAIN_STANDARD = True` is also still outstanding; it is needed only to populate the per-`T` checkpoint-selection columns for already-saved standard-FM runs.

## Open items

- ~~Treat the flow stopping time as a validation-selected hyperparameter.~~ **Implemented and measured** (`04` §16): helps in 18 of 36 conditions, mean +0.0018, best +0.0303, and it converts the DTD/ResNet-18 `full` rolled-out regression into a gain. Reported as an ablation; the required table still classifies `ẑ_T`.
- ~~No paired CI / McNemar on the CLIP branch.~~ **Implemented and measured** as `05` §19: 6 cells favor FM, 28 favor the control, and 2 show no measurable difference.
- ~~No validation-selected `t*` on the CLIP branch.~~ **Implemented and measured** as `05` §20: helps in 17 of 18 conditions (mean +.0239, best +.0520) but flips no control-relative loss into a win.
- **The intervals remain underpowered at `n=3`.** On the image branch, 24 of 72 cells cannot be called either way; on DTD it is 17 of 24. Eight additional excluding-zero cells are degenerate Flowers-102 K=10 results with one effective subset. This is statistical resolution, not the FM layer: the two-sided t-multiplier at `n=3` is 4.30. `ref/stage_2.pdf` fixes the required protocol at Stage 1's 3 repetitions, so the fix is the §14b / §19b supplement rather than a change to the required table.
- **Still outstanding: one pass of `04` with `FORCE_RETRAIN_STANDARD = True`.** `04` §6 writes `best_T{T}.pt` alongside `best.pt` and §7 records `best_val_accuracy_at_T` / `test_accuracy_sel_T`, so `standard T=4` and `rollout T=4` are selected under the same criterion. The last run loaded all 54 standard runs from Drive, so those columns are still absent. Retrained numbers will be reproducible but will not match the stored ones to the last digit — report both.
- Not yet attempted: re-normalizing each Euler state back onto the unit sphere. Both endpoints are unit-norm and the classifier is cosine, but the straight-line path passes through the interior of the ball, so `distance_to_true_prototype` mixes angular and radial movement. Note that renormalizing only the *final* state is provably a no-op — cosine similarity is scale-invariant — so only per-step renormalization is worth testing, and it deviates from the specified integrator, so it would be an ablation.
- Follow-up the controls opened, and now the most informative experiment left: on Aircraft/DINOv2 a plain `direct` MLP captures 97% of FM's gain, while `residual` captures most of FM's advantage over `direct` elsewhere. Isolating the time conditioning alone — `residual` plus a `t` input, nothing else — would say precisely what flow matching contributes beyond iterative shared-weight computation. If it is run, the §19 controls must be swept identically to whatever FM gets, or the comparison stops being fair.
- The optional intermediate-flow-time comparison of samples against prototypes is done. Whether early stopping interacts with the standard-vs-rolled-out choice is answerable from `04` §16: rolled-out models have the earlier `t*` and the larger gain from stopping, consistent with their weaker constraint.
- See `doc/STAGE2_COMPLIANCE.md` for the clause-by-clause map of `ref/stage_2.pdf` and the explicit list of additions that are *not* part of the required experiment.

---

# Stage 3 - FM before the frozen linear classifier

**Run, not yet transcribed.** `06_fm_before_classifier.ipynb` completed end to end on Colab on
2026-08-31 and wrote its artifacts to `outputs/fm_before_classifier` on Drive. Those numbers have not
been read into this file, so **nothing below is a measured Stage 3 result yet** - this section still
records what goes here and what to be careful about when reading it.

The notebook has since gained diagnostic Sections 19-22 and the inline report in Section 24, which
postdate that run. Rerunning populates them without retraining anything: the grid is cached.

## What will be reported

Top-1 test accuracy on the complete official test split for three methods on each of DTD,
FGVC-Aircraft and Flowers-102, all on DINOv2 ViT-S/14 at K=10 with T=12, mean ± std over Stage 1's
subset seeds {0, 1, 2}:

| Dataset | Stage 1 linear probe | End-to-end rollout | ΔAcc | Classifier-guided | ΔAcc |
|---|---|---|---|---|---|
| DTD | .7181 ± .0111 | *pending* | | *pending* | |
| Aircraft | .5096 ± .0138 | *pending* | | *pending* | |
| Flowers-102 | .9932 ± .0000 | *pending* | | *pending* | |

The linear-probe column is not pending - those are the measured Stage 1 K=10 DINOv2 numbers from the
table at the top of this file, and Stage 3 loads exactly those trained heads rather than retraining
them. Two guards in the notebook assert that the reloaded probe reproduces its saved test accuracy
before any FM is trained, so if the Stage 3 baseline column ever disagrees with the Stage 1 table,
the run is wrong and must not be reported.

Also to be reported: representative training and validation curves for both strategies; the joint
PCA and joint t-SNE feature-space comparisons; per-cell paired bootstrap CIs and exact McNemar
tests against the probe; the two variant sweeps; the joint fine-tuning extension; and the two
optional analyses carried over from Stage 2 - samples at intermediate flow times with a
validation-selected stopping time, and the flow in reverse.

The intermediate-flow-time curve is the one to lead with if the headline table is flat, because it
is **anchored at both ends and the notebook asserts it**: identity initialization makes `t=0`
accuracy exactly the linear probe and `t=1` exactly the Stage 3 number above, so the curve is ΔAcc
unrolled and shows *where along the flow* anything happened. Read the logit margin alongside it -
margin moves continuously where accuracy moves in jumps, so a flow that helps confidence without
flipping predictions is visible there and nowhere else.

## Five things to state when the numbers arrive

**1. Validation ΔAcc is meaningless here; only test ΔAcc counts.** The FM is initialized to the exact
identity, and epoch 0 is checkpointed like any other epoch, so the selected checkpoint can never have
worse validation accuracy than the linear probe. Validation ΔAcc ≥ 0 is a property of the selection
rule, not a finding. Quoting it as evidence would be a straightforward error.

**2. Lead with the fixed/broken breakdown, not the mean.** Section 19 sorts every test sample into
`fixed`, `broken`, `both_right`, `both_wrong`. `fixed − broken` is exactly the numerator of ΔAcc, and
`fixed + broken` is the churn underneath it. A ΔAcc of +.004 produced by "fixed 8, broke 0" is a
different finding from the same +.004 produced by "fixed 200, broke 192", and only the breakdown
distinguishes them. Section 19's other two panels then say whether the broken samples were ones the
probe had been confident about — which would be the most important thing on the page.

**3. Report how many of the 18 runs selected epoch 0.** Those FM layers are the identity map - they
learned nothing usable, and the system is exactly the linear probe. A mean that silently includes
them reads as "a small consistent gain" when the real finding may be "it helped on 4 of 18 and did
nothing on the rest". The notebook prints the count and lists the runs.

**4. Flowers-102 K=10 has one effective subset, not three.** The official train split is exactly 1020
images, 10 per class, so all three subset seeds select the identical set and `init_seed` is fixed at
0. Its `std = .0000` is by construction. This already caught out the Stage 2 write-up and applies
unchanged here.

**5. The joint fine-tuning extension must be read against its `head_only` control.** Unfreezing the
classifier adds the FM *and* extra classifier training simultaneously. The control continues the same
head for the same budget with no FM at all, and it is the only honest comparison. Joint-vs-Stage-1
would attribute the control's gain to flow matching.

## What the Stage 2 results predict

Stage 3 is a harder ask than Stage 2 and the existing measurements say so. Stage 2 improved on a
closed-form prototype rule and still did not surpass the linear probe in any setting where the probe
led - it closed a median 59% of the gap. Stage 3 targets that probe directly, and asks a *frozen*
affine classifier to do better on a representation it was itself fit on. On DINOv2 at K=10 the probe
is already at .7181 / .5096 / .9932, and Flowers-102 has essentially no headroom at all.

Two Stage 2 findings transfer as cautions rather than predictions:

- **The control question from `04` §19 applies here too.** On Stage 2's largest-gain setting, a plain
  supervised MLP with no flow reached 97% of the FM gain. Stage 3's Strategy 1 backpropagates a
  classification loss through a rollout, which is a flexible nonlinear map trained on 10 examples per
  class; if it gains, "is this flow matching or just a nonlinear layer?" is the immediate follow-up
  and the write-up should anticipate it.
- **The reverse-flow reference is not a ceiling.** Stage 3 has no class prototypes, so reverse flow
  starts from the frozen classifier's own class templates and from a round trip on the class means.
  For the templates, the untouched value is a *pre-transport reference*: `w_c` need not sit inside its
  class's feature cloud, so reverse flow can legitimately beat it. `RESULTS.md` already carries a
  correction for exactly this mistake in the Stage 2 write-up (see **Correction** above) - it would be
  a poor showing to make it twice. The round-trip anchor is the safer one to lean on, because its
  ideal is unambiguously zero.
- **Displacement is a measured quantity, not an impression.** Every Stage 3 run records
  `||ẑ - z|| / ||z||` on the test split. A large gain with a large displacement, under the `l2`
  transform in particular, is the signature of the FM leaving the unit sphere the head was fit on and
  exploiting the classifier off-manifold rather than improving the representation. Section 14's
  regularization sweep exists to test that, and its result belongs next to the headline table.

A negative or flat result is a legitimate outcome and should be reported plainly with the mechanism,
following the Stage 2 precedent. What would be a genuine problem is a *positive* ΔAcc obtained
against a mis-reconstructed baseline, which is what the two guards exist to prevent.
