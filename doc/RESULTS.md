# Results - Stage 1 and Stage 2

Top-1 accuracy on the complete official test split, every number traced to a saved `metrics.json` / `run_metrics.csv`. 5-shot and 10-shot are mean ± std over subset seeds {0,1,2}; `full` is mean ± std over initialization seeds {0,1,2} for trained methods and a single run for the closed-form image prototypes.

Three baselines were built in Stage 1. Stage 2 adds a flow-matching layer to one of them.

| | Model | Stage 1 | Stage 2 |
|---|---|---|---|
| 1 | Linear probe | done | not applicable (this is the Stage 3 reference) |
| 2 | Image-derived class prototypes | done | **done** - `04_flow_matching.ipynb` |
| 3 | Zero-shot CLIP RN50 text prototypes | done | **done** - `05_flow_matching_clip.ipynb` |

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
| DTD | ResNet-18 | .4723 (+.0072) | .5417 (+.0080) | .5926 (+.0048) |
| DTD | DINOv2 | .6480 (+.0054) | .7083 (+.0145) | .7592 (+.0326) |
| Aircraft | ResNet-18 | .1886 (+.0282) | .2431 (+.0446) | .3180 (+.0660) |
| Aircraft | DINOv2 | .3430 (+.1104) | .4351 (+.1570) | **.5852 (+.2429)** |
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

FM recovers roughly 20–75% of the gap and overtakes the probe only on saturated Flowers/DINOv2, where the difference is inside seed noise. Honest framing: the FM layer is a real improvement over the prototype rule it replaces, not a replacement for a discriminatively trained classifier.

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

`05_flow_matching_clip.ipynb` applies the identical FM layer here: CLIP RN50 image embeddings transported toward the frozen text prototypes, reusing the same `flow_matching` config so the two Stage 2 branches are directly comparable. 108 velocity networks, same protocol.

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

**Finding B — the flow overshoots its own optimum.** Accuracy peaks before `t=1` in 7 of 12 image-branch curves and in **all 6** CLIP-branch curves. Rolled-out models overshoot hardest:

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

Caveats unchanged: the backward pass uses a shifted time grid so it is not the exact inverse, and a contraction is not invertible in principle — this recovers a plausible pre-image, not the original point.

### Animations

Standard and rolled-out side by side from an identical starting feature, with the actual Euler step drawn at each frame, exported as embedded GIFs. The setting is selected programmatically: Aircraft/DINOv2 in `04` (largest ΔAcc, +.2428) and DTD in `05` (largest gain over the control, +.0348).

## Summary

1. **Encoder choice dominates.** DINOv2 ViT-S/14 over ResNet-18 is worth +14 to +31 points — larger than any method difference measured here.
2. **The FM layer works, consistently.** It improves on the prototype baseline in 66 of 72 cells (3 flat, 3 regressions), and the improvement grows with K, up to +.24 on Aircraft/DINOv2 at full data.
3. **It does not beat a linear probe.** FM closes 20–75% of the prototype-to-probe gap and overtakes the probe only where accuracy is already saturated. This is the central honest result for Stage 2.
4. **Standard FM ≥ rolled-out training**, on accuracy (14 of 18 settings) and on compute. Rolled-out training is also the only variant that ever regresses below the baseline, and the one that overshoots hardest (point 8).
5. **T is nearly irrelevant** for standard FM, for a structural reason: the ideal velocity is constant along the path, so the Euler endpoint is T-independent.
6. **On the CLIP branch the fair control reverses the headline.** FM beats zero-shot in 32 of 36 cells but loses to a same-supervision control in 28 of 36. The large Δ vs zero-shot measures the labels, not the layer.
7. **The FM layer repairs a broken classifier rather than sharpening a working one.** Measured at intermediate flow times: cosine to the true prototype rises everywhere, but the *margin* against the best competitor only rises where the baseline was already mis-ranked (Aircraft). Where the baseline ranked correctly (DTD, Flowers) the margin slightly falls and accuracy barely moves.
8. **`t=1` is a convention, not an optimum.** Accuracy peaks before the endpoint in all 6 CLIP-branch curves and 7 of 12 image-branch curves — up to +4.2 points left on the table by rolled-out models on CLIP/Flowers. The stopping time deserves to be a validation-selected hyperparameter.
9. **The reverse flow makes CLIP's modality gap measurable.** Backward-integrating the text prototypes raises their recovery rate from .55/.25/.75 to 1.00/.75/1.00 — the forward rate measures the gap, the reverse pass closes it. On the image branch the same operation barely moves anything, which is what a contraction should do.
10. **Two reporting caveats that must not be silently dropped**: Flowers-102 at K=10 has zero variance by construction (the train split is exactly 10/class), and the Stage 1 image-prototype `full` baseline is a single run, so it has no error bar.

## Open items

- Treat the flow stopping time as a validation-selected hyperparameter. Finding B shows `t=1` is not optimal in any CLIP-branch setting and in 7 of 12 image-branch curves; the gain is up to +4.2 points and costs nothing but a choice.
- Bring `04_flow_matching.ipynb` in line with `05` on `best_val_accuracy`: the standard-FM rows still write the T-averaged selection score into both per-`T` result rows.
- The optional intermediate-flow-time *comparison of samples against prototypes* is done; the remaining unexplored direction is whether early stopping interacts with the standard-vs-rolled-out choice.
