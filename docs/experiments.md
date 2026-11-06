# Experiment notes

Measured results for the defender and discriminator experiments. Every number can be reproduced with the scripts in `scripts/` (the synthetic benchmark, small samples).

## The heuristic defense was completely detectable — measured, not assumed

Running `scripts/train_and_evaluate.py --sweep` on the original blend
approach: at intensity L3 (the default), a trained `DefenseDiscriminator`
separated defended from undefended windows with **AUC = 1.000** — perfect
separation. Investigation found two compounding causes:

- `engine/defender.py` blended every order toward one *fixed* target
  coefficient-of-variation, producing a tight, non-overlapping band of
  defended-window statistics — trivially separable by a threshold.
- `engine/order_flow.py`'s undefended timing generation was hardcoded to
  `500 ± 25ms` for literally every parent order regardless of strategy,
  making the "before" population sit at one unrealistically narrow point in
  feature space, so *any* added variance stood out.

**Fixes applied:** the defender now samples a randomized target profile
*per parent order* (`DefenderAgent.reset_profile`, `SIZE_CV_TARGET_RANGE`,
`IAT_MEAN_TARGET_RANGE_MS`), and the generator samples a randomized cadence
per parent order too (`OrderFlowGenerator._timing_mean_ms`). This measurably
helped (AUC dropped from 1.000 to ~0.94 at low intensity) but **did not
solve the underlying problem** — see below.


## The real finding: obfuscation intensity makes detectability *worse*

Running the intensity sweep after the fixes above:

| Intensity | AUC |
|---|---|
| L0 (off) | 0.500 |
| L1 | 0.957 |
| L2 | 0.982 |
| L3 (default) | 0.993 |
| L4 | 0.997 |
| L5 (max) | 1.000 |

More obfuscation is monotonically *more* detectable, not less. This is the
actual interesting result of this project: a fixed-recipe transformation,
however randomized its target profile, is itself a fingerprint. A
discriminator with enough capacity learns to detect "this looks
obfuscated" instead of "this looks like a TWAP."


## One-shot adversarial optimization against a frozen discriminator didn't help either

`scripts/test_adversarial.py` tested the natural next idea: instead of a
hand-picked target profile, search per-order candidates that directly
minimize a trained discriminator's predicted probability. Evaluated against
a **freshly-trained, held-out** discriminator (never exposed to the
optimization target, to catch overfitting to D1):

- Heuristic defense: AUC = 0.989
- Adversarially-optimized defense: AUC = 0.990

No meaningful difference. This suggests the fix isn't a smarter one-shot
search — it's genuine multi-round co-training (the defender and
discriminator alternately retrained against each other), which is **not
implemented** here and is listed as future work in README.md, not claimed
as done.


## Recalibrating the noise-trader profile against real data — partial progress, not a fix

Following up on #6/#7: `SIZE_CV_TARGET_RANGE`, `IAT_STD_FRACTION_RANGE`, and
`venue_prob` in `engine/defender.py` were hand-picked, never checked against
what undefended flow in this generator actually measures. A feature-importance
check on the trained discriminator showed `size_cv` and `timing_cv` alone
carried ~96% of the detection signal, because the old size_cv target range
(0.10-0.55) started *above* the real empirical ceiling (~0.06-0.07).

`scripts/calibrate_noise_profile.py` now measures the real thing (900 synthetic
parent orders, all three strategies) and the defender's target ranges are set
from that output instead of a guess. A second, more fundamental issue was
found in the process: ICEBERG orders have a genuinely deterministic clip size
— real undefended ICEBERG windows measure `size_cv == 0.000` in 100% of
samples — but the old continuous target-range sampling essentially never lands
on exactly 0, so defended ICEBERG windows were "suspiciously nonzero but
small" 97% of the time. Fixed by sampling `target_size_cv = 0.0` outright with
probability 1/3 (ICEBERG's share of the strategy mix), reproducing the real
point-mass instead of approximating it.

**Result, measured with `scripts/train_and_evaluate.py --sweep`:**

| Intensity | AUC before recalibration | AUC after |
|---|---|---|
| L0 (off) | 0.500 | 0.500 |
| L1 | 0.957 | 0.777 |
| L2 | 0.982 | 0.841 |
| L3 (default) | 0.993 | 0.910 |
| L4 | 0.997 | 0.946 |
| L5 (max) | 1.000 | 0.966 |

A real, measured improvement (15-20 AUC points off at every active intensity)
— but not a fix. Two things are unchanged from #6: the defense is still far
from AUC 0.5 (undetectable) at any active intensity, and detectability is
still monotonically increasing with obfuscation intensity, meaning the
transformation itself is still a fingerprint, just a fainter one.

**Re-ran `scripts/test_adversarial.py` on the recalibrated defender — this
made things worse, not better.** Heuristic defense (D1): AUC = 0.914.
Adversarially-optimized defense against a fresh, held-out judge (D2): AUC =
**1.000**. The one-shot candidate search that previously showed no effect now
actively overfits to D1's decision boundary in a way a fresh discriminator
catches perfectly. This is reported as-is rather than hidden — closing this
loop needs the genuine multi-round co-training described in #7, not a
one-shot search against a static profile, calibrated or not.


## Multi-round adversarial co-training — real progress, still not a fix

Following up on #7 and #10: the single-round test (optimize against one frozen
discriminator, judge with a fresh one) consistently failed or backfired. Two
separate bugs were found and fixed before co-training could be evaluated
honestly:

1. `AdversarialDefender`'s candidate search perturbed size/timing by up to
   ~25-35% relative noise — unmoored from the calibrated ranges in #10, the
   exact same class of bug already fixed once in `engine/defender.py`. Fixed
   by bounding candidate perturbations to `SIZE_CV_TARGET_RANGE` /
   `IAT_STD_FRACTION_RANGE`, and always including a "no perturbation"
   candidate so the search can leave an already-regular order untouched.
2. With that fix alone (still single-round, one frozen discriminator),
   `scripts/test_adversarial.py` still showed no real effect: Δ=+0.020
   (0.898 -> 0.878). Fixing the perturbation scale wasn't the bottleneck —
   optimizing against a single static judge is.

`scripts/adversarial_cotrain.py` implements the actual multi-round loop:
alternately train a fresh discriminator from scratch each round, then
re-defend against the ENSEMBLE of all discriminators trained so far
(fictitious play / historical averaging, the standard fix for "overfits to
fool whichever judge is newest" in adversarial training), repeat for N
rounds, then check with one final discriminator that never contributed to
that ensemble.

**Measured, `--rounds 6 --n-parents 80 --n-candidates 10`:**

| Round | AUC | | Round | AUC |
|---|---|---|---|---|
| 0 (heuristic baseline) | 0.929 | | 4 | 0.574 |
| 1 (single-D, matches earlier failure) | 0.834 | | 5 | 0.775 |
| 2 | 0.782 | | 6 | 0.581 |
| 3 | 0.659 | | **held-out judge** | **0.631** |

Real, honest improvement: Δ=+0.298 vs. the heuristic baseline, confirmed by
a discriminator that never saw the defender's optimization loop. Round 1
alone reproduces the earlier single-round failure (still gets worse than
baseline before the ensemble kicks in at round 2+), which is a useful
internal consistency check that this isn't a fluke of a different random
seed. The trajectory is noisy round-to-round (round 5 spikes back to 0.775)
rather than monotonic, consistent with each new discriminator briefly
finding a fresh gap before the growing ensemble average closes it.

**Still not a fix, stated plainly:** 0.631 is well above the naive/
undetectable baseline of 0.5, and there's no sign in the 6-round trajectory
that it's converging there rather than plateauing around 0.55-0.65. This is
a small-sample proof of concept (80 parent orders, 10 candidates, gradient-
boosted trees as both attacker and defender surrogate) — not evidence that a
production system with a trained generator network and more rounds would
land in the same place, but also not evidence it wouldn't. Report generated
at `data/adversarial_cotrain_report.json` (gitignored, regenerate via
`python scripts/adversarial_cotrain.py`).

## Running this audit yourself

```bash
python scripts/train_and_evaluate.py --n-parents 300 --window 20 --sweep
python scripts/test_adversarial.py
python -m pytest tests/ -v
```

Every number in this document came from one of those three commands.
