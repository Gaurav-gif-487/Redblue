# RedBlue — Adversarial Execution Intelligence

**Status: research prototype on a synthetic benchmark. Not production infrastructure.** Measured results are in [docs/experiments.md](./docs/experiments.md).

## What this actually is

A large institutional order gets sliced into many small "child" orders
(TWAP, VWAP, ICEBERG, ...) to execute without moving the price against
itself. That slicing has a statistical fingerprint — regular sizes, regular
timing, concentrated venues — that a sophisticated counterparty can detect
and trade ahead of. This is a real, known problem in market microstructure
(often discussed as "gaming" or "signal leakage" in execution research).

RedBlue is a small testbed for that problem:

- **`engine/order_flow.py`** — simulates TWAP/VWAP/ICEBERG child-order
  generation with configurable parent order size and cadence.
- **`engine/attacker.py`** — a fast, hand-weighted statistical heuristic
  (entropy, autocorrelation, runs test) that scores how "readable" a stream
  of orders looks in real time. This is explicitly NOT a trained model —
  it's cheap, instant, explainable, and meant for live UX, not benchmarking.
- **`engine/ml_models.py`** — the actual trained, evaluated ML in this
  project: a `StrategyClassifier` (can a model tell TWAP from VWAP from
  ICEBERG?) and a `DefenseDiscriminator` (can a model tell defended flow
  from undefended flow?), both trained with `GroupKFold` cross-validation
  grouped by parent order, so no metaorder's data ever crosses the
  train/test boundary.
- **`engine/defender.py`** — obfuscates child orders by blending their
  size/timing/venue toward a randomized "ambient flow" target profile,
  subject to a hard VWAP deviation cap.
- **`engine/compliance.py`** — generates an audit token per defender
  action, with constraint checks that are either genuinely computed from
  order history or explicitly marked as not evaluated (never faked).
- **`scripts/train_and_evaluate.py`** — the evidence artifact. Trains both
  models, runs an intensity dose-response sweep, and prints/saves an honest
  report. Run this yourself: `python scripts/train_and_evaluate.py --sweep`.
- **`scripts/test_adversarial.py`** — tests whether optimizing the defender
  directly against a trained discriminator beats the hand-designed one.

## The actual result, run it yourself

```
python scripts/train_and_evaluate.py --n-parents 300 --window 20 --sweep
python scripts/adversarial_cotrain.py --rounds 6 --n-parents 80 --n-candidates 10
```

On the synthetic benchmark:
- The `StrategyClassifier` gets ~100% held-out accuracy identifying
  TWAP/VWAP/ICEBERG from a 20-order window (vs. a ~60-66% majority-class
  baseline) — expected, since this generator's three patterns have quite
  different size/timing shapes by construction.
- The `DefenderAgent`'s obfuscation targets were originally hand-picked and
  never checked against what real order flow actually looks like, which let
  a trained `DefenseDiscriminator` separate defended from undefended windows
  with **AUC ≈ 0.99-1.00** at the default intensity, rising monotonically to
  1.00 at max intensity (see docs/experiments.md). Recalibrating those targets
  against the generator's real empirical distribution
  (`scripts/calibrate_noise_profile.py`, docs/experiments.md) measurably helped —
  **AUC ≈ 0.91 at the default intensity (L3), 0.78 at L1** — but detectability
  still rises monotonically with intensity, meaning the obfuscation itself
  remains a fingerprint, just a fainter one.
- A one-shot candidate-search defender optimized directly against a frozen
  discriminator (`scripts/test_adversarial.py`) still does not meaningfully
  improve on the heuristic baseline (Δ≈+0.02, judged by a freshly-trained,
  held-out discriminator that never saw the optimization target) — optimizing
  against a single static judge just overfits to it.
- Genuine multi-round co-training (`scripts/adversarial_cotrain.py`) —
  alternately retraining a fresh discriminator each round and re-optimizing
  the defender against the ensemble of every discriminator trained so far —
  does measurably better: **AUC ≈ 0.63**, confirmed by a discriminator that
  never contributed to that ensemble (down from a 0.93 heuristic baseline in
  the same run; see docs/experiments.md for the round-by-round trajectory).

**Read that as a real, still-negative-on-balance result, not a failure to
hide.** Recalibration and closed-loop co-training both measurably reduce
detectability. But even
the best result here (AUC ≈ 0.63) is well above the undetectable baseline of
0.5, with no sign in a 6-round trajectory that it's converging there rather
than plateauing. This is a small-sample proof of concept (80-300 parent
orders, gradient-boosted trees as both attacker and defender surrogate), not
evidence that a production system with more rounds and a trained generator
network would land in the same place — see the Roadmap below for what's
actually still missing.

## What this is not

- Not a low-latency trading system. The `AdversarialDefender`'s candidate
  search takes single-digit milliseconds per order in Python — nowhere near
  what a real execution venue needs.
- Not backed by real institutional order flow. Everything above is measured
  on a synthetic generator (`engine/order_flow.py`). `scripts/download_sample_data.py`
  can pull real public trade tape (Binance) as a more realistic stand-in for
  ambient market noise, but there is no ground-truth "this was a TWAP order"
  label on real data, so the trained classifiers can only be validated on
  the synthetic benchmark. Real buy-side execution logs aren't public.
- Not a distributed platform. It's a single FastAPI process.

## Running it

```bash
pip install -r requirements.txt
python scripts/train_and_evaluate.py --sweep   # generates the trained models + honest eval report
uvicorn main:app --reload                       # live dashboard at http://localhost:8000
```

The live dashboard's `ml_metrics` panel reports `held_out_accuracy` /
`held_out_auc` from the file above — if you haven't run the training script,
those fields are `null`, not a placeholder number.

Once trained, two more things become available that are silently disabled
before that:
- The **Adversarial Defender** toggle in the top nav switches from
  `DefenderAgent`'s fixed obfuscation recipe to `AdversarialDefender`'s
  discriminator-guided candidate search (see the negative result above —
  this is an experiment, not a claim that it wins).
- `/api/audit/upload` and `/api/audit/analyze-sample` return a
  `validated_models` block with the trained `StrategyClassifier` and
  `DefenseDiscriminator`'s predictions on the uploaded window, alongside the
  fast heuristic score that runs regardless.

## Roadmap (not built)

- Longer, larger-scale adversarial co-training runs (more rounds, more
  parent orders, a trained generator network instead of a candidate-search
  surrogate) to see whether AUC ≈ 0.63 is a genuine floor for this approach
  or just where 6 rounds on 80 synthetic parent orders happens to land — the
  6-round trajectory in docs/experiments.md is noisy and doesn't show clear
  convergence toward 0.5 either way.
- Real market data ingestion beyond a single public trade-tape source, with
  proper handling of the fact that real order-level labels aren't public.
- If this were to become an actual low-latency system: a compiled hot path
  (Rust/C++) for the defend() decision, not Python — but that's a distinct,
  much larger engineering project from what's here, and shouldn't be implied
  to already exist.

## License

MIT — see [LICENSE](./LICENSE).
