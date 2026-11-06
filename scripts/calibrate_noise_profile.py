"""
scripts/calibrate_noise_profile.py

The old NOISE_TRADER_PROFILE ranges in engine/defender.py (SIZE_CV_TARGET_RANGE,
IAT_STD_FRACTION_RANGE) were hand-picked without checking them against what
undefended order flow in this system actually looks like. That gap is exactly
what let a DefenseDiscriminator separate defended from undefended windows with
AUC up to 1.000 (see docs/experiments.md).

This script measures the real thing: it generates a large sample of undefended
child-order windows across all three strategies (TWAP/VWAP/ICEBERG), computes
the same window-level features the discriminator sees, and prints the
empirical percentile range for each feature the defender tries to blend
toward. The output here is what engine/defender.py's target ranges should be
sampled from -- not a guess.

Run: python scripts/calibrate_noise_profile.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engine.order_flow import OrderFlowGenerator, SYMBOLS, VENUES
from engine.ml_models import _window_features, FEATURE_NAMES

N_PARENTS = 900
WINDOW = 20


def collect():
    rng = np.random.default_rng(11)
    patterns = ["TWAP", "VWAP", "ICEBERG"]
    feats = []
    timing_means = []
    for i in range(N_PARENTS):
        pattern = patterns[i % len(patterns)]
        symbol = rng.choice(SYMBOLS)
        gen = OrderFlowGenerator(symbol=symbol, slice_pattern=pattern,
                                  parent_order_size=int(rng.integers(20_000, 300_000)))
        raw = []
        o = gen.generate_child_order(defended=False)
        while o is not None:
            raw.append(o)
            o = gen.generate_child_order(defended=False)
        if len(raw) < WINDOW + 1:
            continue
        timing_means.append(float(np.mean([r.timing_ms for r in raw])))
        for start in range(0, len(raw) - WINDOW, max(1, WINDOW // 2)):
            feats.append(_window_features(raw[start:start + WINDOW]))
    return np.array(feats), np.array(timing_means)


def main():
    feats, timing_means = collect()
    print(f"Collected {len(feats)} undefended windows across TWAP/VWAP/ICEBERG "
          f"(and {len(timing_means)} parent orders for timing-mean stats).\n")

    idx_size_cv = FEATURE_NAMES.index("size_cv")
    idx_timing_cv = FEATURE_NAMES.index("timing_cv")
    idx_venue_entropy = FEATURE_NAMES.index("venue_entropy")

    size_cv = feats[:, idx_size_cv]
    timing_cv = feats[:, idx_timing_cv]
    venue_entropy = feats[:, idx_venue_entropy]

    def report(name, arr, lo_pct=5, hi_pct=95):
        lo, hi = np.percentile(arr, lo_pct), np.percentile(arr, hi_pct)
        print(f"{name:16s} p{lo_pct}={lo:.4f}  p{hi_pct}={hi:.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")
        return lo, hi

    print("-- empirical ranges (undefended, all strategies pooled) --")
    size_lo, size_hi = report("size_cv", size_cv)
    timing_lo, timing_hi = report("timing_cv", timing_cv)
    venue_lo, venue_hi = report("venue_entropy", venue_entropy, lo_pct=25, hi_pct=75)

    timing_mean_lo, timing_mean_hi = np.percentile(timing_means, 5), np.percentile(timing_means, 95)
    print(f"{'timing_ms mean':16s} p5={timing_mean_lo:.1f}  p95={timing_mean_hi:.1f}  "
          f"mean={np.mean(timing_means):.1f}")

    print(f"\nmax possible venue entropy (log2({len(VENUES)})) = {np.log2(len(VENUES)):.3f}")

    print("\n-- suggested replacement constants for engine/defender.py --")
    print(f"SIZE_CV_TARGET_RANGE = ({size_lo:.3f}, {size_hi:.3f})")
    print(f"IAT_MEAN_TARGET_RANGE_MS = ({timing_mean_lo:.1f}, {timing_mean_hi:.1f})")
    # IAT_STD_FRACTION_RANGE approximates timing_cv once blended in, so target
    # it directly at the empirical timing_cv range rather than the mean range.
    print(f"IAT_STD_FRACTION_RANGE = ({timing_lo:.3f}, {timing_hi:.3f})")
    print(f"VENUE_ENTROPY_TARGET_RANGE (25th-75th pct) = ({venue_lo:.3f}, {venue_hi:.3f})  "
          f"-- for reference; current venue routing pushes toward max entropy "
          f"({np.log2(len(VENUES)):.2f}) regardless of this.")


if __name__ == "__main__":
    main()
