"""
scripts/train_and_evaluate.py

The evidence artifact for this project. Generates a labeled dataset from the
order-flow simulator (the only source with ground-truth strategy labels and
guaranteed-identical raw/defended pairs), trains both ML models with
GroupKFold cross-validation (grouped by parent_id, so no metaorder leaks
across the train/test boundary), and prints a plain report of what was
actually measured.

Run: python scripts/train_and_evaluate.py [--n-parents 300] [--window 20]
"""
import argparse
import copy
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engine.order_flow import OrderFlowGenerator, SYMBOLS
from engine.defender import DefenderAgent
from engine.ml_models import StrategyClassifier, DefenseDiscriminator


def intensity_sweep(n_parents: int, window: int):
    """Train a fresh discriminator at each intensity level 1-5 and report AUC.
    This is the honest way to answer "does turning the dial up help at all" —
    it does not assume the answer, it measures it.
    """
    rng = np.random.default_rng(7)
    patterns = ["TWAP", "VWAP", "ICEBERG"]
    results = {}

    for intensity in range(0, 6):
        dw, dl, dg = [], [], []
        for i in range(n_parents):
            pattern = patterns[i % len(patterns)]
            symbol = rng.choice(SYMBOLS)
            gen = OrderFlowGenerator(symbol=symbol, slice_pattern=pattern,
                                      parent_order_size=int(rng.integers(20_000, 300_000)))
            raw_orders = []
            o = gen.generate_child_order(defended=False)
            while o is not None:
                raw_orders.append(o)
                o = gen.generate_child_order(defended=False)
            if len(raw_orders) < window + 1:
                continue

            defender = DefenderAgent(aggressiveness=0.7)
            defender.set_intensity(intensity)
            defender.reset_profile()
            defended_orders = []
            for o in raw_orders:
                oc = copy.deepcopy(o)
                defender.defend(oc)
                defended_orders.append(oc)

            for start in range(0, len(raw_orders) - window, max(1, window // 2)):
                dw.append(raw_orders[start:start + window]); dl.append(False); dg.append(gen.parent_id)
                dw.append(defended_orders[start:start + window]); dl.append(True); dg.append(gen.parent_id)

        disc = DefenseDiscriminator()
        report = disc.fit(dw, dl, dg)
        results[intensity] = {"auc": report.mean_auc, "accuracy": report.mean_accuracy}
        print(f"    L{intensity} ({INTENSITY_NAME.get(intensity,'?')}): "
              f"AUC={report.mean_auc:.3f}  accuracy={report.mean_accuracy:.3f}")
    return results


from engine.defender import INTENSITY_LEVELS as _IL
INTENSITY_NAME = {k: v["name"] for k, v in _IL.items()}


def build_dataset(n_parents: int, window: int, seed: int = 42):
    """Generate n_parents synthetic metaorders and slide a window over each.

    Returns:
        strategy_windows, strategy_labels, strategy_groups
        defense_windows, defense_labels, defense_groups
    """
    rng = np.random.default_rng(seed)
    patterns = ["TWAP", "VWAP", "ICEBERG"]

    strategy_windows, strategy_labels, strategy_groups = [], [], []
    defense_windows, defense_labels, defense_groups = [], [], []

    for i in range(n_parents):
        pattern = patterns[i % len(patterns)]  # balanced classes, not just random
        symbol = rng.choice(SYMBOLS)
        gen = OrderFlowGenerator(
            symbol=symbol,
            slice_pattern=pattern,
            parent_order_size=int(rng.integers(20_000, 300_000)),
        )

        raw_orders = []
        order = gen.generate_child_order(defended=False)
        while order is not None:
            raw_orders.append(order)
            order = gen.generate_child_order(defended=False)

        if len(raw_orders) < window + 1:
            continue

        parent_id = gen.parent_id

        # --- Strategy classification windows (raw stream only) ---
        for start in range(0, len(raw_orders) - window, max(1, window // 2)):
            w = raw_orders[start:start + window]
            strategy_windows.append(w)
            strategy_labels.append(pattern)
            strategy_groups.append(parent_id)

        # --- Defense discrimination: defend a DEEP COPY of the exact same stream ---
        defender = DefenderAgent(aggressiveness=0.7)
        defender.set_intensity(3)
        defender.reset_profile()
        defended_orders = []
        for o in raw_orders:
            o_copy = copy.deepcopy(o)
            defender.defend(o_copy)
            defended_orders.append(o_copy)

        for start in range(0, len(raw_orders) - window, max(1, window // 2)):
            defense_windows.append(raw_orders[start:start + window])
            defense_labels.append(False)
            defense_groups.append(parent_id)

            defense_windows.append(defended_orders[start:start + window])
            defense_labels.append(True)
            defense_groups.append(parent_id)

    return (strategy_windows, strategy_labels, strategy_groups,
            defense_windows, defense_labels, defense_groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-parents", type=int, default=300)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--out", default="data/eval_report.json")
    ap.add_argument("--sweep", action="store_true", help="Also run an intensity 0-5 dose-response sweep")
    args = ap.parse_args()

    if args.sweep:
        print("=" * 60)
        print(" INTENSITY DOSE-RESPONSE SWEEP")
        print("=" * 60)
        sweep_results = intensity_sweep(args.n_parents, args.window)
        os.makedirs("data", exist_ok=True)
        with open("data/intensity_sweep.json", "w") as f:
            json.dump(sweep_results, f, indent=2)
        print()

    print("=" * 60)
    print(" REDBLUE — MODEL TRAINING & HONEST EVALUATION")
    print("=" * 60)
    print(f"\nGenerating {args.n_parents} synthetic parent orders (window={args.window})...")

    (sw, sl, sg, dw, dl, dg) = build_dataset(args.n_parents, args.window)
    print(f"  Strategy-classification windows: {len(sw)} (from {len(set(sg))} parent orders)")
    print(f"  Defense-discrimination windows:  {len(dw)} (from {len(set(dg))} parent orders)")

    print("\n[1] Training StrategyClassifier (TWAP / VWAP / ICEBERG) — GroupKFold by parent_id")
    strat_clf = StrategyClassifier()
    strat_report = strat_clf.fit(sw, sl, sg)
    print(f"    Mean accuracy: {strat_report.mean_accuracy:.3f} (+/- {strat_report.std_accuracy:.3f})")
    print(f"    Majority-class baseline: {strat_report.baseline_majority_class_accuracy:.3f}")
    print(f"    Lift over baseline: {strat_report.mean_accuracy - strat_report.baseline_majority_class_accuracy:+.3f}")
    if strat_report.warning:
        print(f"    WARNING: {strat_report.warning}")

    print("\n[2] Training DefenseDiscriminator (defended vs undefended) — GroupKFold by parent_id")
    disc = DefenseDiscriminator()
    disc_report = disc.fit(dw, dl, dg)
    print(f"    Mean accuracy: {disc_report.mean_accuracy:.3f} (+/- {disc_report.std_accuracy:.3f})")
    print(f"    Mean AUC: {disc_report.mean_auc:.3f}" if disc_report.mean_auc is not None else "    Mean AUC: n/a")
    print(f"    Baseline (majority class): {disc_report.baseline_majority_class_accuracy:.3f}")
    if disc_report.warning:
        print(f"    WARNING: {disc_report.warning}")

    print("\n" + "=" * 60)
    print(" WHAT THIS MEANS")
    print("=" * 60)
    auc = disc_report.mean_auc if disc_report.mean_auc is not None else 1.0
    if auc > 0.75:
        verdict = "Defense is NOT working well — a trained model still separates defended from undefended flow easily."
    elif auc > 0.6:
        verdict = "Defense reduces but does not eliminate detectability. There is a measurable residual signal."
    else:
        verdict = "Defense brings discriminability close to chance (AUC ~0.5) on this synthetic benchmark."
    print(f" {verdict}")
    print(" This is measured on the synthetic generator, not live fund data.")
    print(" Treat it as a benchmark of the method, not a guarantee for any specific")
    print(" real order flow, which will have different statistics.")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "strategy_classification": strat_report.to_dict(),
            "defense_discrimination": disc_report.to_dict(),
            "config": {"n_parents": args.n_parents, "window": args.window},
        }, f, indent=2)
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
