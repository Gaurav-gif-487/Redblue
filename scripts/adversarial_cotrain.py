"""
scripts/adversarial_cotrain.py

The single-round test in scripts/test_adversarial.py (see docs/experiments.md)
found that optimizing the defender against ONE frozen discriminator doesn't
generalize -- a fresh, held-out judge caught it just as well or worse. That's
the textbook overfitting failure mode of adversarial training when only one
side of the game ever updates.

This script closes the actual loop:

  Round 0: heuristic DefenderAgent (calibrated, see engine/defender.py) produces
           the starting defended dataset.
  Round i (i = 1..N):
    (a) Train a FRESH discriminator D_i from scratch on round (i-1)'s defended
        data (GroupKFold, never reuses a previous round's D).
    (b) Re-defend a NEW batch of parent orders with AdversarialDefender,
        searching against the ENSEMBLE of all discriminators trained so far
        (D_1..D_i), not just the newest one. Scoring against the historical
        ensemble (fictitious play) is what stops the defender from just
        re-learning a narrow exploit against whichever D is newest, which is
        the specific failure mode measured in docs/experiments.md (D2 AUC=1.000
        after optimizing against a single D1).

  Final honest check: train ONE more discriminator, D_eval, from scratch on
  FRESH held-out parent orders defended by the round-N policy. D_eval never
  contributes to the ensemble the defender optimized against. If D_eval's AUC
  is still high, co-training didn't generalize either -- report that, don't
  hide it.

Usage: python scripts/adversarial_cotrain.py --rounds 4 --n-parents 150
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
from engine.ml_models import DefenseDiscriminator
from engine.adversarial_defender import AdversarialDefender

WINDOW = 20


def make_raw_dataset(seed, n_parents):
    rng = np.random.default_rng(seed)
    patterns = ["TWAP", "VWAP", "ICEBERG"]
    parents = []
    for i in range(n_parents):
        pattern = patterns[i % len(patterns)]
        symbol = rng.choice(SYMBOLS)
        gen = OrderFlowGenerator(symbol=symbol, slice_pattern=pattern,
                                  parent_order_size=int(rng.integers(20_000, 300_000)))
        raw = []
        o = gen.generate_child_order(defended=False)
        while o is not None:
            raw.append(o)
            o = gen.generate_child_order(defended=False)
        if len(raw) >= WINDOW + 1:
            parents.append((gen.parent_id, raw))
    return parents


def windows_from(parents, defended_fn):
    dw, dl, dg = [], [], []
    for parent_id, raw in parents:
        defended = defended_fn(raw)
        for start in range(0, len(raw) - WINDOW, max(1, WINDOW // 2)):
            dw.append(raw[start:start + WINDOW]); dl.append(False); dg.append(parent_id)
            dw.append(defended[start:start + WINDOW]); dl.append(True); dg.append(parent_id)
    return dw, dl, dg


def heuristic_defend(raw, intensity=3, aggressiveness=0.7):
    defender = DefenderAgent(aggressiveness=aggressiveness)
    defender.set_intensity(intensity)
    defender.reset_profile()
    out = []
    for o in raw:
        c = copy.deepcopy(o)
        defender.defend(c)
        out.append(c)
    return out


def adversarial_defend(raw, discriminators, aggressiveness=0.7, n_candidates=12, seed=None):
    adv = AdversarialDefender(discriminators, aggressiveness=aggressiveness,
                               n_candidates=n_candidates, window=WINDOW, seed=seed)
    return [adv.defend(o) for o in raw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--n-parents", type=int, default=150)
    ap.add_argument("--n-candidates", type=int, default=12)
    ap.add_argument("--intensity", type=int, default=3)
    args = ap.parse_args()

    print("=" * 60)
    print(" MULTI-ROUND ADVERSARIAL CO-TRAINING")
    print("=" * 60)

    history_aucs = []
    ensemble = []  # all discriminators trained so far, used for fictitious play

    print(f"\n[Round 0] Heuristic DefenderAgent (L{args.intensity}, calibrated targets)...")
    parents0 = make_raw_dataset(seed=100, n_parents=args.n_parents)
    dw, dl, dg = windows_from(parents0, lambda r: heuristic_defend(r, intensity=args.intensity))
    d0 = DefenseDiscriminator()
    r0 = d0.fit(dw, dl, dg)
    print(f"    D0 AUC (heuristic baseline): {r0.mean_auc:.3f}")
    history_aucs.append(("round0_heuristic", r0.mean_auc))
    ensemble.append(d0)

    for i in range(1, args.rounds + 1):
        print(f"\n[Round {i}] Defend fresh parent orders against the ensemble of "
              f"{len(ensemble)} discriminator(s) so far...")
        parents_i = make_raw_dataset(seed=200 + i, n_parents=args.n_parents)
        dw_i, dl_i, dg_i = windows_from(
            parents_i,
            lambda r: adversarial_defend(r, ensemble, n_candidates=args.n_candidates, seed=300 + i),
        )

        print(f"    Training a FRESH discriminator D{i} from scratch on this round's data...")
        d_i = DefenseDiscriminator()
        d_i.model_filename = f"discriminator_cotrain_r{i}_model.pkl"
        d_i.scaler_filename = f"discriminator_cotrain_r{i}_scaler.pkl"
        r_i = d_i.fit(dw_i, dl_i, dg_i)
        print(f"    D{i} AUC (in-loop, trained on round {i}'s adversarial data): {r_i.mean_auc:.3f}")
        history_aucs.append((f"round{i}_adversarial", r_i.mean_auc))
        ensemble.append(d_i)

    print(f"\n[Final check] FRESH held-out parent orders, defended by the round-{args.rounds} "
          f"policy (ensemble of all {len(ensemble)} discriminators), judged by a "
          f"discriminator that never contributed to that ensemble...")
    eval_parents = make_raw_dataset(seed=999, n_parents=args.n_parents)
    dw_eval, dl_eval, dg_eval = windows_from(
        eval_parents,
        lambda r: adversarial_defend(r, ensemble, n_candidates=args.n_candidates, seed=42),
    )
    d_eval = DefenseDiscriminator()
    d_eval.model_filename, d_eval.scaler_filename = (
        "discriminator_cotrain_eval_model.pkl", "discriminator_cotrain_eval_scaler.pkl",
    )
    r_eval = d_eval.fit(dw_eval, dl_eval, dg_eval)
    print(f"    D_eval AUC (honest, held-out judge): {r_eval.mean_auc:.3f}")

    print("\n" + "=" * 60)
    print(" RESULT")
    print("=" * 60)
    for label, auc in history_aucs:
        print(f"  {label:28s} AUC = {auc:.3f}")
    print(f"  {'final_held_out_judge':28s} AUC = {r_eval.mean_auc:.3f}")

    baseline = history_aucs[0][1]
    delta = baseline - r_eval.mean_auc
    print()
    if delta > 0.03:
        print(f" -> Co-training measurably reduced detectability vs. the heuristic "
              f"baseline (Δ={delta:+.3f}), confirmed by a judge that never trained "
              f"against the defender's policy.")
    elif delta < -0.03:
        print(f" -> Co-training made it WORSE than the heuristic baseline (Δ={delta:+.3f}). "
              f"Report this, don't hide it.")
    else:
        print(f" -> No meaningful difference vs. the heuristic baseline (Δ={delta:+.3f}).")
    print(f" -> {r_eval.mean_auc:.3f} vs. the naive/undetectable baseline of 0.500 -- "
          f"{'still far from it' if r_eval.mean_auc > 0.6 else 'closing in on it'}.")
    print("=" * 60)

    report = {
        "rounds": args.rounds,
        "n_parents": args.n_parents,
        "n_candidates": args.n_candidates,
        "intensity": args.intensity,
        "history": history_aucs,
        "final_held_out_judge_auc": r_eval.mean_auc,
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "adversarial_cotrain_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
