"""
scripts/test_adversarial.py

Round 1: train a DefenseDiscriminator (D1) on heuristic-defended data.
Round 2: use D1 to guide AdversarialDefender, producing a new defended dataset.
Round 3: train a FRESH discriminator (D2) from scratch on THAT data via
         GroupKFold. D2 is the honest judge — it never saw D1's training
         data or objective, so if D2 still separates easily, the adversarial
         defender didn't generalize, it just overfit to fooling D1.
"""
import copy
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engine.order_flow import OrderFlowGenerator, SYMBOLS
from engine.defender import DefenderAgent
from engine.ml_models import DefenseDiscriminator
from engine.adversarial_defender import AdversarialDefender

N_PARENTS = 200
WINDOW = 20


def make_raw_dataset(seed):
    rng = np.random.default_rng(seed)
    patterns = ["TWAP", "VWAP", "ICEBERG"]
    parents = []
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


def heuristic_defend(raw):
    defender = DefenderAgent(aggressiveness=0.7)
    defender.set_intensity(3)
    defender.reset_profile()
    out = []
    for o in raw:
        c = copy.deepcopy(o)
        defender.defend(c)
        out.append(c)
    return out


def main():
    print("=" * 60)
    print(" ADVERSARIAL DEFENSE — DOES CLOSING THE LOOP HELP?")
    print("=" * 60)

    print("\n[Round 1] Train baseline discriminator D1 on heuristic-defended data...")
    train_parents = make_raw_dataset(seed=1)
    dw, dl, dg = windows_from(train_parents, heuristic_defend)
    d1 = DefenseDiscriminator()
    r1 = d1.fit(dw, dl, dg)
    print(f"    D1 mean AUC (heuristic defense): {r1.mean_auc:.3f}")

    print("\n[Round 2] Generate NEW parent orders, defend using AdversarialDefender guided by D1...")
    eval_parents = make_raw_dataset(seed=2)  # different orders than D1's training set

    def adv_defend(raw):
        adv = AdversarialDefender(d1, aggressiveness=0.7, n_candidates=12, window=WINDOW, seed=None)
        out = []
        for o in raw:
            out.append(adv.defend(o))
        return out

    dw2, dl2, dg2 = windows_from(eval_parents, adv_defend)

    print("\n[Round 3] Train a FRESH discriminator D2 from scratch on the adversarial data (honest judge)...")
    d2 = DefenseDiscriminator()
    d2.model_filename, d2.scaler_filename = "discriminator_adv_model.pkl", "discriminator_adv_scaler.pkl"
    r2 = d2.fit(dw2, dl2, dg2)
    print(f"    D2 mean AUC (adversarial defense, held-out judge): {r2.mean_auc:.3f}")

    print("\n" + "=" * 60)
    print(" RESULT")
    print("=" * 60)
    print(f" Heuristic defense (D1, L3):        AUC = {r1.mean_auc:.3f}")
    print(f" Adversarially-optimized defense:   AUC = {r2.mean_auc:.3f}")
    delta = r1.mean_auc - r2.mean_auc
    if delta > 0.03:
        print(f" -> Adversarial optimization measurably reduced detectability (Δ={delta:+.3f}).")
    elif delta < -0.03:
        print(f" -> Adversarial optimization made it WORSE (Δ={delta:+.3f}). Report this, don't hide it.")
    else:
        print(f" -> No meaningful difference (Δ={delta:+.3f}) at this sample size / candidate count.")
    print("=" * 60)


if __name__ == "__main__":
    main()
