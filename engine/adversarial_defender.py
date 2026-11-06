"""
engine/adversarial_defender.py

The intensity sweep in scripts/train_and_evaluate.py (--sweep) found that the
hand-designed DefenderAgent (engine/defender.py) gets MORE detectable, not
less, as its intensity is turned up: AUC goes from 0.94 at L1 to 1.00 at L5.
That's the real finding of this project, not a bug to quietly patch — a
fixed-recipe obfuscation swaps one fingerprint (the original algorithm's) for
another (the obfuscator's own parametric signature), and a discriminator with
enough trees just learns the new fingerprint instead.

This module tests the thesis: that closing the loop — optimizing
the defender directly against a trained discriminator, instead of against
fixed hand-picked target statistics — does better. It is NOT a production
low-latency system (each order does a small local search, ~5-15ms per order
in this implementation, nowhere near sub-millisecond latency). It is
a proof-of-concept for whether the adversarial framing has legs.

Method: simple candidate search, not a trained MARL policy. For each order,
sample K candidate (size, timing) perturbations, build the resulting recent
window, score each candidate's p_defended under the trained
DefenseDiscriminator, and keep whichever candidate scores lowest — subject to
the same VWAP price-guard constraint as DefenderAgent.

v2 fix, found while wiring up multi-round co-training
(scripts/adversarial_cotrain.py): the original candidate search perturbed
size by N(0, 0.35*aggressiveness) and timing by N(0, 0.5*aggressiveness) —
i.e. up to ~25-35% relative noise at the default aggressiveness. That is the
exact same class of bug already diagnosed and fixed in engine/defender.py's
v1->v2 note: noise scaled off the order's own value, unmoored from any real
reference distribution, rather than bounded to the range real order flow
actually occupies (see scripts/calibrate_noise_profile.py). It explains why
the first single-round test (docs/experiments.md, and the recalibration follow-up in
#10) showed adversarial optimization matching or *beating* the heuristic
defense in the wrong direction — the search could freely wander into
out-of-support candidate values that happened to fool one particular
discriminator's decision boundary, which a fresh discriminator then caught
immediately. Candidate perturbation scale is now bounded by the same
calibrated constants the heuristic defender uses
(defender.SIZE_CV_TARGET_RANGE, defender.IAT_STD_FRACTION_RANGE), and every
candidate set always includes one "no perturbation" candidate so the search
can choose to leave an already-regular order (e.g. ICEBERG) untouched instead
of being forced to inject noise every time.

Also supports scoring against an ENSEMBLE of discriminators (one per prior
co-training round) rather than a single frozen one. This is the standard
fix for the "overfits to fool this one judge" failure mode in adversarial
training (historical averaging / fictitious play): optimizing against the
average of several discriminators trained at different points is harder to
game with a narrow exploit than optimizing against whichever one is newest.
"""
from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Union

import numpy as np

from .order_flow import ChildOrder, VENUES
from .ml_models import DefenseDiscriminator, _window_features
from .defender import SIZE_CV_TARGET_RANGE, IAT_STD_FRACTION_RANGE


class AdversarialDefender:
    def __init__(self, discriminators: Union[DefenseDiscriminator, Sequence[DefenseDiscriminator]],
                 aggressiveness: float = 0.7,
                 n_candidates: int = 12, window: int = 20, max_vwap_deviation_bps: float = 8.0,
                 seed: Optional[int] = None):
        discs = [discriminators] if isinstance(discriminators, DefenseDiscriminator) else list(discriminators)
        if not discs or any(not d.trained for d in discs):
            raise ValueError("AdversarialDefender requires at least one already-trained DefenseDiscriminator")
        self.discriminators = discs
        self.aggressiveness = float(np.clip(aggressiveness, 0.0, 1.0))
        self.n_candidates = n_candidates
        self.window = window
        self.max_vwap_deviation_bps = max_vwap_deviation_bps
        self.rng = np.random.default_rng(seed)
        self._history: List[ChildOrder] = []  # defended history for windowing

        # Candidate perturbation scale bounded to the same empirically-calibrated
        # range the heuristic defender targets (see module docstring), instead
        # of a fraction of the order's own value.
        self._size_cv_max = SIZE_CV_TARGET_RANGE[1]
        self._timing_cv_max = IAT_STD_FRACTION_RANGE[1]

    def _score(self, candidate_window: List[ChildOrder]) -> float:
        if len(candidate_window) < 5:
            return 0.5
        scores = []
        for d in self.discriminators:
            pred = d.predict(candidate_window)
            scores.append(pred.get("p_defended", 0.5))
        return float(np.mean(scores))

    def defend(self, order: ChildOrder) -> ChildOrder:
        """Return a defended copy of `order`, chosen to minimize discriminator confidence."""
        original_price = order.price
        best_candidate = None
        best_score = float("inf")

        recent = self._history[-(self.window - 1):]

        # Candidate 0 is always "no perturbation" — lets the search choose to
        # leave an already-regular order untouched (see v2 note above) rather
        # than being forced to inject noise on every single order.
        candidates_meta = [(0.0, 0.0, False)]
        for _ in range(self.n_candidates - 1):
            size_frac = float(self.rng.normal(0, self._size_cv_max * self.aggressiveness))
            timing_frac = float(self.rng.normal(0, self._timing_cv_max * self.aggressiveness))
            reroute = self.rng.random() < 0.3 * self.aggressiveness
            candidates_meta.append((size_frac, timing_frac, reroute))

        for size_frac, timing_frac, reroute in candidates_meta:
            c = copy.deepcopy(order)
            c.size = max(1, round(order.size * max(0.5, 1.0 + size_frac)))
            c.timing_ms = max(30.0, order.timing_ms * (1.0 + timing_frac))
            if reroute:
                c.venue = self.rng.choice(VENUES)

            # Hard VWAP guard, same rule as DefenderAgent
            dev_bps = abs(c.price - original_price) / original_price * 10_000 if original_price > 0 else 0
            if dev_bps > self.max_vwap_deviation_bps:
                c.price = original_price

            window = recent + [c]
            score = self._score(window)
            if score < best_score:
                best_score = score
                best_candidate = c

        best_candidate.defended = True
        self._history.append(best_candidate)
        if len(self._history) > 200:
            self._history = self._history[-200:]
        return best_candidate

    def reset(self):
        self._history = []

