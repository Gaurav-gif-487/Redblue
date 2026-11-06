"""
Defender Agent — order flow obfuscation engine.
Applies statistical transformations to break detectable patterns
while preserving execution quality (VWAP <= max_vwap_deviation_bps).

v2 design note: the original version perturbed each field with noise
proportional to a fixed fraction of the order's own value (e.g.
size_noise_std = order.size * 0.3). That overshoots for regular strategies
like TWAP (near-zero baseline variance) and undershoots for already-noisy
ones, and it doesn't target any real reference distribution — it just makes
the stream "noisier than before," which a discriminator trained on
size_cv/timing_cv finds trivially separable (measured AUC 1.00 on the
benchmark in scripts/train_and_evaluate.py before this fix).

v3 design note: v2 fixed the "grows without bound" problem by blending
toward a per-parent-order NOISE_TRADER_PROFILE instead of scaling noise off
the order's own value. But the profile's target ranges (SIZE_CV_TARGET_RANGE,
IAT_STD_FRACTION_RANGE below) were still hand-picked, not measured — and a
feature-importance check on the resulting discriminator showed size_cv and
timing_cv alone carried ~96% of the detection signal. The reason: the old
size_cv range (0.10-0.55) started above the real undefended ceiling
(empirically ~0.06, see scripts/calibrate_noise_profile.py), so blending
toward it moved every defended window into a region undefended windows
almost never occupy — a free threshold for any classifier. Same problem,
smaller magnitude, on the IAT std fraction (old range 0.25-0.60 vs. real
timing_cv topping out around 0.10-0.12).

The ranges below now come directly from scripts/calibrate_noise_profile.py,
which measures the actual empirical distribution of undefended TWAP/VWAP/
ICEBERG windows rather than guessing at one. Re-run that script if
order_flow.py's generation logic changes, and re-run
scripts/train_and_evaluate.py --sweep after changing these constants to see
the discriminator AUC actually respond — do not trust a "score reduced"
number that isn't backed by that evaluation.

Known remaining gap, stated honestly: venue routing (see `venue_prob` below)
still pushes toward a uniform random venue choice at high intensity, and
real venue_entropy sits lower and narrower (25th-75th pct ~1.90-2.26 out of
a max of 3.0 for 8 venues) than what near-uniform routing produces. Capping
venue_prob is a partial mitigation, not a fix — a real fix would route with
weights matched to the empirical venue distribution instead of uniformly,
which is not implemented here.
"""
import time
import uuid
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .order_flow import ChildOrder, VENUES, SYMBOLS
from .compliance import ComplianceEngine, LPAToken


# Obfuscation intensity levels. `blend` is how far each order's parameters are
# pulled toward NOISE_TRADER_PROFILE (0 = untouched, 1 = fully resampled from
# the target profile). This replaces the old independent per-field noise
# fractions, which compounded into an inconsistent, more-detectable signal.
# `venue_prob` ceilings were lowered from the old 0.15-0.90 range as a partial
# mitigation for the venue_entropy leak described above (uncapped routing
# pushed toward near-uniform entropy ~2.7-3.0, well above the real 25th-75th
# pct range of ~1.90-2.26). This does not fully close that gap — see the
# module docstring.
INTENSITY_LEVELS = {
    0: {"name": "MONITORING", "blend": 0.0, "venue_prob": 0.0, "decoy": False},
    1: {"name": "MINIMAL", "blend": 0.15, "venue_prob": 0.08, "decoy": False},
    2: {"name": "LOW", "blend": 0.30, "venue_prob": 0.16, "decoy": False},
    3: {"name": "MEDIUM", "blend": 0.50, "venue_prob": 0.27, "decoy": False},
    4: {"name": "HIGH", "blend": 0.70, "venue_prob": 0.38, "decoy": True},
    5: {"name": "MAXIMUM", "blend": 0.90, "venue_prob": 0.50, "decoy": True},
}

# Range that each metaorder's target profile is sampled from, per parent
# order (see DefenderAgent.reset_profile). A FIXED target CV was tried first
# and measurably failed: it produces a tight, non-overlapping band of
# defended-window statistics that a trained discriminator separates from
# undefended windows with a single threshold (AUC 1.00, see docs/experiments.md).
# Sampling a fresh target per parent order spreads defended-window statistics
# across the same range real strategy diversity already occupies, so a
# classifier can't split on CV/entropy alone.
#
# These specific bounds come from scripts/calibrate_noise_profile.py, which
# measures the empirical distribution of undefended TWAP/VWAP/ICEBERG windows
# (see the v3 design note above for why the previous hand-picked bounds
# failed). Re-run that script and update these if order_flow.py changes.
SIZE_CV_TARGET_RANGE = (0.00, 0.06)
IAT_MEAN_TARGET_RANGE_MS = (280.0, 860.0)
IAT_STD_FRACTION_RANGE = (0.03, 0.10)  # std sampled as this fraction of the mean

# Even after the range fix above, a feature-importance check on the resulting
# discriminator showed size_cv STILL carrying ~85% of the detection signal.
# Cause: ICEBERG orders have a genuinely deterministic clip size — real
# undefended ICEBERG windows measure size_cv == 0.000 in 100% of samples
# (scripts/calibrate_noise_profile.py). But SIZE_CV_TARGET_RANGE is a
# continuous uniform draw, which is essentially never exactly 0 — so a
# defended window can practically never land on the exact-zero point-mass
# that a third of real strategies (ICEBERG, in this generator's even 3-way
# strategy mix) sit on. Any classifier can split on "suspiciously nonzero
# but small" alone and pick up most ICEBERG-origin windows for free.
# Fix: reproduce that point-mass directly by sampling target_size_cv == 0.0
# outright with probability matching the real zero-variance strategy share,
# instead of drawing continuously and hoping to land close enough.
ZERO_SIZE_CV_PROBABILITY = 1 / 3  # matches ICEBERG's share of the 3-way TWAP/VWAP/ICEBERG mix


class DefenderAction:
    """Records a single defender intervention with full audit trail."""

    def __init__(self):
        self.original: Dict = {}
        self.modified: Dict = {}
        self.modifications: List[Dict] = []
        self.lpa: Optional[LPAToken] = None
        self.latency_ns: int = 0
        self.latency_ms: float = 0.0
        self.timestamp: float = time.time()
        self.intensity_level: int = 2
        self.decoy_signals: List[Dict] = []
        self.score_before: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "original": self.original,
            "modified": self.modified,
            "modifications": self.modifications,
            "lpa": self.lpa.to_dict() if self.lpa else None,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "intensity_level": self.intensity_level,
            "intensity_name": INTENSITY_LEVELS.get(self.intensity_level, {}).get("name", "UNKNOWN"),
            "decoy_signals": self.decoy_signals,
            "score_before": self.score_before,
            # No "score_after_estimate" — see the note in defend() for why a
            # single-order post-defense score can't honestly be computed here.
        }


class DefenderAgent:
    """
    Prescribes real-time modifications to make order flow
    statistically indistinguishable from noise.

    Enhanced with 6 obfuscation techniques and intensity levels L0-L5.
    """

    def __init__(self, aggressiveness: float = 0.7, max_vwap_deviation_bps: float = 8.0):
        self.aggressiveness = np.clip(aggressiveness, 0.0, 1.0)
        self.max_vwap_deviation_bps = max_vwap_deviation_bps
        self.rng = np.random.default_rng()
        self.compliance = ComplianceEngine()
        self.actions: List[DefenderAction] = []
        self.intensity_level: int = 2  # Default L2
        self._recent_sizes: List[int] = []
        self._recent_timings: List[float] = []
        self._recent_venues: List[str] = []
        self._recent_orders: List = []  # rolling ChildOrder history for real compliance checks (C001, C005)
        self.reset_profile()

    def reset_profile(self):
        """Resample this defender's target noise-trader profile.

        Call this at the start of each new parent/metaorder (main.py does
        this in new_parent_orders()) so that consecutive metaorders don't
        share one detectable target distribution.
        """
        if self.rng.random() < ZERO_SIZE_CV_PROBABILITY:
            self._target_size_cv = 0.0
        else:
            self._target_size_cv = float(self.rng.uniform(*SIZE_CV_TARGET_RANGE))
        self._target_iat_mean = float(self.rng.uniform(*IAT_MEAN_TARGET_RANGE_MS))
        self._target_iat_std = self._target_iat_mean * float(self.rng.uniform(*IAT_STD_FRACTION_RANGE))

    def defend(self, order: ChildOrder, attacker_analysis: Dict = None) -> DefenderAction:
        """
        Apply obfuscation to a single child order.
        Returns DefenderAction with full modification audit trail.
        LPA token is generated BEFORE any modification is surfaced.
        """
        start_ns = time.perf_counter_ns()
        action = DefenderAction()
        action.original = order.model_dump()
        action.intensity_level = self.intensity_level
        modifications = []

        # Get intensity parameters
        params = INTENSITY_LEVELS.get(self.intensity_level, INTENSITY_LEVELS[2])

        # Record attacker score before
        if attacker_analysis:
            action.score_before = attacker_analysis.get("detection_score", 0)

        # L0 = monitoring only
        if self.intensity_level == 0:
            action.modified = order.model_dump()
            action.modifications = []
            latency_ns = time.perf_counter_ns() - start_ns
            lpa = self.compliance.generate_lpa(
                order_id=order.id, modifications=["MONITORING_ONLY"],
                vwap_deviation_bps=0.0, max_allowed_bps=self.max_vwap_deviation_bps,
                latency_ns=latency_ns, recent_orders=self._recent_orders,
            )
            action.lpa = lpa
            action.latency_ns = latency_ns
            action.latency_ms = round(latency_ns / 1e6, 4)
            self.actions.append(action)
            return action

        blend = params["blend"] * self.aggressiveness

        # ---- 1. Size: blend toward NOISE_TRADER_PROFILE size distribution ----
        original_size = order.size
        baseline_mean = float(np.mean(self._recent_sizes[-10:])) if len(self._recent_sizes) >= 3 else order.size
        target_std = baseline_mean * self._target_size_cv
        resampled_size = max(1.0, baseline_mean + float(self.rng.normal(0, target_std)))
        new_size = max(1, round((1 - blend) * order.size + blend * resampled_size))

        if new_size != original_size:
            modifications.append({
                "type": "SIZE_BLEND",
                "from": original_size,
                "to": new_size,
                "rationale": f"Blended {blend:.0%} toward this metaorder's target size CV={self._target_size_cv:.2f}, "
                             f"vs. baseline mean {baseline_mean:.0f} — breaks size regularity without discarding order intent",
            })
            order.size = new_size

        # ---- 2. Timing: blend toward this metaorder's target inter-arrival distribution ----
        original_timing = order.timing_ms
        target_iat = float(self.rng.normal(self._target_iat_mean, self._target_iat_std))
        target_iat = max(50.0, min(target_iat, 3000.0))
        new_timing = (1 - blend) * order.timing_ms + blend * target_iat

        if abs(new_timing - original_timing) > 5:
            modifications.append({
                "type": "TIMING_BLEND",
                "from": round(original_timing, 1),
                "to": round(new_timing, 1),
                "rationale": f"Blended {blend:.0%} toward this metaorder's target IAT "
                             f"(mean={self._target_iat_mean:.0f}ms) to destroy scheduling periodicity",
            })
            order.timing_ms = new_timing

        # ---- 3. Venue: route toward near-uniform distribution (entropy target) ----
        if self.rng.random() < params["venue_prob"] * self.aggressiveness:
            original_venue = order.venue
            new_venue = self.rng.choice(VENUES)
            if new_venue != original_venue:
                modifications.append({
                    "type": "VENUE_ROUTE",
                    "from": original_venue,
                    "to": new_venue,
                    "rationale": "Stochastic re-routing toward uniform venue distribution; best-execution compliant",
                })
                order.venue = new_venue

        # ---- 4. Decoy Trade Signals (cosmetic; logged for audit, does not
        #          touch this order's own statistical fingerprint) ----
        decoy_signals = []
        if params["decoy"] and self.rng.random() < 0.3 * self.aggressiveness:
            correlated = [s for s in SYMBOLS if s != order.symbol]
            decoy_sym = self.rng.choice(correlated)
            decoy_size = max(100, int(order.size * 0.05))
            decoy_dir = "SELL" if order.direction == "BUY" else "BUY"
            decoy_signals.append({
                "symbol": decoy_sym,
                "direction": decoy_dir,
                "size": decoy_size,
                "rationale": f"Hedging decoy in correlated asset {decoy_sym}; <=0.1% ADV; orthogonal direction",
            })
            modifications.append({
                "type": "DECOY_SIGNAL",
                "from": f"{order.symbol} {order.direction}",
                "to": f"{decoy_sym} {decoy_dir} x{decoy_size}",
                "rationale": f"Decoy trade in {decoy_sym} disrupts sector-level flow correlation",
            })
        action.decoy_signals = decoy_signals

        # ---- Price Guard (hard VWAP constraint) ----
        original_price = action.original["price"]
        price_dev_bps = abs(order.price - original_price) / original_price * 10_000 if original_price > 0 else 0

        if price_dev_bps > self.max_vwap_deviation_bps:
            order.price = original_price
            modifications.append({
                "type": "PRICE_GUARD",
                "from": f"{price_dev_bps:.1f}bps",
                "to": "0.0bps",
                "rationale": f"Deviation {price_dev_bps:.1f}bps exceeded {self.max_vwap_deviation_bps}bps hard limit",
            })
            price_dev_bps = 0.0

        # NOTE: there is deliberately no "score_after_estimate" here. A single
        # order's post-defense detectability isn't measurable in isolation —
        # detection is a property of a WINDOW of orders. The honest measurement
        # is: re-run AttackerAgent.analyze() on the accumulated defended stream
        # (see main.py), and for the real efficacy number, the trained
        # DefenseDiscriminator's AUC on held-out windows (see
        # scripts/train_and_evaluate.py). There is deliberately no per-order
        # "after" score.

        # ---- Generate LPA Token (BEFORE surfacing recommendation) ----
        latency_ns = time.perf_counter_ns() - start_ns

        lpa = self.compliance.generate_lpa(
            order_id=order.id,
            modifications=[m["type"] for m in modifications],
            vwap_deviation_bps=price_dev_bps,
            max_allowed_bps=self.max_vwap_deviation_bps,
            latency_ns=latency_ns,
            recent_orders=self._recent_orders,
        )

        order.defended = True

        action.modified = order.model_dump()
        action.modifications = modifications
        action.lpa = lpa
        action.latency_ns = latency_ns
        action.latency_ms = round(latency_ns / 1e6, 4)

        # Track history for sequence breaking
        self._recent_sizes.append(order.size)
        self._recent_timings.append(order.timing_ms)
        self._recent_venues.append(order.venue)
        self._recent_orders.append(order)
        if len(self._recent_sizes) > 50:
            self._recent_sizes = self._recent_sizes[-50:]
            self._recent_timings = self._recent_timings[-50:]
            self._recent_venues = self._recent_venues[-50:]
            self._recent_orders = self._recent_orders[-50:]

        self.actions.append(action)
        return action

    def set_aggressiveness(self, value: float):
        self.aggressiveness = float(np.clip(value, 0.0, 1.0))

    def set_intensity(self, level: int):
        """Set obfuscation intensity level (L0-L5)."""
        self.intensity_level = max(0, min(5, level))
