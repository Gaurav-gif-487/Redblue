"""
Compliance Engine — LPA (Legal Pre-Approval) token generation and audit trail.
The LPA token is generated before any recommendation is surfaced to the EMS.
Legal boundaries are hard-coded constraints, not policy documents.

Enhanced with 8 constraint checks (C001-C012 subset) and enriched LPA token fields.
"""
import uuid
import time
from typing import List, Dict, Optional
from datetime import datetime, timezone


# Rationale categories 
RATIONALE_CATEGORIES = [
    "BEST_EXECUTION",
    "LIQUIDITY_SEEKING",
    "ADVERSE_SELECTION_AVOIDANCE",
    "HEDGING",
    "RISK_REDUCTION",
]

# Modification type to rationale mapping
MOD_RATIONALE_MAP = {
    "SIZE_BLEND": ("Size blended toward target size distribution to reduce pattern regularity", "BEST_EXECUTION"),
    "TIMING_BLEND": ("Timing blended toward ambient inter-arrival distribution to break scheduling periodicity", "ADVERSE_SELECTION_AVOIDANCE"),
    "VENUE_ROUTE": ("Stochastic venue routing toward uniform distribution", "BEST_EXECUTION"),
    "DECOY_SIGNAL": ("Hedging decoy disrupts directional flow signal", "HEDGING"),
    "PRICE_GUARD": ("Price guard enforces VWAP deviation hard limit", "BEST_EXECUTION"),
    "MONITORING_ONLY": ("Monitoring mode: no modifications applied", "BEST_EXECUTION"),
}


class LPAToken:
    """Legal Pre-Approval token — generated before every Defender recommendation.
    Enhanced with economic_rationale, rationale_category, and constraint_checks fields.
    """

    def __init__(
        self,
        order_id: str,
        modifications: List[str],
        vwap_deviation_bps: float,
        max_allowed_bps: float,
        latency_ns: int,
        recent_orders: Optional[List] = None,
    ):
        self.token = f"LPA-{uuid.uuid4().hex[:8].upper()}"
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.order_id = order_id
        self.modifications = modifications
        self.vwap_deviation_bps = round(vwap_deviation_bps, 4)
        self.max_allowed_bps = max_allowed_bps
        self.latency_ms = round(latency_ns / 1e6, 4)
        self.schema_version = "1.3.0"

        # Generate economic rationale (machine-readable plain English)
        self.economic_rationale = self._generate_rationale(modifications)
        self.rationale_category = self._classify_rationale(modifications)

        # Run all constraint checks
        self.constraint_checks = self._run_constraint_checks(
            modifications, vwap_deviation_bps, max_allowed_bps, recent_orders or []
        )
        # A check that was never evaluated does not count as failed OR passed
        # for the overall `compliant` flag — but it is surfaced explicitly so
        # nobody mistakes "not evaluated" for "verified".
        evaluated = [c for c in self.constraint_checks if c["passed"] is not None]
        self.compliant = all(c["passed"] for c in evaluated)
        self.unevaluated_checks = [c["constraint_id"] for c in self.constraint_checks if c["passed"] is None]

        # LPA score over only the checks that were actually evaluated
        passed_count = sum(1 for c in evaluated if c["passed"])
        self.lpa_score = round(passed_count / max(len(evaluated), 1), 4)

        # Hard-coded compliance checks 
        self._validate_no_spoof(modifications)

    def _generate_rationale(self, modifications: List[str]) -> str:
        """Generate machine-readable plain-English rationale."""
        if not modifications:
            return "No modifications recommended; original order preserved."

        rationales = []
        for mod in modifications:
            rationale_text, _ = MOD_RATIONALE_MAP.get(mod, (f"Applied {mod} transformation", "BEST_EXECUTION"))
            rationales.append(rationale_text)

        return "; ".join(rationales) + "."

    def _classify_rationale(self, modifications: List[str]) -> str:
        """Classify the overall rationale category."""
        categories = []
        for mod in modifications:
            _, cat = MOD_RATIONALE_MAP.get(mod, ("", "BEST_EXECUTION"))
            categories.append(cat)

        if "HEDGING" in categories:
            return "HEDGING"
        elif "ADVERSE_SELECTION_AVOIDANCE" in categories:
            return "ADVERSE_SELECTION_AVOIDANCE"
        elif "RISK_REDUCTION" in categories:
            return "RISK_REDUCTION"
        else:
            return "BEST_EXECUTION"

    def _run_constraint_checks(self, modifications: List[str], vwap_dev: float, max_bps: float,
                                recent_orders: List) -> List[Dict]:
        """Run constraint checks.

        Some of these ARE genuinely computed from `recent_orders` (the
        defender's rolling order history). Others require data this
        simulation doesn't model at all (cancellations, a real NBBO feed,
        wall-clock session boundaries) — those are marked `passed: None`
        with `evaluated: False` rather than defaulting to `True`.
        """
        checks = []

        # C001: No same-day round-trip (opposite-direction order for the
        # same symbol within this session's recent order history).
        if recent_orders:
            symbols_seen = {}
            round_trip = False
            for o in recent_orders:
                prev_dir = symbols_seen.get(o.symbol)
                if prev_dir and prev_dir != o.direction:
                    round_trip = True
                symbols_seen[o.symbol] = o.direction
            checks.append({
                "constraint_id": "C001", "name": "No same-day round-trip",
                "passed": not round_trip, "evaluated": True,
                "observed_value": "round-trip detected" if round_trip else "0 round-trips",
                "limit": "0 allowed",
            })
        else:
            checks.append({
                "constraint_id": "C001", "name": "No same-day round-trip",
                "passed": None, "evaluated": False,
                "observed_value": "n/a — no order history in context",
                "limit": "0 allowed",
            })

        # C002: Cancellation rate <=15% — NOT MODELED. This simulation never
        # generates or tracks order cancellations at all, so there is no
        # honest way to compute an observed cancellation rate. Reporting 0%
        # here (as the old code did) would imply the system verified there
        # were no cancellations, when really nothing was checked.
        checks.append({
            "constraint_id": "C002", "name": "Cancellation rate <=15%",
            "passed": None, "evaluated": False,
            "observed_value": "not modeled — this system does not generate cancel events",
            "limit": "15%",
        })

        # C004: Minimum lot size. This one IS a genuine structural invariant:
        # DefenderAgent.defend() clamps size with max(1, round(...)) before
        # any order reaches here, so size >= 1 is guaranteed by code, not
        # assumed.
        checks.append({
            "constraint_id": "C004", "name": "Minimum lot size",
            "passed": True, "evaluated": True,
            "observed_value": ">=1 share (enforced by max(1, ...) clamp in DefenderAgent.defend)",
            "limit": "1 share",
        })

        # C005: No simultaneous opposing orders in the same instrument within
        # 1 second — computed for real from recent_orders' timestamps.
        if recent_orders:
            conflict = False
            for i, o in enumerate(recent_orders[-20:]):
                for o2 in recent_orders[-20:][i + 1:]:
                    if (o.symbol == o2.symbol and o.direction != o2.direction
                            and abs(o.timestamp - o2.timestamp) < 1.0):
                        conflict = True
            checks.append({
                "constraint_id": "C005", "name": "No simultaneous buy-sell",
                "passed": not conflict, "evaluated": True,
                "observed_value": "conflict detected" if conflict else "no conflict in last 20 orders",
                "limit": "No same-instrument opposing orders within 1s",
            })
        else:
            checks.append({
                "constraint_id": "C005", "name": "No simultaneous buy-sell",
                "passed": None, "evaluated": False,
                "observed_value": "n/a — no order history in context",
                "limit": "No same-instrument opposing orders within 1s",
            })

        # C007: Venue best execution — NOT MODELED. There is no real NBBO
        # feed in this system; venues are a fixed list with no live quote
        # comparison, so "NBBO compliant" can't honestly be asserted.
        checks.append({
            "constraint_id": "C007", "name": "Venue best execution",
            "passed": None, "evaluated": False,
            "observed_value": "not modeled — no live NBBO feed in this system",
            "limit": "<=3bps from NBBO",
        })

        # C008: LPA score >= 0.95 is checked by the caller after this method
        # returns (self.lpa_score, computed over evaluated checks only) — not
        # duplicated here to avoid a self-referential check.

        # C009: Timing within session — NOT MODELED. Orders here use a
        # simulation clock (time.time() at generation), not a real exchange
        # session calendar, so "within 16:00 ET" can't be verified against
        # anything real.
        checks.append({
            "constraint_id": "C009", "name": "Timing within session",
            "passed": None, "evaluated": False,
            "observed_value": "not modeled — no real session calendar in this system",
            "limit": "Before session close - 60s",
        })

        # C011: VWAP deviation <= max_bps — computed from order history.
        checks.append({
            "constraint_id": "C011", "name": "VWAP deviation limit",
            "passed": vwap_dev <= max_bps, "evaluated": True,
            "observed_value": f"{vwap_dev:.2f}bps",
            "limit": f"<={max_bps}bps",
        })

        return checks

    def _validate_no_spoof(self, modifications: List[str]):
        """Hard constraint: system cannot generate spoofing patterns."""
        forbidden = {"LAYERING", "SPOOFING", "WASH_TRADE", "FRONT_RUN"}
        for mod in modifications:
            if mod.upper() in forbidden:
                self.compliant = False
                raise ValueError(f"COMPLIANCE VIOLATION: Forbidden modification type '{mod}' detected. "
                                 f"System is architecturally constrained from generating this pattern.")

    def to_dict(self) -> Dict:
        return {
            "token": self.token,
            "timestamp": self.timestamp,
            "order_id": self.order_id,
            "modifications": self.modifications,
            "vwap_deviation": f"{self.vwap_deviation_bps:.2f}bps",
            "compliant": self.compliant,
            "unevaluated_checks": self.unevaluated_checks,
            "latency_ms": self.latency_ms,
            "lpa_score": self.lpa_score,
            "economic_rationale": self.economic_rationale,
            "rationale_category": self.rationale_category,
            "constraint_checks": self.constraint_checks,
            "schema_version": self.schema_version,
        }


class ComplianceEngine:
    """Manages LPA generation and maintains the compliance audit trail."""

    def __init__(self):
        self.audit_log: List[LPAToken] = []

    def generate_lpa(
        self,
        order_id: str,
        modifications: List[str],
        vwap_deviation_bps: float,
        max_allowed_bps: float,
        latency_ns: int,
        recent_orders: Optional[List] = None,
    ) -> LPAToken:
        """Generate an LPA token. This MUST be called before any recommendation is surfaced."""
        lpa = LPAToken(
            order_id=order_id,
            modifications=modifications,
            vwap_deviation_bps=vwap_deviation_bps,
            max_allowed_bps=max_allowed_bps,
            latency_ns=latency_ns,
            recent_orders=recent_orders,
        )
        self.audit_log.append(lpa)
        return lpa

    @property
    def total_lpas(self) -> int:
        return len(self.audit_log)

    @property
    def compliance_rate(self) -> float:
        if not self.audit_log:
            return 100.0
        compliant_count = sum(1 for lpa in self.audit_log if lpa.compliant)
        return (compliant_count / len(self.audit_log)) * 100

    def get_audit_log(self, limit: int = 50) -> List[Dict]:
        return [lpa.to_dict() for lpa in reversed(self.audit_log[-limit:])]
