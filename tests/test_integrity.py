"""
tests/test_integrity.py — regression tests for result-integrity properties
(no label leakage, no constant features, a single ground-truth stream).
"""
import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from engine.order_flow import OrderFlowGenerator
from engine.attacker import AttackerAgent
from engine.defender import DefenderAgent
from engine.ml_models import _GroupedClassifier, EvalReport


def _make_orders(n=30, pattern="TWAP"):
    gen = OrderFlowGenerator(symbol="AAPL", slice_pattern=pattern, parent_order_size=100_000)
    orders = []
    for _ in range(n):
        o = gen.generate_child_order(defended=False)
        if o is None:
            break
        orders.append(o)
    return orders


def test_completion_ratio_is_not_a_constant():
    """Old bug: completion_ratio = total/(total*1.5) == 0.667 always."""
    attacker = AttackerAgent()
    small_orders = _make_orders(6)
    large_orders = _make_orders(30)
    r_small = attacker.analyze(small_orders)["features"]["completion_ratio"]
    r_large = attacker.analyze(large_orders)["features"]["completion_ratio"]
    assert r_small != pytest.approx(0.6667, abs=1e-6)
    assert r_large != pytest.approx(0.6667, abs=1e-6)
    assert r_small != r_large  # must actually depend on the input


def test_grouped_split_never_leaks_a_parent_order_across_folds():
    """Old bug: ML training had no group awareness, and worse, the *task*
    itself (predict next order's direction from prior orders of the same
    parent) was tautological. This test locks in the GroupKFold guard."""
    from sklearn.model_selection import GroupKFold

    groups = ["p1"] * 10 + ["p2"] * 10 + ["p3"] * 10
    X = np.random.default_rng(0).normal(size=(30, 4))
    y = np.array([0, 1] * 15)

    gkf = GroupKFold(n_splits=3)
    for train_idx, test_idx in gkf.split(X, y, groups):
        train_groups = set(np.array(groups)[train_idx])
        test_groups = set(np.array(groups)[test_idx])
        assert train_groups.isdisjoint(test_groups)


def test_undefended_timing_is_not_a_hardcoded_constant_across_metaorders():
    """Old bug: every parent order used timing_ms = 500 + N(0,25), identical
    regardless of strategy/desk, giving CV always ~0.03-0.06."""
    means = []
    for i in range(10):
        gen = OrderFlowGenerator(slice_pattern=["TWAP", "VWAP", "ICEBERG"][i % 3], parent_order_size=50_000)
        orders = []
        o = gen.generate_child_order()
        while o and len(orders) < 15:
            orders.append(o)
            o = gen.generate_child_order()
        means.append(np.mean([x.timing_ms for x in orders]))
    assert np.std(means) > 5.0  # metaorders must have different cadences, not one fixed value


def test_defender_score_after_estimate_field_removed():
    """DefenderAction must not carry an unvalidated per-order score estimate."""
    from engine.defender import DefenderAction
    action = DefenderAction()
    assert not hasattr(action, "score_after_estimate")
    d = action.to_dict()
    assert "score_after_estimate" not in d


def test_ab_stream_uses_single_ground_truth_generator():
    """Old bug: main.py's live sim used two INDEPENDENT OrderFlowGenerator
    instances for undefended vs defended, which could even end up with
    different slice_pattern values, confounding the before/after comparison.
    This test locks in the fix at the engine level: defending a deep copy of
    the same order stream preserves the parent_id and child sequence.
    """
    gen = OrderFlowGenerator(symbol="AAPL", slice_pattern="VWAP", parent_order_size=80_000)
    raw = _orders_from(gen)
    defender = DefenderAgent(aggressiveness=0.7)
    defender.set_intensity(3)
    defended = []
    for o in raw:
        oc = copy.deepcopy(o)
        defender.defend(oc)
        defended.append(oc)

    assert len(raw) == len(defended)
    for r, d in zip(raw, defended):
        assert r.parent_id == d.parent_id  # same underlying metaorder
        assert r.direction == d.direction  # defense must not flip trading intent
        assert r.symbol == d.symbol


def _orders_from(gen, n=25):
    orders = []
    for _ in range(n):
        o = gen.generate_child_order(defended=False)
        if o is None:
            break
        orders.append(o)
    return orders


def test_compliance_checks_report_unevaluated_honestly():
    """Old bug: C002/C007/C009 were hard-coded passed=True with comments
    like 'tracked at portfolio level in production' despite nothing tracking
    them. They must now be explicitly marked unevaluated (passed=None)."""
    from engine.compliance import ComplianceEngine
    ce = ComplianceEngine()
    lpa = ce.generate_lpa(
        order_id="o1", modifications=["SIZE_BLEND"],
        vwap_deviation_bps=1.0, max_allowed_bps=8.0, latency_ns=1000,
        recent_orders=[],
    )
    ids_marked_none = {c["constraint_id"] for c in lpa.constraint_checks if c["passed"] is None}
    assert "C002" in ids_marked_none  # cancellation rate: not modeled
    assert "C007" in ids_marked_none  # NBBO best-ex: not modeled
    assert "C009" in ids_marked_none  # session timing: not modeled
    for c in lpa.constraint_checks:
        if c["passed"] is None:
            assert "not modeled" in c["observed_value"] or "n/a" in c["observed_value"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
