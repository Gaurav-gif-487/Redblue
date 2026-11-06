"""
Metrics Engine — Computes all benchmark metrics comparing
undefended vs defended order flow performance.
"""
import numpy as np
from typing import List, Dict, Any
from .order_flow import ChildOrder


class MetricsEngine:
    """Computes and tracks all performance metrics."""

    def __init__(self):
        self.snapshots: List[Dict] = []

    def compute(
        self,
        undefended_orders: List[ChildOrder],
        defended_orders: List[ChildOrder],
        attacker_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute a full metrics snapshot comparing the two streams."""

        u_sizes = np.array([o.size for o in undefended_orders], dtype=np.float64) if undefended_orders else np.array([])
        d_sizes = np.array([o.size for o in defended_orders], dtype=np.float64) if defended_orders else np.array([])
        u_timings = np.array([o.timing_ms for o in undefended_orders], dtype=np.float64) if undefended_orders else np.array([])
        d_timings = np.array([o.timing_ms for o in defended_orders], dtype=np.float64) if defended_orders else np.array([])
        u_venues = [o.venue for o in undefended_orders]
        d_venues = [o.venue for o in defended_orders]

        u_attack = attacker_results.get("undefended", {})
        d_attack = attacker_results.get("defended", {})

        # VWAP
        u_vwap = self._calc_vwap(undefended_orders)
        d_vwap = self._calc_vwap(defended_orders)
        vwap_dev_bps = abs(d_vwap - u_vwap) / u_vwap * 10_000 if u_vwap > 0 else 0.0

        # Detection
        det_before = u_attack.get("detection_score", 0)
        det_after = d_attack.get("detection_score", 0)
        det_reduction = ((det_before - det_after) / det_before * 100) if det_before > 0 else 0

        snapshot = {
            "detection": {
                "before": round(det_before, 2),
                "after": round(det_after, 2),
                "reduction": round(det_reduction, 2),
            },
            "entropy": {
                "size_before": self._shannon_entropy(u_sizes, bucket_width=100),
                "size_after": self._shannon_entropy(d_sizes, bucket_width=100),
                "venue_before": self._cat_entropy(u_venues),
                "venue_after": self._cat_entropy(d_venues),
                "timing_before": self._shannon_entropy(u_timings, bucket_width=100),
                "timing_after": self._shannon_entropy(d_timings, bucket_width=100),
            },
            "autocorrelation": {
                "size_before": round(abs(self._autocorrelation(u_sizes)), 4),
                "size_after": round(abs(self._autocorrelation(d_sizes)), 4),
                "timing_before": round(abs(self._autocorrelation(u_timings)), 4),
                "timing_after": round(abs(self._autocorrelation(d_timings)), 4),
            },
            "vwap": {
                "undefended": round(u_vwap, 4),
                "defended": round(d_vwap, 4),
                "deviation_bps": round(vwap_dev_bps, 4),
            },
            "volume": {
                "undefended": int(np.sum(u_sizes)) if len(u_sizes) else 0,
                "defended": int(np.sum(d_sizes)) if len(d_sizes) else 0,
            },
            "order_count": {
                "undefended": len(undefended_orders),
                "defended": len(defended_orders),
            },
            "patterns_detected": {
                "before": len(u_attack.get("patterns", [])),
                "after": len(d_attack.get("patterns", [])),
            },
        }

        self.snapshots.append(snapshot)
        return snapshot

    def _calc_vwap(self, orders: List[ChildOrder]) -> float:
        if not orders:
            return 0.0
        total_pv = sum(o.price * o.size for o in orders)
        total_v = sum(o.size for o in orders)
        return total_pv / total_v if total_v > 0 else 0.0

    def _autocorrelation(self, arr: np.ndarray, lag: int = 1) -> float:
        if len(arr) < lag + 2:
            return 0.0
        mean = np.mean(arr)
        denom = np.sum((arr - mean) ** 2)
        if denom == 0:
            return 0.0
        num = np.sum((arr[:-lag] - mean) * (arr[lag:] - mean))
        return float(num / denom)

    def _shannon_entropy(self, arr: np.ndarray, bucket_width: float = 100) -> float:
        if len(arr) == 0:
            return 0.0
        buckets = np.floor(arr / bucket_width).astype(int)
        _, counts = np.unique(buckets, return_counts=True)
        probs = counts / len(arr)
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        max_ent = np.log2(len(counts)) if len(counts) > 1 else 1.0
        return round(float(entropy / max_ent) if max_ent > 0 else 0.0, 4)

    def _cat_entropy(self, arr: list) -> float:
        if not arr:
            return 0.0
        _, counts = np.unique(arr, return_counts=True)
        probs = counts / len(arr)
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        max_ent = np.log2(len(counts)) if len(counts) > 1 else 1.0
        return round(float(entropy / max_ent) if max_ent > 0 else 0.0, 4)

    def generate_benchmark_report(self, defender_actions: list) -> Dict[str, Any]:
        """Generate a full benchmark report against target metrics."""
        latencies = [a.latency_ms for a in defender_actions] if defender_actions else [0]
        lat_arr = np.array(latencies)

        vwap_devs = [s["vwap"]["deviation_bps"] for s in self.snapshots] if self.snapshots else [0]
        det_before = [s["detection"]["before"] for s in self.snapshots] if self.snapshots else [0]
        det_after = [s["detection"]["after"] for s in self.snapshots] if self.snapshots else [0]

        return {
            "orders_processed": len(defender_actions),
            "latency": {
                "mean_ms": round(float(np.mean(lat_arr)), 4),
                "p50_ms": round(float(np.percentile(lat_arr, 50)), 4),
                "p95_ms": round(float(np.percentile(lat_arr, 95)), 4),
                "p99_ms": round(float(np.percentile(lat_arr, 99)), 4),
                "max_ms": round(float(np.max(lat_arr)), 4),
            },
            "vwap_deviation": {
                "mean_bps": round(float(np.mean(vwap_devs)), 4),
                "max_bps": round(float(np.max(vwap_devs)), 4),
                "within_tolerance_pct": round(
                    float(np.sum(np.array(vwap_devs) <= 8) / len(vwap_devs) * 100), 2
                ),
            },
            "detection": {
                "before_mean": round(float(np.mean(det_before)), 2),
                "after_mean": round(float(np.mean(det_after)), 2),
                "reduction_pct": round(
                    float((np.mean(det_before) - np.mean(det_after)) / np.mean(det_before) * 100)
                    if np.mean(det_before) > 0 else 0, 2
                ),
            },
            "compliance": {
                "total_lpas": len(defender_actions),
                "all_compliant": all(a.lpa.compliant for a in defender_actions) if defender_actions else True,
                "compliance_rate_pct": round(
                    sum(1 for a in defender_actions if a.lpa.compliant) / len(defender_actions) * 100
                    if defender_actions else 100, 2
                ),
            },
        }
