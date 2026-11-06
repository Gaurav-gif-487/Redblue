"""
Attacker Agent — heuristic, hand-weighted statistical detector.

This module is deliberately NOT a trained model. It computes real statistics
(autocorrelation, Shannon entropy, Wald-Wolfowitz runs test, skew/kurtosis)
on an order window and combines them with a fixed linear weighting into a
0-100 "detection score". It exists to give a cheap, instant, explainable
signal on every tick.

The actual trained, evaluated ML lives in engine/ml_models.py
(StrategyClassifier, DefenseDiscriminator) and is validated with grouped
cross-validation in scripts/train_and_evaluate.py. Do not read the numbers
from this module as evidence of predictive accuracy — they are a heuristic
prior, useful for real-time UX, not a benchmark result.
"""
import time
import numpy as np
from scipy import stats as sp_stats
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any, Optional
from .order_flow import ChildOrder


# Strategy fingerprint signatures
STRATEGY_SIGNATURES = {
    "TWAP": {"size_cv_range": (0, 0.15), "timing_cv_range": (0, 0.12), "label": "Time-Weighted Avg Price"},
    "VWAP": {"size_cv_range": (0.1, 0.4), "timing_cv_range": (0.05, 0.2), "label": "Volume-Weighted Avg Price"},
    "ICEBERG": {"size_cv_range": (0, 0.08), "timing_cv_range": (0.15, 0.5), "label": "Iceberg / Hidden"},
    "MOMENTUM": {"size_cv_range": (0.3, 0.8), "timing_cv_range": (0.3, 0.8), "label": "Momentum Execution"},
    "IS": {"size_cv_range": (0.2, 0.6), "timing_cv_range": (0.1, 0.4), "label": "Implementation Shortfall"},
}


class AttackerAgent:
    """Adversarial ML agent that detects patterns in institutional order flow."""

    def __init__(self):
        self.analysis_history: List[Dict] = []
        self.isolation_forest = IsolationForest(
            n_estimators=50, contamination=0.1, random_state=42
        )
        self._if_fitted = False

    def analyze(self, orders: List[ChildOrder]) -> Dict[str, Any]:
        """Run full pattern detection analysis on an order stream."""
        start_ns = time.perf_counter_ns()

        if len(orders) < 5:
            return {
                "detection_score": 0.0,
                "confidence": 0.0,
                "patterns": [],
                "features": {},
                "latency_ns": 0,
                "order_count": len(orders),
                "strategy_classification": "UNKNOWN",
                "risk_level": "LOW",
                "top_features": [],
            }

        sizes = np.array([o.size for o in orders], dtype=np.float64)
        timings = np.array([o.timing_ms for o in orders], dtype=np.float64)
        prices = np.array([o.price for o in orders], dtype=np.float64)
        venues = [o.venue for o in orders]
        directions = [o.direction for o in orders]
        timestamps = np.array([o.timestamp for o in orders], dtype=np.float64)

        # ---- Feature Extraction  ----
        features = {}

        # --- Temporal Features (F001-F012) ---
        # F001: Inter-Arrival Time
        if len(timestamps) >= 2:
            iats = np.diff(timestamps) * 1000  # Convert to ms
            features["iat_mean"] = float(np.mean(iats))
            features["iat_std"] = float(np.std(iats))
        else:
            iats = np.array([500.0])
            features["iat_mean"] = 500.0
            features["iat_std"] = 0.0

        # F002: IAT Z-Score (normalized deviation)
        if len(iats) >= 20:
            rolling_mean = np.mean(iats[-20:])
            rolling_std = np.std(iats[-20:])
            features["iat_zscore"] = float((iats[-1] - rolling_mean) / (rolling_std + 1e-12))
        else:
            features["iat_zscore"] = 0.0

        # F007: IAT Autocorrelation-5
        features["iat_autocorr_5"] = abs(self._autocorrelation(iats, 1)) if len(iats) >= 3 else 0.0

        # F008: IAT Trend (slope of last 10 IATs)
        if len(iats) >= 10:
            x = np.arange(min(10, len(iats)))
            slope, _, _, _, _ = sp_stats.linregress(x, iats[-len(x):])
            features["iat_trend"] = float(slope)
        else:
            features["iat_trend"] = 0.0

        # F009: Order Burst Flag
        if len(timestamps) >= 3:
            recent_iats = np.diff(timestamps[-4:]) * 1000
            features["burst_flag"] = 1.0 if np.any(recent_iats < 500) and len(recent_iats) >= 3 else 0.0
        else:
            features["burst_flag"] = 0.0

        # F011: Hour Entropy (order timing across 30-min buckets)
        if len(timings) >= 5:
            timing_buckets = (timings / 1800000).astype(int) % 13  # 13 half-hour buckets in trading day
            features["hour_entropy"] = self._shannon_entropy(timing_buckets.tolist())
        else:
            features["hour_entropy"] = 0.0

        # --- Size & Volume Features (F013-F023) ---
        # Autocorrelation (lag-1 and lag-2)
        features["size_autocorr_1"] = abs(self._autocorrelation(sizes, 1))
        features["size_autocorr_2"] = abs(self._autocorrelation(sizes, 2))
        features["timing_autocorr_1"] = abs(self._autocorrelation(timings, 1))
        features["price_autocorr_1"] = abs(self._autocorrelation(prices, 1))

        # F015: Size Z-Score
        if len(sizes) >= 20:
            rolling_mean = np.mean(sizes[-20:])
            rolling_std = np.std(sizes[-20:])
            features["size_zscore"] = float((sizes[-1] - rolling_mean) / (rolling_std + 1e-12))
        else:
            features["size_zscore"] = float((sizes[-1] - np.mean(sizes)) / (np.std(sizes) + 1e-12)) if len(sizes) > 1 else 0.0

        # F018: Completion Ratio — filled volume vs. the parent order's declared size.
        # Completion ratio uses the generator's parent_order_size when available
        # on the orders; otherwise it is "unknown" (0.0).
        total_volume = float(np.sum(sizes))
        parent_size = getattr(orders[-1], "parent_order_size", None)
        if parent_size and parent_size > 0:
            features["completion_ratio"] = float(min(1.0, total_volume / parent_size))
        else:
            features["completion_ratio"] = 0.0

        # F019: Size Autocorrelation-5
        if len(sizes) >= 6:
            features["size_autocorr_5"] = abs(self._autocorrelation(sizes[-6:], 1))
        else:
            features["size_autocorr_5"] = features["size_autocorr_1"]

        # F020: Size Rounding Score
        round_lots = sum(1 for s in sizes[-10:] if s % 100 == 0 or s % 500 == 0)
        features["size_rounding_score"] = round_lots / min(10, len(sizes))

        # F021: Volume Participation Deviation
        mean_size = np.mean(sizes)
        features["volume_participation_dev"] = float(np.std(sizes) / (mean_size + 1e-12))

        # Coefficient of variation
        features["size_cv"] = float(np.std(sizes) / np.mean(sizes)) if np.mean(sizes) > 0 else 0
        features["timing_cv"] = float(np.std(timings) / np.mean(timings)) if np.mean(timings) > 0 else 0

        # --- Venue & Routing Features (F024-F032) ---
        # Entropy
        features["venue_entropy"] = self._shannon_entropy(venues)
        features["direction_entropy"] = self._shannon_entropy(directions)

        # F026: Dark Pool Ratio
        dark_pools = {"IEX", "EDGX"}  # Simplified dark pool classification
        dark_count = sum(1 for v in venues[-20:] if v in dark_pools)
        features["dark_pool_ratio"] = dark_count / min(20, len(venues))

        # F027: Lit/Dark Switch
        if len(venues) >= 2:
            switches = sum(1 for i in range(1, len(venues))
                          if (venues[i] in dark_pools) != (venues[i-1] in dark_pools))
            features["lit_dark_switch_rate"] = switches / (len(venues) - 1)
        else:
            features["lit_dark_switch_rate"] = 0.0

        # F028: Venue Autocorrelation (venue stickiness)
        if len(venues) >= 2:
            same_venue = sum(1 for i in range(1, len(venues)) if venues[i] == venues[i-1])
            features["venue_autocorr"] = same_venue / (len(venues) - 1)
        else:
            features["venue_autocorr"] = 0.0

        # --- Statistical Tests ---
        # Runs test for randomness (direction)
        features["direction_runs_z"] = abs(self._runs_test_z(directions))

        # Kurtosis and skewness of sizes
        features["size_kurtosis"] = float(sp_stats.kurtosis(sizes)) if len(sizes) > 3 else 0
        features["size_skewness"] = float(sp_stats.skew(sizes)) if len(sizes) > 3 else 0

        # Isolation Forest anomaly score (on size/timing feature matrix)
        features["anomaly_score"] = self._anomaly_score(sizes, timings)

        # ---- Pattern Detection ----
        patterns = []

        if features["size_cv"] < 0.3:
            patterns.append({
                "name": "SIZE_REGULARITY",
                "severity": "HIGH",
                "score": round((1 - features["size_cv"]) * 100, 1),
                "detail": f"Order sizes show low variation (CV={features['size_cv']:.4f}). Consistent with algorithmic slicing.",
            })

        if features["timing_cv"] < 0.25:
            patterns.append({
                "name": "TIMING_PERIODICITY",
                "severity": "HIGH",
                "score": round((1 - features["timing_cv"]) * 100, 1),
                "detail": f"Inter-order timing is highly periodic (CV={features['timing_cv']:.4f}). Likely TWAP/scheduled execution.",
            })

        if features["venue_entropy"] < 0.6:
            unique_venues = len(set(venues))
            patterns.append({
                "name": "VENUE_CONCENTRATION",
                "severity": "MEDIUM",
                "score": round((1 - features["venue_entropy"]) * 100, 1),
                "detail": f"Flow concentrated on {unique_venues} venues (H={features['venue_entropy']:.4f}). Predictable routing.",
            })

        if features["size_autocorr_1"] > 0.3:
            patterns.append({
                "name": "SIZE_AUTOCORRELATION",
                "severity": "HIGH",
                "score": round(features["size_autocorr_1"] * 100, 1),
                "detail": f"Sequential sizes correlated (ρ₁={features['size_autocorr_1']:.4f}). Pattern is learnable by adversary.",
            })

        if features["direction_runs_z"] > 1.96:
            patterns.append({
                "name": "DIRECTION_PERSISTENCE",
                "severity": "MEDIUM",
                "score": round(min(features["direction_runs_z"] * 25, 100), 1),
                "detail": f"Directional clustering detected (Z={features['direction_runs_z']:.2f}). Non-random buy/sell sequencing.",
            })

        if features["timing_autocorr_1"] > 0.25:
            patterns.append({
                "name": "TIMING_AUTOCORRELATION",
                "severity": "MEDIUM",
                "score": round(features["timing_autocorr_1"] * 100, 1),
                "detail": f"Timing intervals correlated (ρ₁={features['timing_autocorr_1']:.4f}). Execution cadence is predictable.",
            })

        if features["venue_autocorr"] > 0.5:
            patterns.append({
                "name": "VENUE_STICKINESS",
                "severity": "MEDIUM",
                "score": round(features["venue_autocorr"] * 100, 1),
                "detail": f"Venue stickiness P(same)={features['venue_autocorr']:.3f}. Routing pattern is predictable.",
            })

        if features["size_rounding_score"] > 0.5:
            patterns.append({
                "name": "ROUND_LOT_PATTERN",
                "severity": "LOW",
                "score": round(features["size_rounding_score"] * 100, 1),
                "detail": f"{features['size_rounding_score']*100:.0f}% round lots detected. Indicates algorithmic origin.",
            })

        if features.get("burst_flag", 0) > 0:
            patterns.append({
                "name": "ORDER_BURST",
                "severity": "HIGH",
                "score": 85.0,
                "detail": "Burst detected: ≥3 orders within 500ms. Aggressive algorithmic execution.",
            })

        # ---- Composite Detection Score ----
        weights = {
            "size_autocorr_1": 0.14,
            "timing_autocorr_1": 0.10,
            "venue_entropy": 0.12,
            "size_cv": 0.14,
            "timing_cv": 0.10,
            "direction_entropy": 0.04,
            "price_autocorr_1": 0.03,
            "size_autocorr_2": 0.04,
            "anomaly_score": 0.08,
            "venue_autocorr": 0.06,
            "size_rounding_score": 0.05,
            "dark_pool_ratio": 0.03,
            "iat_autocorr_5": 0.04,
            "burst_flag": 0.03,
        }

        detection_score = 0.0
        feature_contributions = {}
        for feat, weight in weights.items():
            val = features.get(feat, 0.0)
            if feat in ("venue_entropy", "direction_entropy"):
                contrib = weight * (1 - val) * 100
            elif feat in ("size_cv", "timing_cv"):
                contrib = weight * max(0, (1 - val * 2)) * 100
            elif feat == "dark_pool_ratio":
                contrib = weight * (1 - val) * 100  # Low dark pool usage = predictable
            else:
                contrib = weight * val * 100
            detection_score += contrib
            feature_contributions[feat] = round(contrib, 2)

        detection_score = float(np.clip(detection_score, 0, 100))
        confidence = min(len(orders) / 30, 1.0) * 100

        # ---- Strategy Classification ----
        strategy_class = self._classify_strategy(features)

        # ---- Risk Level ----
        if detection_score > 62:
            risk_level = "HIGH"
        elif detection_score > 50:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # ---- Top-3 Feature Attribution ----
        sorted_contribs = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
        top_features = [{"feature": f, "contribution": c} for f, c in sorted_contribs[:3]]

        latency_ns = time.perf_counter_ns() - start_ns

        result = {
            "detection_score": round(detection_score, 2),
            "confidence": round(confidence, 1),
            "patterns": patterns,
            "features": {k: round(v, 6) for k, v in features.items()},
            "latency_ns": latency_ns,
            "latency_ms": round(latency_ns / 1e6, 4),
            "order_count": len(orders),
            "strategy_classification": strategy_class,
            "risk_level": risk_level,
            "top_features": top_features,
            "feature_contributions": feature_contributions,
        }

        self.analysis_history.append(result)
        return result

    def _classify_strategy(self, features: Dict) -> str:
        """Classify the detected execution strategy (multi-task output)."""
        size_cv = features.get("size_cv", 0.5)
        timing_cv = features.get("timing_cv", 0.5)

        best_match = "UNKNOWN"
        best_score = float('inf')

        for strategy, sig in STRATEGY_SIGNATURES.items():
            size_dist = max(0, sig["size_cv_range"][0] - size_cv) + max(0, size_cv - sig["size_cv_range"][1])
            time_dist = max(0, sig["timing_cv_range"][0] - timing_cv) + max(0, timing_cv - sig["timing_cv_range"][1])
            total_dist = size_dist + time_dist
            if total_dist < best_score:
                best_score = total_dist
                best_match = strategy

        return best_match

    def _autocorrelation(self, arr: np.ndarray, lag: int = 1) -> float:
        """Compute autocorrelation at given lag."""
        if len(arr) < lag + 2:
            return 0.0
        mean = np.mean(arr)
        denom = np.sum((arr - mean) ** 2)
        if denom == 0:
            return 0.0
        num = np.sum((arr[:-lag] - mean) * (arr[lag:] - mean))
        return float(num / denom)

    def _shannon_entropy(self, arr: list) -> float:
        """Normalized Shannon entropy ∈ [0, 1]."""
        if not arr:
            return 0.0
        if isinstance(arr[0], (float, np.floating)):
            arr = [int(x) for x in arr]
        _, counts = np.unique(arr, return_counts=True)
        probs = counts / len(arr)
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def _runs_test_z(self, arr: list) -> float:
        """Wald-Wolfowitz runs test Z-score for randomness."""
        if len(arr) < 10:
            return 0.0
        n = len(arr)
        unique_vals = list(set(arr))
        if len(unique_vals) != 2:
            return 0.0

        n1 = arr.count(unique_vals[0])
        n2 = arr.count(unique_vals[1])

        runs = 1
        for i in range(1, n):
            if arr[i] != arr[i - 1]:
                runs += 1

        expected_runs = (2 * n1 * n2) / n + 1
        var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n ** 2 * (n - 1))

        if var_runs <= 0:
            return 0.0
        z = (runs - expected_runs) / np.sqrt(var_runs)
        return float(abs(z))

    def _anomaly_score(self, sizes: np.ndarray, timings: np.ndarray) -> float:
        """Isolation Forest anomaly detection — returns regularity score ∈ [0, 1]."""
        if len(sizes) < 10:
            return 0.0
        X = np.column_stack([sizes, timings])
        try:
            self.isolation_forest.fit(X)
            scores = self.isolation_forest.decision_function(X)
            # Higher scores = more "normal" (more regular/predictable)
            # We want to return high when data is very regular (easy to detect)
            regularity = float(np.clip(np.mean(scores) + 0.5, 0, 1))
            return regularity
        except Exception:
            return 0.0
