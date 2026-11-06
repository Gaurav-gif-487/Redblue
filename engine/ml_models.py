"""
engine/ml_models.py — the actual trained, evaluated ML in this project.

Two tasks:

1. StrategyClassifier — given a window of child orders, predict which
   execution algorithm produced them (TWAP / VWAP / ICEBERG). This is the
   business question ("does my slicing algorithm leak its own
   identity?") and it is not trivially solvable: within one window all
   orders come from ONE parent order, but the label is not encoded in the
   window's own history the way direction was — it must be inferred from
   the statistical shape (size CV, timing CV, autocorrelation) of the slicing
   pattern itself.

2. DefenseDiscriminator — given a window of child orders, predict whether it
   is the raw (undefended) stream or the RedBlue-obfuscated copy of that same
   stream. This is a direct measurement of "does the defense
   actually work": if a trained discriminator can still tell defended and
   undefended windows apart with much-better-than-chance accuracy, the
   defense has failed regardless of what the heuristic detector says.

Both models are evaluated with GroupKFold, grouped by parent_id, so that no
window from a given parent metaorder ever appears in both the train and test
fold. That is the leakage guard the old code was missing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.stats import skew
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from .order_flow import ChildOrder

MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)


def _window_features(orders: List[ChildOrder]) -> np.ndarray:
    """Shared feature extractor for a window of child orders.

    Deliberately excludes `direction` and any field that trivially encodes
    the label being predicted for either downstream task.
    """
    sizes = np.array([o.size for o in orders], dtype=float)
    timings = np.array([o.timing_ms for o in orders], dtype=float)
    venues = [o.venue for o in orders]
    timestamps = np.array([o.timestamp for o in orders], dtype=float)

    iat = np.diff(timestamps) * 1000 if len(orders) > 1 else np.array([500.0])

    def autocorr(a):
        if len(a) < 3:
            return 0.0
        m = np.mean(a)
        d = np.sum((a - m) ** 2)
        return float(np.sum((a[:-1] - m) * (a[1:] - m)) / d) if d > 0 else 0.0

    size_cv = float(np.std(sizes) / np.mean(sizes)) if np.mean(sizes) > 0 else 0.0
    timing_cv = float(np.std(timings) / np.mean(timings)) if np.mean(timings) > 0 else 0.0

    venue_counts: Dict[str, int] = {}
    for v in venues:
        venue_counts[v] = venue_counts.get(v, 0) + 1
    probs = np.array(list(venue_counts.values())) / len(venues)
    venue_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    dark_ratio = sum(1 for v in venues[-10:] if v in {"IEX", "EDGX"}) / min(10, len(venues))
    round_score = sum(1 for s in sizes[-10:] if s % 100 == 0 or s % 500 == 0) / min(10, len(sizes))

    size_skew = float(skew(sizes)) if len(sizes) > 3 and np.std(sizes) > 1e-9 else 0.0
    if not np.isfinite(size_skew):
        size_skew = 0.0

    feats = np.array([
        size_cv,
        timing_cv,
        venue_entropy,
        autocorr(sizes),
        autocorr(timings),
        autocorr(iat),
        float(np.std(iat) / (np.mean(iat) + 1e-12)),
        dark_ratio,
        round_score,
        float(len(set(venues))),
        float(np.mean(sizes[-5:]) / np.mean(sizes)) if len(sizes) >= 5 else 1.0,
        size_skew,
        float(len(sizes)),
        float(np.min(iat)),
        float(np.max(iat)),
    ])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


FEATURE_NAMES = [
    "size_cv", "timing_cv", "venue_entropy", "size_autocorr", "timing_autocorr",
    "iat_autocorr", "iat_cv", "dark_ratio", "round_lot_score", "unique_venues",
    "recent_size_ratio", "size_skew", "window_len", "iat_min", "iat_max",
]


@dataclass
class EvalReport:
    """Honest, held-out evaluation report. No number in here is estimated."""
    task: str
    n_samples: int
    n_groups: int
    n_folds: int
    fold_accuracy: List[float] = field(default_factory=list)
    fold_auc: List[Optional[float]] = field(default_factory=list)
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    mean_auc: Optional[float] = None
    confusion: Optional[List[List[int]]] = None
    classes: Optional[List[str]] = None
    baseline_majority_class_accuracy: float = 0.0
    warning: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "task": self.task,
            "n_samples": self.n_samples,
            "n_parent_orders": self.n_groups,
            "n_folds": self.n_folds,
            "fold_accuracy": [round(a, 4) for a in self.fold_accuracy],
            "fold_auc": [round(a, 4) if a is not None else None for a in self.fold_auc],
            "mean_accuracy": round(self.mean_accuracy, 4),
            "std_accuracy": round(self.std_accuracy, 4),
            "mean_auc": round(self.mean_auc, 4) if self.mean_auc is not None else None,
            "confusion_matrix": self.confusion,
            "classes": self.classes,
            "baseline_majority_class_accuracy": round(self.baseline_majority_class_accuracy, 4),
            "lift_over_baseline": round(self.mean_accuracy - self.baseline_majority_class_accuracy, 4),
            "warning": self.warning,
        }


class _GroupedClassifier:
    """Shared training/eval scaffolding for both models. Not used directly."""

    model_filename = "model.pkl"
    scaler_filename = "scaler.pkl"

    def __init__(self):
        self.model: Optional[GradientBoostingClassifier] = None
        self.scaler = StandardScaler()
        self.trained = False
        self.classes_: Optional[List[str]] = None
        self.last_eval: Optional[EvalReport] = None
        self._load_if_exists()

    def _model_path(self, name):
        return os.path.join(MODEL_DIR, name)

    def _load_if_exists(self):
        mp, sp = self._model_path(self.model_filename), self._model_path(self.scaler_filename)
        if os.path.exists(mp) and os.path.exists(sp):
            try:
                self.model = joblib.load(mp)
                self.scaler = joblib.load(sp)
                self.classes_ = list(self.model.classes_)
                self.trained = True
            except Exception:
                pass

    def _save(self):
        joblib.dump(self.model, self._model_path(self.model_filename))
        joblib.dump(self.scaler, self._model_path(self.scaler_filename))

    def _evaluate_grouped(self, X, y, groups, task_name: str, n_folds: int = 5) -> EvalReport:
        """GroupKFold evaluation: no parent_id ever appears in both train and test."""
        n_groups = len(set(groups))
        n_folds = max(2, min(n_folds, n_groups))
        gkf = GroupKFold(n_splits=n_folds)

        classes = sorted(set(y))
        binary = len(classes) == 2

        fold_acc, fold_auc = [], []
        all_true, all_pred = [], []

        for train_idx, test_idx in gkf.split(X, y, groups):
            train_groups = set(np.array(groups)[train_idx])
            test_groups = set(np.array(groups)[test_idx])
            assert train_groups.isdisjoint(test_groups), "leakage: a parent_id is in both train and test"

            scaler = StandardScaler().fit(X[train_idx])
            X_tr, X_te = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
            y_tr, y_te = np.array(y)[train_idx], np.array(y)[test_idx]

            clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)

            fold_acc.append(accuracy_score(y_te, preds))
            all_true.extend(y_te.tolist())
            all_pred.extend(preds.tolist())

            if binary:
                probs = clf.predict_proba(X_te)[:, 1]
                try:
                    fold_auc.append(roc_auc_score(y_te, probs))
                except ValueError:
                    fold_auc.append(None)
            else:
                fold_auc.append(None)

        valid_auc = [a for a in fold_auc if a is not None]
        vals, counts = np.unique(y, return_counts=True)
        baseline = float(np.max(counts) / len(y))

        report = EvalReport(
            task=task_name,
            n_samples=len(y),
            n_groups=n_groups,
            n_folds=n_folds,
            fold_accuracy=fold_acc,
            fold_auc=fold_auc,
            mean_accuracy=float(np.mean(fold_acc)),
            std_accuracy=float(np.std(fold_acc)),
            mean_auc=float(np.mean(valid_auc)) if valid_auc else None,
            confusion=confusion_matrix(all_true, all_pred, labels=classes).tolist(),
            classes=[str(c) for c in classes],
            baseline_majority_class_accuracy=baseline,
        )
        if n_groups < 20:
            report.warning = (
                f"Only {n_groups} distinct parent orders in the training set. "
                "Treat this evaluation as directional, not a reliable estimate — "
                "collect more parent orders before citing these numbers externally."
            )
        return report


class StrategyClassifier(_GroupedClassifier):
    """Predicts the slicing algorithm (TWAP/VWAP/ICEBERG) from a window of child orders."""

    model_filename = "strategy_model.pkl"
    scaler_filename = "strategy_scaler.pkl"

    def fit(self, windows: List[List[ChildOrder]], labels: List[str], groups: List[str]) -> EvalReport:
        X = np.array([_window_features(w) for w in windows])
        y = np.array(labels)

        report = self._evaluate_grouped(X, y, groups, task_name="strategy_classification")
        self.last_eval = report

        # Fit final model on ALL data for deployment (eval above is what you trust,
        # this refit is just so predict() has a model to use in production).
        self.scaler = StandardScaler().fit(X)
        self.model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        self.model.fit(self.scaler.transform(X), y)
        self.classes_ = list(self.model.classes_)
        self.trained = True
        self._save()
        return report

    def predict(self, orders: List[ChildOrder]) -> Dict:
        if not self.trained or len(orders) < 5:
            return {"predicted_strategy": "UNKNOWN", "confidence": 0.0, "trained": self.trained}
        X = self.scaler.transform(_window_features(orders).reshape(1, -1))
        probs = self.model.predict_proba(X)[0]
        idx = int(np.argmax(probs))
        return {
            "predicted_strategy": self.classes_[idx],
            "confidence": round(float(probs[idx]), 3),
            "class_probabilities": {c: round(float(p), 3) for c, p in zip(self.classes_, probs)},
            "trained": True,
        }


class DefenseDiscriminator(_GroupedClassifier):
    """Predicts whether a window of child orders is defended or undefended.

    This is the real efficacy metric for the Defender: train it on paired
    (raw, defended-copy-of-the-same-stream) windows and see if it can still
    tell them apart. AUC near 0.5 = the obfuscation is working. AUC near 1.0
    = the defense is cosmetic and a sophisticated observer would still see
    through it.
    """

    model_filename = "discriminator_model.pkl"
    scaler_filename = "discriminator_scaler.pkl"

    def fit(self, windows: List[List[ChildOrder]], is_defended: List[bool], groups: List[str]) -> EvalReport:
        X = np.array([_window_features(w) for w in windows])
        y = np.array([1 if d else 0 for d in is_defended])

        report = self._evaluate_grouped(X, y, groups, task_name="defense_discrimination")
        self.last_eval = report

        self.scaler = StandardScaler().fit(X)
        self.model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        self.model.fit(self.scaler.transform(X), y)
        self.classes_ = list(self.model.classes_)
        self.trained = True
        self._save()
        return report

    def predict(self, orders: List[ChildOrder]) -> Dict:
        if not self.trained or len(orders) < 5:
            return {"p_defended": 0.5, "trained": self.trained}
        X = self.scaler.transform(_window_features(orders).reshape(1, -1))
        prob = float(self.model.predict_proba(X)[0, 1])
        return {"p_defended": round(prob, 3), "trained": True}
