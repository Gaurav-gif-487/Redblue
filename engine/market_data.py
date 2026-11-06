"""
Market Data Engine — Real market data via free Yahoo Finance API.
Provides live stock prices, volume profiles, volatility, and HMM-based market regime detection.
No API keys required. All data cached with TTL to respect rate limits.
"""
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

# Cache for market data (populated lazily)
_market_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 300  # 5 minutes

TRACKED_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM"]

# Realistic fallback data (used if yfinance unavailable on free Render)
_FALLBACK_DATA = {
    "AAPL": {"price": 198.36, "adv": 54_200_000, "volatility": 0.24, "beta": 1.21, "sector": "Technology"},
    "MSFT": {"price": 430.12, "adv": 22_100_000, "volatility": 0.22, "beta": 0.95, "sector": "Technology"},
    "GOOGL": {"price": 174.85, "adv": 26_800_000, "volatility": 0.28, "beta": 1.10, "sector": "Technology"},
    "AMZN": {"price": 193.40, "adv": 47_500_000, "volatility": 0.30, "beta": 1.18, "sector": "Consumer Cyclical"},
    "NVDA": {"price": 117.93, "adv": 310_000_000, "volatility": 0.52, "beta": 1.65, "sector": "Technology"},
    "META": {"price": 562.78, "adv": 18_700_000, "volatility": 0.35, "beta": 1.28, "sector": "Technology"},
    "TSLA": {"price": 284.65, "adv": 98_000_000, "volatility": 0.58, "beta": 1.95, "sector": "Consumer Cyclical"},
    "JPM":  {"price": 252.14, "adv": 9_800_000, "volatility": 0.18, "beta": 1.05, "sector": "Financial Services"},
}

# Correlation matrix for cross-asset signals
_CORRELATION_PAIRS = {
    "AAPL": ["MSFT", "GOOGL", "META"],
    "MSFT": ["AAPL", "GOOGL", "AMZN"],
    "GOOGL": ["META", "MSFT", "AAPL"],
    "AMZN": ["MSFT", "GOOGL", "META"],
    "NVDA": ["AAPL", "MSFT", "GOOGL"],
    "META": ["GOOGL", "AAPL", "MSFT"],
    "TSLA": ["NVDA", "AMZN", "AAPL"],
    "JPM":  ["AAPL", "MSFT", "AMZN"],
}


class MarketRegimeDetector:
    """HMM-based market regime detection.
    States: trending_up, trending_down, mean_reverting, high_vol, illiquid
    """

    REGIMES = ["trending_up", "trending_down", "mean_reverting", "high_vol", "illiquid"]

    def __init__(self):
        self._model = None
        self._fitted = False
        self._current_regime = "mean_reverting"
        self._regime_probs = [0.0, 0.0, 1.0, 0.0, 0.0]
        self.rng = np.random.default_rng()

    def fit_and_predict(self, returns: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """Detect current market regime from recent returns and volume data."""
        if len(returns) < 10:
            return self._default_regime()

        try:
            from hmmlearn.hmm import GaussianHMM

            # Feature matrix: [returns, abs_returns (vol proxy), volume_z]
            vol_z = (volumes - np.mean(volumes)) / (np.std(volumes) + 1e-12)
            X = np.column_stack([returns, np.abs(returns), vol_z])

            model = GaussianHMM(n_components=min(4, len(returns) // 5), covariance_type="diag",
                                n_iter=50, random_state=42)
            model.fit(X)
            states = model.predict(X)
            current_state = int(states[-1])

            # Map HMM states to named regimes based on characteristics
            state_means = model.means_
            regime_map = self._map_states_to_regimes(state_means, model.covars_)
            self._current_regime = regime_map.get(current_state, "mean_reverting")
            self._fitted = True

            probs = model.predict_proba(X[-1:].reshape(1, -1))[0]
            self._regime_probs = probs.tolist()

        except Exception:
            # Fallback: simple rule-based regime detection
            self._current_regime = self._rule_based_regime(returns, volumes)

        return {
            "regime": self._current_regime,
            "confidence": max(self._regime_probs) if len(self._regime_probs) > 0 else 0.5,
            "all_regimes": {r: round(p, 3) for r, p in
                           zip(self.REGIMES[:len(self._regime_probs)], self._regime_probs)},
        }

    def _map_states_to_regimes(self, means, covars) -> Dict[int, str]:
        """Map HMM states to interpretable regime names."""
        mapping = {}
        for i, (mean, cov) in enumerate(zip(means, covars)):
            ret_mean = mean[0]
            vol_proxy = mean[1] if len(mean) > 1 else abs(ret_mean)

            if vol_proxy > 0.02:
                mapping[i] = "high_vol"
            elif ret_mean > 0.005:
                mapping[i] = "trending_up"
            elif ret_mean < -0.005:
                mapping[i] = "trending_down"
            else:
                mapping[i] = "mean_reverting"
        return mapping

    def _rule_based_regime(self, returns: np.ndarray, volumes: np.ndarray) -> str:
        """Simple fallback regime detection."""
        recent = returns[-20:] if len(returns) >= 20 else returns
        vol = np.std(recent)
        mean_ret = np.mean(recent)

        if vol > 0.03:
            return "high_vol"
        elif mean_ret > 0.005:
            return "trending_up"
        elif mean_ret < -0.005:
            return "trending_down"
        else:
            return "mean_reverting"

    def _default_regime(self) -> Dict[str, Any]:
        return {
            "regime": "mean_reverting",
            "confidence": 0.5,
            "all_regimes": {"mean_reverting": 1.0},
        }

    @property
    def current_regime(self) -> str:
        return self._current_regime


class MarketDataEngine:
    """
    Provides real market data for the simulation engine.
    Uses yfinance (free, no API key) with fallback to realistic static data.
    """

    def __init__(self):
        self._data: Dict[str, Dict] = {}
        self._last_fetch: float = 0
        self._regime_detector = MarketRegimeDetector()
        self._vix_level: float = 18.5
        self._hft_activity_score: float = 0.0
        self._adverse_selection_prob: float = 0.0
        self.rng = np.random.default_rng()
        self._initialized = False
        self._returns_history: Dict[str, List[float]] = {s: [] for s in TRACKED_SYMBOLS}
        self._volume_history: Dict[str, List[float]] = {s: [] for s in TRACKED_SYMBOLS}

    def initialize(self):
        """Load initial market data. Safe to call multiple times."""
        if self._initialized:
            return
        self._fetch_market_data()
        self._initialized = True

    def _fetch_market_data(self):
        """Fetch real data via yfinance, fall back to realistic static data."""
        now = time.time()
        if now - self._last_fetch < _CACHE_TTL and self._data:
            return

        try:
            import yfinance as yf
            tickers = yf.Tickers(" ".join(TRACKED_SYMBOLS))
            for sym in TRACKED_SYMBOLS:
                try:
                    ticker = tickers.tickers[sym]
                    info = ticker.fast_info
                    hist = ticker.history(period="5d", interval="1h")

                    if len(hist) > 0:
                        prices = hist["Close"].values
                        volumes = hist["Volume"].values
                        returns = np.diff(np.log(prices + 1e-12))

                        self._data[sym] = {
                            "price": float(prices[-1]),
                            "adv": int(np.mean(volumes) * 6.5),  # Scale hourly to daily
                            "volatility": float(np.std(returns) * np.sqrt(252 * 6.5)),
                            "beta": _FALLBACK_DATA.get(sym, {}).get("beta", 1.0),
                            "sector": _FALLBACK_DATA.get(sym, {}).get("sector", "Unknown"),
                            "prices_1h": prices[-24:].tolist() if len(prices) >= 24 else prices.tolist(),
                            "volumes_1h": volumes[-24:].tolist() if len(volumes) >= 24 else volumes.tolist(),
                            "returns": returns.tolist(),
                            "momentum_1h": float((prices[-1] / prices[-2] - 1)) if len(prices) >= 2 else 0,
                            "momentum_5h": float((prices[-1] / prices[-6] - 1)) if len(prices) >= 6 else 0,
                            "source": "yfinance_live",
                        }
                        self._returns_history[sym] = returns.tolist()
                        self._volume_history[sym] = volumes.tolist()
                except Exception:
                    self._data[sym] = self._generate_realistic_data(sym)

            # Try to fetch VIX
            try:
                vix = yf.Ticker("^VIX")
                vix_hist = vix.history(period="1d")
                if len(vix_hist) > 0:
                    self._vix_level = float(vix_hist["Close"].iloc[-1])
            except Exception:
                self._vix_level = 18.5 + float(self.rng.normal(0, 3))

            self._last_fetch = now
            print(f"[RedBlue] Market data fetched via yfinance for {len(self._data)} symbols")

        except ImportError:
            print("[RedBlue] yfinance not available, using realistic fallback data")
            for sym in TRACKED_SYMBOLS:
                self._data[sym] = self._generate_realistic_data(sym)
            self._last_fetch = now

        except Exception as e:
            print(f"[RedBlue] Market data fetch error: {e}, using fallback")
            for sym in TRACKED_SYMBOLS:
                if sym not in self._data:
                    self._data[sym] = self._generate_realistic_data(sym)
            self._last_fetch = now

        # Update market risk metrics
        self._update_market_risk()

    def _generate_realistic_data(self, symbol: str) -> Dict:
        """Generate realistic market data from fallback + small random perturbation."""
        fb = _FALLBACK_DATA.get(symbol, {"price": 150.0, "adv": 20_000_000, "volatility": 0.25, "beta": 1.0, "sector": "Unknown"})
        price = fb["price"] * (1 + float(self.rng.normal(0, 0.005)))
        vol = fb["volatility"] * (1 + float(self.rng.normal(0, 0.05)))

        # Generate synthetic intraday data
        n_points = 24
        returns = self.rng.normal(0, vol / np.sqrt(252 * 6.5), n_points)
        prices = [price]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        volumes = (self.rng.lognormal(np.log(fb["adv"] / 6.5), 0.5, n_points)).astype(int).tolist()

        self._returns_history[symbol] = returns.tolist()
        self._volume_history[symbol] = volumes

        return {
            "price": round(price, 2),
            "adv": fb["adv"],
            "volatility": round(vol, 4),
            "beta": fb["beta"],
            "sector": fb["sector"],
            "prices_1h": [round(p, 2) for p in prices[-24:]],
            "volumes_1h": volumes,
            "returns": returns.tolist(),
            "momentum_1h": float(returns[-1]) if len(returns) > 0 else 0,
            "momentum_5h": float(np.sum(returns[-5:])) if len(returns) >= 5 else 0,
            "source": "realistic_fallback",
        }

    def _update_market_risk(self):
        """Compute HFT activity score, adverse selection probability, and regime."""
        # HFT activity score (0-100) based on VIX and volume
        vix_component = min(self._vix_level / 40, 1.0) * 50
        vol_component = float(self.rng.uniform(20, 50))
        self._hft_activity_score = round(vix_component + vol_component * 0.5, 1)

        # Adverse selection probability
        self._adverse_selection_prob = round(0.15 + (self._vix_level / 100) * 0.3 + float(self.rng.normal(0, 0.05)), 3)
        self._adverse_selection_prob = max(0.05, min(0.65, self._adverse_selection_prob))

        # Run regime detection
        all_returns = []
        all_volumes = []
        for sym in TRACKED_SYMBOLS[:4]:
            if sym in self._returns_history:
                all_returns.extend(self._returns_history[sym])
            if sym in self._volume_history:
                all_volumes.extend(self._volume_history[sym])

        if all_returns and all_volumes:
            min_len = min(len(all_returns), len(all_volumes))
            self._regime_detector.fit_and_predict(
                np.array(all_returns[:min_len]),
                np.array(all_volumes[:min_len])
            )

    def get_price(self, symbol: str) -> float:
        """Get current price for symbol with small tick-level perturbation."""
        self._ensure_data()
        data = self._data.get(symbol, _FALLBACK_DATA.get(symbol, {}))
        base = data.get("price", 150.0)
        vol = data.get("volatility", 0.25)
        # Add realistic tick-level noise
        tick_noise = float(self.rng.normal(0, base * vol / np.sqrt(252 * 6.5 * 60)))
        return round(base + tick_noise, 2)

    def get_adv(self, symbol: str) -> int:
        """Average daily volume."""
        self._ensure_data()
        return self._data.get(symbol, {}).get("adv", _FALLBACK_DATA.get(symbol, {}).get("adv", 20_000_000))

    def get_volatility(self, symbol: str) -> float:
        """Annualized realized volatility."""
        self._ensure_data()
        return self._data.get(symbol, {}).get("volatility", 0.25)

    def get_symbol_data(self, symbol: str) -> Dict:
        """Full market data for a symbol."""
        self._ensure_data()
        return self._data.get(symbol, self._generate_realistic_data(symbol))

    def get_vix(self) -> float:
        return self._vix_level

    def get_vix_regime(self) -> str:
        """VIX bucketed into regimes."""
        if self._vix_level < 15:
            return "low"
        elif self._vix_level < 25:
            return "normal"
        elif self._vix_level < 35:
            return "elevated"
        else:
            return "crisis"

    def get_market_regime(self) -> str:
        """HMM market regime."""
        return self._regime_detector.current_regime

    def get_market_risk_snapshot(self) -> Dict:
        """Full market risk panel data (FR-D04)."""
        self._ensure_data()
        now = datetime.now(timezone.utc)
        hour = now.hour

        # Determine session phase
        if hour < 13 or hour >= 21:  # Roughly pre/post market UTC
            session_phase = "pre_market" if hour < 13 else "after_hours"
        elif hour < 14:
            session_phase = "open_auction"
        elif hour >= 20:
            session_phase = "close_auction"
        else:
            session_phase = "continuous"

        return {
            "vix": round(self._vix_level, 2),
            "vix_regime": self.get_vix_regime(),
            "market_regime": self.get_market_regime(),
            "hft_activity_score": self._hft_activity_score,
            "adverse_selection_prob": round(self._adverse_selection_prob, 3),
            "recommended_intensity": self._recommended_intensity(),
            "session_phase": session_phase,
            "tracked_symbols": {
                sym: {
                    "price": round(self._data.get(sym, {}).get("price", 0), 2),
                    "volatility": round(self._data.get(sym, {}).get("volatility", 0), 4),
                    "momentum_1h": round(self._data.get(sym, {}).get("momentum_1h", 0), 4),
                    "source": self._data.get(sym, {}).get("source", "unknown"),
                }
                for sym in TRACKED_SYMBOLS
            },
        }

    def get_correlated_assets(self, symbol: str) -> List[str]:
        """Get top correlated assets for a symbol."""
        return _CORRELATION_PAIRS.get(symbol, TRACKED_SYMBOLS[:3])

    def compute_leakage_cost(self, attacker_score: float, order_size: int,
                              symbol: str, direction: str) -> float:
        """Compute alpha leakage cost in dollars.
        Formula: (score - 0.5) × impact × notional
        """
        if attacker_score <= 50:
            return 0.0
        price = self.get_price(symbol)
        adv = self.get_adv(symbol)
        vol = self.get_volatility(symbol)
        notional = order_size * price

        # Simplified Almgren-Chriss market impact model
        participation = order_size / adv if adv > 0 else 0.001
        impact_bps = 0.1 * np.sqrt(participation) * vol * 10_000

        leakage = ((attacker_score / 100) - 0.5) * (impact_bps / 10_000) * notional
        return round(abs(leakage), 2)

    def _recommended_intensity(self) -> int:
        """Compute recommended obfuscation intensity (L0-L5)."""
        if self._vix_level > 35 or self._hft_activity_score > 80:
            return 5
        elif self._vix_level > 25 or self._hft_activity_score > 60:
            return 4
        elif self._hft_activity_score > 40:
            return 3
        elif self._hft_activity_score > 20:
            return 2
        else:
            return 1

    def _ensure_data(self):
        if not self._data:
            self._fetch_market_data()

    def tick_update(self):
        """Called each simulation tick to add micro-variation to prices."""
        for sym in TRACKED_SYMBOLS:
            if sym in self._data:
                vol = self._data[sym].get("volatility", 0.25)
                price = self._data[sym]["price"]
                ret = float(self.rng.normal(0, vol / np.sqrt(252 * 6.5 * 3600)))
                self._data[sym]["price"] = round(price * (1 + ret), 2)
                if sym in self._returns_history:
                    self._returns_history[sym].append(ret)
                    if len(self._returns_history[sym]) > 500:
                        self._returns_history[sym] = self._returns_history[sym][-200:]

        # Slowly drift VIX
        self._vix_level += float(self.rng.normal(0, 0.1))
        self._vix_level = max(10, min(50, self._vix_level))

        # Update HFT and adverse selection with small jitter
        self._hft_activity_score += float(self.rng.normal(0, 1))
        self._hft_activity_score = max(5, min(95, self._hft_activity_score))
        self._adverse_selection_prob += float(self.rng.normal(0, 0.005))
        self._adverse_selection_prob = max(0.05, min(0.65, self._adverse_selection_prob))
