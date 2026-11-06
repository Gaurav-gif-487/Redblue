"""
Order Flow Generator — Simulates realistic institutional order slicing.
Generates (timestamp, size, venue, direction) tuples matching real-world patterns.
"""
import time
import uuid
import numpy as np
from typing import Optional, List, Dict
from pydantic import BaseModel

VENUES = ["NYSE", "NASDAQ", "BATS", "IEX", "EDGX", "ARCA", "CHX", "MEMX"]
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM"]
DIRECTIONS = ["BUY", "SELL"]

VENUE_WEIGHTS_CONCENTRATED = np.array([0.35, 0.30, 0.15, 0.08, 0.05, 0.03, 0.02, 0.02])
VENUE_WEIGHTS_UNIFORM = np.ones(8) / 8


class ChildOrder(BaseModel):
    id: str
    timestamp: float
    symbol: str
    direction: str
    size: float
    price: float
    venue: str
    timing_ms: float
    parent_id: str
    parent_order_size: Optional[float] = None  # declared size of the parent metaorder, if known
    defended: bool = False


class OrderFlowGenerator:
    """Generates child orders from a parent order using realistic slicing algorithms."""

    def __init__(
        self,
        parent_order_size: Optional[int] = None,
        symbol: Optional[str] = None,
        direction: Optional[str] = None,
        base_price: Optional[float] = None,
        slice_pattern: str = "TWAP",
    ):
        self.rng = np.random.default_rng()
        self.parent_order_size = parent_order_size or int(self.rng.integers(50_000, 500_000))
        self.symbol = symbol or self.rng.choice(SYMBOLS)
        self.direction = direction or self.rng.choice(DIRECTIONS)
        self.base_price = base_price or float(self.rng.uniform(100, 500))
        self.slice_pattern = slice_pattern
        self.parent_id = str(uuid.uuid4())[:12]
        self.filled = 0
        self.child_count = 0
        self.child_orders: List[ChildOrder] = []

        # Per-metaorder timing cadence, sampled per parent order.
        # Real execution algorithms vary their cadence across desks, venues and
        # urgency; a fixed cadence would let the DefenseDiscriminator win on
        # timing alone (see docs/experiments.md).
        self._timing_mean_ms = float(self.rng.uniform(250.0, 900.0))
        self._timing_std_frac = float(self.rng.uniform(0.03, 0.10))

    def generate_child_order(self, defended: bool = False) -> Optional[ChildOrder]:
        """Generate the next child order. Returns None if parent is fully filled."""
        if self.filled >= self.parent_order_size:
            return None

        remaining = self.parent_order_size - self.filled

        if not defended:
            size, venue, timing_ms, price_offset = self._undefended_params(remaining)
        else:
            size, venue, timing_ms, price_offset = self._undefended_params(remaining)

        size = max(1, min(size, remaining))
        self.filled += size
        self.child_count += 1

        order = ChildOrder(
            id=str(uuid.uuid4())[:10],
            timestamp=time.time(),
            symbol=self.symbol,
            direction=self.direction,
            size=size,
            price=round(self.base_price + price_offset, 2),
            venue=venue,
            timing_ms=timing_ms,
            parent_id=self.parent_id,
            parent_order_size=float(self.parent_order_size),
            defended=defended,
        )
        self.child_orders.append(order)
        return order

    def _undefended_params(self, remaining: int):
        """Generate predictable (detectable) order parameters."""
        if self.slice_pattern == "TWAP":
            target_slices = 20
            base_size = self.parent_order_size // target_slices
            size = base_size + int(self.rng.normal(0, base_size * 0.05))
        elif self.slice_pattern == "VWAP":
            volume_curve = np.sin((self.child_count / 20) * np.pi) * 0.5 + 0.5
            size = int((self.parent_order_size / 15) * volume_curve) + int(self.rng.normal(0, 30))
        else:  # ICEBERG
            size = self.parent_order_size // 25

        size = max(100, min(size, remaining))

        # Concentrated venue selection (predictable)
        venue_idx = self.rng.choice(len(VENUES), p=VENUE_WEIGHTS_CONCENTRATED)
        venue = VENUES[venue_idx]

        # Regular timing, cadence fixed per metaorder, low variance within it
        timing_ms = self._timing_mean_ms + float(self.rng.normal(0, self._timing_mean_ms * self._timing_std_frac))
        timing_ms = max(20.0, timing_ms)

        # Small price impact
        price_offset = float(self.rng.normal(0, 0.02))

        return size, venue, timing_ms, price_offset

    @property
    def progress(self) -> float:
        return self.filled / self.parent_order_size if self.parent_order_size > 0 else 0.0

    def reset(self):
        self.filled = 0
        self.child_count = 0
        self.child_orders = []
        self.parent_id = str(uuid.uuid4())[:12]
