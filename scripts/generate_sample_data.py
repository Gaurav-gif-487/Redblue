# scripts/generate_sample_data.py
# Generates realistic institutional equity order flow for the demo
import numpy as np
import pandas as pd
import time
import os

def generate_equity_data():
    rng = np.random.default_rng(42)
    orders = []
    
    # symbol, base_price, approx_daily_volume
    symbols = [
        ('AAPL', 195.0, 54e6),
        ('MSFT', 430.0, 22e6),
        ('NVDA', 118.0, 310e6)
    ]

    print("[DataGen] Generating institutional equity orders...")

    for sym, price, adv in symbols:
        # Simulate 5-10 TWAP metaorders per symbol
        for metaorder in range(rng.integers(5, 10)):
            direction = rng.choice(['BUY', 'SELL'])
            # 0.05% to 0.2% of ADV (typical institutional slice)
            parent_size = int(rng.uniform(0.0005, 0.002) * adv)
            n_slices = rng.integers(20, 40)
            
            # Start time: sometime in the last 2 hours
            base_ts = (time.time() * 1000) - rng.uniform(0, 7200000)
            
            # Typical venues
            venues_conc = ['NYSE', 'NYSE', 'NASDAQ', 'NASDAQ', 'BATS', 'IEX', 'ARCA', 'EDGX']
            
            for i in range(n_slices):
                # TWAP with slight size noise
                child_size = int(parent_size / n_slices + rng.normal(0, parent_size / n_slices * 0.1))
                child_size = max(100, (child_size // 100) * 100) # Round to lot size
                
                # ~500ms to 2s interval with low jitter (classic TWAP)
                ts = base_ts + i * (rng.uniform(500, 2000))
                
                venue = rng.choice(venues_conc)
                
                # Price with slight micro-volatility
                p = price * (1 + rng.normal(0, 0.0005))
                
                orders.append({
                    'timestamp': int(ts),
                    'symbol': sym,
                    'side': direction,
                    'size': child_size,
                    'price': round(p, 2),
                    'venue': venue
                })

    df = pd.DataFrame(orders).sort_values('timestamp').reset_index(drop=True)
    
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    
    df.to_csv('data/sample_orders.csv', index=False)
    print(f"[DataGen] Success: Generated {len(df)} orders saved to data/sample_orders.csv")

if __name__ == "__main__":
    generate_equity_data()
