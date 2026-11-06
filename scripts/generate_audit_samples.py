# scripts/generate_audit_samples.py
import pandas as pd
import numpy as np
import time
import os

def create_samples():
    os.makedirs('data/audit_samples', exist_ok=True)
    
    # 1. Vulnerable Trades (Highly Predictable TWAP)
    vulnerable = []
    base_ts = int(time.time() * 1000)
    for i in range(100):
        vulnerable.append({
            'timestamp': base_ts + (i * 1000), # Exactly 1 second apart
            'symbol': 'AAPL',
            'side': 'BUY',
            'size': 500, # Constant size
            'price': 190.0 + np.random.normal(0, 0.05),
            'venue': 'NASDAQ' # Single venue
        })
    pd.DataFrame(vulnerable).to_csv('data/audit_samples/vulnerable_trades.csv', index=False)
    
    # 2. Protected Trades (Randomized/Obfuscated)
    protected = []
    for i in range(100):
        # Randomized timing
        ts_step = np.random.uniform(500, 2500)
        base_ts += ts_step
        
        # Randomized size (mean 500)
        size = int(np.random.normal(500, 150))
        size = max(100, (size // 10) * 10)
        
        protected.append({
            'timestamp': int(base_ts),
            'symbol': 'AAPL',
            'side': 'BUY',
            'size': size,
            'price': 190.0 + np.random.normal(0, 0.05),
            'venue': np.random.choice(['NYSE', 'NASDAQ', 'BATS', 'IEX'])
        })
    pd.DataFrame(protected).to_csv('data/audit_samples/protected_trades.csv', index=False)
    
    print("[AuditGen] Created sample files in data/audit_samples/")
    print(" - vulnerable_trades.csv (Should yield HIGH detection score)")
    print(" - protected_trades.csv (Should yield LOW detection score)")

if __name__ == "__main__":
    create_samples()
