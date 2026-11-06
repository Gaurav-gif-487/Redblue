import pandas as pd
import sys
import os
import uuid

# Add parent dir to path so we can import engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.attacker import AttackerAgent
from engine.defender import DefenderAgent
from engine.order_flow import ChildOrder

def run_backtest(csv_path="data/sample_orders.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}.")
        print("Please run 'python scripts/download_sample_data.py' first.")
        return

    print("==================================================")
    print(" REDBLUE OPEN-SOURCE DATA BACKTESTING ENGINE")
    print("==================================================")
    
    # 1. Load Data
    print(f"\n[1] Loading real high-frequency tick data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    raw_orders = []
    # Binance timestamps are typically ms epoch
    start_time = df['timestamp'].min()
    
    for idx, row in df.iterrows():
        # normalize timestamp to seconds offset for the engine
        t_sec = (row['timestamp'] - start_time) / 1000.0
        
        # Create ChildOrder object
        order = ChildOrder(
            id=str(uuid.uuid4()),
            timestamp=t_sec,
            symbol=row['symbol'],
            direction=row['side'],
            size=float(row['size']),
            price=float(row['price']),
            venue=row['venue'],
            timing_ms=500.0, # Default estimate
            parent_id="backtest_parent"
        )
        raw_orders.append(order)
        
    print(f"    Loaded {len(raw_orders)} raw orders.")

    # 2. Instantiate Engines
    print("\n[2] Initializing Adversarial AI (Attacker) and Defender Engine...")
    attacker = AttackerAgent()
    defender = DefenderAgent() 
    defender.set_intensity(3) # Use Level 3 (Timing Jitter + Fragmentation)

    # 3. Baseline Scan
    print("\n[3] Running Baseline Vulnerability Scan on raw open-source data...")
    baseline_analysis = attacker.analyze(raw_orders)
    baseline_score = baseline_analysis.get("detection_score", 0)
    print(f"    Baseline Detection Score: {baseline_score:.2f}%")
    print(f"    Detected Strategy: {baseline_analysis.get('strategy_classification')}")
    top_feats = [f["feature"] if isinstance(f, dict) else str(f) for f in baseline_analysis.get('top_features', [])]
    print(f"    Top Vulnerabilities: {', '.join(top_feats)}")
    
    # 4. Run Defender
    print("\n[4] Running RedBlue Defender (Intensity L3) over historical data...")
    defended_orders = []
    lpa_tokens = 0
    total_vwap_dev = 0.0
    
    # Simulate streaming
    for order in raw_orders:
        # defend() modifies the order in-place and returns an action
        action = defender.defend(order, attacker_analysis=baseline_analysis)
        defended_orders.append(order)
        if action.lpa:
            lpa_tokens += 1
            total_vwap_dev += action.lpa.vwap_deviation_bps

    avg_vwap_dev = total_vwap_dev / lpa_tokens if lpa_tokens > 0 else 0
    print(f"    Generated {lpa_tokens} Legal Pre-Approval (LPA) tokens.")
    print(f"    Average VWAP Deviation: {avg_vwap_dev:.2f} bps (Limit: <8.0 bps)")

    # 5. Defended Scan
    print("\n[5] Running Adversarial Scan on Defended output...")
    defended_analysis = attacker.analyze(defended_orders)
    defended_score = defended_analysis.get("detection_score", 0)
    print(f"    Defended Detection Score: {defended_score:.2f}%")

    # 6. Conclusion
    print("\n==================================================")
    print(" BACKTEST RESULTS SUMMARY")
    print("==================================================")
    print(f" Reduction in Predictability: {max(0, baseline_score - defended_score):.2f}%")
    print(f" Safe Harbor Compliance:      {'PASS' if avg_vwap_dev <= 8.0 else 'FAIL'}")
    print("==================================================\n")


if __name__ == "__main__":
    run_backtest()
