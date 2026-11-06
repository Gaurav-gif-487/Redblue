import requests
import pandas as pd
import time
import os

def download_binance_trades(symbol="BTCUSDT", limit=1000):
    print(f"Downloading {limit} real recent trades for {symbol} from Binance...")
    url = f"https://api.binance.com/api/v3/trades?symbol={symbol}&limit={limit}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Convert to standard format: timestamp, symbol, side, size, price, venue
        orders = []
        for row in data:
            orders.append({
                "timestamp": row["time"],
                "symbol": symbol,
                "side": "SELL" if row["isBuyerMaker"] else "BUY",
                "size": float(row["qty"]),
                "price": float(row["price"]),
                "venue": "BINANCE"
            })
            
        df = pd.DataFrame(orders)
        
        os.makedirs("data", exist_ok=True)
        out_path = os.path.join("data", "sample_orders.csv")
        df.to_csv(out_path, index=False)
        print(f"Successfully saved {len(df)} real trades to {out_path}.")
        print("You can now run 'python scripts/backtest.py' to test the RedBlue models against this open-source data.")
        
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_binance_trades()
