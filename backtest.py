import pandas as pd
import numpy as np
from tqdm import tqdm # You might need to pip install tqdm
import matplotlib.pyplot as plt
from pipeline import PSXDataPipeline
from scanner import StructuralScanner

# Configuration
BACKTEST_YEARS = 2
HOLDING_PERIODS = [5, 10, 20] # Check returns after 5, 10, and 20 days
MIN_LIQUIDITY = 10_000_000

class Backtester:
    def __init__(self):
        self.pipeline = PSXDataPipeline()
        self.results = []

    def run(self):
        # 1. Fetch Data
        print("📥 Fetching historical data...")
        # Force a fresh update to ensure we have full history
        data_cache = self.pipeline.update_universe()
        
        print(f"🔄 Starting Backtest on {len(data_cache)} symbols...")
        print("⏳ This simulates every single day for the past 2 years. It may take a few minutes.")

        # 2. Iterate through every stock
        for symbol, df in tqdm(data_cache.items(), desc="Analyzing Universe"):
            if len(df) < 250: continue # Skip young stocks

            # We start from index 200 to ensure enough data for moving averages
            # We end 20 days before the last date so we can calculate the outcome
            start_index = len(df) - (250 * BACKTEST_YEARS)
            if start_index < 200: start_index = 200
            
            end_index = len(df) - 20 

            # 3. Time Travel Loop
            for i in range(start_index, end_index):
                # Slice the dataframe to simulate "Today is day i"
                # We interpret df.iloc[i] as "Today's Close"
                
                # OPTIMIZATION: 
                # Calculating zones is heavy. We only run the full scanner 
                # if the price is effectively moving (Volatility check) 
                # to speed up the loop.
                
                # Create the 'past' view
                df_past = df.iloc[:i+1].copy()
                
                # Run the Scanner
                scanner = StructuralScanner(df_past, min_liquidity_pkr=MIN_LIQUIDITY)
                
                # We catch the candidates
                candidates = scanner.evaluate_breakout()
                
                if candidates:
                    for signal in candidates:
                        # 4. Calculate The Outcome (The "Peek" into the future)
                        entry_price = signal['level'] # Assuming we bought the breakout level
                        # Or use closing price: entry_price = df.iloc[i]['close']
                        
                        future_close_5 = df.iloc[i+5]['close']
                        future_close_10 = df.iloc[i+10]['close']
                        future_close_20 = df.iloc[i+20]['close']
                        
                        res_5d = (future_close_5 - entry_price) / entry_price
                        res_10d = (future_close_10 - entry_price) / entry_price
                        res_20d = (future_close_20 - entry_price) / entry_price

                        self.results.append({
                            'Date': df.index[i].date(),
                            'Symbol': symbol,
                            'Signal_Score': signal['compression_score'],
                            'Vol_Expansion': signal['vol_expansion'],
                            'Return_5D': res_5d,
                            'Return_10D': res_10d,
                            'Return_20D': res_20d
                        })

    def analyze(self):
        if not self.results:
            print("❌ No signals found in the backtest period.")
            return

        df_res = pd.DataFrame(self.results)
        
        print("\n" + "="*40)
        print("📊 BACKTEST RESULTS (Last 2 Years)")
        print("="*40)
        print(f"Total Signals: {len(df_res)}")
        
        # Win Rate (Positive return after 10 days)
        win_rate = len(df_res[df_res['Return_10D'] > 0]) / len(df_res) * 100
        avg_return = df_res['Return_10D'].mean() * 100
        
        print(f"🎯 Win Rate (10-Day): {win_rate:.1f}%")
        print(f"📈 Avg Return (10-Day): {avg_return:.2f}%")
        
        # Best Performers
        print("\n🏆 Top Performing Signals:")
        print(df_res.sort_values(by='Return_10D', ascending=False)[['Date', 'Symbol', 'Return_10D']].head(5))

        # Worst Performers
        print("\n💀 Worst Failures:")
        print(df_res.sort_values(by='Return_10D', ascending=True)[['Date', 'Symbol', 'Return_10D']].head(5))
        
        # Export
        df_res.to_csv("backtest_results.csv", index=False)
        print("\n💾 Detailed logs saved to 'backtest_results.csv'")

if __name__ == "__main__":
    tester = Backtester()
    tester.run()
    tester.analyze()