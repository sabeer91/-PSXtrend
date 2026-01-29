import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

class StructuralScanner:
    def __init__(self, df, min_liquidity_pkr=10_000_000):
        self.df = df
        self.min_liquidity = min_liquidity_pkr

    def _calculate_metrics(self):
        """Computes Technicals: ATR, Squeeze, Volume SMA."""
        # True Range
        self.df['tr'] = np.maximum(
            self.df['high'] - self.df['low'],
            np.maximum(
                abs(self.df['high'] - self.df['close'].shift(1)),
                abs(self.df['low'] - self.df['close'].shift(1))
            )
        )
        # ATRs
        self.df['atr_14'] = self.df['tr'].rolling(14).mean()
        self.df['atr_5'] = self.df['tr'].rolling(5).mean()
        self.df['atr_20'] = self.df['tr'].rolling(20).mean()
        
        # Volatility Compression Ratio (VCR)
        # < 1.0 means volatility is shrinking relative to monthly average
        self.df['compression_ratio'] = self.df['atr_5'] / self.df['atr_20']
        
        # Volume SMA
        self.df['vol_sma_20'] = self.df['volume'].rolling(20).mean()

    def _find_structural_zones(self, lookback=250, tolerance=0.02):
        """
        Identifies horizontal zones with multiple touches using Fractal Highs.
        """
        # Slice recent history
        data = self.df.iloc[-lookback:].copy()
        
        # Find local maxima (peaks)
        # order=5 means it's the highest point 5 days before and 5 days after
        high_idx = argrelextrema(data['high'].values, np.greater_equal, order=5)[0]
        highs = data.iloc[high_idx]['high']
        
        zones = []
        
        # Cluster the highs
        for price in highs:
            matched = False
            for i, zone in enumerate(zones):
                # Check if price is within X% of an existing zone
                if abs(zone['level'] - price) / zone['level'] <= tolerance:
                    zones[i]['touches'] += 1
                    # Keep the higher level to avoid false breakouts
                    zones[i]['level'] = max(zones[i]['level'], price) 
                    matched = True
                    break
            
            if not matched:
                zones.append({'level': price, 'touches': 1})

        # Filter: Only strong zones (3+ touches)
        valid_zones = [z for z in zones if z['touches'] >= 3]
        return sorted(valid_zones, key=lambda x: x['level'])

    def evaluate_breakout(self):
        """
        Main logic checks: Liquidity -> Structure -> Breakout -> Volume -> Context
        """
        self._calculate_metrics()
        
        if len(self.df) < 25: return None
        
        today = self.df.iloc[-1]
        yesterday = self.df.iloc[-2]
        
        # 1. Liquidity Check (Turnover > 10M PKR)
        if (today['close'] * today['volume']) < self.min_liquidity:
            return None

        zones = self._find_structural_zones()
        breakout_candidates = []

        for zone in zones:
            level = zone['level']
            
            # --- The "Commitment" Logic ---
            
            # A. Price Logic: Crossed and Closed above
            # Ensure it wasn't already above yesterday (avoid repeat alerts)
            if yesterday['close'] < level and today['close'] > level:
                
                # B. Extension Filter (Must not be a tiny wick break)
                # Breakout distance >= 0.75 ATR
                dist_mult = (today['close'] - level) / today['atr_14']
                if dist_mult < 0.75: continue
                
                # C. Volume Expansion (Institutional Footprint)
                vol_mult = today['volume'] / today['vol_sma_20']
                if vol_mult < 1.8: continue
                
                # D. Compression Context
                # We want the breakout to come from a "squeeze", not an already extended run
                # Check compression of the day BEFORE breakout
                prev_compression = yesterday['compression_ratio']
                if prev_compression > 1.0: continue # Rejected: Volatility was already high

                breakout_candidates.append({
                    'level': level,
                    'touches': zone['touches'],
                    'vol_expansion': round(vol_mult, 2),
                    'atr_extension': round(dist_mult, 2),
                    'compression_score': round(1 - prev_compression, 2) # Higher is tighter
                })
        
        return breakout_candidates