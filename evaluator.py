import pandas as pd
import numpy as np

class AgenticEvaluator:
    def __init__(self, pipeline_instance):
        self.pipeline = pipeline_instance
        self.market_regime = self._assess_market_regime()

    def _assess_market_regime(self):
        """
        Determines global market health (Risk-On / Risk-Off).
        """
        df_index = self.pipeline.get_market_regime()
        
        # Default safe state if data fails
        if df_index is None or df_index.empty:
            return {'status': 'NEUTRAL', 'vol_mult': 1.0}

        # Calculate Index Technicals
        df_index['sma_200'] = df_index['close'].rolling(200).mean()
        
        # Simple RSI
        delta = df_index['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df_index['rsi'] = 100 - (100 / (1 + rs))

        current_close = df_index['close'].iloc[-1]
        current_sma = df_index['sma_200'].iloc[-1] if not np.isnan(df_index['sma_200'].iloc[-1]) else current_close
        current_rsi = df_index['rsi'].iloc[-1] if not np.isnan(df_index['rsi'].iloc[-1]) else 50

        # --- Regime Logic ---
        # Bear Market: Below 200 SMA + Low Momentum
        if current_close < current_sma and current_rsi < 45:
            return {
                'status': 'RISK_OFF', 
                'vol_mult': 1.4  # Require 40% MORE volume to trust breakout
            }
        
        # Overheated: Way above SMA + High RSI
        elif current_close > (current_sma * 1.15) and current_rsi > 75:
            return {
                'status': 'OVEREXTENDED', 
                'vol_mult': 1.2  # Caution required
            }

        return {'status': 'RISK_ON', 'vol_mult': 1.0}

    def evaluate_signal(self, ticker, signal_data, structural_zones):
        """
        Final Gatekeeper: Decides if signal is worthy of an alert.
        """
        # 1. Regime Filter
        # Adjust volume requirement based on market health
        required_vol = 1.8 * self.market_regime['vol_mult']
        
        if signal_data['vol_expansion'] < required_vol:
            # Signal rejected: Volume insufficient for current regime
            return None

        # 2. Location Context (Blue Sky vs Overhead Supply)
        breakout_level = signal_data['level']
        
        # Find next resistance strictly above current price
        next_resistance = None
        for zone in structural_zones:
            if zone['level'] > (breakout_level * 1.03): # At least 3% away
                next_resistance = zone['level']
                break
        
        # 3. Location Filter
        # If next resistance is too close (< 5%), reject unless it's a massive volume event
        if next_resistance:
            upside = ((next_resistance - breakout_level) / breakout_level) * 100
            
            # Reject low R:R trades in bad markets
            if upside < 5.0 and self.market_regime['status'] != 'RISK_ON':
                return None
                
            signal_data['next_resistance'] = f"{next_resistance:.2f} (+{upside:.1f}%)"
        else:
            signal_data['next_resistance'] = "Blue Sky (No Structure Detected)"

        return self._generate_narrative(ticker, signal_data)

    def _generate_narrative(self, ticker, data):
        """
        Constructs the Prompt for the LLM.
        """
        return f"""
        ACT AS: Senior Proprietary Trader for PSX.
        TASK: Summarize this technical breakout for the internal desk.
        STYLE: Clinical, Data-Driven, No Hype.

        DATA:
        - Ticker: {ticker}
        - Setup: Breakout from Volatility Compression
        - Key Level: {data['level']:.2f} (Tested {data['touches']} times)
        - Extension: Price is {data['atr_extension']}x ATR above level.
        - Volume: {data['vol_expansion']}x vs 20-Day Avg.
        - Compression Score: {data['compression_score']} (Higher is tighter coil)
        - Overhead Supply: {data['next_resistance']}
        - Market Environment: {self.market_regime['status']}

        OUTPUT TEMPLATE:
        🚨 **STRUCTURE BREAK: {{Ticker}}**
        
        **The Setup**
        [1 sentence on the compression/coil context]

        **The Trigger**
        [1 sentence on the volume and extension strength]

        **Context**
        [Comment on resistance clearance and market regime]

        *Quality Score: [0-10 based on metrics]*
        """