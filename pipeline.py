import yfinance as yf
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta

class PSXDataPipeline:
    def __init__(self, storage_path="./data_store", suffix=".KA"):
        """
        Args:
            storage_path (str): Directory for cache (not strictly needed for GitHub Actions but good for local dev).
            suffix (str): Exchange suffix for Yahoo Finance (default '.KA' for Karachi).
        """
        self.storage_path = Path(storage_path)
        self.suffix = suffix
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core Liquid Universe (Add/Remove as needed)
        self.universe = [
            "SYS", "TRG", "LUCK", "ENGRO", "OGDC", "PPL", "HUBC", "UBL", "MCB", "HBL",
            "MEBL", "PSO", "ATRL", "NRL", "SEARLE", "UNITY", "NETSOL", "AVN", "PAEL",
            "GGL", "TELE", "TPL", "PIOC", "DGKC", "CHCC", "FCCL", "KAPCO", "EFERT",
            "OCTOPUS", "AIRLINK", "SHEL", "EPCL", "LOTECHEM"
        ]

    def _normalize_symbol(self, symbol):
        symbol = symbol.upper().strip()
        if not symbol.endswith(self.suffix):
            symbol += self.suffix
        return symbol

    def _validate_data(self, df, symbol):
        """
        Hygiene Checks:
        1. Non-empty
        2. Minimum history length
        3. Drop zero-volume days (holidays/suspensions)
        """
        if df is None or df.empty:
            return None
            
        # Drop days with 0 volume (distorts volatility/ATR)
        df = df[df['Volume'] > 0].copy()
        
        # Require at least 200 days for SMA200 / Regime filters
        if len(df) < 200:
            return None

        # Standardize columns to lowercase
        df.columns = [c.lower() for c in df.columns]
        return df

    def update_universe(self):
        """
        Fetches fresh data for the entire universe.
        Returns: Dictionary of {symbol: dataframe} to keep in memory (optimized for Cloud/GitHub Actions).
        """
        normalized_symbols = [self._normalize_symbol(s) for s in self.universe]
        data_cache = {}
        
        print(f"🔄 Fetching data for {len(normalized_symbols)} symbols...")
        
        # Batch download (Much faster)
        try:
            data = yf.download(
                normalized_symbols, 
                period="2y", 
                group_by='ticker', 
                threads=True,
                progress=False,
                auto_adjust=True
            )
        except Exception as e:
            print(f"❌ Batch download failed: {e}")
            return {}

        for symbol in normalized_symbols:
            clean_name = symbol.replace(self.suffix, "")
            
            # Extract specific ticker DF from the multi-index
            if len(normalized_symbols) > 1:
                try:
                    df_sym = data[symbol].copy()
                except KeyError:
                    continue
            else:
                df_sym = data.copy()
            
            # Validate
            clean_df = self._validate_data(df_sym, symbol)
            
            if clean_df is not None:
                # Save to memory dict
                data_cache[clean_name] = clean_df
                # Optional: Save to disk for local debugging
                clean_df.to_parquet(self.storage_path / f"{clean_name}.parquet")
        
        return data_cache

    def load_data(self, symbol, memory_cache=None):
        """
        Loads data either from the passed memory_cache (Cloud mode) or disk (Local mode).
        """
        # Priority 1: Memory Cache (from update_universe return)
        if memory_cache and symbol in memory_cache:
            return memory_cache[symbol]
            
        # Priority 2: Disk Cache
        file_path = self.storage_path / f"{symbol}.parquet"
        if file_path.exists():
            return pd.read_parquet(file_path)
            
        return None

    def get_market_regime(self):
        """
        Fetches KSE-100 data to determine broad market health.
        """
        try:
            # ^KSE is the ticker for KSE-100 Index on Yahoo
            # If unavailable, you might use a major ETF or proxy like 'OGDC.KA'
            kse100 = yf.download("^KSE", period="1y", progress=False, auto_adjust=True)
            if not kse100.empty:
                kse100.columns = [c.lower() for c in kse100.columns]
                return kse100
        except:
            pass
        return None