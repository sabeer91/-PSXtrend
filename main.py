import os
import json
import requests
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Import modules
from pipeline import PSXDataPipeline
from scanner import StructuralScanner
from evaluator import AgenticEvaluator

# Load environment variables (for local dev)
# On GitHub Actions, these are injected automatically from Secrets
load_dotenv()

# --- CONFIGURATION ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HISTORY_FILE = "alert_history.json"

class TelegramSender:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
    
    def send(self, message):
        """Sends message to Telegram. Prints to console if creds are missing."""
        if not self.token or not self.chat_id:
            print(f"\n📢 [CONSOLE ALERT] \n{message}\n")
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code != 200:
                print(f"❌ Telegram Error: {resp.text}")
        except Exception as e:
            print(f"❌ Connection Error: {e}")

class AlertManager:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.history = self._load_history()

    def _load_history(self):
        """Loads JSON history. Returns empty dict if file is missing/corrupt."""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("⚠️ History file corrupted. Starting fresh.")
                return {}
        return {}

    def is_cooling_down(self, symbol, cooldown_days=5):
        """Returns True if symbol was alerted recently."""
        if symbol in self.history:
            try:
                last_date_str = self.history[symbol]['date']
                last_date = datetime.fromisoformat(last_date_str)
                days_diff = (datetime.now() - last_date).days
                if days_diff < cooldown_days:
                    return True
            except (ValueError, KeyError):
                pass # Invalid date format in file, ignore
        return False

    def log_alert(self, symbol, data):
        """Updates the history file."""
        self.history[symbol] = {
            "date": datetime.now().isoformat(),
            "level": data['level'],
            "score": data.get('compression_score', 0)
        }
        # Atomic write to avoid corruption
        with open(self.filepath, 'w') as f:
            json.dump(self.history, f, indent=4)

def generate_llm_summary(prompt):
    """
    Sends the prompt to OpenAI. 
    Falls back to returning the prompt itself if Key is missing or API fails.
    """
    if not OPENAI_API_KEY:
        return f"⚠️ **(No AI Key)** - Raw Signal:\n{prompt}"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4", # Or "gpt-3.5-turbo" for lower cost
            messages=[
                {"role": "system", "content": "You are a succinct financial risk analyst. Output in Markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=250
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        return f"⚠️ **AI Error** - Raw Signal:\n{prompt}"

def main():
    print("🚀 Starting PSX Regime Shift Detector...")
    
    # 1. Init Components
    pipeline = PSXDataPipeline()
    telegram = TelegramSender(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    manager = AlertManager(HISTORY_FILE)
    
    # 2. Fetch Data (In-Memory)
    # This returns a Dict { 'SYS': dataframe, ... }
    # optimized for cloud so we don't read/write to disk
    market_data = pipeline.update_universe()
    
    if not market_data:
        print("❌ No data fetched. Aborting.")
        return

    # 3. Assess Market Regime
    evaluator = AgenticEvaluator(pipeline)
    regime = evaluator.market_regime
    print(f"🌍 Market Regime: {regime['status']} (Vol Multiplier: {regime['vol_mult']}x)")
    
    alerts_triggered = 0

    # 4. Scan the Universe
    for symbol, df in market_data.items():
        # A. Check Cooldown
        if manager.is_cooling_down(symbol):
            continue

        # B. Run Math Scanner
        scanner = StructuralScanner(df)
        breakout_candidates = scanner.evaluate_breakout()
        
        if not breakout_candidates:
            continue

        # C. Get Zones (for context)
        zones = scanner._find_structural_zones()

        # D. Agentic Evaluation
        for candidate in breakout_candidates:
            # The Evaluator applies the "Context" filter (Regime + Location)
            alert_prompt = evaluator.evaluate_signal(symbol, candidate, zones)
            
            if alert_prompt:
                print(f"🔔 Breakout Detected: {symbol}")
                
                # E. LLM Narrative
                final_message = generate_llm_summary(alert_prompt)
                
                # F. Send & Log
                telegram.send(final_message)
                manager.log_alert(symbol, candidate)
                alerts_triggered += 1
                
                # Sleep briefly to avoid Telegram rate limits
                time.sleep(1)

    # 5. Final Report
    if alerts_triggered == 0:
        print("✅ Scan Complete. No structural breakouts detected today.")
        # Optional: Send a "Heartbeat" message to know it ran
        # telegram.send(f"✅ Daily Scan Complete. Market: {regime['status']}. No signals.")
    else:
        print(f"✅ Scan Complete. Sent {alerts_triggered} alerts.")

if __name__ == "__main__":
    main()
