import os
import json
import time
import datetime
from pathlib import Path
from dotenv import load_dotenv
import requests

# Import our custom modules
from pipeline import PSXDataPipeline
from scanner import StructuralScanner
from evaluator import AgenticEvaluator

# Load environment variables (API Keys)
load_dotenv()

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOG_FILE = "alert_history.json"

class TelegramSender:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}/sendMessage"

    def send(self, message):
        if not self.token or not self.chat_id:
            print("⚠️ Telegram credentials missing. Printing to console.")
            print(f"\n--- ALERT ---\n{message}\n-------------\n")
            return

        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            requests.post(self.base_url, json=payload)
        except Exception as e:
            print(f"❌ Failed to send Telegram alert: {e}")

class AlertManager:
    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.history = self._load_history()

    def _load_history(self):
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return {}

    def is_cooling_down(self, symbol, cooldown_days=5):
        """Prevents spamming the same symbol within X days."""
        if symbol in self.history:
            last_alert = datetime.datetime.fromisoformat(self.history[symbol]['date'])
            if (datetime.datetime.now() - last_alert).days < cooldown_days:
                return True
        return False

    def log_alert(self, symbol, data):
        """Saves alert to history for future performance review."""
        self.history[symbol] = {
            "date": datetime.datetime.now().isoformat(),
            "trigger_price": data['level'],
            "type": "Breakout",
            "score": data.get('score', 0) 
        }
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=4)

def llm_wrapper(prompt):
    """
    Simple wrapper to call OpenAI. 
    If no key is found, it returns the structured prompt (Good for debugging).
    """
    if not OPENAI_API_KEY:
        return f"⚠️ [NO AI KEY] Raw Logic Output:\n\n{prompt}"
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a succinct financial risk analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Generation Failed: {e}\n\nFalling back to raw prompt:\n{prompt}"

def main():
    print("🚀 Starting PSX Regime Shift Detector...")
    
    # 1. Initialize Components
    pipeline = PSXDataPipeline()
    telegram = TelegramSender(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    manager = AlertManager(LOG_FILE)
    
    # 2. Update Data
    # pipeline.update_universe() # Uncomment this to run the download daily
    
    # 3. Analyze Market Regime
    evaluator = AgenticEvaluator(pipeline)
    print(f"🌍 Market Regime: {evaluator.market_regime['status']}")
    
    # 4. Iterate Universe
    alerts_generated = 0
    
    for symbol in pipeline.universe:
        clean_symbol = symbol.replace(".KA", "") # Internal use
        
        # A. Check Cooldown
        if manager.is_cooling_down(clean_symbol):
            continue

        # B. Load Data
        df = pipeline.load_data(clean_symbol)
        if df is None: continue

        # C. Run Quantitative Scanner
        scanner = StructuralScanner(df)
        breakout_candidates = scanner.evaluate_breakout()
        
        if not breakout_candidates:
            continue

        # D. Run Structural Zones (Need this for context in Evaluator)
        zones = scanner._find_structural_zones()

        # E. Evaluate Candidates
        for candidate in breakout_candidates:
            # The Evaluator decides if it's worthy and creates the Prompt
            alert_prompt = evaluator.evaluate_signal(clean_symbol, candidate, zones)
            
            if alert_prompt:
                print(f"🔔 Signal Detected: {clean_symbol}")
                
                # F. Generate Narrative via LLM
                final_message = llm_wrapper(alert_prompt)
                
                # G. Send & Log
                telegram.send(final_message)
                manager.log_alert(clean_symbol, candidate)
                alerts_generated += 1
    
    print(f"✅ Run Complete. Generated {alerts_generated} alerts.")

if __name__ == "__main__":
    main()