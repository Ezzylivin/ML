import logging
from datetime import datetime
import numpy as np
import pandas as pd

# --- Standardized Import ---
try:
    from .utils2 import clean_float
except ImportError:
    def clean_float(x): 
        try: return float(x) if x is not None else 0.0
        except: return 0.0

class MongoLogHandler(logging.Handler):
    """
    ENGINE v7.0: AI-Aware Logging
    Captures ML inference probabilities and model states into MongoDB.
    """
    def __init__(self, db_collection, bot_id):
        super().__init__()
        self.collection = db_collection
        self.bot_id = bot_id

    def emit(self, record):
        try:
            level = record.levelname.lower()
            log_type = 'status'
            if level in ['warning', 'error', 'critical']:
                log_type = level

            # Detect extra AI metadata if passed in the log record
            inference_data = getattr(record, 'inference', None)

            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": log_type,
                "message": self.format(record),
                "prob": clean_float(inference_data) if inference_data else None
            }
            
            if self.collection is not None:
                try:
                    self.collection.update_one(
                        {"botId": self.bot_id},
                        {"$push": {
                            "logs": {
                                "$each": [log_entry], 
                                "$slice": -1000 
                            }
                        }}, 
                        upsert=True
                    )
                except:
                    pass 
        except: 
            self.handleError(record)

# --- UPGRADED LOGGING HELPERS ---

def log_inference_step(logger, price, prob, adx):
    """
    Logs the specific ML prediction result.
    This replaces log_intent for the modular setup.
    """
    sentiment = "NEUTRAL"
    if prob >= 0.65: sentiment = "BULLISH"
    elif prob <= 0.35: sentiment = "BEARISH"
    
    logger.info(
        f"🧠 ML INFERENCE | Sentiment: {sentiment} | Conf: {prob:.2f} | ADX: {adx:.1f} | Price: ${price:,.2f}",
        extra={'inference': prob}
    )

def log_monitoring_phase(logger, current_price, position=None, thought=None):
    """Standardized reporting for the bot's 'Idle/Monitoring' state."""
    msg_parts = []
    
    if position:
        entry = clean_float(position.entry_price if hasattr(position, 'entry_price') else position.get('entry', 0))
        # Support for the new Manager.Position object or legacy dict
        side_val = getattr(position, 'side', position.get('side', 1))
        side_label = 'LONG' if side_val == 1 or str(side_val).lower() == 'long' else 'SHORT'
        
        try:
            pnl_percent = ((current_price - entry) / entry * 100) if entry > 0 else 0
            if side_label == 'SHORT': pnl_percent = -pnl_percent
            msg_parts.append(f"💎 MANAGING {side_label} | PnL: {pnl_percent:+.2f}% | Price: ${current_price:,.2f}")
        except:
            msg_parts.append(f"💎 MANAGING | Price: ${current_price:,.2f}")
    else:
        msg_parts.append(f"⏳ SCANNING | Price: ${current_price:,.2f}")

    if thought:
        msg_parts.append(f"| 💭 {thought}")

    logger.info(" ".join(msg_parts))

def log_decision(logger, price, prob, action, reason="Model Threshold"):
    """
    MODULAR DECISION LOG:
    Logs the exact action taken based on probabilistic thresholds.
    """
    if action == 0:
        return

    action_label = "LONG" if action == 1 else "SHORT"
    logger.info(
        f"📊 DECISION | Action: {action_label} | Reason: {reason} | Conf: {prob:.2f} | Price: ${price:,.2f}"
    )
