import logging
from datetime import datetime

# --- Standardized Import ---
try:
    from .utils import clean_float
except ImportError:
    def clean_float(x): 
        try: return float(x) if x is not None else 0.0
        except: return 0.0

class MongoLogHandler(logging.Handler):
    """
    Custom logging handler that saves logs to a MongoDB collection.
    Uses a capped array ($slice) to keep history to the last 1000 entries.
    """
    def __init__(self, db_collection, bot_id):
        super().__init__()
        self.collection = db_collection
        self.bot_id = bot_id

    def emit(self, record):
        try:
            # Map logging levels to frontend-friendly types
            level = record.levelname.lower()
            log_type = 'status'
            if level in ['warning', 'error', 'critical']:
                log_type = level

            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": log_type,
                "message": self.format(record)
            }
            
            # Atomic update to the bot's log array
            if self.collection is not None:
                try:
                    self.collection.update_one(
                        {"botId": self.bot_id},
                        {"$push": {
                            "logs": {
                                "$each": [log_entry], 
                                "$slice": -1000 # Keeps only the 1000 most recent logs
                            }
                        }}, 
                        upsert=True
                    )
                except:
                    pass # Silent fail to prevent logging loops
        except: 
            self.handleError(record)

# --- LOGGING HELPER FUNCTIONS ---

def log_monitoring_phase(logger, current_price, position=None, thought=None):
    """Standardized reporting for the bot's 'Idle/Monitoring' state."""
    msg_parts = []
    
    if position:
        entry = clean_float(position.get('entry', 0))
        side = position.get('side', 'unknown')
        try:
            # Calculate floating PnL for the log message
            pnl_percent = ((current_price - entry) / entry * 100) if entry > 0 else 0
            if side == 'short': pnl_percent = -pnl_percent
            msg_parts.append(f"💎 MANAGING | PnL: {pnl_percent:+.2f}% | Price: ${current_price:,.2f}")
        except:
            msg_parts.append(f"💎 MANAGING | Price: ${current_price:,.2f}")
    else:
        msg_parts.append(f"⏳ SCANNING | Price: ${current_price:,.2f}")

    if thought:
        msg_parts.append(f"| 💭 {thought}")

    logger.info(" ".join(msg_parts))

def log_intent(logger, strategies, price):
    """Logs the start of a signal calculation cycle."""
    names = [s.get("code", "Unknown") for s in strategies]
    logger.info(f"🎯 INTENT | Price: ${price:,.2f} | Evaluating: {', '.join(names)}")

def log_decision(logger, price, decision, strategy_results):
    """Logs the final trade logic and supporting indicator data."""
    action = decision.get('decision', 'NO_TRADE')
    if action == "NO_TRADE":
        return

    logger.info(f"📊 DECISION | Action: {action} | Price: ${price:,.2f} | Conf: {decision.get('confidence', 0)}")

    if isinstance(strategy_results, list):
        for r in strategy_results:
            if not r or r.get('signal', 0) == 0: 
                continue
            
            # Format target and stop for the log table
            tgu = f"${r.get('target'):,.2f}" if r.get('target') else "N/A"
            stu = f"${r.get('stop'):,.2f}" if r.get('stop') else "N/A"
            
            logger.info(
                f" • {r.get('code', 'STG')} | {r.get('reason', 'Signal')} | "
                f"Conf: {r.get('confidence', 0)} | TP: {tgu} | SL: {stu}"
            )
