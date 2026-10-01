import logging
from datetime import datetime

# ==========================================
# 1. THE STRATEGY TRANSLATOR (The "Brain")
# ==========================================
# This dictionary maps technical strategy codes to plain English explanations.
# It covers RSI, MACD, Bollinger Bands, Moving Averages, and ML.
STRATEGY_TRANSLATOR = {
    # --- RSI Strategies ---
    "RSI_OVERBOUGHT": "The market is overheated (RSI High). Buyers are exhausted, which usually precedes a price drop.",
    "RSI_OVERSOLD":   "The market is undervalued (RSI Low). Sellers are exhausted, often leading to a bounce up.",
    "RSI_DIVERGENCE": "Price is moving one way, but momentum (RSI) is moving the other. A reversal is likely.",

    # --- MACD Strategies ---
    "MACD_CROSS_BULL": "Momentum Shift: The MACD line crossed above the signal line, indicating upward momentum.",
    "MACD_CROSS_BEAR": "Momentum Shift: The MACD line crossed below the signal line, indicating downward pressure.",

    # --- Moving Averages (SMA/EMA) ---
    "SMA_CROSS_GOLDEN": "Golden Cross: Short-term trend has crossed above the long-term average. Strongly Bullish.",
    "SMA_CROSS_DEATH":  "Death Cross: Short-term trend has dropped below the long-term average. Strongly Bearish.",
    "TREND_FOLLOWING":  "Price is consistently above the moving average. We are riding the trend.",

    # --- Bollinger Bands ---
    "BB_SQUEEZE":   "Volatility is dangerously low. A massive explosive move is imminent.",
    "BB_BREAKOUT":  "Price has broken outside the statistical bands. This signals a continuation of the breakout.",
    "BB_REVERSAL":  "Price touched the outer band and is snapping back to the mean.",

    # --- Machine Learning ---
    "ML_CONFIDENCE": "The AI model has detected a high-probability pattern based on historical training data.",
    
    # --- Candlestick Patterns ---
    "HAMMER":       "Reversal Pattern: Sellers pushed price down, but buyers rejected it aggressively.",
    "ENGULFING":    "Momentum Shift: The current candle completely overwhelmed the previous one.",
    
    # --- Fallback ---
    "UNKNOWN":      "Technical setup detected matching strategy parameters."
}

def get_dynamic_explanation(strategy_name, raw_message=""):
    """
    Scans the strategy name and message to find the best human explanation.
    """
    # 1. Direct Lookup
    if strategy_name in STRATEGY_TRANSLATOR:
        return STRATEGY_TRANSLATOR[strategy_name]

    # 2. Fuzzy Search (Scan for keywords)
    # E.g., if strategy_name is "RSI_DIVERGENCE_1H", it matches "RSI_DIVERGENCE"
    for key, text in STRATEGY_TRANSLATOR.items():
        keywords = key.split('_')
        # Check if ALL keywords exist in the strategy name OR the raw message
        if all(k in strategy_name.upper() or k in raw_message.upper() for k in keywords):
            return text

    return "Technical signal detected. Market conditions meet strategy entry criteria."


# ==========================================
# 2. THE LOGGING HANDLER (The "Bridge")
# ==========================================
# This class automatically pushes every log to your MongoDB so the UI sees it.
class MongoLogHandler(logging.Handler):
    def __init__(self, db_collection, bot_id):
        super().__init__()
        self.collection = db_collection
        self.bot_id = bot_id

    def emit(self, record):
        try:
            # Format the message
            log_message = self.format(record)
            
            # Create the log object
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": record.levelname,  # INFO, ERROR, WARNING
                "message": log_message
            }

            # Push to MongoDB (Efficient $push)
            self.collection.update_one(
                {"botId": self.bot_id},
                {"$push": {"logs": log_entry}}
            )
        except Exception:
            self.handleError(record)


# ==========================================
# 3. HELPER FUNCTIONS (The "Action")
# ==========================================
# Call these functions inside your main bot loop.

def log_decision_phase(logger, strategy_name, signal_details, is_blocked=False, block_reason=""):
    """
    PART 1: Logs the decision logic with human context.
    """
    # 1. Get the Explanation
    explanation = get_dynamic_explanation(strategy_name, signal_details)

    # 2. Log The Thought Process
    logger.info(f"DECISION | 🧠 THOUGHTS: {strategy_name.upper()} Triggered ({signal_details})")
    logger.info(f"DECISION | 🗣️ CONTEXT: {explanation}")

    # 3. Log The Verdict
    if is_blocked:
        logger.info(f"DECISION | ⚖️ VERDICT: Signal Generated but Blocked.")
        logger.info(f"DECISION | 🚫 REASON: {block_reason}")
    else:
        logger.info(f"DECISION | ⚖️ VERDICT: GO! Signal Executed.")


def log_monitoring_phase(logger, current_price, position=None):
    """
    PART 2: Logs the heartbeat with distances to targets.
    """
    # Standard Price Log
    logger.info(f"⏳ MONITORING | Price: ${current_price:,.2f}")

    if position:
        # 1. Extract Data
        entry = float(position.get('entry_price', 0))
        tp = float(position.get('take_profit', 0))
        sl = float(position.get('stop_loss', 0))
        side = position.get('side', 'long').lower()
        
        # 2. Calculate Math
        dist_tp = abs(current_price - tp)
        dist_sl = abs(current_price - sl)
        
        # Calculate PnL Percentage
        if entry > 0:
            raw_pnl = (current_price - entry) / entry * 100
            pnl_percent = raw_pnl if side == 'long' else -raw_pnl
        else:
            pnl_percent = 0.0

        # 3. Determine Status Emoji
        emoji = "🟢" if pnl_percent >= 0 else "🔴"
        action = "Buying" if side == 'long' else "Shorting"

        # 4. Log the "Trader Narrative"
        logger.info(f"💎 STRATEGY  | {emoji} {action.upper()} from ${entry:,.2f} (PnL: {pnl_percent:+.2f}%)")
        logger.info(f"🎯 TARGETS   | To Win: Hit ${tp:,.2f} (${dist_tp:.2f} away) | Safety: ${sl:,.2f} (${dist_sl:.2f} away)")
    else:
        # If no position, just simple status
        logger.info("💤 STATUS     | No active trades. Scanning for setups...")
