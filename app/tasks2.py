import logging
import traceback
from .celery_app import celery_app
from .bot2 import LiveExecutiveBot
from .config2 import bots_collection, ML_CONFIG

# NEW MODULAR IMPORTS
from predictors.model_factory import ModelFactory

logger = logging.getLogger("CeleryWorker")

@celery_app.task(bind=True, name="run_bot_task", acks_late=True, track_started=True)
def run_bot_task(self, config):
    """
    CELERY TASK v7.0: AI-Aware Worker
    Features: Model Pre-loading, Graceful Retry, and State Syncing.
    """
    bot_id = config.get("botId")
    model_type = config.get("params", {}).get("model_type", "XGBoost")
    
    logger.info(f"🌀 [TASK START] Launching Bot {bot_id} with {model_type}")
    
    # 1. Update Task State in DB (So UI shows "Starting...")
    self.update_state(state='PROGRESS', meta={'status': 'Loading AI Model...'})
    
    try:
        # 2. PRE-LOAD MODEL CACHE (Upgrade #1 Integration)
        # This calls the Factory. If the model is already in RAM, it's instant.
        # If not, it loads it once so the bot loop doesn't lag on its first candle.
        ModelFactory.load_model(model_type, bot_id=bot_id)
        
        # 3. Initialize Modular Bot
        bot = LiveExecutiveBot(config)
        
        # 4. State Handover
        # We tell the bot to check this task instance for a shutdown signal
        self.update_state(state='STARTED', meta={'status': 'Bot Loop Active'})
        
        # This loops until the user clicks 'Stop' in the UI or a crash occurs
        bot.sync_and_trade() 
        
        return {
            "status": "Success",
            "botId": bot_id,
            "msg": "Bot stopped gracefully via user command."
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"🚨 [CRITICAL] Bot {bot_id} Crashed!\n{error_trace}")
        
        # Update MongoDB with the specific crash reason for user debugging
        bots_collection.update_one(
            {"botId": bot_id},
            {"$set": {"status": "CRASHED", "last_error": str(e)}}
        )
        
        # Celery Retry Logic (Optional: Only retry on network errors)
        if "Connection" in str(e) or "Timeout" in str(e):
            logger.info(f"🔄 Retrying Bot {bot_id} in 60s...")
            raise self.retry(exc=e, countdown=60, max_retries=5)
            
        raise e
