from .celery_app import celery_app
from .bot import LiveExecutiveBot
import logging

@celery_app.task(bind=True, name="run_bot_task", acks_late=True)
def run_bot_task(self, config):
    """
    Celery task that initializes and runs the bot loop.
    This runs in a separate process/container.
    """
    bot_id = config.get("botId")
    logger = logging.getLogger("CeleryWorker")
    logger.info(f"Received Task: Starting Bot {bot_id}")
    
    try:
        bot = LiveExecutiveBot(config)
        bot.sync_and_trade() # This will loop until stopped via DB
        return f"Bot {bot_id} Stopped Gracefully"
    except Exception as e:
        logger.error(f"Bot {bot_id} Crashed: {str(e)}")
        raise e
