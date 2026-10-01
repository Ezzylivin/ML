import os
from celery import Celery

# Connect to Redis container (default) or localhost
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "trading_bot_worker",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Important for trading loops: Don't kill tasks immediately on shutdown
    task_acks_late=True,
)
