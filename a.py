# 🚀 UPGRADE: v21.0 - "Optimizer Supreme Upgraded"
# Includes the requested enhancements:
#   1. Improved modularity and code reusability.
#   2. Dynamic scalability for parallel jobs.
#   4. Enhanced Monte Carlo analysis.
#   5. Improved edge case handling.
#   6. Multi-objective optimization with additional risk metrics.
#   7. User-friendly CLI updates.
#   8. Real-time logging and monitoring improvements.

import requests
import json
import pandas as pd
import numpy as np
import time
import optuna
from datetime import datetime, timedelta, timezone
import os
import random
import argparse
import logging
import traceback
import hashlib
from copy import deepcopy
from colorama import Fore, Style, init
import warnings
import uuid
from typing import Dict, List, Tuple, Any, Optional

# ===================================================
# CONFIGURATIONS AND LOGGING
# ===================================================
ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://127.0.0.1:8000")
RESULTS_DIR = os.path.join(os.getcwd(), "data", "optimizer_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    filename="optimizer_errors.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("Optimizer")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("optimizer_history.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(file_handler)

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)

DEFAULT_PARALLEL_JOBS = max(1, (os.cpu_count() or 4) - 1)
TIMEOUT_SECONDS = 600
TRADE_FREQ_MULTIPLIERS = {"1h": 0.04, "4h": 0.02, "15m": 0.08}

# ===================================================
# MULTI-OBJECTIVE OPTIMIZATION TARGET METRICS
# ===================================================
# Supports Pareto optimization for additional objectives such as risk/reward ratios.
ADDITIONAL_METRICS = {
    'sharpe_ratio': lambda metrics: metrics.get('totalReturn', 0) / max(1.0, metrics.get('stdDev', 1.0)),
    'risk_of_ruin': lambda metrics: metrics.get('prob_ruin', 0),
}

# DEFAULT STRATEGY POOLS
TREND_POOL = [
    "sma_crossover",
    "macd_crossover",
    "ichimoku_cloud",
    "psar_signal",
    "obv_signal",
    "atr_breakout",
]
RANGE_POOL = [
    "rsi_divergence",
    "stochastic_crossover",
    "bollinger_bands",
    "cci_oversold",
]
ALL_AVAILABLE_MODELS = []

# ===================================================
# UTIL FUNCTIONS
# ===================================================

def print_header(text: str):
    print(f"\n{Fore.CYAN}{'=' * 60}\n {text.center(58)} \n{'=' * 60}{Style.RESET_ALL}")


def robust_request(
    method: str, url: str, json_data: Optional[Dict] = None, retries: int = 3
) -> requests.Response:
    for i in range(retries):
        try:
            if method == "GET":
                r = requests.get(url, timeout=15)
            else:
                r = requests.post(url, json=json_data, timeout=TIMEOUT_SECONDS)
                
            # Retry on server errors (status 5xx)
            if r.status_code == 200:
                return r
            if r.status_code >= 500:
                sleep_time = 2**i
                time.sleep(sleep_time)
            else:
                return r
        except Exception:
            time.sleep(2**i)
    raise Exception(f"Request failed after {retries} retries: {url}")


