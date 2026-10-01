import os
import logging
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Internal Imports
from .trading_env import TradingEnv
from app.config import MODEL_STORAGE_DIR, LOG_DIR

logger = logging.getLogger("RLAgent")

class RLAgent:
    """
    UPGRADE #1 (Part B): The Reinforcement Learning Agent.
    Uses PPO (Proximal Policy Optimization) to learn trading behavior.
    """
    def __init__(self, model_id="BTC_1h_PPO"):
        self.model_id = model_id
        self.save_path = os.path.join(MODEL_STORAGE_DIR, f"{self.model_id}.zip")
        self.model = None

    def train(self, df, total_timesteps=100000):
        """
        Trains the agent inside the custom Trading Environment.
        """
        logger.info(f"🏋️  Starting RL Training for {self.model_id}...")
        
        # 1. Create and Wrap Environment
        env = DummyVecEnv([lambda: TradingEnv(df)])

        # 2. Initialize PPO Model
        # MlpPolicy = Multi-layer Perceptron (Standard for vector data)
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=0.0003, # standard for stable training
            n_steps=2048,
            batch_size=64,
            gamma=0.99,           # Discount factor for future rewards
            tensorboard_log=os.path.join(LOG_DIR, "tensorboard")
        )

        # 3. Training with Checkpoints
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, 
            save_path=os.path.join(MODEL_STORAGE_DIR, "checkpoints"),
            name_prefix=self.model_id
        )

        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        self.model.save(self.save_path)
        logger.info(f"💾 Model trained and saved to {self.save_path}")

    def load(self):
        """Loads a trained brain from disk."""
        if os.path.exists(self.save_path):
            self.model = PPO.load(self.save_path)
            return True
        return False

    def predict_bullish_prob(self, observation):
        """
        MODULAR ADAPTER: Converts RL actions into a 'Bullish Probability' (0-1).
        This allows the RL Agent to plug into your existing threshold-based Backtester.
        """
        if self.model is None and not self.load():
            return 0.5  # Neutral fallback
            
        # Get Action Probability Distribution
        # observation shape must match training (num_features,)
        obs_tensor = np.array([observation])
        
        # predict() usually returns the discrete action, 
        # but for the modular system, we want the raw 'thought' process:
        action, _states = self.model.predict(obs_tensor, deterministic=True)
        
        # Mapping RL Actions to Probabilities:
        # 0: Hold -> 0.5
        # 1: Long -> 1.0 (Confident Bullish)
        # 2: Short -> 0.0 (Confident Bearish)
        if action == 1: return 0.95
        if action == 2: return 0.05
        return 0.50

if __name__ == "__main__":
    # Example: How to trigger training for any new dataset
    data = pd.read_csv("data/BTC-USD-1h.csv")
    # ... perform feature engineering ...
    agent = RLAgent(model_id="BTC_1H_V1")
    agent.train(data)
