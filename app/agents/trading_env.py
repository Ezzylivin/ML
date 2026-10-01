import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    UPGRADE #1 (Part B): Reinforcement Learning Environment.
    A high-fidelity simulation 'Gym' for BTC-USD trading.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=1000, commission=0.0006, slippage=0.0001):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index()
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage

        # --- 🎮 ACTION SPACE ---
        # 0: Hold/Neutral, 1: Long, 2: Short
        self.action_space = spaces.Discrete(3)

        # --- 👀 OBSERVATION SPACE ---
        # The agent sees a window of data (e.g., OHLCV + Technical Indicators)
        # We define this as a continuous box of values
        num_features = len(self.df.select_dtypes(include=[np.number]).columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Restarts the 'Game' at a random point in history for better learning."""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.current_step = np.random.randint(0, len(self.df) - 500) # Start with buffer

        return self._next_observation(), {}

    def _next_observation(self):
        """Gets the current market 'state' for the agent to analyze."""
        obs = self.df.select_dtypes(include=[np.number]).iloc[self.current_step].values
        return obs.astype(np.float32)

    def step(self, action):
        """The 'Tick' of the engine: Executes an action and returns the reward."""
        current_price = self.df.loc[self.current_step, 'close']
        prev_net_worth = self.net_worth

        # 1. EXECUTE ACTIONS
        # Simplification: RL Agent flips entire balance into position
        if action == 1: # LONG
            if self.shares_held == 0:
                buy_price = current_price * (1 + self.slippage)
                self.shares_held = (self.balance * 0.95) / buy_price
                self.balance -= (self.shares_held * buy_price) * (1 + self.commission)
                self.cost_basis = buy_price

        elif action == 2: # SHORT (Simplified to 'Exit/Cash Out' for base env)
            if self.shares_held > 0:
                sell_price = current_price * (1 - self.slippage)
                self.balance += (self.shares_held * sell_price) * (1 - self.commission)
                self.shares_held = 0
                self.cost_basis = 0

        # 2. UPDATE STATE
        self.current_step += 1
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # 3. CALCULATE REWARD (The Heart of RL)
        # Reward is the % change in net worth (encourages growth, penalizes drawdowns)
        reward = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth > 0 else 0
        
        # Add a small penalty for every step to encourage efficiency (Time decay)
        reward -= 0.00001 

        # 4. CHECK TERMINATION
        terminated = self.net_worth < (self.initial_balance * 0.5) # Bankruptcy at -50%
        truncated = self.current_step >= len(self.df) - 1

        return self._next_observation(), reward, terminated, truncated, {}
