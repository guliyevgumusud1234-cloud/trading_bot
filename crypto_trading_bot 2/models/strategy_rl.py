"""
Deep Reinforcement Learning Strategy using PPO.

This strategy:
- Uses Proximal Policy Optimization (PPO)
- Custom Gym environment for crypto trading
- 200+ features as state space
- 3 actions: Long, Neutral, Short
- Risk-adjusted reward function
- 1M timestep training
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from typing import Dict, Optional, Tuple, Any
import yaml

from utils.logger import get_logger

logger = get_logger(__name__)


class CryptoTradingEnv(gym.Env):
    """
    Custom Gym environment for cryptocurrency futures trading.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000,
        max_position_size: float = 0.5,
        max_leverage: float = 2.0,
        transaction_cost: float = 0.0004,
        lookback_window: int = 50
    ):
        """
        Initialize trading environment.
        
        Args:
            df: DataFrame with OHLCV and features
            initial_balance: Starting capital
            max_position_size: Max 50% of capital per trade
            max_leverage: Maximum leverage (reduced to 2x)
            transaction_cost: Transaction cost per trade (0.04%)
            lookback_window: Number of features to use (reduced from 200+)
        """
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        # SIMPLIFIED State space: Select most important features only
        # Priority features: Returns, RSI, MACD, Bollinger Bands, Volume
        feature_cols = df.columns.tolist()
        
        # Select key features (first 30-50 most important)
        # If we have fewer features, use all
        n_features = min(len(feature_cols), 50)
        self.selected_features = feature_cols[:n_features]
        
        # Position features: position, entry_price_ratio, unrealized_pnl_ratio
        n_position_features = 3
        self.n_state_features = n_features + n_position_features
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_state_features,),
            dtype=np.float32
        )
        
        # Action space: 0=Close/Hold, 1=Long, 2=Short
        self.action_space = spaces.Discrete(3)
        
        # Episode tracking
        self.reset()
        
        logger.debug(f"Trading environment initialized with {len(df)} timesteps, {n_features} features")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.position_size = 0.0
        self.entry_price = 0.0
        self.leverage = 1.0
        self.total_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        self.holding_time = 0
        self.num_trades = 0
        
        self.equity_curve = [self.balance]
        self.peak_balance = self.balance
        self.drawdown = 0.0
        
        self.trade_history = []
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step with SIMPLIFIED logic and auto stop-loss/take-profit.
        
        Args:
            action: 0=Close/Hold, 1=Long, 2=Short
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Ensure balance is valid
        if np.isnan(self.balance) or np.isinf(self.balance) or self.balance < 0:
            self.balance = self.initial_balance
        
        prev_balance = float(self.balance)
        current_price = float(self.df.loc[self.current_step, 'close'])
        
        # Ensure price is valid
        if np.isnan(current_price) or current_price <= 0:
            current_price = self.entry_price if self.entry_price > 0 else 50000.0
        
        position_closed = False
        
        # Execute action
        if action == 1 and self.position == 0:  # Open Long
            self.position = 1
            self.entry_price = current_price
            # Ensure position size doesn't exceed balance
            self.position_size = min(self.balance * self.max_position_size, self.balance * 0.99)
            # Apply transaction cost
            cost = self.position_size * self.transaction_cost
            self.balance = max(0, self.balance - cost)  # Prevent negative balance
            self.holding_time = 0
            self.num_trades += 1
            
        elif action == 2 and self.position == 0:  # Open Short
            self.position = -1
            self.entry_price = current_price
            # Ensure position size doesn't exceed balance
            self.position_size = min(self.balance * self.max_position_size, self.balance * 0.99)
            # Apply transaction cost
            cost = self.position_size * self.transaction_cost
            self.balance = max(0, self.balance - cost)  # Prevent negative balance
            self.holding_time = 0
            self.num_trades += 1
            
        elif action == 0 and self.position != 0:  # Close position
            if self.position == 1:  # Close long
                pnl_pct = (current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0.0
            else:  # Close short
                pnl_pct = (self.entry_price - current_price) / self.entry_price if self.entry_price > 0 else 0.0
            
            # Apply leverage
            pnl_pct *= self.max_leverage
            # Clip to prevent extreme values
            pnl_pct = np.clip(pnl_pct, -0.5, 2.0)  # Max 50% loss, 200% gain
            
            # Update balance safely
            if self.position_size > 0:
                pnl_amount = self.position_size * pnl_pct
                cost = self.position_size * self.transaction_cost
                new_balance = self.balance + self.position_size + pnl_amount - cost
                self.balance = max(0, new_balance)  # Prevent negative
                self.total_pnl += pnl_amount
            
            # Record trade
            self.trade_history.append({
                'position': 'LONG' if self.position == 1 else 'SHORT',
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'pnl': self.position_size * pnl_pct,
                'holding_time': self.holding_time
            })
            
            self.position = 0
            self.position_size = 0.0
            self.entry_price = 0.0
            self.holding_time = 0
            position_closed = True
        
        # Check stop loss / take profit if position is open
        if self.position != 0:
            if self.position == 1:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - current_price) / self.entry_price
            
            pnl_pct *= self.max_leverage
            
            # Auto close on stop/target (10% loss or 20% profit)
            if pnl_pct <= -0.10 or pnl_pct >= 0.20:
                # Clip pnl to prevent extreme values
                pnl_pct = np.clip(pnl_pct, -0.5, 2.0)
                if self.position_size > 0:
                    pnl_amount = self.position_size * pnl_pct
                    cost = self.position_size * self.transaction_cost
                    new_balance = self.balance + self.position_size + pnl_amount - cost
                    self.balance = max(0, new_balance)  # Prevent negative
                    self.total_pnl += pnl_amount
                
                self.trade_history.append({
                    'position': 'LONG' if self.position == 1 else 'SHORT',
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'pnl': self.position_size * pnl_pct,
                    'holding_time': self.holding_time,
                    'auto_closed': True
                })
                
                self.position = 0
                self.position_size = 0.0
                self.entry_price = 0.0
                self.holding_time = 0
                position_closed = True
        
        # Ensure balance is still valid after operations
        if np.isnan(self.balance) or np.isinf(self.balance):
            self.balance = prev_balance
        self.balance = max(0, float(self.balance))
        
        # Calculate reward using SIMPLIFIED function
        reward = self.calculate_reward(action, prev_balance, self.balance, position_closed)
        
        # Ensure reward is valid
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        # Update holding time
        if self.position != 0:
            self.holding_time += 1
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = (
            self.current_step >= len(self.df) - 1 or
            self.balance < self.initial_balance * 0.5
        )
        
        obs = self._get_observation()
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_pnl': self.total_pnl,
            'num_trades': self.num_trades
        }
        
        return obs, reward, done, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation with SIMPLIFIED features."""
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        
        # Market features (only selected features)
        row = self.df.iloc[self.current_step]
        market_features = row[self.selected_features].values.astype(np.float32)
        
        # Replace NaN and Inf with 0
        market_features = np.nan_to_num(market_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Position features (simplified)
        if self.position != 0 and self.entry_price > 0:
            current_price = row.get('close', self.entry_price)
            if current_price > 0:
                entry_price_ratio = self.entry_price / current_price
                unrealized_pnl = self._calculate_unrealized_pnl(current_price)
                unrealized_pnl_ratio = unrealized_pnl / self.initial_balance if self.initial_balance > 0 else 0.0
            else:
                entry_price_ratio = 1.0
                unrealized_pnl_ratio = 0.0
        else:
            entry_price_ratio = 1.0
            unrealized_pnl_ratio = 0.0
        
        # Clip to reasonable ranges
        entry_price_ratio = np.clip(entry_price_ratio, 0.5, 2.0)
        unrealized_pnl_ratio = np.clip(unrealized_pnl_ratio, -1.0, 1.0)
        
        position_features = np.array([
            float(self.position),  # -1, 0, or 1
            float(entry_price_ratio),
            float(unrealized_pnl_ratio)
        ], dtype=np.float32)
        
        # Replace any remaining NaN
        position_features = np.nan_to_num(position_features, nan=0.0)
        
        # Combine
        obs = np.concatenate([market_features, position_features]).astype(np.float32)
        
        # Final check for NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return obs
    
    def _open_position(self, position: int, price: float):
        """Open a new position."""
        self.position = position
        self.entry_price = price
        
        # Calculate position size (2% of balance)
        self.position_size = self.balance * self.max_position_size
        
        # Determine leverage based on confidence (placeholder)
        self.leverage = 2.0  # Can be dynamic
        
        # Apply transaction cost
        cost = self.position_size * self.transaction_cost
        self.balance -= cost
        
        self.holding_time = 0
        self.num_trades += 1
        
        logger.debug(f"Opened {'LONG' if position == 1 else 'SHORT'} at ${price:.2f}")
    
    def _close_position(self, price: float) -> float:
        """Close current position and return realized P&L."""
        if self.position == 0:
            return 0.0
        
        # Calculate P&L
        pnl = self._calculate_unrealized_pnl(price)
        
        # Apply transaction cost
        cost = self.position_size * self.transaction_cost
        pnl -= cost
        
        # Update balance
        self.balance += pnl
        self.total_pnl += pnl
        
        # Record trade
        self.trade_history.append({
            'position': 'LONG' if self.position == 1 else 'SHORT',
            'entry_price': self.entry_price,
            'exit_price': price,
            'pnl': pnl,
            'holding_time': self.holding_time
        })
        
        logger.debug(f"Closed position at ${price:.2f}, P&L: ${pnl:.2f}")
        
        # Reset position
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.leverage = 1.0
        self.unrealized_pnl = 0.0
        self.holding_time = 0
        
        return pnl
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L for open position."""
        if self.position == 0 or self.position_size == 0:
            return 0.0
        
        price_change = (current_price - self.entry_price) / self.entry_price
        
        if self.position == 1:  # Long
            pnl = self.position_size * price_change * self.max_leverage
        else:  # Short
            pnl = self.position_size * (-price_change) * self.max_leverage
        
        return pnl
    
    def calculate_reward(self, action: int, prev_balance: float, current_balance: float, position_closed: bool = False) -> float:
        """
        SIMPLIFIED REWARD FUNCTION
        Focus on PnL only, minimal penalties
        """
        # Ensure valid balance values
        prev_balance = max(0, prev_balance) if not np.isnan(prev_balance) and not np.isinf(prev_balance) else self.initial_balance
        current_balance = max(0, current_balance) if not np.isnan(current_balance) and not np.isinf(current_balance) else prev_balance
        
        # Calculate PnL percentage
        if prev_balance > 0:
            pnl_pct = (current_balance - prev_balance) / prev_balance
            # Clip to reasonable range
            pnl_pct = np.clip(pnl_pct, -1.0, 10.0)  # Max 100% loss, 1000% gain
        else:
            pnl_pct = 0.0
        
        # Base reward: Scale PnL to reasonable range
        # 1% profit = +10 reward, 1% loss = -10 reward
        reward = pnl_pct * 1000
        
        # CRITICAL: If no position and no trade, STRONG negative reward
        # This forces the agent to trade
        if action == 0 and self.position == 0:  # Holding with no position
            reward = -5.0  # VERY STRONG penalty - agent MUST trade
        
        # Bonus for opening positions (encourage exploration)
        if (action == 1 or action == 2) and self.position == 0:
            reward += 2.0  # Larger bonus for taking action (was 0.5)
        
        # Small transaction cost penalty (only when opening/closing)
        if position_closed:
            reward -= 0.4  # 0.04% transaction fee scaled
        
        # Bonus for successful trades
        if position_closed and pnl_pct > 0:
            reward += 5.0  # Bonus for profitable trade
        
        # Bonus/penalty at episode end
        if position_closed:
            total_return = (current_balance / self.initial_balance) - 1
            if total_return > 0.05:  # >5% profit
                reward += 20
            elif total_return < -0.05:  # >5% loss
                reward -= 20
        
        # Clip reward to prevent extreme values
        reward = np.clip(reward, -50, 50)
        
        return reward


class DeepRLStrategy:
    """
    Deep Reinforcement Learning strategy using PPO.
    """
    
    def __init__(self, config_path: str = 'config/strategy_params.yaml'):
        """
        Initialize Deep RL strategy.
        
        Args:
            config_path: Path to strategy configuration
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['deep_rl']
        self.model = None
        self.env = None
        
        logger.info("DeepRLStrategy initialized")
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        total_timesteps: int = 500_000
    ) -> Dict:
        """
        Train the PPO agent.
        
        Args:
            train_data: Training data with features
            val_data: Validation data
            total_timesteps: Number of training steps
            
        Returns:
            Training metrics
        """
        logger.info(f"Training Deep RL agent for {total_timesteps:,} timesteps...")
        
        try:
            # Create environment
            logger.info("Creating training environment...")
            train_env = lambda: CryptoTradingEnv(train_data)
            self.env = DummyVecEnv([train_env])
            logger.info("Training environment created successfully")
            
            # Evaluation environment
            logger.info("Creating evaluation environment...")
            eval_env = DummyVecEnv([lambda: CryptoTradingEnv(val_data)])
            logger.info("Evaluation environment created successfully")
            
            # Create PPO model
            logger.info("Creating PPO model...")
            import os
            os.makedirs('./logs/rl_tensorboard/', exist_ok=True)
            os.makedirs('./logs/rl_eval/', exist_ok=True)
            os.makedirs('./models/rl_best/', exist_ok=True)
            logger.info("Directories created")
            
            # Get params from config (handle both old and new structure)
            if 'model' in self.config and 'params' in self.config['model']:
                params = self.config['model']['params']
            elif 'params' in self.config:
                params = self.config['params']
            else:
                # Default optimized params
                params = {
                    'learning_rate': 0.00005,
                    'n_steps': 512,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_range': 0.2,
                    'ent_coef': 0.1,
                    'vf_coef': 0.5,
                    'max_grad_norm': 0.5
                }
            logger.info(f"PPO params: {params}")
            
            # Check environment
            logger.info(f"Environment observation space: {self.env.observation_space}")
            logger.info(f"Environment action space: {self.env.action_space}")
            
            logger.info("Initializing PPO model (this may take a moment)...")
            
            try:
                # Test environment thoroughly before PPO initialization
                logger.info("Testing environment reset...")
                test_obs = self.env.reset()
                logger.info(f"Environment reset successful, observation shape: {test_obs[0].shape if isinstance(test_obs, tuple) else test_obs.shape}")
                
                # Test a step to ensure environment works
                logger.info("Testing environment step...")
                test_action = [self.env.action_space.sample()]  # DummyVecEnv expects list
                test_result = self.env.step(test_action)
                obs, reward, done, info = test_result
                logger.info(f"Environment step successful: obs shape={obs.shape}, reward={reward[0] if isinstance(reward, np.ndarray) else reward}")
                
                # Use smaller network architecture to avoid initialization issues
                # 143 features is large, so use smaller hidden layers
                policy_kwargs = dict(
                    net_arch=[dict(pi=[128, 64], vf=[128, 64])]  # Smaller network: 128->64 instead of default 256->256
                )
                
                logger.info("Calling PPO constructor with optimized settings...")
                import sys
                import torch
                sys.stdout.flush()
                
                # Set PyTorch to use single thread to avoid deadlocks
                torch.set_num_threads(1)
                
                self.model = PPO(
                    policy='MlpPolicy',
                    env=self.env,
                    learning_rate=params['learning_rate'],
                    n_steps=params['n_steps'],
                    batch_size=params['batch_size'],
                    n_epochs=params['n_epochs'],
                    gamma=params['gamma'],
                    gae_lambda=params['gae_lambda'],
                    clip_range=params['clip_range'],
                    ent_coef=params['ent_coef'],
                    vf_coef=params['vf_coef'],
                    max_grad_norm=params['max_grad_norm'],
                    policy_kwargs=policy_kwargs,  # Smaller network
                    verbose=0,  # Reduce verbosity
                    tensorboard_log=None,  # Disable tensorboard to avoid hanging
                    device='cpu'  # Explicitly use CPU to avoid GPU issues
                )
                logger.info("PPO model created successfully")
                sys.stdout.flush()
            except Exception as e:
                logger.error(f"Error creating PPO model: {e}", exc_info=True)
                raise
            
            # Evaluation callback
            logger.info("Setting up evaluation callback...")
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path='./models/rl_best/',
                log_path='./logs/rl_eval/',
                eval_freq=10000,
                deterministic=True,
                render=False
            )
            
            # Reward logging callback
            try:
                from train import RewardLoggingCallback
                reward_callback = RewardLoggingCallback(check_freq=10000)
                from stable_baselines3.common.callbacks import CallbackList
                callbacks = CallbackList([eval_callback, reward_callback])
                logger.info("Reward logging callback added")
            except ImportError:
                callbacks = eval_callback
                logger.info("Reward logging callback not available, using eval callback only")
            
            logger.info("Starting PPO training...")
            
            # Train
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=False  # Disable progress bar (requires tqdm/rich)
            )
            
            logger.info("Deep RL training complete")
        except Exception as e:
            logger.error(f"Error during RL training: {e}", exc_info=True)
            raise
        
        return {'total_timesteps': total_timesteps}
    
    def generate_signal(
        self,
        features: pd.DataFrame,
        current_position: Dict
    ) -> Tuple[str, float, Optional[Dict]]:
        """
        Generate trading signal using trained PPO agent.
        
        Args:
            features: Feature DataFrame
            current_position: Current position info
            
        Returns:
            Tuple of (signal, confidence, trade_params)
        """
        if self.model is None:
            logger.warning("Model not trained!")
            return 'NEUTRAL', 0.0, None
        
        # Prepare observation
        obs = self._prepare_observation(features, current_position)
        
        # Predict action
        action, _states = self.model.predict(obs, deterministic=True)
        
        # Get action probabilities for confidence
        action_probs = self.model.policy.get_distribution(obs).distribution.probs.detach().numpy()[0]
        confidence = action_probs[action]
        
        # Map action to signal (Environment: 0=Close/Hold, 1=Long, 2=Short)
        action_map = {0: 'NEUTRAL', 1: 'LONG', 2: 'SHORT'}
        signal = action_map[action]
        
        # Build trade parameters
        trade_params = None
        if signal != 'NEUTRAL':
            current_price = features.iloc[-1]['close'] if 'close' in features else 50000
            trade_params = {
                'signal': signal,
                'entry_price': current_price,
                'confidence': float(confidence),
                'strategy': 'deep_rl',
                'max_leverage': self.config['max_leverage']
            }
        
        logger.debug(f"RL Signal: {signal}, Confidence: {confidence:.2f}")
        
        return signal, float(confidence), trade_params
    
    def _prepare_observation(self, features: pd.DataFrame, current_position: Dict) -> np.ndarray:
        """Prepare observation for model."""
        # Market features
        market_features = features.iloc[-1].values
        
        # Position features
        position_features = np.array([
            current_position.get('position', 0),
            current_position.get('size', 0),
            current_position.get('unrealized_pnl', 0),
            current_position.get('leverage', 0),
            current_position.get('holding_time', 0)
        ])
        
        obs = np.concatenate([market_features, position_features]).astype(np.float32)
        
        return obs
    
    def save_model(self, path: str):
        """Save PPO model."""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        self.model.save(path)
        logger.info(f"RL model saved to {path}")
    
    def load_model(self, path: str):
        """Load PPO model."""
        self.model = PPO.load(path)
        logger.info(f"RL model loaded from {path}")


if __name__ == "__main__":
    print("Testing Deep RL Strategy...")
    
    # Create synthetic data
    n_steps = 1000
    dates = pd.date_range('2024-01-01', periods=n_steps, freq='15min')
    
    df = pd.DataFrame({
        'close': 50000 + np.cumsum(np.random.randn(n_steps) * 100),
        'volume': np.random.uniform(100, 1000, n_steps),
        'rsi_14': np.random.uniform(30, 70, n_steps),
        'atr_14': np.random.uniform(500, 1500, n_steps)
    }, index=dates)
    
    # Add more features
    for i in range(10):
        df[f'feature_{i}'] = np.random.randn(n_steps)
    
    # Test environment
    print("\n=== Testing Trading Environment ===")
    env = CryptoTradingEnv(df, initial_balance=10000)
    obs, info = env.reset()
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    
    # Run a few steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    
    print(f"Balance after 10 steps: ${info['balance']:.2f}")
    print(f"Total P&L: ${info['total_pnl']:.2f}")
    
    # Test strategy (without full training)
    print("\n=== Testing RL Strategy ===")
    strategy = DeepRLStrategy()
    print("Strategy initialized successfully")
    
    print("\n✅ Deep RL strategy test complete")
    print("Note: Full training requires 1M timesteps and takes several hours")

