import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import logging
from gymnasium import spaces
from datetime import datetime
from collections import deque

class CooperativeFarmEnv(gym.Env):
    """Single farm environment with cooperation mechanisms"""
    def __init__(self, farm_id: int, market, shared_resources, knowledge_pool,
                 crop_features: List[str], weather_features: List[str],
                 action_features: List[str], costs_nitrogen: float,
                 years: List[int], locations: List[Tuple[float, float]],
                 action_space: spaces.Box, action_multiplier: float = 1.0,
                 seed: Optional[int] = None, reward: Optional[str] = None,
                 max_yield: float = 1000.0, max_steps: int = 100,
                 exclude_crop_features=None, exclude_weather_features=None,
                 delay_steps=0, **kwargs):
        super().__init__()
        
        # Default to empty list if no exclusions are provided
        if exclude_crop_features is None:
            exclude_crop_features = []
        if exclude_weather_features is None:
            exclude_weather_features = []

        # Define original features
        self.original_crop_features = ['DVS', 'TGROWTH', 'LAI', 'NUPTT', 'TRAN', 'TNSOIL', 'TRAIN', 'TRANRF', 'WSO']
        self.original_weather_features = ['IRRAD', 'TMIN', 'RAIN']

        # Set excluded features (mask them during observation)
        self.exclude_crop_features = exclude_crop_features
        self.exclude_weather_features = exclude_weather_features

        # Create masks for excluded features
        self.crop_feature_mask = [feature not in self.exclude_crop_features for feature in self.original_crop_features]
        self.weather_feature_mask = [feature not in self.exclude_weather_features for feature in self.original_weather_features]


        # Store cooperative components
        self.farm_id = farm_id
        self.market = market
        self.shared_resources = shared_resources
        self.knowledge_pool = knowledge_pool
        
        # Store farm parameters
        self.crop_features = crop_features
        self.weather_features = weather_features
        self.action_features = action_features
        self.costs_nitrogen = costs_nitrogen
        self.years = sorted(years) 
        self.locations = locations
        self.action_multiplier = action_multiplier
        self.reward_type = reward
        self.max_yield = max_yield
        self.max_steps = max_steps
        self.current_step = 0
        
        # Initialize date tracking
        self.start_date = datetime(min(self.years), 1, 1)
        self.days_per_step = 365 // max_steps
        self._date = self.start_date  # Use private attribute for date
        
        # Initialize location tracking
        self.current_location_idx = 0
        self._loc = self.locations[0] if self.locations else (0, 0)
        
        # RNG
        self.np_random = None
        if seed is not None:
            self.reset(seed=seed)
        
        # Define action space
        self.action_space = action_space
        
        # Define observation space as a Dict space
        obs_spaces = {
            'crop_state': spaces.Box(
                low=0.0, high=1.0,
                shape=(len(crop_features),),
                dtype=np.float32
            ),
            'weather': spaces.Box(
                low=0.0, high=1.0,
                shape=(len(weather_features),),
                dtype=np.float32
            ),
            # Previous action features - always use shape (1,) for consistency
            'prev_action': spaces.Box(
                low=0.0, high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            # Cooperative features
            'cooperative': spaces.Box(
                low=0.0, high=1.0,
                shape=(3,),  # market_price, total_supply, sharing_farms
                dtype=np.float32
            )
        }
        self.observation_space = spaces.Dict(obs_spaces)
        
        self.logger = logging.getLogger(__name__)


        # Set delay steps
        self.delay_steps = delay_steps
        self.observation_buffer = deque(maxlen=delay_steps)  # Buffer to store past observations
        
        
        # Initialize farm state
        self._initialize_farm_state()

        print(f"FARM {self.farm_id}\nPERCEPTION DELAY: {self.delay_steps}\nFEATURES:\ncrops: {self.crop_features} EXCLUDING: {self.exclude_crop_features}\nweather: {self.weather_features} EXCLUDING: {self.exclude_weather_features}\nactions:{self.action_features}")



    @property
    def date(self):
        """Get current date"""
        return self._date
        
    @date.setter
    def date(self, value):
        """Set current date"""
        self._date = value
        
    @property
    def loc(self):
        """Get current location"""
        return self._loc
        
    @loc.setter
    def loc(self, value):
        """Set current location"""
        self._loc = value
        
    def _initialize_farm_state(self):
        """Initialize farm-specific state variables"""
        self.current_year_idx = 0
        self.current_location_idx = 0
        if self.np_random is not None:
            self.current_weather = self.np_random.random(len(self.weather_features))
            self.current_crop_state = self.np_random.random(len(self.crop_features))
        else:
            self.current_weather = np.random.random(len(self.weather_features))
            self.current_crop_state = np.random.random(len(self.crop_features))
        self.previous_action = np.zeros(1, dtype=np.float32)
        self.previous_yield = 0.0
        
        # Reset date and location
        self._date = datetime(self.years[self.current_year_idx], 1, 1)
        self._loc = self.locations[self.current_location_idx] if self.locations else (0, 0)
        
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        try:
            self.current_step += 1
            

            # Update farm state
            self._update_farm_state(action)
            
            # Calculate base yield with clipping
            base_yield = np.clip(self._calculate_yield(action), 0.0, self.max_yield)
            
            # Apply knowledge bonus with safety checks
            knowledge_bonus = self.knowledge_pool.get_knowledge_bonus(
                self._get_flat_observation(), action
            )
            knowledge_bonus = np.clip(knowledge_bonus, -0.5, 2.0)  # Limit bonus range
            current_yield = base_yield * (1.0 + knowledge_bonus)
            current_yield = np.clip(current_yield, 0.0, self.max_yield)
            
            # Update market and get price with safety checks
            try:
                market_price = np.clip(
                    self.market.update_price(current_yield),
                    self.market.min_price,
                    self.market.base_price * 2.0
                )
            except Exception as e:
                self.logger.error(f"Market update failed: {e}")
                market_price = np.clip(
                    self.market.base_price,
                    self.market.min_price,
                    self.market.base_price * 2.0
                )
            
            # Calculate costs with safety checks
            try:
                operating_cost = np.clip(
                    self.shared_resources.get_cost(self.farm_id),
                    0.0,
                    self.costs_nitrogen * 10.0
                )
                fertilizer_cost = np.clip(
                    self.costs_nitrogen * np.sum(action),
                    0.0,
                    self.costs_nitrogen * 10.0
                )
                total_cost = operating_cost + fertilizer_cost
            except Exception as e:
                self.logger.error(f"Cost calculation failed: {e}")
                total_cost = self.costs_nitrogen * np.sum(action)
            
            # Calculate reward with safety checks
            market_reward = np.clip(current_yield * market_price, 0.0, self.max_yield * self.market.base_price * 2.0)
            final_reward = np.clip(market_reward - total_cost, -1e6, 1e6)
            
            # Update knowledge pool
            self.knowledge_pool.add_experience(
                self._get_flat_observation(), action, final_reward
            )
            
            # Update cooperation status
            self.shared_resources.update_sharing_status(
                self.farm_id, 
                final_reward > 0  # Share if profitable
            )
            
            # Update state
            self.previous_yield = current_yield
            self.previous_action = np.array([np.mean(action)], dtype=np.float32)
            
            # Create observation

            current_observation = self._get_observation()
    
            # Add the current observation to the buffer for delayed observation
            self.observation_buffer.append(current_observation)
            
            # If delay > 0, return the delayed observation from the buffer
            if self.delay_steps > 0 and len(self.observation_buffer) >= self.delay_steps:
                delayed_observation = self.observation_buffer[0]  # Oldest observation in the buffer
            else:
                delayed_observation = current_observation  # No delay, return current observation
            
            # Check for episode termination
            terminated = self.current_step >= self.max_steps
            truncated = False
            
            info = {
                'market_price': float(market_price),
                'operating_cost': float(operating_cost),
                'fertilizer_cost': float(fertilizer_cost),
                'knowledge_bonus': float(knowledge_bonus),
                'yield': float(current_yield)
            }
            

            return delayed_observation, float(final_reward), terminated, truncated, info
        
        except Exception as e:
            self.logger.error(f"Error in step: {e}")
            return self._get_observation(), 0.0, True, False, {}


    
    def _update_farm_state(self, action: np.ndarray) -> None:
        """Update farm state based on action and time"""
        # Update weather
        if self.np_random is not None:
            self.current_weather = self.np_random.random(len(self.weather_features))
        else:
            self.current_weather = np.random.random(len(self.weather_features))
        
        # Update crop state
        self.current_crop_state = np.clip(
            self.current_crop_state + 0.1 * (action - 0.5),
            0.0, 1.0
        )
        
        # Update year and location indices
        if self.current_step % (self.max_steps // len(self.years)) == 0:
            self.current_year_idx = (self.current_year_idx + 1) % len(self.years)
            self.current_location_idx = (self.current_location_idx + 1) % len(self.locations)
            # Update date and location
            self._date = datetime(self.years[self.current_year_idx], self.date.month, self.date.day)
            self._loc = self.locations[self.current_location_idx] if self.locations else (0, 0)
    
    def _calculate_yield(self, action: np.ndarray) -> float:
        """Calculate crop yield based on current state and action"""
        # Clip input factors to prevent extreme values
        weather_factor = np.clip(np.mean(self.current_weather), 0.0, 1.0)
        crop_factor = np.clip(np.mean(self.current_crop_state), 0.0, 1.0)
        action_factor = np.clip(np.mean(action), 0.0, 1.0)
        
        # Calculate yield with safety checks
        base_yield = self.max_yield * weather_factor * crop_factor * action_factor
        base_yield = np.clip(base_yield * self.action_multiplier, 0.0, self.max_yield)
        
        return float(base_yield)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Create observation dictionary"""
        # Normalize and bound crop state
        crop_state = np.clip(self.current_crop_state, 0.0, 1.0)
        crop_state = np.nan_to_num(crop_state, nan=0.0)
        
        # Normalize and bound weather
        weather = np.clip(self.current_weather, 0.0, 1.0)
        weather = np.nan_to_num(weather, nan=0.0)
        
        # Normalize and bound previous action
        prev_action = np.clip(self.previous_action, 0.0, 1.0)
        prev_action = np.nan_to_num(prev_action, nan=0.0)
        

        # Mask excluded crop and weather features (set them to NaN or 0.0)
        full_crop_state = np.copy(crop_state)  # Copy the original state
        full_weather = np.copy(weather)  # Copy the original state
        
        # Apply masks: set excluded crop/weather features to NaN or 0.0
        full_crop_state[~np.array(self.crop_feature_mask)] = 0.0  # Exclude features (set to NaN or 0.0)
        full_weather[~np.array(self.weather_feature_mask)] = 0.0  # Exclude features (set to NaN or 0.0)
 
        # Calculate cooperative features with numerical stability
        market_price = float(self.market.price_history[-1] if self.market.price_history else self.market.base_price)
        market_price = np.clip(market_price / (2 * self.market.base_price), 0.0, 1.0)
        
        total_supply = float(self.market.total_supply)
        max_possible_supply = float(self.max_yield * self.max_steps)
        supply_ratio = np.clip(total_supply / max_possible_supply, 0.0, 1.0)
        
        sharing_ratio = np.clip(float(len(self.shared_resources.sharing_farms)) / float(self.max_steps), 0.0, 1.0)
        
        cooperative_features = np.array([
            market_price,
            supply_ratio,
            sharing_ratio
        ], dtype=np.float32)
        
        # Handle any potential NaN values
        cooperative_features = np.nan_to_num(cooperative_features, nan=0.0)
        
        obs = {
            'crop_state': crop_state.astype(np.float32),
            'weather': weather.astype(np.float32),
            'prev_action': prev_action.astype(np.float32),
            'cooperative': cooperative_features
        }
        
        # Ensure all arrays are properly shaped and bounded
        for key, value in obs.items():
            if not isinstance(value, np.ndarray):
                obs[key] = np.array(value, dtype=np.float32)
            if len(value.shape) == 0:
                obs[key] = value.reshape(1)
            elif len(value.shape) == 1:
                obs[key] = value.reshape(-1)
            # Final safety check for bounds
            obs[key] = np.clip(obs[key], 0.0, 1.0)
                
        return obs
    
    def _get_flat_observation(self) -> np.ndarray:
        """Get flattened observation for knowledge pool"""
        obs = self._get_observation()
        flat_obs = np.concatenate([
            obs[key].flatten()
            for key in sorted(obs.keys())
        ])
        return flat_obs
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment and optionally set a new random seed"""
        super().reset(seed=seed)
        
        # Initialize RNG if seed is provided
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        self.current_step = 0
        
        # Reset components
        self.market.reset()
        self.shared_resources.reset()
        self.knowledge_pool.reset()
        
        # Reset farm state
        self._initialize_farm_state()
        
        return self._get_observation(), {}

