from project.gavinwheat import GavinWheat
import gymnasium as gym
import numpy as np

class CooperativeGavinWheat(GavinWheat):
    """Extension of GavinWheat with cooperative observation space"""
    
    def _get_observation_space(self):
        nvars = (
            len(self.crop_features)
            + len(self.action_features)
            + len(self.weather_features) * self.timestep
        )
        return gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(nvars,),
            dtype=np.float32
        )